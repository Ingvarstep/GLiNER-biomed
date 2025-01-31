"""
GLiNER Model Evaluation Script

This script evaluates a GLiNER model's performance on a Named Entity Recognition (NER) task.
It calculates precision, recall, F1-score, and sample-level accuracy.

Usage:
1. Configure data paths and model parameters
2. Specify entity types to evaluate
3. Run: python evaluate_gliner.py

Output:
- JSON file with evaluation metrics (results.json)
- Raw prediction/ground truth comparisons (raw_results.json)
"""

import json
import torch
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from gliner import GLiNER
from gliner.data_processing.collator import DataCollator


class NEREvaluator:
    """Core evaluation logic for NER tasks"""
    def __init__(self, dataset: List[Dict], true_entities: List, pred_entities: List, 
                 span_based: bool = False):
        self.dataset = dataset
        self.true_entities = true_entities
        self.pred_entities = pred_entities
        self.span_based = span_based
        self.raw_comparisons = []
        
    def _process_ground_truth(self, tokens, entities: List) -> List:
        """Format ground truth entities"""
        processed = []
        for start, end, _, label in entities:
            label = ' '.join(label.split('_')).lower()
            if self.span_based:
                processed.append((start, end, label))
            else:
                entity_text = ' '.join(tokens[start:end+1])
                processed.append((entity_text, label))
        return processed

    def _process_predictions(self, tokens: List[str], entities: List) -> List:
        """Format predicted entities"""
        processed = []
        for start, end, label, _ in entities:
            label = ' '.join(label.split('_')).lower()
            if self.span_based:
                processed.append((start, end, label))
            else:
                entity_text = ' '.join(tokens[start:end+1])
                processed.append((entity_text, label))
        return processed

    def _align_data(self) -> tuple:
        """Align ground truth and predictions for comparison"""
        aligned_true, aligned_pred = [], []
        for idx, (true, pred) in enumerate(zip(self.true_entities, self.pred_entities)):
            tokens = self.dataset[idx]['tokenized_text']
            aligned_true.append(self._process_ground_truth(tokens, true))
            aligned_pred.append(self._process_predictions(tokens, pred))
        return aligned_true, aligned_pred

    def _calculate_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate precision, recall, and F1 per entity type"""
        true, pred = self._align_data()
        metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

        for t, p in zip(true, pred):
            true_set = set(t)
            pred_set = set(p)
            
            # Record raw comparisons for analysis
            self.raw_comparisons.append({
                'ground_truth': list(true_set),
                'predictions': list(pred_set)
            })

            # Update metrics counts
            for item in pred_set:
                metrics[item[2] if self.span_based else item[1]]['tp' if item in true_set else 'fp'] += 1
                
            for item in true_set:
                if item not in pred_set:
                    metrics[item[2] if self.span_based else item[1]]['fn'] += 1

        # Calculate final metrics
        results = {}
        for label, counts in metrics.items():
            tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[label] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'support': tp + fn
            }

        return results

    def evaluate(self) -> Dict:
        """Run full evaluation pipeline"""
        metrics = self._calculate_metrics()
        
        # Calculate sample-level accuracy
        correct = sum(1 for comp in self.raw_comparisons 
                     if set(comp['predictions']) == set(comp['ground_truth']))
        accuracy = correct / len(self.raw_comparisons) if self.raw_comparisons else 0
        
        # Save diagnostics
        with open('raw_results.json', 'w') as f:
            json.dump(self.raw_comparisons, f, indent=2)

        return {
            'entity_metrics': metrics,
            'accuracy': round(accuracy, 4),
            'total_samples': len(self.raw_comparisons)
        }


class GLiNEREvaluationPipeline:
    """End-to-end evaluation pipeline for GLiNER models"""
    def __init__(self, model_path: str, data_path: Union[str, Path], device: str = 'cuda:0'):
        self.model = GLiNER.from_pretrained(model_path).to(device)
        self.test_data = self._load_and_preprocess_data(data_path)
        
    def _load_and_preprocess_data(self, data_path: Union[str, Path]) -> List[Dict]:
        """Load and format evaluation data"""
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        processed = []
        for item in raw_data:
            tokens = item['tokenized_text']
            entities = [
                [start, end, ' '.join(tokens[start:end+1]), str(label)]
                for start, end, label in item['ner']
            ]
            if tokens and entities:
                processed.append({
                    'tokenized_text': tokens,
                    'ner': entities
                })
        return processed

    def run_evaluation(
        self,
        threshold: float = 0.5,
        batch_size: int = 8,
        flat_ner=False,
        multi_label=False,
        entity_types: Optional[List[str]] = None,
        span_evaluation: bool = False
    ) -> Dict:
        """Execute model evaluation with specified parameters"""
        self.model.eval()
        collator = DataCollator(
            self.model.config,
            data_processor=self.model.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            prepare_labels=False,
            entity_types=entity_types
        )

        loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=False
        )

        predictions, ground_truth = [], []
        for batch in tqdm(loader, desc="Evaluating batches"):

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.model.device)

            # Generate predictions
            with torch.no_grad():
                outputs = self.model(**batch)[0]

            if not isinstance(outputs, torch.Tensor):
                model_output = torch.from_numpy(model_output)

            # Decode model outputs
            decoded = self.model.decoder.decode(
                batch["tokens"],
                batch["id_to_classes"],
                outputs,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label
            )
            predictions.extend(decoded)
            ground_truth.extend(batch["entities"])

        # Calculate final metrics
        evaluator = NEREvaluator(
            dataset=self.test_data,
            true_entities=ground_truth,
            pred_entities=predictions,
            span_based=span_evaluation
        )
        return evaluator.evaluate()


if __name__ == "__main__":
    # Configuration
    CONFIG = {
        "model_path": "gliner-community/gliner_small-v2.5",
        "data_path": "your data in gliner format",
        "entity_types": None,  # List specific types to evaluate or None for all
        "threshold": 0.3,
        "batch_size": 1, #currently supported only 1
        "span_evaluation": False #true if you want to evaluate at the level of span positions 
    }

    # Initialize and run pipeline
    pipeline = GLiNEREvaluationPipeline(
        model_path=CONFIG["model_path"],
        data_path=CONFIG["data_path"]
    )
    
    results = pipeline.run_evaluation(
        threshold=CONFIG["threshold"],
        batch_size=CONFIG["batch_size"],
        entity_types=CONFIG["entity_types"],
        span_evaluation=CONFIG["span_evaluation"]
    )

    # Save and display results
    print("\nEvaluation Results:")
    print(f"Sample Accuracy: {results['accuracy']}")
    print(f"Total Samples: {results['total_samples']}")
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nPer-entity metrics saved in evaluation_results.json")