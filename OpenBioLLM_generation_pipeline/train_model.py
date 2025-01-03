import os
import json
import copy
import logging
import argparse
from tqdm import tqdm

import bitsandbytes as bnb
import torch
from torch.utils.data import Dataset

from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
)

from messages import SINGLE_SYSTEM_MESSAGE, CHAT_TEMPLATE

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define the dataset class
class Text2JSONDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, max_length=1024):
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = self.collect_dataset()

    def __len__(self):
        return len(self.dataset)

    def collect_dataset(self):
        batches = [os.path.join(self.dataset_path, batch) 
                        for batch in os.listdir(self.dataset_path) if batch.endswith('.json')]

        dataset = []
        for batch in batches:
            with open(batch, 'r', encoding='utf-8') as f:
                data = json.load(f)
                dataset.extend(data)
        return dataset

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['input_text']

        output = json.dumps(item['generated_json'], indent=2)

        input_llm =  (
            f'Here is a text input: "{text}" '
            "Analyze this text, identify the entities, and extract their relationships as per your instructions."
        )

        chat = copy.deepcopy(SINGLE_SYSTEM_MESSAGE)
        chat.extend([
            {"role": "user", "content": input_llm}, 
            {"role": "assistant", "content": str(output)}
        ])

        input_ = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=False
        )

        inputs = self.tokenizer(
            input_, return_tensors="pt", max_length=self.max_length, truncation=True, padding='max_length'
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        labels_ids = input_ids.clone().squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels_ids
        }

# Helper function to find linear module names
def find_all_linear_names(model, quantize=False):
    cls = bnb.nn.Linear4bit if quantize else torch.nn.Linear

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

# Main function
def main(args):
    logger.info("Starting script with configuration: %s", args)

    QUANTIZE = args.quantize
    USE_LORA = args.use_lora
    
    model_path = args.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if QUANTIZE:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype="float32",
        quantization_config=bnb_config,
        trust_remote_code=True,
        token=args.hf_token,
        attn_implementation="flash_attention_2"
    )

    if USE_LORA:
        modules = find_all_linear_names(model, quantize=QUANTIZE)

        peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=modules,
        )
    else:
        peft_config = None

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.chat_template = CHAT_TEMPLATE

    train_dataset = Text2JSONDataset(args.train_data_path, tokenizer, max_length=args.max_length)
    test_dataset = Text2JSONDataset(args.test_data_path, tokenizer, max_length=args.max_length)

    logger.info("Dataset lengths - Train: %d, Test: %d", len(train_dataset), len(test_dataset))

    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_32bit",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=False,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to="none",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    trainer.train()

    if peft_config is not None:
        logger.info("Merging LoRA weights into the base model (since use_lora=True).")
        trainer.model = trainer.model.merge_and_unload()
        trainer.model.save_pretrained(os.path.join(args.output_dir, 'merged'))
        tokenizer.save_pretrained(os.path.join(args.output_dir, 'merged'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text2JSON Dataset Training Script")

    parser.add_argument('--model_path', type=str, required=True, help="Path to the model.")
    parser.add_argument('--train_data_path', type=str, required=True, help="Path to the training dataset.")
    parser.add_argument('--test_data_path', type=str, required=True, help="Path to the testing dataset.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save trained models.")
    parser.add_argument('--hf_token', type=str, required=True, help="Hugging Face authentication token.")
    parser.add_argument('--max_length', type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument('--num_train_epochs', type=int, default=3, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=2, help="Training batch size.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument('--gradient_checkpointing', action='store_true', help="Enable gradient checkpointing.")
    parser.add_argument('--learning_rate', type=float, default=3e-5, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay.")
    parser.add_argument('--fp16', action='store_true', help="Enable FP16 training.")
    parser.add_argument('--bf16', action='store_true', help="Enable BF16 training.")
    parser.add_argument('--max_grad_norm', type=float, default=0.9, help="Maximum gradient norm.")
    parser.add_argument('--max_steps', type=int, default=-1, help="Maximum training steps.")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument('--logging_steps', type=int, default=1, help="Logging steps.")
    parser.add_argument('--eval_steps', type=int, default=200, help="Evaluation steps.")
    parser.add_argument('--save_steps', type=int, default=100, help="Save steps.")
    parser.add_argument('--quantize', action='store_true', help="Enable quantization.")
    parser.add_argument('--use_lora', action='store_true', help="Enable LoRA training.")

    args = parser.parse_args()

    main(args)
