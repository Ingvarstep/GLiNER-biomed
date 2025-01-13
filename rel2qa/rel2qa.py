import argparse
import os
import json
from typing import Any, List, Dict
from copy import deepcopy
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import ast


def load_files_from_folder(folder_path: str) -> List[str]:
    """
    Loads all JSON file paths from a specified folder.

    Args:
        folder_path (str): Path to the folder containing JSON files.

    Returns:
        List[str]: A list of file paths for all JSON files in the folder.
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"The folder '{folder_path}' does not exist.")

    # List all JSON files in the directory
    batch_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".json")]
    if not batch_files:
        raise ValueError(f"No JSON files found in the folder '{folder_path}'.")
    
    return batch_files


def extract_triplets(generated_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract triplets from the generated JSON with strict validation for head, tail, and relation.

    Args:
        generated_json (dict): A JSON structure that must contain
                               'entities' (list) and 'relations' (list).

    Returns:
        List[dict]: A list of triplets, where each triplet is represented
                    by a dictionary containing "id", "from", "relation", and "to".
    """
    try:
        # Ensure the input JSON has required keys
        if 'entities' not in generated_json or 'relations' not in generated_json:
            raise ValueError("Input JSON must contain 'entities' and 'relations' keys.")

        # Ensure entities and relations are lists
        if not isinstance(generated_json['entities'], list) or not isinstance(generated_json['relations'], list):
            raise ValueError("'entities' and 'relations' must be lists.")

        # Create a dictionary for entity lookup by ID
        entity_dict = {
            entity['id']: entity['text']
            for entity in generated_json['entities']
            if 'id' in entity and 'text' in entity
        }

        # Extract triplets
        triplets = []
        counter = 0
        for relation in generated_json['relations']:
            # Check for required keys in each relation
            if 'head' not in relation or 'tail' not in relation or 'type' not in relation:
                continue

            head_id = relation['head']
            tail_id = relation['tail']
            relation_type = relation['type']

            # Validate the presence of head and tail in the entity dictionary
            if head_id not in entity_dict or tail_id not in entity_dict:
                continue  # Skip relations with invalid head or tail

            # Add the triplet
            triplets.append({
                "id": counter,
                "from": entity_dict[head_id],
                "relation": relation_type,
                "to": entity_dict[tail_id]
            })
            counter += 1

        return triplets

    except Exception as e:
        print(f"An error occurred while processing the JSON: {e}")
        return []


def process_batch(file_path: str, output_folder: str, llm, tokenizer, sampling_params, fs_chat):
    """
    Process a single JSON batch file to extract triplets and generate QA pairs.

    Args:
        file_path (str): Path to the input batch file.
        output_folder (str): Path to the output folder for saving results.
        llm: LLM model instance for QA generation.
        tokenizer: Tokenizer instance for preparing model input.
        sampling_params: Sampling parameters for the LLM.
        fs_chat: Predefined system chat template.
    """
    try:
        # Load the batch data
        with open(file_path, "r", encoding="utf-8") as f:
            instances = json.load(f)

        updated_instances = []
        queries = []  # Collect all queries for batch processing

        # Step 1: Prepare all queries
        for instance in instances:
            input_text = instance.get("input_text", "")
            generated_json = instance.get("generated_json", {})

            # Extract triplets
            triplets = extract_triplets(generated_json)

            # Build the LLM conversation
            run_chat = deepcopy(fs_chat)
            run_chat.append({
                "role": "user",
                "content": f"input_text = \"{input_text}\"\ntriplets = {json.dumps(triplets)}"
            })

            # Convert the conversation into the model's prompt
            query = tokenizer.apply_chat_template(run_chat, tokenize=False, add_generation_prompt=True)
            queries.append(query)

        # Step 2: Generate QA pairs for all queries
        outputs = llm.generate(queries, sampling_params)

        # Step 3: Map results back to instances
        for instance, output in zip(instances, outputs):
            try:
                qa_result = output.outputs[0].text.strip()  # Extract the first output
                instance["QA"] = ast.literal_eval(qa_result)  # Safely evaluate the string as a Python object
                updated_instances.append(instance)  # Add instance only if parsing succeeds
            except (ValueError, SyntaxError) as e:
                # Skip the instance if parsing fails
                print(f"Skipping instance due to parsing error: {e}")

        # Step 4: Save the updated batch
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"{os.path.basename(file_path).replace('.json', '_qa.json')}")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(updated_instances, f, indent=2, ensure_ascii=False)

        print(f"Saved results to {output_file}.")

    except Exception as e:
        print(f"Error processing batch {file_path}: {e}")



def main():
    """
    Main pipeline function for batch-wise processing and saving.
    """
    # Argument parser
    parser = argparse.ArgumentParser(description="Batch process JSON files to extract triplets and generate QA pairs.")
    parser.add_argument(
        "--input_folder",
        type=str,
        default="./data/synthetic_data/sample",
        help="Path to the input folder containing JSON files. Default is './data/synthetic_data/sample'."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./results",
        help="Path to the output folder for saving processed JSON files. Default is './results'."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/users/y/yazdani0/ds4dhxkg/Llama-3.1-8B-Instruct-bio-awq",
        help="Path to the model used for LLM processing. Default is '/home/users/y/yazdani0/ds4dhxkg/Llama-3.1-8B-Instruct-bio-awq'."
    )
    args = parser.parse_args()

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Load the list of batch files
    batch_files = load_files_from_folder(args.input_folder)
    print(f"Found {len(batch_files)} batch files in '{args.input_folder}'.")

    # Sampling parameters for vLLM
    sampling_params = SamplingParams(
        n=1,
        best_of=1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        seed=42,
        max_tokens=2048,
    )

    # Instantiate the LLM and tokenizer
    llm = LLM(
        model=args.model_path,
        max_model_len=16_000
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Predefined FS chat prompt
    fs_chat = [
        {"role": "system", "content": """You are an expert assistant designed to convert structured entity-relation triplet data into concise question-answer (QA) pairs. Your task is to generate clear and meaningful questions (`q`) from the `from` and `relation` fields of each triplet, with answers (`a`) directly derived from the `to` field.
    
    Follow these guidelines:
    1. Use the `from` and `relation` fields to craft precise and logically sound questions. The structure of the question should closely reflect the relation in the triplet, even if this results in similar or repeated phrasing for similar triplets.
    2. The `to` field should be directly used as the answer (`a`) for the question.
    3. Maintain a consistent style for clarity, ensuring the phrasing remains uniform for similar relations across triplets.
    4. Include the triplet `id` in each QA pair for traceability.
    5. If the `input_text` provides useful context to disambiguate relationships, ensure the generated QA aligns logically with it.
    
    Your output should strictly follow this format:
    ```json
    [{"id": <triplet_id>, "q": "<generated_question>", "a": "<triplet_to>"}]
    ```"""},
        {
            "role": "user",
            "content": """input_text = \"In well-characterized samples of patients undergoing breast, caesarian-section or abdominal surgery, the efficacy and safety of perioperative pain care organized by ComfortSafe Pyramid approaches will be evaluated. In addition, patient satisfaction with this care will be evaluated. Post-operative pain and opioid-related outcomes will be operationalized using standardized symptom (nausea, vomiting, respiratory depression, confusion, acute and chronic pain) severity measures, prn analgesic (opioid and non-opioid) consumption, and urine toxicology. Together, these original data will have important implications for the clinical management of post-operative pain utilizing opioid-sparing and multi-modal approaches.\n\nA total of 60 self-selected surgical patients will be enrolled in the study, 20 undergoing breast surgery, 20 undergoing caesarian-section, and 20 undergoing abdominal surgery. All will receive anesthesia and pain management consistent with the ComfortSafe Pyramid which emphasizes opioid-sparing and multimodal approaches. Side and adverse opioid-related effects and post-operative pain will be assessed at regular intervals during the immediate post-operative period (24 hours). To evaluate the incidence of opioid misuse/abuse and the development of chronic pain, an opioid use history, urine toxicology and Brief Pain Inventory will be collected at 1 month, 3 months and 6 months following discharge. Patient satisfaction measures will be obtained at discharge, 1 month, and 3 months follow-up timepoint.\".\ntriplets = [{"id": 0, "from": "ComfortSafe Pyramid", "relation": "organizes", "to": "perioperative pain care"}, {"id": 1, "from": "perioperative pain care", "relation": "evaluated by", "to": "patient satisfaction"}, {"id": 2, "from": "perioperative pain care", "relation": "evaluated by", "to": "post-operative pain"}, {"id": 3, "from": "perioperative pain care", "relation": "evaluated by", "to": "opioid-related outcomes"}, {"id": 4, "from": "post-operative pain", "relation": "includes", "to": "nausea"}, {"id": 5, "from": "post-operative pain", "relation": "includes", "to": "vomiting"}, {"id": 6, "from": "post-operative pain", "relation": "includes", "to": "respiratory depression"}, {"id": 7, "from": "post-operative pain", "relation": "includes", "to": "confusion"}, {"id": 8, "from": "post-operative pain", "relation": "includes", "to": "acute pain"}, {"id": 9, "from": "post-operative pain", "relation": "includes", "to": "chronic pain"}, {"id": 10, "from": "opioid-related outcomes", "relation": "operationalized by", "to": "prn analgesic consumption"}, {"id": 11, "from": "opioid-related outcomes", "relation": "operationalized by", "to": "urine toxicology"}, {"id": 12, "from": "ComfortSafe Pyramid", "relation": "emphasizes", "to": "opioid-sparing"}, {"id": 13, "from": "ComfortSafe Pyramid", "relation": "emphasizes", "to": "multi-modal approaches"}, {"id": 14, "from": "60 surgical patients", "relation": "undergoing", "to": "breast surgery"}, {"id": 15, "from": "60 surgical patients", "relation": "undergoing", "to": "caesarian-section"}, {"id": 16, "from": "60 surgical patients", "relation": "undergoing", "to": "abdominal surgery"}, {"id": 17, "from": "anesthesia", "relation": "received by", "to": "60 surgical patients"}]""",
        },
        {
            "role": "assistant",
            "content": """[{"id": 0, "q": "What does the ComfortSafe Pyramid organize?", "a": "perioperative pain care"}, {"id": 1, "q": "What is perioperative pain care evaluated by?", "a": "patient satisfaction"}, {"id": 2, "q": "What is perioperative pain care evaluated by?", "a": "post-operative pain"}, {"id": 3, "q": "What is perioperative pain care evaluated by?", "a": "opioid-related outcomes"}, {"id": 4, "q": "What does post-operative pain include?", "a": "nausea"}, {"id": 5, "q": "What does post-operative pain include?", "a": "vomiting"}, {"id": 6, "q": "What does post-operative pain include?", "a": "respiratory depression"}, {"id": 7, "q": "What does post-operative pain include?", "a": "confusion"}, {"id": 8, "q": "What does post-operative pain include?", "a": "acute pain"}, {"id": 9, "q": "What does post-operative pain include?", "a": "chronic pain"}, {"id": 10, "q": "What are opioid-related outcomes operationalized by?", "a": "prn analgesic consumption"}, {"id": 11, "q": "What are opioid-related outcomes operationalized by?", "a": "urine toxicology"}, {"id": 12, "q": "What does the ComfortSafe Pyramid emphasize?", "a": "opioid-sparing"}, {"id": 13, "q": "What does the ComfortSafe Pyramid emphasize?", "a": "multi-modal approaches"}, {"id": 14, "q": "What are 60 surgical patients undergoing?", "a": "breast surgery"}, {"id": 15, "q": "What are 60 surgical patients undergoing?", "a": "caesarian-section"}, {"id": 16, "q": "What are 60 surgical patients undergoing?", "a": "abdominal surgery"}, {"id": 17, "q": "Who receives anesthesia?", "a": "60 surgical patients"}]""",
        },
        {
            "role": "user",
            "content": """input_text = \"Disclosed are compositions that include anti-CD74 immunoconjugates and optionally a therapeutic and/or diagnostic agent. In preferred embodiments, the immunoconjugates comprise one or more anti-CD74 antibodies or antigen-binding fragments thereof, conjugated to a carrier such as a polymer, nanoparticle, complex or micelle. Also disclosed are methods for preparing the immunoconjugates and using the immunoconjugates in diagnostic and therapeutic procedures. In certain preferred embodiments, the therapeutic methods comprise administering to a subject with a CD74-expressing disease an anti-CD74 immunoconjugate and thereby inducing cell death of CD74-expressing cells. In more preferred embodiments, the CD74 immunoconjugate is capable of inducing cell death in the absence of any other therapeutic agent, although such agents may be optionally administered prior to, together with or subsequent to administration of the anti-CD74 immunoconjugate. The compositions may be part of a kit for administering the anti-CD74 immunoconjugates or compositions.\".\ntriplets = [{"id": 0, "from": "anti-CD74 immunoconjugates", "relation": "includes", "to": "therapeutic and/or diagnostic agent"}, {"id": 1, "from": "anti-CD74 immunoconjugates", "relation": "comprises", "to": "anti-CD74 antibodies"}, {"id": 2, "from": "anti-CD74 immunoconjugates", "relation": "comprises", "to": "antigen-binding fragments"}, {"id": 3, "from": "anti-CD74 immunoconjugates", "relation": "conjugated to", "to": "carrier"}, {"id": 4, "from": "carrier", "relation": "such as", "to": "polymer"}, {"id": 5, "from": "carrier", "relation": "such as", "to": "nanoparticle"}, {"id": 6, "from": "carrier", "relation": "such as", "to": "complex"}, {"id": 7, "from": "carrier", "relation": "such as", "to": "micelle"}, {"id": 8, "from": "anti-CD74 immunoconjugates", "relation": "targets", "to": "CD74-expressing disease"}, {"id": 9, "from": "subject", "relation": "has", "to": "CD74-expressing disease"}, {"id": 10, "from": "anti-CD74 immunoconjugates", "relation": "administered to", "to": "subject"}]""",
        },
        {
            "role": "assistant",
            "content": """[{"id": 0, "q": "What do anti-CD74 immunoconjugates include?", "a": "therapeutic and/or diagnostic agent"}, {"id": 1, "q": "What do anti-CD74 immunoconjugates comprise?", "a": "anti-CD74 antibodies"}, {"id": 2, "q": "What do anti-CD74 immunoconjugates comprise?", "a": "antigen-binding fragments"}, {"id": 3, "q": "What are anti-CD74 immunoconjugates conjugated to?", "a": "carrier"}, {"id": 4, "q": "What is an example of a carrier?", "a": "polymer"}, {"id": 5, "q": "What is an example of a carrier?", "a": "nanoparticle"}, {"id": 6, "q": "What is an example of a carrier?", "a": "complex"}, {"id": 7, "q": "What is an example of a carrier?", "a": "micelle"}, {"id": 8, "q": "What do anti-CD74 immunoconjugates target?", "a": "CD74-expressing disease"}, {"id": 9, "q": "What does a subject have?", "a": "CD74-expressing disease"}, {"id": 10, "q": "To whom are anti-CD74 immunoconjugates administered?", "a": "subject"}]""",
        },
        {
            "role": "user",
            "content": """input_text = \"In a 104-week carcinogenicity study in rats at doses up to 120 mg/kg/day, which is approximately 90 times clinical exposure on a mg/m 2basis, the incidences of benign adenomas of the thyroid and liver were increased in male rats. No evidence of carcinogenicity was found in female rats. The higher incidences of thyroid and liver tumors in male rats were not seen at lower dose of 30 mg/kg/day and 60 mg/kg/day respectively (which are over 20 and 45 times, respectively, clinical exposures on a mg/m 2basis). In a 104-week carcinogenicity study in mice at doses up to 500 mg/kg/day, no evidence of carcinogenicity was found in mice (which is approximately 187 times clinical exposure on a mg/m 2basis). \n\nRepaglinide was non-genotoxic in a battery of in vivoand in vitrostudies: Bacterial mutagenesis (Ames test), in vitro forward cell mutation assay in V79 cells (HGPRT), in vitrochromosomal aberration assay in human lymphocytes, unscheduled and replicating DNA synthesis in rat liver, and in vivo mouse and rat micronucleus tests. \n\nIn a rat fertility study, repaglinide was administered to male and female rats at doses up to 300 and 80 mg/kg/day, respectively. No adverse effects on fertility were observed (which are over 60 times clinical exposure on a mg/m 2basis).\".\ntriplets = [{"id": 0, "from": "carcinogenicity study", "relation": "conducted on", "to": "rats"}, {"id": 1, "from": "carcinogenicity study", "relation": "administered at", "to": "doses up to 120 mg/kg/day"}, {"id": 2, "from": "doses up to 120 mg/kg/day", "relation": "related to", "to": "clinical exposure"}, {"id": 3, "from": "benign adenomas of the thyroid and liver", "relation": "observed in", "to": "male rats"}, {"id": 4, "from": "benign adenomas of the thyroid and liver", "relation": "not observed in", "to": "female rats"}, {"id": 5, "from": "lower dose of 30 mg/kg/day and 60 mg/kg/day", "relation": "associated with lower incidence of", "to": "benign adenomas of the thyroid and liver"}, {"id": 6, "from": "mice", "relation": "administered at", "to": "doses up to 500 mg/kg/day"}, {"id": 7, "from": "doses up to 500 mg/kg/day", "relation": "related to", "to": "clinical exposure"}, {"id": 8, "from": "Repaglinide", "relation": "assessed as", "to": "non-genotoxic"}, {"id": 9, "from": "non-genotoxic", "relation": "based on", "to": "in vivo and in vitro studies"}, {"id": 10, "from": "in vivo and in vitro studies", "relation": "includes", "to": "Ames test"}, {"id": 11, "from": "in vivo and in vitro studies", "relation": "includes", "to": "in vitro forward cell mutation assay in V79 cells (HGPRT) and in vitro chromosomal aberration assay in human lymphocytes"}, {"id": 12, "from": "in vivo and in vitro studies", "relation": "includes", "to": "unscheduled and replicating DNA synthesis in rat liver"}, {"id": 13, "from": "in vivo and in vitro studies", "relation": "includes", "to": "in vivo mouse and rat micronucleus tests"}, {"id": 14, "from": "rat fertility study", "relation": "conducted on", "to": "male and female rats"}, {"id": 15, "from": "rat fertility study", "relation": "administered at", "to": "doses up to 300 and 80 mg/kg/day"}]""",
        },
        {
            "role": "assistant",
            "content": """[{"id": 0, "q": "On which species was the carcinogenicity study conducted?", "a": "rats"}, {"id": 1, "q": "At what doses was the carcinogenicity study administered?", "a": "doses up to 120 mg/kg/day"}, {"id": 2, "q": "What were doses up to 120 mg/kg/day related to?", "a": "clinical exposure"}, {"id": 3, "q": "In which species were benign adenomas of the thyroid and liver observed?", "a": "male rats"}, {"id": 4, "q": "In which species were benign adenomas of the thyroid and liver not observed?", "a": "female rats"}, {"id": 5, "q": "What was associated with lower incidences of benign adenomas of the thyroid and liver?", "a": "lower dose of 30 mg/kg/day and 60 mg/kg/day"}, {"id": 6, "q": "At what doses were mice administered?", "a": "doses up to 500 mg/kg/day"}, {"id": 7, "q": "What were doses up to 500 mg/kg/day related to?", "a": "clinical exposure"}, {"id": 8, "q": "How was Repaglinide assessed?", "a": "non-genotoxic"}, {"id": 9, "q": "What was non-genotoxic based on?", "a": "in vivo and in vitro studies"}, {"id": 10, "q": "What do in vivo and in vitro studies include?", "a": "Ames test"}, {"id": 11, "q": "What do in vivo and in vitro studies include?", "a": "in vitro forward cell mutation assay in V79 cells (HGPRT) and in vitro chromosomal aberration assay in human lymphocytes"}, {"id": 12, "q": "What do in vivo and in vitro studies include?", "a": "unscheduled and replicating DNA synthesis in rat liver"}, {"id": 13, "q": "What do in vivo and in vitro studies include?", "a": "in vivo mouse and rat micronucleus tests"}, {"id": 14, "q": "On which species was the rat fertility study conducted?", "a": "male and female rats"}, {"id": 15, "q": "At what doses was the rat fertility study administered?", "a": "doses up to 300 and 80 mg/kg/day"}]""",
        }
    ]
    
    # Process each batch
    for batch_file in batch_files:
        process_batch(batch_file, output_folder, llm, tokenizer, sampling_params, fs_chat)


if __name__ == "__main__":
    main()