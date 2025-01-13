from datasets import load_dataset
from transformers import AutoTokenizer
import random

# Load the Medical LLaMA3 Instruct Dataset (Short)
dataset_name = "Shekswess/medical_llama3_instruct_dataset_short"
data = load_dataset(dataset_name)["train"]

# Load the tokenizer
model_path = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Set a random seed for reproducibility
random.seed(42)

# Randomly sample 128 examples from the dataset
sample_size = 128
random_samples = random.sample(list(data), sample_size)

# Apply the chat template using the tokenizer
chat_templates = []
for sample in random_samples:
    instruction = sample["instruction"]
    input_text = sample.get("input", "")
    output_text = sample["output"]

    # Create a chat-like format using a proper template
    chat = [
        {"role": "system", "content": "You are a helpful assistant." + " " + instruction.strip()},
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text}
    ]

    formatted_string = tokenizer.apply_chat_template(chat, tokenize=False)
    chat_templates.append(formatted_string)

from awq import AutoAWQForCausalLM

quant_path = "Llama-3.1-8B-Instruct-bio-awq"
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# Leverage 8 GPUs
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    use_cache=False
)

# Quantization with multi-GPU and custom parameters
model.quantize(
    tokenizer=tokenizer,
    quant_config=quant_config,
    calib_data=chat_templates,
    max_calib_samples=128,
    max_calib_seq_len=512
)

# Save the quantized model and tokenizer
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)