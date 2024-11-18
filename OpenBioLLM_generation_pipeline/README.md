# OpenBioLLM Generation Pipeline

## Download the GGUF Model File

Use the following code to download the GGUF model file from Hugging Face:

```
from huggingface_hub import hf_hub_download

repo_id = "mradermacher/OpenBioLLM-Llama3-70B-GGUF"
filename = "OpenBioLLM-Llama3-70B.Q5_K_M.gguf"
model_path = hf_hub_download(repo_id, filename=filename)
```

`model_path` should be passed to `OblConfig.py` to load the model locally with VLLM.

## Configuration in `OblConfig.py`

Adjust these parameters as needed:

```
model_name = *model_path*  # Path to the downloaded model
tokenizer_name = "aaditya/Llama3-OpenBioLLM-70B"  # Tokenizer name
max_model_len = 8192  # Maximum model length (tokens); 8192 is both the maximum we set and the upper limit for the model
tensor_parallel_size = 8  # Number of available GPUs
max_gen_tokens = 2048  # Maximum tokens for generation
```

## Run the `text2graph.py` Script

To generate graphs, run `text2graph.py` with the appropriate arguments. Make sure to review the script for details on the required inputs.
