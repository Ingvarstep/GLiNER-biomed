### QA Generation

---

#### Purpose
Processes JSON files with entity-relation triplets to generate question-answer (QA) pairs using an LLM.

---

#### Model
LLM available at: [Llama-3.1-8B-Instruct-BioQA-AWQ](https://huggingface.co/anthonyyazdaniml/Llama-3.1-8B-Instruct-BioQA-AWQ)

---

#### Usage
```bash
python script.py --input_folder <input_path> --output_folder <output_path> --model_path <model_path>
```
- Defaults:
  - `--input_folder`: `./data/synthetic_data/sample`
  - `--output_folder`: `./results`
  - `--model_path`: `/home/users/y/yazdani0/ds4dhxkg/Llama-3.1-8B-Instruct-bio-awq`

---

#### Output
Processed JSON files saved in the specified output folder with `_qa.json` suffix.
