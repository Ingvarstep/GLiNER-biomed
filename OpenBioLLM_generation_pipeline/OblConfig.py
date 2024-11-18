class OblConfig:
    model_name = '/home/users/y/yazdani0/.cache/huggingface/hub/models--mradermacher--OpenBioLLM-Llama3-70B-GGUF/snapshots/84e74f043dcac633a9dc7077d7d02fa9f53e7e6a/OpenBioLLM-Llama3-70B.Q5_K_M.gguf'
    tokenizer_name = "aaditya/Llama3-OpenBioLLM-70B"
    max_model_len = 8192
    tensor_parallel_size = 8
    max_gen_tokens = 2048
    
    chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|end_of_text|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    
    one_shot_messages = [
        {"role": "system", "content": """You are an AI language model specialized in biomedical text analysis. Your task is to extract all significant entities and their relationships from the provided text and output them in JSON format according to the schema below.
    
    **Instructions:**
    
    - **Entity Extraction ("nodes")**:
      - Identify named entities.
      - For each entity, provide:
        - **id**: Exact name from the text.
        - **type**: Broad category.
    
    - **Relation Extraction ("edges")**:
      - Identify all relationships between the entities.
      - For each relationship, provide:
        - **from**: The "id" of the source entity.
        - **to**: The "id" of the target entity.
        - **label**: Descriptive label of the relationship.
    
    - **Constraints**:
      - No duplicate entities or relationships.
      - Every entity referenced in "edges" must exist in "nodes".
      - Adhere strictly to the JSON schema provided.
    
    **JSON Schema**:
    
    ```json
    {"nodes": [{"id": "string", "type": "string"}], "edges": [{"from": "string", "to": "string", "label": "string"}]}
    ```
    
    **Output Requirement**:
    
    - Provide only the JSON output without any additional text or explanations."""},
        {"role": "user", "content": """Please create a JSON for this text: "Pseudomonas aeruginosa (Pa) infection in cystic fibrosis (CF) patients is associated with worse long-term pulmonary disease and shorter survival, and chronic Pa infection (CPA) is associated with reduced lung function, faster rate of lung decline, increased rates of exacerbations and shorter survival. By using exome sequencing and extreme phenotype design, it was recently shown that isoforms of dynactin 4 (DCTN4) may influence Pa infection in CF, leading to worse respiratory disease. The purpose of this study was to investigate the role of DCTN4 missense variants on Pa infection incidence, age at first Pa infection and chronic Pa infection incidence in a cohort of adult CF patients from a single centre. Polymerase chain reaction and direct sequencing were used to screen DNA samples for DCTN4 variants. A total of 121 adult CF patients from the Cochin Hospital CF centre have been included, all of them carrying two CFTR defects: 103 developed at least 1 pulmonary infection with Pa, and 68 patients of them had CPA. DCTN4 variants were identified in 24% (29/121) CF patients with Pa infection and in only 17% (3/18) CF patients with no Pa infection. Of the patients with CPA, 29% (20/68) had DCTN4 missense variants vs 23% (8/35) in patients without CPA. Interestingly, p.Tyr263Cys tend to be more frequently observed in CF patients with CPA than in patients without CPA (4/68 vs 0/35), and DCTN4 missense variants tend to be more frequent in male CF patients with CPA bearing two class II mutations than in male CF patients without CPA bearing two class II mutations (P = 0.06). Our observations reinforce that DCTN4 missense variants, especially p.Tyr263Cys, may be involved in the pathogenesis of CPA in male CF.\""""},
        {"role": "assistant", "content": """{"nodes": [{"id": "Pseudomonas aeruginosa (Pa)", "type": "bacterium"}, {"id": "cystic fibrosis (CF)", "type": "disease"}, {"id": "Chronic Pa infection (CPA)", "type": "medical condition"}, {"id": "dynactin 4 (DCTN4)", "type": "gene"}, {"id": "DCTN4 missense variants", "type": "genetic variation"}, {"id": "p.Tyr263Cys", "type": "mutation"}, {"id": "CFTR defects", "type": "genetic mutation"}, {"id": "male CF patients", "type": "patient group"}, {"id": "exome sequencing", "type": "technique"}, {"id": "extreme phenotype design", "type": "study design"}, {"id": "polymerase chain reaction", "type": "technique"}, {"id": "direct sequencing", "type": "technique"}, {"id": "Cochin Hospital CF centre", "type": "organization"}, {"id": "lung function", "type": "clinical parameter"}, {"id": "pulmonary disease", "type": "disease"}, {"id": "survival", "type": "clinical outcome"}, {"id": "rate of lung decline", "type": "clinical parameter"}, {"id": "rates of exacerbations", "type": "clinical parameter"}, {"id": "Pa infection incidence", "type": "clinical parameter"}, {"id": "age at first Pa infection", "type": "clinical parameter"}, {"id": "chronic Pa infection incidence", "type": "clinical parameter"}, {"id": "class II mutations", "type": "genetic mutation"}, {"id": "pathogenesis of CPA in male CF patients", "type": "clinical concept"}, {"id": "CF patients", "type": "patient group"}, {"id": "CF patients with CPA", "type": "patient group"}, {"id": "male CF patients with CPA bearing two class II mutations", "type": "patient group"}, {"id": "DCTN4 variants", "type": "genetic variation"}], "edges": [{"from": "Pseudomonas aeruginosa (Pa)", "to": "cystic fibrosis (CF)", "label": "infection associated with worse outcomes in"}, {"from": "Pseudomonas aeruginosa (Pa)", "to": "pulmonary disease", "label": "associated with worse"}, {"from": "Pseudomonas aeruginosa (Pa)", "to": "survival", "label": "associated with shorter"}, {"from": "Chronic Pa infection (CPA)", "to": "lung function", "label": "associated with reduced"}, {"from": "Chronic Pa infection (CPA)", "to": "rate of lung decline", "label": "associated with faster"}, {"from": "Chronic Pa infection (CPA)", "to": "rates of exacerbations", "label": "associated with increased"}, {"from": "Chronic Pa infection (CPA)", "to": "survival", "label": "associated with shorter"}, {"from": "dynactin 4 (DCTN4)", "to": "Pa infection incidence", "label": "may influence"}, {"from": "DCTN4 missense variants", "to": "Pa infection incidence", "label": "may influence"}, {"from": "DCTN4 missense variants", "to": "age at first Pa infection", "label": "may influence"}, {"from": "DCTN4 missense variants", "to": "chronic Pa infection incidence", "label": "may influence"}, {"from": "polymerase chain reaction", "to": "DCTN4 variants", "label": "used to screen for"}, {"from": "direct sequencing", "to": "DCTN4 variants", "label": "used to screen for"}, {"from": "Cochin Hospital CF centre", "to": "CF patients", "label": "treated at"}, {"from": "CF patients", "to": "CFTR defects", "label": "carry"}, {"from": "p.Tyr263Cys", "to": "CF patients with CPA", "label": "more frequently observed in"}, {"from": "DCTN4 missense variants", "to": "male CF patients with CPA bearing two class II mutations", "label": "more frequent in"}, {"from": "DCTN4 missense variants", "to": "pathogenesis of CPA in male CF patients", "label": "may be involved in"}]}"""}
    ]