import re
import os
import json
import torch
import random
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from vllm import LLM, SamplingParams


def create_chat(text, tokenizer):
    messages = [
        {"role": "system", "content": """
    A chat between a curious user and an artificial intelligence Assistant. The Assistant is an expert at identifying entities and relationships in text. The Assistant responds in JSON output only.

    The User provides text in the format:

    -------Text begin-------
    <User provided text>
    -------Text end-------

    The Assistant follows the following steps before replying to the User:

    1. **identify the most important entities** The Assistant identifies the most important entities in the text. These entities are listed in the JSON output under the key "nodes", they follow the structure of a list of dictionaries where each dict is:

    "nodes":[{"id": <entity N>, "type": <type>, "detailed_type": <detailed type>}, ...]

    where "type": <type> is a broad categorization of the entity. "detailed type": <detailed_type>  is a very descriptive categorization of the entity.

    2. **determine relationships** The Assistant uses the text between -------Text begin------- and -------Text end------- to determine the relationships between the entities identified in the "nodes" list defined above. These relationships are called "edges" and they follow the structure of:

    "edges":[{"from": <entity 1>, "to": <entity 2>, "label": <relationship>}, ...]

    The <entity N> must correspond to the "id" of an entity in the "nodes" list.

    The Assistant never repeats the same node twice. The Assistant never repeats the same edge twice.
    The Assistant responds to the User in JSON only, according to the following JSON schema:

    {"type":"object","properties":{"nodes":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"type":{"type":"string"},"detailed_type":{"type":"string"}},"required":["id","type","detailed_type"],"additionalProperties":false}},"edges":{"type":"array","items":{"type":"object","properties":{"from":{"type":"string"},"to":{"type":"string"},"label":{"type":"string"}},"required":["from","to","label"],"additionalProperties":false}}},"required":["nodes","edges"],"additionalProperties":false}
        """},
        {"role": "user", "content": """
    -------Text begin-------
    {}
    -------Text end-------
    """}
    ]
    messages[-1]['content'] = messages[-1]['content'].format(text)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def generate_response_vllm(pipe, chats, generation_args):
    responses = pipe.generate(chats, generation_args, use_tqdm=False)
    results = [response.outputs[0].text for response in responses]
    return results

def generate_response_hf(pipe, chat, generation_args):
    responses = pipe(chat, **generation_args)
    results = [response[0]['generated_text'] for response in responses]
    return results

def convert_to_json(responses, texts, *args):
    batch_results = []
    for i, response in enumerate(responses):
        try:
            result = json.loads(response)
            result['text'] = texts[i]
        except:
            result = None
        batch_results.append(result)
    return batch_results


def generate_dataset(pipe, tokenizer, generation_args, texts, generate_response=generate_response_hf, process_response=None, batch_size=8, max_lines=None):
    batch_texts = []
    batch_chats = []
    final_results = []
    line_num = 0

    try:
        for id, text in tqdm(enumerate(texts)):
            if max_lines is not None and id >= max_lines:
                break

            batch_texts.append(text)

            chat = create_chat(text, tokenizer)
            batch_chats.append(chat)

            if len(batch_texts) == batch_size:
                try:
                    responses = generate_response(pipe, batch_chats, generation_args)
                except Exception as err:
                    print(f"Error in generating response: {err}")
                    continue

                if process_response is not None:
                    try:
                        batch_results = process_response(responses, batch_texts)
                    except Exception as err:
                        print(f"Error in processing response: {err}")
                        continue
                else:
                    batch_results = [response.outputs[0].text for response in responses]

                final_results.extend(batch_results)
                batch_texts = []
                batch_chats = []

            line_num += 1

    except KeyboardInterrupt:
        print("Execution interrupted. Returning results gathered so far...")

    return final_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default= "data/raw_texts.json")
    parser.add_argument('--save_path', type=str, default= "data/rex")
    parser.add_argument('--model', type=str, default= "EmergentMethods/Phi-3-mini-128k-instruct-graph")#"Qwen/Qwen2.5-32B-Instruct")#"NousResearch/Hermes-2-Pro-Llama-3-8B")
    parser.add_argument('--quantization', type=str, default= "fp8")
    parser.add_argument('--hf_token', type=str, default= "")
    parser.add_argument('--max_examples', type=int, default= 100)
    parser.add_argument('--use_vllm', type=bool, default= True)
    args = parser.parse_args()

    with open(args.data_path, 'r') as f:
        texts = json.load(f)
        random.shuffle(texts)
        texts = texts[:args.max_examples]
        print('Texts count: ', len(texts))

    tokenizer = AutoTokenizer.from_pretrained(args.model, token = args.hf_token)

    
    if not args.use_vllm:
        print('Using vLLM...')
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            token = args.hf_token
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        
        generate_response = generate_response_hf
    else:
        pipe = LLM(model=args.model,
                        max_model_len = 32000, 
                        tensor_parallel_size=1, dtype="half", #kv_cache_dtype="fp8", 
                                            gpu_memory_utilization = 0.9, quantization = args.quantization)

        generation_args = SamplingParams(temperature = 0.25, repetition_penalty = 1.1, top_k=100, max_tokens=512, top_p=0.8, stop="<end>")

        generate_response = generate_response_vllm

    results = generate_dataset(pipe, tokenizer, generation_args, texts, generate_response = generate_response, 
                                                        process_response = convert_to_json, batch_size=8)

    synthetic_dataset = []
    for id, result in enumerate(results):
        if result is not None:
            item = {'text': texts[id], "nodes": result['nodes'], "edges": result["edges"]}
            synthetic_dataset.append(item)

    save_path = os.path.join(args.save_path, f"text2graph.json")

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)