import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import lqr_utils_seq as lqr
from functools import partial
from datasets import load_dataset
import random
import pickle
import time
from data_handling import ContrastiveBuilder


device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"device: {device}")


def load_model(model_name, quant=False):

    if quant:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,          # or load_in_8bit=True
            # load_in_8bit=True,
            bnb_4bit_compute_dtype=th.float16,
            bnb_4bit_quant_type="nf4",  # best for LLMs
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant_config, dtype=th.float32, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else: 
        model = AutoModelForCausalLM.from_pretrained(
            model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer
        
def get_refused_prompts():
    dataset_name = "walledai/AdvBench"
    dataset = load_dataset(dataset_name)

    return dataset['train'][:]["prompt"]


def get_safe_prompts():
    dataset = load_dataset("tatsu-lab/alpaca")
    return dataset['train'][:]["instruction"]


def get_safe_prompt_with_response(tokenizer):
    dataset = load_dataset("tatsu-lab/alpaca")
    harmful_prompts = dataset['train'][:]["instruction"]
    ps = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p["instruction"]}],
        tokenize=False,
        add_generation_prompt=True
    ) + p["output"] for p in dataset['train']]
    return ps


def main():
    # model_name = "google/gemma-2-2b"
    # model_name = "Qwen/Qwen2.5-3B-Instruct"
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "google/gemma-2-9b-it"
    # model_name = "Qwen/Qwen2.5-14B-Instruct"
    model, tokenizer = load_model(model_name, quant=True)
    # messages = [
    #     {"role": "user", "content": p} for p in prompt
    # ]
    # print(messages)

    harmful_prompts = get_refused_prompts()[:416]

    safe_prompts = get_safe_prompts()
    # formatted_harmful_prompts = [tokenizer.apply_chat_template(
    formatted_harmful_prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True
    ) for p in harmful_prompts]

    # formatted_safe_prompts = [tokenizer.apply_chat_template(
    #     [{"role": "user", "content": p}],
    #     tokenize=False,
    #     add_generation_prompt=True
    # ) for p in safe_prompts]


    safe_with_response = get_safe_prompt_with_response(tokenizer)


    dataguy = ContrastiveBuilder(model, tokenizer)
    # filename = "Llama-3.1-8B-Instruct-ref"
    # dataguy.collect_data_batch(formatted_harmful_prompts, 416, filename, batch_size=12)
    # # dataguy.collect_data_batch(formatted_harmful_prompts, 1, filename)
    # print("done with refused")

    # filename = "Llama-3.1-8B-Instruct-nonref"
    # dataguy.collect_data_batch(safe_with_response, 200, filename, batch_size=12)
    # # dataguy.collect_data_batch(formatted_safe_prompts, 1, filename)
    # print("done with not refused")

    # # for i in range(10):
    filename = f"Llama-3.1-8B-Instruct-nonref_jac"
    dataguy.collect_jacobians(safe_with_response, 15, filename, max_ctx=32)
    # dataguy.collect_jacobians(formatted_safe_prompts, 1, filename)
    # print(f"done with jac")

if __name__ == "__main__":
    main()
