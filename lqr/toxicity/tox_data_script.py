import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import lqr.lqr_utils as lqr
from functools import partial
from datasets import load_dataset
import random
import pickle
import time
from lqr.data_handling import ContrastiveBuilder
import gc

device = th.device("cuda" if th.cuda.is_available() else "cpu")
dataset_name = "allenai/real-toxicity-prompts"
dataset = load_dataset(dataset_name)


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
        
def get_tox_prompts(lb, ub):
    assert(lb >= 0 and lb <= ub)
    assert(ub <= 1)

    data = dataset["train"]
    prompts = [
                item["text"]
                for item in data["prompt"]
                if item["toxicity"] is not None and item["toxicity"] <= ub and item["toxicity"] >= lb
            ]
    return prompts

def get_alternate_prompts():
    ds = load_dataset("tcapelle/jigsaw-toxic-comment-classification-challenge")['test']
    prompts = [p['comment_text'] for p in ds]
    return prompts

def main():

    model_keys = {  
        "google/gemma-2-2b": "gemma-2-2b",
        "meta-llama/Meta-Llama-3-8B": "Llama-3-8B",
        "Qwen/Qwen2.5-3B": "Qwen2.5-3B"

    }

    models = [
        # "google/gemma-2-2b",
        # "meta-llama/Meta-Llama-3-8B",
        "Qwen/Qwen2.5-3B",
        # "Qwen/Qwen2.5-14B",
    #    "Qwen/Qwen2.5-32B"
    ]

    for model_name in models:
        print(f"running model: {model_name}")

        model, tokenizer = load_model(model_name, quant=True)



        lqr.print_curr_mem("just the model")
        toxic_prompts = get_tox_prompts(0.8, 1)
        nontoxic_prompts = get_tox_prompts(0, 0.1)

        dataguy = ContrastiveBuilder(model, tokenizer)
        # filename = "Qwen2.5-14B-tox"
        # filename = "Llama-3.1-8B-Instruct-tox"
        
        # filename = "Qwen2.5-32B-tox"
        # # filename = "Qwen2.5-3B-tox"
        # dataguy.collect_data_batch(toxic_prompts, 200, filename)
        # # print("done with dtox")

        # # filename = "Qwen2.5-3B-nontox"
        # filename = "Qwen2.5-32B-nontox"
        # dataguy.collect_data_batch(nontoxic_prompts, 200, filename)
        # print("done with nontox")

        # filename = "Llama-3.2-1B-nontox_jac"
        filename = "test_jac"
        dataguy.collect_jacobians_vram(nontoxic_prompts, 1, filename, max_ctx=24)
        print("done with jac")

        lqr.print_curr_mem("final GPU results")

if __name__ == "__main__":
    print(f"device: {device}")
    main()
