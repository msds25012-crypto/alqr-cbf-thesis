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

def main():
    # model_name = "google/gemma-2-9b"
    model_name = "Qwen/Qwen2.5-14B"
    model, tokenizer = load_model(model_name, quant=True)
    toxic_prompts = get_tox_prompts(0.8, 1)
    nontoxic_prompts = get_tox_prompts(0, 0.1)

    dataguy = ContrastiveBuilder(model, tokenizer)
    filename = "Qwen2.5-14B-tox"
    # filename = "gemma-2-2b_tox"

    dataguy.collect_data_batch(toxic_prompts, 1000, filename)
    print("done with nontox")

    filename = "Qwen2.5-14B-nontox"
    dataguy.collect_data_batch(nontoxic_prompts, 1000, filename)
    print("done with tox")

    filename = "Qwen2.5-14B-nontox_jac"
    dataguy.collect_jacobians(nontoxic_prompts, 50, filename)
    print("done with jac")

if __name__ == "__main__":
    print(f"device: {device}")
    main()
