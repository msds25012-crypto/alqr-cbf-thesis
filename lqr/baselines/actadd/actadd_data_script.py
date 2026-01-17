import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import random
import pickle
import time
from data_handling_actadd import ContrastiveBuilder


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
        

def main():
    model_name= "meta-llama/Llama-3.2-1B"
    # model_name = "google/gemma-2-2b"
    # model_name = "Qwen/Qwen2.5-3B"
    # model_name ="meta-llama/Meta-Llama-3-8B"
    # model_name = "google/gemma-2-9b"
    # model_name = "Qwen/Qwen2.5-14B"
    model, tokenizer = load_model(model_name, quant=True)
    dataset_name = "allenai/real-toxicity-prompts"
    dataset=load_dataset(dataset_name)
    data = dataset["train"]

    tox_prompts = [
        item["text"]
        for item in data["prompt"]
        if item["toxicity"] is not None and item["toxicity"] <= 1.0 and item["toxicity"] >= 0.8
    ]
    dataguy = ContrastiveBuilder(model, tokenizer)
    filename = "llama1b_actadd_tox_vec"
    dataguy.collect_data_batch(tox_prompts,100,filename)

    # nontox_prompts = [
    #     item["text"]
    #     for item in data["prompt"]
    #     if item["toxicity"] is not None and item["toxicity"] <= 0.1 and item["toxicity"] >= 0.0
    # ]
    # dataguy = ContrastiveBuilder(model, tokenizer)
    # filename = "llama1b_actadd_nontox_vec"
    # dataguy.collect_data_batch(nontox_prompts,100,filename)


if __name__ == "__main__":
    main()
