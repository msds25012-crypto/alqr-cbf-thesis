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
import csv

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def load_model(model_name, quant=False):

    if quant:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,          # or load_in_8bit=True
            # load_in_8bit=True,
            bnb_4bit_compute_dtype=th.float32,
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
        
# def get_concept_and_non_sentences(concepts, target):
#     sentences = []
#     others = {}
#     for c in concepts:
#         if c != target:
#             others[c] = []
#     with open('concepts/filtered_sentences.csv', mode='r', newline='', encoding='utf-8') as file:
#         csv_reader = csv.DictReader(file)
#         for row in csv_reader:
#             # Each 'row' is a dictionary (e.g., {'name': 'John Smith', 'department': 'Accounting', ...})
#             if row["concept"] == target+".NOUN":
#                 sentences.append(row["sentence"])
#             else:
#                 others[row["concept"][:-len(".NOUN")]].append(row["sentence"])
        
#     print(others)
#     p = min(len(o) for o in others)

#     other_sen = []
#     for i in range(p):
#         for c in others:
#             other_sen.append(c[i])
#     return sentences, other_sen




def get_target_and_other_sentences(csv_path, target):
    sentences = []  # Target concept sentences
    others = {}     # Other concepts: {concept_name: [sentences]}

    # Read CSV and separate sentences
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            concept_name = row["concept"]
            sentence = row["sentence"]

            if concept_name == f"{target}.NOUN":
                sentences.append(sentence)
            else:
                # Initialize list if not exists
                if concept_name not in others:
                    others[concept_name] = []
                others[concept_name].append(sentence)

    # Find the minimum length among other concepts
    min_len = min(len(sents) for sents in others.values())

    # Build the alternating list
    other_sentences = []
    for i in range(min_len):
        for concept in others:
            other_sentences.append(others[concept][i])

    return sentences, other_sentences

def main():
    model_name = "google/gemma-2-2b"
    # model_name = "meta-llama/Meta-Llama-3-8B"
    # model_name = "google/gemma-2-9b"
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "meta-llama/Llama-3.2-1B"
    # model_name = "Qwen/Qwen2.5-3B"

    concepts = [
        "football",
        "circus",
        "church",
        "dog",
        "balloon"
    ]

    sen, other = get_target_and_other_sentences('concepts/filtered_sentences.csv', "dog")

    for i in range(10):
        print(sen[i])
        print(f"other: {other[i]}\n")

    model, tokenizer = load_model(model_name, quant=True)

    dataguy = ContrastiveBuilder(model, tokenizer)
    
    # filename = "gemma-2-2b-dog"
    # dataguy.collect_data_batch(sen, 200, filename)
    # print("done with dtox")

    filename = "gemma-2-2b-nondog"
    dataguy.collect_data_batch(other, 200, filename)
    print("done with nondog")

    # filename = "gemma-2-2b-dog_jac"
    # dataguy.collect_jacobians(sen, 50, filename)
    # print("done with jac")

if __name__ == "__main__":
    print(f"device: {device}")
    main()
