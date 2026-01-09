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
        
def true_prompts(tokenizer):
    mc = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
    ds_mc = mc["validation"]
    shuffled = ds_mc.shuffle(seed=None)  

    # print(shuffled[0])
    questions = [shuffled[i]["question"] for i in range(len(shuffled))]

    answers = []
    for item in shuffled:
        for i in range(len(item["mc1_targets"]["choices"])):
            if item["mc1_targets"]["labels"][i] == 1:
                answers.append(item["mc1_targets"]["choices"][i])
                break

    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True
    ) + answers[i] for i, p in enumerate(questions)]

    return prompts

def false_prompts(tokenizer):
    mc = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
    ds_mc = mc["validation"]
    shuffled = ds_mc.shuffle(seed=None)  

    # print(shuffled[0])
    questions = [shuffled[i]["question"] for i in range(len(shuffled))]

    answers = []
    for item in shuffled:
        for i in range(len(item["mc1_targets"]["choices"])):
            if item["mc1_targets"]["labels"][i] == 0:
                answers.append(item["mc1_targets"]["choices"][i])
                break
            
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True
    ) + answers[i] for i, p in enumerate(questions)]

    return prompts


def get_questions(tokenizer):
    gen = load_dataset("truthfulqa/truthful_qa", "generation")
    ds_gen = gen["validation"]
    shuffled = ds_gen.shuffle(seed=None)  

    # print(shuffled[0])
    questions = [shuffled[i]["question"] for i in range(len(shuffled)) if shuffled[i]["type"] == "Adversarial"]

    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True
    ) for p in questions]

    return prompts


def main():
    # model_name = "google/gemma-2-2b"
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "google/gemma-2-9b-it"
    # model_name = "Qwen/Qwen2.5-14B-Instruct"
    model, tokenizer = load_model(model_name, quant=True)
    # messages = [
    #     {"role": "user", "content": p} for p in prompt
    # ]
    # print(messages)

    t_prompts = true_prompts(tokenizer)
    f_prompts = false_prompts(tokenizer)
    # print(promptywompty[0])
    # print(promptywompty[816])
    # print(len(promptywompty))


    dataguy = ContrastiveBuilder(model, tokenizer)
    # filename = "Qwen2.5-3B-Instruct-false"
    # dataguy.collect_data_batch(f_prompts, 200, filename)
    # # dataguy.collect_data_batch(formatted_harmful_prompts, 1, filename)
    # print("done with refused")

    # filename = "Qwen2.5-3B-Instruct-true"
    # dataguy.collect_data_batch(t_prompts, 200, filename)
    # # dataguy.collect_data_batch(formatted_safe_prompts, 1, filename)
    # print("done with not refused")

    # # for i in range(10):
    filename = f"Qwen2.5-3B-Instruct-true_jac"
    dataguy.collect_jacobians(t_prompts, 50, filename)
    # dataguy.collect_jacobians(formatted_safe_prompts, 1, filename)
    print(f"done with jac")

if __name__ == "__main__":
    main()
