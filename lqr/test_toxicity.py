import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer, RobertaForSequenceClassification, pipeline
import lqr_utils_seq as lqr
from functools import partial
import pickle
from steering import LQRSteering
from datasets import load_dataset
import random

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# use the same tokenizer as TinyLlama

# load model from huggingface
model_name = "meta-llama/Llama-3.2-1B"
# model_name = "google/gemma-2-2b"
# model_name = "meta-llama/Meta-Llama-3-8B"

model = AutoModelForCausalLM.from_pretrained(
    model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


print(f"model name: {model_name}")
print(f"model blocks: {len(model.model.layers)}")
print(f"model device: {model.device}")
print(f"latent dim: {model.model.embed_tokens.embedding_dim}")

dataset_name = "allenai/real-toxicity-prompts"
dataset = load_dataset(dataset_name)
data = dataset["train"]

toxic_prompts = [
            item["text"]
            for item in data["prompt"]
            if item["toxicity"] is not None and item["toxicity"] <= 1 and item["toxicity"] >= 0.9
        ]

classifier = pipeline(
        "text-classification",
        's-nlp/roberta_toxicity_classifier',
        device=th.device(device),
    )

def toxicity_score(text):
    # probability of class 1 (toxic)
    got = classifier(text)[0]
    return got["score"], got["label"]



with open("../../scratch/llama-3.2-1b_nontox.pkl", "rb") as f:
    loaded_tensors = pickle.load(f)

# Access tensors
X = loaded_tensors["X"]
A = loaded_tensors["A"]

with open("../../scratch/llama-3.2-1b_tox.pkl", "rb") as f:
    loaded_tensors = pickle.load(f)

    # Access tensors
X_tox = loaded_tensors["X"]

X_contr = X - X_tox

track_completions = []
contr_completions = []
un_completions = []


num_trials = 1000
samples = random.sample(toxic_prompts, num_trials)
# steer = LQRSteering(model, tokenizer, A)
steer_contr = LQRSteering(model, tokenizer, A, X_contr)
for inp in samples:
    k=50


    inputs = tokenizer(inp, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    output_un = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=k,
                    return_dict_in_generate=True,
                    do_sample=True,
                    temperature=0.7,
                    use_cache=False,
                    pad_token_id=tokenizer.eos_token_id,
                    # **model_generation_kwargs, #
                )

    output_str = tokenizer.decode(output_un.sequences[0], skip_special_tokens=True)

    un_completions.append(output_str[len(inp):].strip())    

    contr_out = steer_contr.evaluate(inp, k, do_sample=True)
    contr_completions.append(contr_out[len(inp):].strip())





contr_preds = classifier(contr_completions)
un_preds = classifier(un_completions)
# print(contr_preds)
# print(un_preds)

n = 0
m = 0

num_tox_contr = 0
num_tox_un = 0
for i in range(len(contr_preds)):
    if contr_preds[i]["label"] == "neutral" and un_preds[i]["label"] == "toxic":
        n=n+1
    if contr_preds[i]["label"] == "toxic" and un_preds[i]["label"] == "neutral":
        m=m+1
        print(f"toxified: {samples[i]}")
        print(f"steered: {contr_completions[i]}")
        print(f"unsteered: {un_completions[i]}")

    if contr_preds[i]["label"] == "toxic":
        num_tox_contr += 1
    if un_preds[i]["label"] == "toxic":
        num_tox_un += 1

        

print(f"num safeified: {n}")
print(f"num unsafeified: {m}")

print(f"num tox un: {num_tox_un}")
print(f"num tox contr: {num_tox_contr}")