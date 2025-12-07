import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer, RobertaForSequenceClassification, pipeline, BitsAndBytesConfig
import lqr_utils_seq as lqr
from functools import partial
import pickle
from steering import LQRSteering
from datasets import load_dataset
import random
import time

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# use the same tokenizer as TinyLlama
# tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-step-50K-105b")

# load model from huggingface
# model_name = "PY007/TinyLlama-1.1B-step-50K-105b"
# model_name = "meta-llama/Llama-3.2-1B"
model_name = "google/gemma-2-2b"
# model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "Qwen/Qwen2.5-3B"
# model = LlamaForCausalLM.from_pretrained(
    # model_name).to(device)


# model = AutoModelForCausalLM.from_pretrained(
#     model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,          # or load_in_8bit=True
    bnb_4bit_compute_dtype=th.float16,
    bnb_4bit_quant_type="nf4",  # best for LLMs
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=quant_config, dtype=th.float32, device_map="auto")



print(f"model name: {model_name}")
print(f"model blocks: {len(model.model.layers)}")
print(f"model device: {model.device}")
print(f"latent dim: {model.model.embed_tokens.embedding_dim}")

dataset_name = "allenai/real-toxicity-prompts"
dataset = load_dataset(dataset_name)
data = dataset["train"]
# toxic_ds = data.filter(
#     lambda item: item["prompt"]["toxicity"] is not None and item["prompt"]["toxicity"] > 0.7,
#     num_proc=8  # increase or decrease based on CPU cores
# )
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

def calculate_dist_n(texts_list, n=1):
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Collect all tokens
    tokens = []
    for text_item in texts_list:
        tokens.extend(text_item.lower().split())

    if len(tokens) < n:
        return 0.0

    # Build n-grams
    ngrams = [
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    ]

    unique_ngrams = set(ngrams)

    return len(unique_ngrams) / len(ngrams)


nontox_filename = "gemma-2-2b_nontox"
# nontox_filename = "qwen2.5-3b_nontox"
# nontox_filename = "llama-3.2-1b_nontox"
# nontox_filename = "llama-3-8b_nontox"
with open("../../scratch/"+nontox_filename+".pkl", "rb") as f:
    loaded_tensors = pickle.load(f)

# Access tensors
X = loaded_tensors["X"]
A = loaded_tensors["A"]
print(f"X shape: {X.shape}")
print(f"A shape: {A.shape}")

tox_filename = "gemma-2-2b_tox"
# tox_filename = "qwen2.5-3b_tox"
# tox_filename = "llama-3-8b_tox"
# tox_filename = "llama-3.2-1b_tox"
with open("../../scratch/"+tox_filename+".pkl", "rb") as f:
    loaded_tensors = pickle.load(f)

    # Access tensors
X_tox = loaded_tensors["X"]

X_contr = X - X_tox
print(f"X contr: {X_contr.shape}")



num_trials = 1000
samples = random.sample(toxic_prompts, num_trials)
# steer = LQRSteering(model, tokenizer, A)

temp = 0.7
start_time = time.perf_counter()


q_list = [0.1, 1, 10, 100, 1000]
r_list = [0.1, 1, 10, 100, 1000]
qf_list = [0.1, 1, 10, 100, 1000]
for q in q_list:
    for r in r_list:
        for qf in qf_list:
            steer_contr = LQRSteering(model, tokenizer, q=q,r=r,qf=qf, A=A, contrastive_vecs=X_contr)
            for l in [0.5, 1, 1.5, 2, 2.5]:
                track_completions = []
                contr_completions = []
                un_completions = []
                print(f"LAMBDA: {l}, q: {q}, r: {r}, qf: {qf}")
                counter = 0
                k=50


                inputs = tokenizer(
                        samples, 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                    ).to(device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                output_un = model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=k,
                                return_dict_in_generate=True,
                                do_sample=True,
                                temperature=temp,
                                use_cache=False,
                                pad_token_id=tokenizer.eos_token_id,
                                # **model_generation_kwargs, #
                            )

                output_str = tokenizer.batch_decode(output_un.sequences, skip_special_tokens=True)


                contr_out = steer_contr.track_setpoint(samples, k, lmbda=l, do_sample=True, temp = temp)
                for i, inp in enumerate(samples):
                    un_completions.append(output_str[i][len(inp):].strip())
                    contr_completions.append(contr_out[i][len(inp):].strip())
                    
                # print(f"PROMPT: {samples}")
                # print(f"\nSteered output: {contr_out}")
                # print(f"\nUnsteered output: {output_str}")

                contr_preds = classifier(contr_completions)
                un_preds = classifier(un_completions)
                # print(contr_preds)
                # print(un_preds)
                dist_1_steered = calculate_dist_n(contr_completions, n=1)
                dist_2_steered = calculate_dist_n(contr_completions, n=2)
                dist_3_steered = calculate_dist_n(contr_completions, n=3)

                dist_1_base = calculate_dist_n(un_completions, n=1)
                dist_2_base = calculate_dist_n(un_completions, n=2)
                dist_3_base = calculate_dist_n(un_completions, n=3)
                n = 0
                m = 0

                num_tox_contr = 0
                num_tox_un = 0
                for i in range(len(contr_preds)):
                    if contr_preds[i]["label"] == "neutral" and un_preds[i]["label"] == "toxic":
                        n=n+1
                    if contr_preds[i]["label"] == "toxic" and un_preds[i]["label"] == "neutral":
                        m=m+1
                        print(f"\ntoxified: {samples[i]}")
                        print(f"steered: {contr_completions[i]}")
                        print(f"toxicity score: {contr_preds[i]['score']}\n")
                        print(f"unsteered: {un_completions[i]}\n")

                    if contr_preds[i]["label"] == "toxic":
                        num_tox_contr += 1
                    if un_preds[i]["label"] == "toxic":
                        num_tox_un += 1

                        
                print("\n\n\n\n\n\n\n\n")
                print(f"num safeified: {n}")
                print(f"num unsafeified: {m}")

                print(f"num tox un: {num_tox_un}")
                print(f"num tox contr: {num_tox_contr}")
                print("")

                print(f"dist 1 base: {dist_1_base}")
                print(f"dist 2 base: {dist_2_base}")
                print(f"dist 3 base: {dist_3_base}\n")
                print(f"dist 1 steered: {dist_1_steered}")
                print(f"dist 2 steered: {dist_2_steered}")
                print(f"dist 3 steered: {dist_3_steered}")

end_time = time.perf_counter()
print(f"runtime: {end_time - start_time}")