import torch as th
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
import lqr_utils_seq as lqr
from functools import partial
import pickle
from steering import LQRSteering

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# use the same tokenizer as TinyLlama
# tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-step-50K-105b")

# load model from huggingface
# model_name = "PY007/TinyLlama-1.1B-step-50K-105b"
model_name = "meta-llama/Llama-3.2-1B"
# model_name = "google/gemma-2-2b"
# model_name = "meta-llama/Meta-Llama-3-8B"
# model = LlamaForCausalLM.from_pretrained(
    # model_name).to(device)
model = AutoModelForCausalLM.from_pretrained(
    model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


print(f"model name: {model_name}")
print(f"model blocks: {len(model.model.layers)}")
print(f"model device: {model.device}")
print(f"latent dim: {model.model.embed_tokens.embedding_dim}")

# T = len(model.model.layers)
# n = model.model.embed_tokens.embedding_dim
# print(f"n: {n}")
# m = n


# with open("../../scratch/coherent.pkl", "rb") as f:
#     loaded_tensors = pickle.load(f)

# Access tensors
# X_nom = loaded_tensors["X"].to(device)
# A = loaded_tensors["A"].to(device)
# B = th.eye(m).unsqueeze(0).repeat(T, 1, 1).to(device)


# with open("../../scratch/sad_data.pkl", "rb") as f:
#     loaded_tensors = pickle.load(f)

# X_sad = loaded_tensors["X"].to(device)
# A_sad = loaded_tensors["A"].to(device)


# X_cntr = X_nom - X_sad

# text = "hello. "
# k = 10

# steer = LQRSteering(model, tokenizer, A, X_cntr)

# temp=0.7
# output = steer.evaluate(text, k, do_sample=True, temp=temp)

# print(output)

# inputs = tokenizer(text, return_tensors="pt").to(device)
# input_ids = inputs["input_ids"]
# attention_mask = inputs["attention_mask"]
# output_un = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 max_new_tokens=k,
#                 return_dict_in_generate=True,
#                 do_sample=True,
#                 temperature=temp,
#                 use_cache=False,
#                 pad_token_id=tokenizer.eos_token_id,
#                 # **model_generation_kwargs, #
#             )

# output_str = tokenizer.decode(output_un.sequences[0], skip_special_tokens=True)
# print(f"unsteered: {output_str}")


nom = "The thing about cats "
inp = "The political climate in the U.S. "
steer = LQRSteering(model, tokenizer, q=1, r=0.1, qf=1000)
k=100
steer.track_tokens(nom, inp, k=k)

inputs = tokenizer(inp, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print("running unsteered")
output_un = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=k,
                return_dict_in_generate=True,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
                # **model_generation_kwargs, #
            )

output_str = tokenizer.decode(output_un.sequences[0], skip_special_tokens=True)
print(f"unsteered: {output_str}")