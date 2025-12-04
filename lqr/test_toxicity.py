import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer, RobertaForSequenceClassification, pipeline, BitsAndBytesConfig
import lqr_utils_seq as lqr
from functools import partial
import pickle
from steering import LQRSteering
from datasets import load_dataset
import random

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# use the same tokenizer as TinyLlama
# tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-step-50K-105b")

# load model from huggingface
# model_name = "PY007/TinyLlama-1.1B-step-50K-105b"
# model_name = "meta-llama/Llama-3.2-1B"
# model_name = "google/gemma-2-2b"
# model_name = "meta-llama/Meta-Llama-3-8B"
model_name = "Qwen/Qwen2.5-3B"
# model = LlamaForCausalLM.from_pretrained(
    # model_name).to(device)


# model = AutoModelForCausalLM.from_pretrained(
#     model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    got = classifier(text)[0]
    return got["score"], got["label"]


# nontox_filename = "gemma-2-2b_nontox"
nontox_filename = "qwen2.5-3b_nontox"
# nontox_filename = "llama-3.2-1b_nontox"
with open("../../scratch/"+nontox_filename+".pkl", "rb") as f:
    loaded_tensors = pickle.load(f)

# Access tensors
X = loaded_tensors["X"]
A = loaded_tensors["A"]

# tox_filename = "gemma-2-2b_tox"
tox_filename = "qwen2.5-3b_tox"
# tox_filename = "llama-3.2-1b_tox"
with open("../../scratch/"+tox_filename+".pkl", "rb") as f:
    loaded_tensors = pickle.load(f)

X_tox = loaded_tensors["X"]

X_contr = X - X_tox




num_trials = 1000
samples = random.sample(toxic_prompts, num_trials)
steer_contr = LQRSteering(model, tokenizer, q=1,r=0.1,qf=1000, A=A, contrastive_vecs=X_contr)
temp = 0.7
for l in [0.5, 1, 1.5, 2]:
    track_completions = []
    contr_completions = []
    un_completions = []
    print(f"LAMBDA: {l}")
    counter = 0
    for i, inp in enumerate(samples):

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
                        temperature=temp,
                        use_cache=False,
                        pad_token_id=tokenizer.eos_token_id,
                        # **model_generation_kwargs, #
                    )

        output_str = tokenizer.decode(output_un.sequences[0], skip_special_tokens=True)

        un_completions.append(output_str[len(inp):].strip())    

        contr_out = steer_contr.track_setpoint(inp, k, lmbda=l, do_sample=True, temp = temp)
        contr_completions.append(contr_out[len(inp):].strip())
        if l == 2 and counter % 60 == 0:
            
            print(f"PROMPT: {inp}")
            print(f"\nSteered output: {contr_out}")
            print(f"\nUnsteered output: {output_str}\n")
        
        counter+=1



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
            print(f"\ntoxified: {samples[i]}")
            print(f"steered: {contr_completions[i]}")
            print(f"toxicity score: {contr_preds[i]['score']}\n")
            print(f"unsteered: {un_completions[i]}\n")

        if contr_preds[i]["label"] == "toxic":
            num_tox_contr += 1
        if un_preds[i]["label"] == "toxic":
            num_tox_un += 1

            
    print("\n\n\n\n\n\n\n\n") # in case you do not want to be forced to look at the wonderfully horrific outputs
    print(f"num safeified: {n}")
    print(f"num unsafeified: {m}")
    print(f"num tox un: {num_tox_un}")
    print(f"num tox contr: {num_tox_contr}\n")