import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer, RobertaForSequenceClassification, pipeline, BitsAndBytesConfig
import lqr.lqr_utils as lqr
from functools import partial
import pickle
from lqr.steering_aff import LQRSteering
from datasets import load_dataset
import random
import time
import lqr.toxicity.tox_data_script as utils
import json
import yaml
import argparse

from lqr.config import config
PATH = Path(__file__).resolve().parent / config["environment"]["tox_data_path"] ["environment"]["tox_data_path"]
PICKLE_JAR = config_data["environment"]["pickle_jar"]

# ppl = load("perplexity", module_type="metric")

device = th.device("cuda" if th.cuda.is_available() else "cpu")

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


# def load_file(filename):
#     with open("../../scratch/"+filename+".pkl", "rb") as f:
#         loaded_tensors = pickle.load(f)
#     return loaded_tensors

def load_file(filename):
    try:
        with open(PICKLE_JAR + filename + ".pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    

def run_trials(model, tokenizer, toxic_prompts, num_trials, A, X_nom, X_contr, l_list=[1], q_list=[0.1], r_list=[10], qf_list=[0.1], BATCH_SZ=500):
    samples = random.sample(toxic_prompts, num_trials)
    
    do_sample = True
    temp = 1

    # headers: 
    print("lambda,q,r,qf,num_safeified,num_unsafeified,num_tox_un,num_tox_contr,dist1_base,dist2_base,dist3_base,dist1_steered,dist2_steered,dist3_steered, ppl_steered, ppl_base")

    start_time = time.perf_counter()

    # for start in range(0, len(samples), BATCH_SZ):
                
        # batch = samples[start:start+BATCH_SZ]
    inputs = tokenizer(
            samples, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    un_completions = []
    k=10
    with th.no_grad():
        output_un = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=k,
                        return_dict_in_generate=True,
                        do_sample=do_sample,
                        top_p=0.3,
                        repetition_penalty=1.2,
                        temperature=temp,
                        use_cache=False,
                        pad_token_id=tokenizer.eos_token_id,
                        # **model_generation_kwargs, #
                    )

    output_str = tokenizer.batch_decode(output_un.sequences, skip_special_tokens=True)
    postbase_time = time.perf_counter()

    # print(f"base output: {output_str}")

    for i, inp in enumerate(samples):
        un_completions.append(output_str[i][len(inp):].strip())
    un_preds = classifier(un_completions)

    dist_1_base = calculate_dist_n(un_completions, n=1)
    dist_2_base = calculate_dist_n(un_completions, n=2)
    dist_3_base = calculate_dist_n(un_completions, n=3)
    
    sweep = []
    signals = []
    for q in q_list:
        for r in r_list:
            for qf in qf_list:
                # if q == 0.1 and r == 0.1:
                    # continue
                
                # steer_contr = LQRSteering(model, tokenizer, q=q, r=r, qf=qf, contrastive_vecs=X_contr, preserve_mem=True, aff=True)
                steer_contr = LQRSteering(model, tokenizer, q=q, r=r, qf=qf, A=A, contrastive_vecs=X_contr, preserve_mem=True, aff=True)
                # steer_contr = LQRSteering(model, tokenizer, q=q,r=r,qf=qf, A=A, contrastive_vecs=X_contr, preserve_mem=True)
                post_init_time = time.perf_counter()
                
                for l in l_list:
                    contr_completions = []

                    contr_out = []
                    for i in range(0, len(samples), BATCH_SZ):
                        batch = samples[i:i+BATCH_SZ]
                        contr = steer_contr.track_affine(batch, X_nom, k, lmbda=l, do_sample=do_sample)
                        # contr = steer_contr.track_setpoint(batch, k, lmbda=l, do_sample=do_sample)
                        contr_out.extend(contr)
                        senales = steer_contr.setpoint_signals
                        signals.append(senales)
                    # contr_out = steer_contr.track_setpoint(samples, k, lmbda=l, do_sample=do_sample, temp = temp)
                    print(f"steered output: {contr_out}")
                    # contr_out = steer_contr.track_traj(X.unsqueeze(1),inp, k, do_sample=True)
                    # print("")
                    # if (l == 1.25) and counter % 60 == 0:
                    # if counter % 60 == 0:
                    for i, inp in enumerate(samples):
                        un_completions.append(output_str[i][len(inp):].strip())
                        # contr_completions.append(contr_out[i][len(inp):].strip())
                        
                    # print(f"PROMPT: {samples}")
                    # print(f"\nSteered output: {contr_out}")
                    # print(f"\nUnsteered output: {output_str}")
                    end_time = time.perf_counter()
                    # print(f"runtimesdfsd: {end_time - postbase_time}")

                    contr_preds = classifier(contr_completions)
                    
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

                        if contr_preds[i]["label"] == "toxic":
                            num_tox_contr += 1
                    ppl_steered = 0
                    ppl_unsteered = 0

                    data = {
                        "lambda": l,
                        "Q": q,
                        "R": r,
                        "Qf": qf,
                        "prompts": samples,
                        "unsteered output": output_str, 
                        "steered output": contr_out,
                        "num safeified": n,
                        "num unsafeified": m,
                        "num tox unsteered": num_tox_un,
                        "num tox steered": num_tox_contr,
                        "dist 1 base": dist_1_base,
                        "dist 2 base": dist_2_base,
                        "dist 3 base": dist_3_base,
                        "dist 1 steered": dist_1_steered,
                        "dist 2 steered": dist_2_steered,
                        "dist 3 steered": dist_3_steered,
                        }
                    sweep.append(data)
                    print(l,q,r,qf,n,m,num_tox_un,num_tox_contr,dist_1_base,dist_2_base,dist_3_base,dist_1_steered,dist_2_steered,dist_3_steered,ppl_steered,ppl_unsteered, sep=",")

                del steer_contr

    # steer_contr.plot_unorms(f"lqr_unorms")
    end_time = time.perf_counter()
    print(f"runtime: {end_time - start_time}")
    return sweep, signals


def main():
    models = {
            "gemma2b": 'google/gemma-2-2b',
            "llama8b": "meta-llama/Meta-Llama-3-8B",
            "qwen3b": "Qwen/Qwen2.5-3B",
            "gemma9b": 'google/gemma-2-9b',
        }

    model_keys = {  
            'google/gemma-2-2b': 'gemma-2-2b',
            "meta-llama/Meta-Llama-3-8B": "Meta-Llama-3-8B",
            "Qwen/Qwen2.5-3B": "Qwen2.5-3B",
            'google/gemma-2-9b': 'gemma-2-9b'
        }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["qwen3b", "llama8b", "gemma2b", "gemma9b"],
        default="gemma2b",
    )

    args = parser.parse_args()
    if args.model in models:
        model_name = models[args.model]
        print(f"Running model: {model_name}")
    else:
        raise ValueError("vro...")

    m = model_name.split("/")[-1]
    output_filename = f"{m}_affine"
    model, tokenizer = utils.load_model(model_name, quant=True)

    print(f"Running test_toxicity.py: {model_name}")
    print(f"model dtype: {model.dtype}")
    key = model_keys[model_name]

    tox = load_file(key + "-tox")
    nontox = load_file(key + "-nontox")
    jac = load_file(key + "-nontox_jac")

    X = nontox["X"]
    A = jac["A"]
    X_tox = tox["X"]
    X_contr = X - X_tox

    print(X_contr.shape)
    toxic_prompts = utils.get_tox_prompts(0.9, 1)
    # toxic_prompts = utils.get_tox_prompts(0.0, 1)
    l_list = [0.5, 0.75, 1]
    q_list = [0.1, 1]
    r_list = [0.1, 1]
    qf_list = [0.1, 1]

    sweeps = []

    print(f"running test_toxicity.py: {model_name}")
    for i in range(1):
        num_trials = 100
        s, signals = run_trials(
            model, 
            tokenizer, 
            toxic_prompts, 
            num_trials, 
            A, 
            X,
            X_contr, 
            l_list, 
            q_list, 
            r_list, 
            qf_list
        )
        sweeps.extend(s)

    print(sweeps[0])
    with open(PATH + output_filename + ".txt", 'w', encoding='utf-8') as json_file:
        json.dump(sweeps, json_file, indent=4)

if __name__ == "__main__":
    main()

