import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer, RobertaForSequenceClassification, pipeline, BitsAndBytesConfig
import lqr_utils_seq as lqr
from functools import partial
import pickle
from steering import LQRSteering
from PIDsteering import PIDSteering
from datasets import load_dataset
import random
import time
import tqa_data_script as utils
import json
import csv
import yaml

with open('config/config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)
PICKLE_JAR = config_data["environment"]["pickle_jar"]

device = th.device("cuda" if th.cuda.is_available() else "cpu")

def load_file(filename):
    try:
        with open(PICKLE_JAR + filename + ".pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def run_trials(model, tokenizer, num_trials, A, X_contr, l_list=[1], q_list=[0.1], r_list=[1], qf_list=[0.1], k=100, do_sample=True, filename="json_out", batch_size=100):
    # do_sample = False
    # print("lambda,q,r,qf,num_safeified,num_unsafeified,num_tox_un,num_tox_contr,dist1_base,dist2_base,dist3_base,dist1_steered,dist2_steered,dist3_steered, ppl_base, ppl_steered")
    # ds = load_dataset("HumanLLMs/Human-Like-DPO-Dataset")["train"]

    # prompts = []
    # for i in range(len(ds)):
    #     prompts.append(ds[i]["prompt"])
    # samples = random.sample(prompts, num_trials)

    samples = []
    for i in range(num_trials):
        samples.append("Once upon a time")

    print(samples[0:3])

    output_str = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        start_time = time.perf_counter()
        # k=50
        inputs = tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
            ).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        data_list = []
        with th.no_grad():
            output_un = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=k,
                            return_dict_in_generate=True,
                            do_sample=do_sample,
                            top_p=0.3,
                            repetition_penalty=1.2,
                            temperature=1,
                            use_cache=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )

            output = tokenizer.batch_decode(output_un.sequences, skip_special_tokens=True)
            output_str.extend(output)
            postbase_time = time.perf_counter()

    print(output_str)

    data_list = []
    for q in q_list:
        for r in r_list:
            for qf in qf_list:
                steer_contr = LQRSteering(model, tokenizer, q=q,r=r,qf=qf, A=A, contrastive_vecs=X_contr, perserve_mem=True)
                temp_data = []
                for l in l_list:
                    contr_completions = []
                    un_completions = []
                    
                    contr_out = []
                    for i in range(0, len(samples), batch_size):
                        batch = samples[i:i+batch_size]
                        contr = steer_contr.track_setpoint(batch, k, lmbda=l, do_sample=do_sample)
                        contr_out.extend(contr)
                    # contr = steer_contr.track_setpoint(prompt, k, lmbda=l, do_sample=do_sample)
                    # contr_out.extend(contr)

                    print(contr)

                    for i in range(len(contr_out)):
                        data_list.append({
                            "lambda": l,
                            "steered": contr_out[i],
                            "unsteered": output_str[i]
                        })
                    # data = {
                    #     "lambda": l,
                    #     "steered": contr_out,
                    #     "unsteered": output_str
                    # }
                    # data_list.append(data)
    # file_path = "concepts/" + filename + ".txt"
    # with open(file_path, 'w') as file:
    #     json.dump(data_list, file, indent=4)
    file_path = "concepts/" + filename + ".csv"

    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = ["lambda", "steered", "unsteered"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data_list)



def main():
    # prompts = utils.get_refused_prompts()
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model_name = "google/gemma-2-2b"
    # model_name = "Qwen/Qwen2.5-3B-Instruct"
    # model_name = "meta-llama/Meta-Llama-3.1-8B"
    # model_name = "Qwen/Qwen2.5-14B-Instruct"

    # output_filename = "Llama-3.1-8B-Instruct-sweep"
    # it_format = qwen_it_format

    print("Running test_con.py:", model_name)

    model, tokenizer = utils.load_model(model_name, quant=True)
    # prompts = utils.get_questions(tokenizer)
    # prompts = utils.get_questions_no_it()
    prompt = "Once upon a time"
    


    dog = load_file("gemma-2-2b-dog")
    notdog = load_file("gemma-2-2b-nondog")
    jac = load_file("gemma-2-2b-dog_jac")
    

    X = dog["X"]
    X_f = notdog["X"]
    A = jac["A"]
    print(f"X device {X.device}")

    print(f"X shape: {X.shape}")
    print(f"X_ref shape: {X_f.shape}")
    # print(f"A shape: {A.shape}")

    X_contr = X - X_f
    del X
    del X_f

    print(X_contr)
    # l_list = [0.5, 1, 1.5, 2, 2.5]
    # l_list = [3, 3.5, 4]
    l_list = [1,2,3]

    # q_list = [0.1]
    # r_list = [1]
    # qf_list = [0.1]
    kp = 0.5
    ki = 0.5
    kd = 0.01

    # q_list = [0.1, 1]
    # r_list = [0.1, 1, 10]
    # qf_list = [0.1, 1, 10]

    # q_list = [0.1, 1]
    # r_list = [0.1, 1, 10]
    # qf_list = [0.1, 1, 10]
    # q_list = [1, 10]
    # r_list = [1, 10]
    # qf_list = [0.1, 1, 10]
    # q_list = [0.1]
    # r_list = [1]
    # qf_list = [1]

    num_trials = 100
    output_filename = "gemma-2-2b-vague-out"
    # num_trials = 437
    # num_trials = 15
    run_trials(
        model, 
        tokenizer, 
        num_trials,
        A, 
        X_contr, 
        l_list, 
        # q_list, 
        # r_list, 
        # qf_list,
        filename=output_filename
    )



if __name__ == "__main__":
    main()

