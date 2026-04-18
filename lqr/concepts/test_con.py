import torch as th
import pickle
from lqr.steering import LQRSteering
import time
import con_data_script as utils
import json
import csv
import yaml
import os

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

def run_trials(model, tokenizer, num_trials, A, X_contr, l_list=[1], q_list=[0.1], r_list=[1], qf_list=[0.1], k=100, do_sample=True, filename="json_out", batch_size=10):
    # do_sample = False
    # print("lambda,q,r,qf,num_safeified,num_unsafeified,num_tox_un,num_tox_contr,dist1_base,dist2_base,dist3_base,dist1_steered,dist2_steered,dist3_steered, ppl_base, ppl_steered")
    # ds = load_dataset("HumanLLMs/Human-Like-DPO-Dataset")["train"]

    # prompts = []
    # for i in range(len(ds)):
    #     prompts.append(ds[i]["prompt"])
    # samples = random.sample(prompts, num_trials)

    print(A.device)
    print(X_contr.device)

    samples = []
    for i in range(num_trials):
        samples.append("Once upon a time")

    print(samples[0:3])

    output_str = []
    # for i in range(0, len(samples), batch_size):
    #     batch = samples[i:i+batch_size]
    #     start_time = time.perf_counter()
    #     # k=50
    #     inputs = tokenizer(
    #             batch, 
    #             return_tensors="pt", 
    #             padding=True,
    #             truncation=True,
    #         ).to(device)
    #     input_ids = inputs["input_ids"]
    #     attention_mask = inputs["attention_mask"]

    #     data_list = []
    #     with th.no_grad():
    #         output_un = model.generate(
    #                         input_ids=input_ids,
    #                         attention_mask=attention_mask,
    #                         max_new_tokens=k,
    #                         return_dict_in_generate=True,
    #                         do_sample=do_sample,
    #                         top_p=0.3,
    #                         repetition_penalty=1.2,
    #                         temperature=1,
    #                         use_cache=False,
    #                         pad_token_id=tokenizer.eos_token_id,
    #                     )

    #         output = tokenizer.batch_decode(output_un.sequences, skip_special_tokens=True)
    #         output_str.extend(output)
    #         postbase_time = time.perf_counter()

    # print(output_str)


    data_list = []
    for q in q_list:
        for r in r_list:
            for qf in qf_list:
                steer_contr = LQRSteering(model, tokenizer, q=q,r=r,qf=qf, A=A, contrastive_vecs=X_contr, preserve_mem=True)
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

                    print(contr_out)

                    for i in range(len(contr_out)):
                        data_list.append({
                            "lambda": l,
                            "steered": contr_out[i],
                            # "unsteered": output_str[i]
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
    file_path = "new_concepts/" + filename + ".csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = ["lambda", "steered", "unsteered"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data_list)

def test_mutlisteer(model, tokenizer, num_trials, A, X_contr, X_contr_alt, l_list=[1], q_list=[0.1], r_list=[1], qf_list=[0.1], k=100, do_sample=True, filename="json_out", batch_size=100):
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

    data_list = []
    for q in q_list:
        for r in r_list:
            for qf in qf_list:
                steer_contr = LQRSteering(model, tokenizer, q=q,r=r,qf=qf, A=A, contrastive_vecs=X_contr, preserve_mem=True)
                temp_data = []
                for l in l_list:
                    contr_completions = []
                    un_completions = []
                    
                    contr_out = []
                    for i in range(0, len(samples), batch_size):
                        batch = samples[i:i+batch_size]
                        contr = steer_contr.multisteer(batch, k, alt_contr=X_contr_alt, lmbda=l, alt_lmbda=1.5, do_sample=do_sample)
                        contr_out.extend(contr)
                    # contr = steer_contr.track_setpoint(prompt, k, lmbda=l, do_sample=do_sample)
                    # contr_out.extend(contr)

                    print(contr)

                    for i in range(len(contr_out)):
                        data_list.append({
                            "lambda": l,
                            "steered": contr_out[i],
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
    file_path = "new_concepts/" + filename + ".csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
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
    


    dog = load_file("gemma-2-2b-football")
    notdog = load_file("gemma-2-2b-nonfootball")

    church = load_file("gemma-2-2b-balloon")
    notchurch = load_file("gemma-2-2b-balloon")

    jac = load_file("gemma-2-2b-football_jac")
    

    X = dog["X"]
    X_f = notdog["X"]
    A = jac["A"]


    X_c = church['X']
    X_nc = notchurch['X']

    print(f"X device {X.device}")

    print(f"X shape: {X.shape}")
    print(f"X_ref shape: {X_f.shape}")
    # print(f"A shape: {A.shape}")

    X_contr = X - X_f
    # X_contr_dog = X - X_f
    # X_contr_church = X_c - X_nc
    del X
    del X_f

    # a = 0.5
    # X_contr = a*X_contr_church + (1-a)*X_contr_dog
    # print(X_contr)
    # l_list = [0.5, 1, 1.5, 2, 2.5]
    # l_list = [3, 3.5, 4]
    l_list = [1.5]

    # q_list = [0.1]
    # r_list = [1]
    # qf_list = [0.1]
    kp = 0.5
    ki = 0.5
    kd = 0.01

    # q_list = [0.1, 1]
    # r_list = [0.1, 1, 10]
    # qf_list = [0.1, 1, 10] 

    q_list = [1]
    r_list = [1]
    qf_list = [0.1]
    # q_list = [1, 10]
    # r_list = [1, 10]
    # qf_list = [0.1, 1, 10]
    # q_list = [0.1]
    # r_list = [1]
    # qf_list = [1]


    A_lr = th.zeros_like(A, device=device)
    for i, At in enumerate(A):
        rank = th.linalg.matrix_rank(At)
        print(f"layer {i} true rank: {rank}")
        U, S, Vh = th.linalg.svd(At, full_matrices=False)
        tol = 1.2  # or relative threshold
        rank = (S > tol).sum()
        print(f"reduced rank: {rank}")
        # tol = S.max() * 1e-6
        mask = S > tol
        U_k = U[:, mask]
        S_k = S[mask]
        Vh_k = Vh[mask, :]

        A_lr[i] = (U_k * S_k) @ Vh_k

    # num_trials = 3
    output_filename = "gemma-2-2b-lowrank"
    num_trials = 500
    # num_trials = 15
    run_trials(
        model, 
        tokenizer, 
        num_trials,
        A_lr, 
        X_contr, 
        l_list, 
        k=300,
        q_list=q_list, 
        r_list=r_list, 
        qf_list=qf_list,
        filename=output_filename
    )

    # test_mutlisteer(
    #     model, 
    #     tokenizer, 
    #     num_trials,
    #     A, 
    #     X_contr=X_contr_church,
    #     X_contr_alt=X_contr_dog,
    #     l_list=l_list, 
    #     k=300,
    #     # q_list, 
    #     # r_list, 
    #     # qf_list,
    #     filename=output_filename
    # )


if __name__ == "__main__":
    main()

