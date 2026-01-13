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
import tqa_data_script as utils
import json
import yaml
import matplotlib.pyplot as plt

with open('config/config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)
PICKLE_JAR = config_data["environment"]["pickle_jar"]
PATH = config_data["environment"]["tqa_data_path"]

device = th.device("cuda" if th.cuda.is_available() else "cpu")



def load_file(filename):
    with open(PICKLE_JAR+filename+".pkl", "rb") as f:
        loaded_tensors = pickle.load(f)
    return loaded_tensors

def run_trials(model, tokenizer, prompts, A, X_contr, l_list=[1], q=0.1, r=10, qf=0.1, k=50, do_sample=True, filename="json_out", batch_size=50):
    # do_sample = False
    # print("lambda,q,r,qf,num_safeified,num_unsafeified,num_tox_un,num_tox_contr,dist1_base,dist2_base,dist3_base,dist1_steered,dist2_steered,dist3_steered, ppl_base, ppl_steered")
    steer = LQRSteering(model, tokenizer, q=q,r=r,qf=qf, A=A, contrastive_vecs=X_contr, perserve_mem=True)

    output = []    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        signals = steer.generate_and_collect(batch, 1, do_sample)
        output.append(signals)

    # print(f"unsteered: {output}")
    
    signals_per_lambda = []
    for l in l_list:
        steered_signals = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            contr = steer.track_setpoint(batch, 1, lmbda=l, do_sample=do_sample)
            curr_signals = steer.setpoint_signals
            
            steered_signals.append(curr_signals)

        # print(f"steered (l={l}): {steered_signals}")
        # print(f"len: {len(steered_signals[0])}")
        signals_per_lambda.append(th.mean(th.tensor(steered_signals), dim=0).tolist())

    print(signals_per_lambda)
    print(len(signals_per_lambda[0]))
    plt.plot(output[0][1:], label=f"unsteered")
    for i, row in enumerate(signals_per_lambda):
        # temp = [n / signals_per_lambda[1][1+i] for i, n in enumerate(row[1:])]

        # plt.plot(temp, label=f"lambda = {l_list[i]}")
        plt.plot(row[1:], label=f"lambda = {l_list[i]}")
        plt.axhline(y=l_list[i], color='red', linestyle='--', linewidth=1)

    # plt.plot(mean_list, linewidth=3, color="black", label="Mean")

    plt.legend()
    # plt.savefig("llama_lambda_track_ratio.png")
    plt.savefig("llama_lambda_track.png")
    
                    


    # data_list.append({"sweeps": sweep_data_list})
    # file_path = PATH + filename + ".txt"
    # with open(file_path, 'w') as file:
    #     json.dump(data_list, file, indent=4)
    
    # end_time = time.perf_counter()
    # print(f"runtime: {end_time - start_time}")

def main():
    # prompts = utils.get_refused_prompts()
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "google/gemma-2-9B-it"
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "Qwen/Qwen2.5-3B-Instruct"
    # model_name = "Qwen/Qwen2.5-14B-Instruct"

    # output_filename = "Llama-3.1-8B-Inst-advers-sweep"

    model, tokenizer = utils.load_model(model_name, quant=True)

    print(f"model layers: {len(model.model.layers)}")
    prompts = utils.get_questions_no_it()
    # prompts = utils.get_questions(tokenizer)
    
    print(prompts[0])
    print(len(prompts))

    
    # true = load_file("Qwen2.5-3B-Instruct-true")
    # false = load_file("Qwen2.5-3B-Instruct-false")
    # jac = load_file("Qwen2.5-3B-Instruct-true_jac")

    true = load_file("Llama-3.1-8B-Instruct-true")
    false = load_file("Llama-3.1-8B-Instruct-false")
    jac = load_file("Llama-3.1-8B-Instruct-true_jac")

    X = true["X"]
    X_f = false["X"]
    A = jac["A"]
    print(f"X device {X.device}")

    print(f"X shape: {X.shape}")
    print(f"X_ref shape: {X_f.shape}")
    print(f"A shape: {A.shape}")

    X_contr = X - X_f
    print(f"X: {X}")
    print(f"X norm: {th.norm(X,dim=1)}")
    print(f"Xf: {X_f}")
    print(f"Xf norm: {th.norm(X_f,dim=1)}")
    print(f"contr: {X_contr}")
    print(f"contr norm: {th.norm(X_contr,dim=1)}")
    del X
    del X_f
    # l_list = [0.5, 1, 1.5, 2, 2.5]
    l_list = [0.5, 1, 1.5, 2, 2.5]

    # q_list = [0.1, 1]
    # r_list = [0.1, 1, 10]
    # qf_list = [0.1, 1]

    q = 1
    r = 10
    qf = 10

    # num_trials = 817
    # num_trials = 437
    # num_trials = 15

    run_trials(
        model, 
        tokenizer, 
        prompts[:25], 
        A, 
        X_contr, 
        l_list, 
        q, 
        r, 
        qf,
    )

# def run_trials(model, tokenizer, prompts, it_format, num_trials, A, X_contr, l_list=[1], q_list=[0.1], r_list=[10], qf_list=[0.1], k=50, do_sample=False, filename="json_out"):



if __name__ == "__main__":
    main()

