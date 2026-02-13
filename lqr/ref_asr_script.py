import test_ref as tref
import torch as th
import json
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
import io
import argparse

def get_harmful_instructions():
    """Load harmful instructions from AdvBench dataset."""
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url)
    dataset = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    instructions = dataset["goal"].tolist()
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test




def main():
    models = {
            "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
            "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
            "gemma9b": "google/gemma-2-9b-it"
        }

    model_keys = {  
            "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct",
            "google/gemma-2-9b-it": "gemma-2-9b-it"
        }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["qwen3b", "llama8b", "gemma9b"],
        default="llama8b",
    )

    parser.add_argument(
        "--setpoint",
        choices=["lfs", "ang"],
        default="lfs",
    )

    args = parser.parse_args()

    if args.model in models:
        model_name = models[args.model]
        print(f"Running model: {model_name}")
    else:
        raise ValueError("vro...")
    # prompts = utils.get_refused_prompts()
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "Qwen/Qwen2.5-3B-Instruct"
    # model_name = "Qwen/Qwen2.5-14B-Instruct"
    # model_name = "google/gemma-2-9B-it"

    # output_filename = "gemma-2-9b-it-TEST"
    # output_filename = "llama-8b-it-angle-qrqf-sweep"
    # output_filename = "gemma-2-9B-it-one-token-lfs-sanity"

    # l_list = [0.5, 1, 1.5, 2, 2.5]
    # l_list = [0.7,0.8,0.9,1,1.1,1.2,1.3]
    l_list = [1.5]
    # l_list = [1, 1.5, 2]

    # q_list = [0.1]
    # r_list = [1]
    # qf_list = [10]
    # q_list = [0.1]
    # r_list = [1]
    # qf_list = [10]
    q_list = [1]
    r_list = [1]
    qf_list = [0.1]
    m = model_name.split("/")[-1]
    output_filename = f"{m}_{args.setpoint}_{q_list[0]}_{r_list[0]}_{qf_list[0]}"

# q: 0.1, r: 1, qf: 10

    key = model_keys[model_name]
    
    # ref = tref.load_file("gemma-2-9b-it-ref")
    # nonref = tref.load_file("gemma-2-9b-it-nonref")
    # # nonref = tref.load_file("gemma-2-9b-it-compliant")
    # jac = tref.load_file("gemma-2-9b-it-nonref_jac")
    
    # ref = tref.load_file("Qwen2.5-3B-Instruct-ref")
    # nonref = tref.load_file("Qwen2.5-3B-Instruct-nonref")
    # jac = tref.load_file("Qwen2.5-3B-Instruct-nonref_jac")

    # ref = load_file("Qwen2.5-14B-Instruct-ref")
    # nonref = load_file("Qwen2.5-14B-Instruct-nonref")
    # jac = load_file("Qwen2.5-14B-Instruct-nonref_jac")

    ref = tref.load_file(key + "-ref")
    nonref = tref.load_file(key + "-nonref")
    jac = tref.load_file(key + "-nonref_jac")


    model, tokenizer = tref.utils.load_model(model_name, quant=True)
    # harmful_prompts = tref.utils.get_refused_prompts()[416:]
    _, harmful_prompts = get_harmful_instructions()
    formatted_harmful_prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p + "\n\n"}],
        tokenize=False,
        add_generation_prompt=True
    ) for p in harmful_prompts]
    print(formatted_harmful_prompts[0])

    X = nonref["X"]
    X_ref = ref["X"]
    A = jac["A"]
    print(f"X device {X.device}")

    print(f"X shape: {X.shape}")
    print(f"X_ref shape: {X_ref.shape}")
    print(f"A shape: {A.shape}")


    # X_contr_norm = X_contr #/ th.norm(X_contr)
    # prefix_sum = th.cumsum(X_contr_norm, dim=0)
    # shifted_refusal_dirs = X_contr_norm.roll(1, dims=0)
    # shifted_refusal_dirs[0] = X_contr_norm[0]
    # diff_from_first = X_contr_norm - shifted_refusal_dirs
    
    # X_contr = 0.9*X_contr_norm + 0.01*prefix_sum + 0.01*diff_from_first
    
    
    k=512
    

    num_trials = 104
    # num_trials = 1
    

    if args.setpoint == 'lfs':
        X_contr = X - X_ref
        del X
        del X_ref
        tref.run_trials_lfs(
            model, 
            tokenizer, 
            formatted_harmful_prompts, 
            num_trials, 
            A, 
            X_contr, 
            l_list, 
            q_list, 
            r_list, 
            qf_list,
            k=k,
            do_sample=False,
            filename=output_filename,
            batch_size=50
        )
    elif args.setpoint == 'ang':
        X_contr = X_ref - X
        del X
        del X_ref
        tref.run_trials_ang(
            model, 
            tokenizer, 
            formatted_harmful_prompts, 
            num_trials, 
            A, 
            X_contr, 
            l_list, 
            q_list, 
            r_list, 
            qf_list,
            k=k,
            do_sample=False,
            filename=output_filename,
            batch_size=50
        )
    else:
        raise ValueError("bubby")


    # del X_contr
    print("__________________________________________\nFinished Initial Sweep\n__________________________________________")

    import asr as asr

    it_format = asr.llama_it_format

    data_file_path = asr.PATH + output_filename + ".txt"
    with open(data_file_path, 'r') as file:
        data = json.load(file)

    print("UNSTEERED")
    inps = asr.get_classifier_inputs(it_format, data[0]["unsteered output"])
    u_score = asr.harmbench_judge(inps,batch_size=4)
    data[0]["unsteered ASR"] = u_score

    print("STEERED")
    for d in data[1]["sweeps"]:
        q = d["Q"]
        r = d["R"]
        qf = d["Qf"]
        l = d["lambda"]
        print(f"lambda: {l}, q: {q}, r: {r}, qf: {qf}")
        inps = asr.get_classifier_inputs(it_format, d["steered output"])
        s_score = asr.harmbench_judge(inps,batch_size=4)

        d["Steered ASR"] = s_score

    with open(data_file_path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    main()