import asr as asr
import test_ref as tref
import json

def main():
    # prompts = utils.get_refused_prompts()
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    # model_name = "Qwen/Qwen2.5-14B-Instruct"

    output_filename = "test"

    l_list = [0.5, 0.75, 1, 1.25, 1.5]
    # l_list = [0.75]

    # q_list = [0.1]
    # r_list = [1]
    # qf_list = [0.1]

    q_list = [0.1, 1, 10]
    r_list = [0.1, 1, 10]
    qf_list = [0.1, 1, 10]

    num_trials = 104

    
    # ref = load_file("llama-3.1-8B-it-ref")
    # nonref = load_file("llama-3.1-8B-it-nonref")
    # jac = load_file("llama-3.1-8B-it-nonref_jac")
    
    ref = tref.load_file("Qwen2.5-3B-Instruct-ref")
    nonref = tref.load_file("Qwen2.5-3B-Instruct-nonref")
    jac = tref.load_file("Qwen2.5-3B-Instruct-nonref_jac")

    # ref = load_file("Qwen2.5-14B-Instruct-ref")
    # nonref = load_file("Qwen2.5-14B-Instruct-nonref")
    # jac = load_file("Qwen2.5-14B-Instruct-nonref_jac")


    model, tokenizer = tref.utils.load_model(model_name, quant=True)
    harmful_prompts = tref.utils.get_refused_prompts()[416:]
    formatted_harmful_prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True
    ) for p in harmful_prompts]

    X = nonref["X"]
    X_ref = ref["X"]
    A = jac["A"]
    print(f"X device {X.device}")

    print(f"X shape: {X.shape}")
    print(f"X_ref shape: {X_ref.shape}")
    print(f"A shape: {A.shape}")

    X_contr = X - X_ref
    del X
    del X_ref
    
    # num_trials = 10
    tref.run_trials(
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
        output_filename
    )

    print("__________________________________________\nFinished Initial Sweep\n__________________________________________")


    it_format = asr.qwen_it_format

    data_file_path = asr.PATH + output_filename + ".txt"
    with open(data_file_path, 'r') as file:
        data = json.load(file)

    print("UNSTEERED")
    inps = asr.get_classifier_inputs(it_format, data[0]["unsteered output"])
    u_score = asr.harmbench_judge(inps)
    data[0]["unsteered ASR"] = u_score

    print("STEERED")
    for d in data[1]["sweeps"]:
        q = d["Q"]
        r = d["R"]
        qf = d["Qf"]
        l = d["lambda"]
        print(f"lambda: {l}, q: {q}, r: {r}, qf: {qf}")
        inps = asr.get_classifier_inputs(it_format, d["steered output"])
        s_score = asr.harmbench_judge(inps)

        d["Steered ASR"] = s_score

    with open(data_file_path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    main()