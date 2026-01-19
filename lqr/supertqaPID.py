from test_tqa import PATH, load_file, run_trials, run_base_model, no_it_format
from testMMLU import test_mmlu
import tqa_data_script as utils
import json
import os.path


def generation(models, params):
    prompts = utils.get_all_questions_no_it()
    for model_name in models:
        print(f"Running Generation: {model_name}")
        model, tokenizer = utils.load_model(model_name, quant=True)
        l_list = params[model_name]['l_list']
        lqr_p = params[model_name]['LQR_params']
        q = lqr_p[0]
        r = lqr_p[1]
        qf = lqr_p[2]
        print(f"q:{q}, r:{r}, qf:{qf}")

        f_pfix = params[model_name]['filename_pfix']
        false = load_file(f_pfix + "-false")
        true = load_file(f_pfix + "-true")
        jac = load_file(f_pfix + "-true_jac")

        if true is None or false is None or jac is None:
            print(f"Skipping model -- contrastive vectors or jacobians not found")
            continue
        
        X = true["X"]
        A = jac["A"]
        X_f = false["X"]
        X_contr = X - X_f

        output_filename= f_pfix + "_tqa_eval"
        for i in range(5):
            # print(f"running test_toxicity.py: {model_name}")
            num_trials = 817
            run_trials(
                model, 
                tokenizer, 
                prompts, 
                no_it_format,
                num_trials, 
                A, 
                X_contr, 
                l_list, 
                [q], 
                [r], 
                [qf],
                filename=output_filename
            )
        del model
        model=None
        del tokenizer
        tokenizer=None
        print(f"Finish generation: {model_name}, output to {f_pfix}_tqa_eval.txt")
        print("___________________________________________")


def base_model_generation(models, params):
    prompts = utils.get_all_questions_no_it()
    for model_name in models:
        print(f"Running Generation: {model_name}")
        model, tokenizer = utils.load_model(model_name, quant=True)
        f_pfix = params[model_name]['filename_pfix']
        output_filename= f_pfix + "_tqa_BASE_MODEL_eval"

        sweeps = []
        for i in range(5):
            # print(f"running test_toxicity.py: {model_name}")
            num_trials = 817
            s = run_base_model(
                model, 
                tokenizer, 
                prompts, 
                no_it_format,
                num_trials, 
                filename=output_filename
            )
            sweeps.append(s)
        with open(PATH + output_filename + ".txt", 'w', encoding='utf-8') as json_file:
            json.dump(sweeps, json_file, indent=4)

        print(f"Finish generation: {model_name} BASE MODEL, output to {output_filename}.txt")
        print("___________________________________________")

    # def run_base_model(model, tokenizer, prompts, it_format, num_trials, k=50, do_sample=True, filename="json_out", batch_size=100):

    
def mmlu(models, params):
    mmlu_filename = "MMLU_trials_TQA"
    try:
        with open(PATH + mmlu_filename + ".txt", "r") as f:
            mmlu_data = json.load(f)
    except FileNotFoundError:
        mmlu_data = {}

    for model_name in models:
        print(f"Running MMLU: {model_name}")
        model, tokenizer = utils.load_model(model_name, quant=True)
        l_list = params[model_name]['l_list']
        lqr_p = params[model_name]['LQR_params']
        instruct = params[model_name]['instruct']
        q = lqr_p[0]
        r = lqr_p[1]
        qf = lqr_p[2]
        print(f"q:{q}, r:{r}, qf:{qf}")

        f_pfix = params[model_name]['filename_pfix']
        false = load_file(f_pfix + "-false")
        true = load_file(f_pfix + "-true")
        jac = load_file(f_pfix + "-true_jac")

        if true is None or false is None or jac is None:
            print(f"Skipping model MMLU -- contrastive vectors or jacobians not found")
            continue

        X = true["X"]
        A = jac["A"]
        X_f = false["X"]
        X_contr = X - X_f

        out = test_mmlu(model, tokenizer, X_contr, A, lambda_list=l_list, q=q, r=r, qf=qf, N_PROMPTS=10, N_LOOP=100, BATCH_SIZE=4, N_SHOTS = 5, INSTRUCT=instruct)
        mmlu_data[model_name] = out

        print(f"Finished MMLU model {model_name}")
    with open(PATH + mmlu_filename + ".txt", 'w') as file:
        json.dump(mmlu_data, file, indent=4)


def main():
    models = [
        'google/gemma-2-2b',
        "meta-llama/Meta-Llama-3-8B",
        'google/gemma-2-9b',
        "Qwen/Qwen2.5-3B",
        "meta-llama/Llama-3.2-1B",
        "Qwen/Qwen2.5-14B",
    ]

    params = {
        'google/gemma-2-2b': {'filename_pfix': 'gemma-2-2b', 
                              'l_list': [2, 2.5, 3.5], 
                              'LQR_params': [0.1,1,0.1],
                              'instruct': False},
        "meta-llama/Meta-Llama-3-8B": {'filename_pfix': 'Llama-3-8B', 
                              'l_list': [2, 2.5, 3.5], 
                              'LQR_params': [0.1,10,10],
                              'instruct': False},
        'google/gemma-2-9b': {'filename_pfix': 'gemma-2-9b', 
                              'l_list': [2, 2.5, 3.5], 
                              'LQR_params': [0.1,1,0.1],
                              'instruct': False},
        "Qwen/Qwen2.5-14B": {'filename_pfix': 'Qwen2.5-14B', 
                              'l_list': [2, 2.5, 3.5], 
                              'LQR_params': [1,1,1],
                              'instruct': False},
        "Qwen/Qwen2.5-3B": {'filename_pfix': 'Qwen2.5-3B', 
                              'l_list': [2, 2.5, 3.5], 
                              'LQR_params': [0.1,1,0.1],
                              'instruct': False},
        "meta-llama/Llama-3.2-1B": {'filename_pfix': 'Llama-3.2-1B', 
                              'l_list': [2, 2.5, 3.5], 
                              'LQR_params': [0.1,1,1],
                              'instruct': False},
    }

    # Generate outputs, measure toxicity, and measure Dist 1,2,3
    # generation(models, params)
    # print("Done with all generations")

    # Get MMLU performance
    # mmlu(models, params)
    # print("finish all MMLU")

    # base model evaluation
    base_model_generation(models, params)

if __name__ == "__main__":
    main()

    