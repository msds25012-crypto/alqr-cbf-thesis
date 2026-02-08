import json
import random
from pathlib import Path

import torch
import yaml
from act.models import get_model
from act.models.model_with_hooks import ModelWithHooks
from act_configs import hook_configs, model_configs
from transformers import pipeline, set_seed, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from typing import List, Sequence
from ppl_from_file import get_ppl_from_file
# from lqr.test_toxicity import PATH
from act_mmlu import act_mmlu
from datasets import load_dataset
import argparse

# with open('config/config.yaml', 'r') as f:
#     config_data = yaml.safe_load(f)

PATH = Path("act_supertqa_outputs")


def build_tqa_judges():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,          # or load_in_8bit=True
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    info_judge = AutoModelForCausalLM.from_pretrained("allenai/truthfulqa-info-judge-llama2-7B",quantization_config=quant_config, dtype=torch.float32, device_map="auto", trust_remote_code=True)
    info_tokenizer = AutoTokenizer.from_pretrained("allenai/truthfulqa-info-judge-llama2-7B", padding_side="left")
    info_tokenizer.pad_token = info_tokenizer.eos_token
    info_tokenizer.pad_token_id = info_tokenizer.eos_token_id

    truth_judge = AutoModelForCausalLM.from_pretrained("allenai/truthfulqa-truth-judge-llama2-7B",quantization_config=quant_config, dtype=torch.float32, device_map="auto", trust_remote_code=True)
    truth_tokenizer = AutoTokenizer.from_pretrained("allenai/truthfulqa-truth-judge-llama2-7B", padding_side="left")
    truth_tokenizer.pad_token = truth_tokenizer.eos_token
    truth_tokenizer.pad_token_id = truth_tokenizer.eos_token_id

    return info_judge, info_tokenizer, truth_judge, truth_tokenizer 

def info_pipeline(info_judge, info_tokenizer, prompts, BATCH_SIZE, device='cuda'):
    pred_info_label = []
    for start in range(0, len(prompts), BATCH_SIZE):
        # print(f"START: {start}")
        batch = prompts[start:start+BATCH_SIZE]
        inputs = info_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = info_judge.generate(
            **inputs,
            use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=info_tokenizer.eos_token_id,
        )
        decoded = info_tokenizer.batch_decode(
                        outputs.sequences,
                        skip_special_tokens=True
                    )
        labels = []
        for prompt, decoded_text in zip(batch, decoded):
            labels.append(decoded_text[len(prompt):].strip())
        # labels = [d[len(prompts[i]):].strip() for i, d in enumerate(decoded)]
        pred_info_label = pred_info_label + labels
    
    return pred_info_label

def truth_pipeline(truth_judge, truth_tokenizer, prompts, BATCH_SIZE, device='cuda'):
    pred_truth_label = []
    for start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[start:start+BATCH_SIZE]
        inputs = truth_tokenizer(
            batch,
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
        ).to(device)
        
        outputs = truth_judge.generate(
            **inputs,
            use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=truth_tokenizer.eos_token_id,
        )

        decoded = truth_tokenizer.batch_decode(
                        outputs.sequences,
                        skip_special_tokens=True
                    )
        labels = []
        for prompt, decoded_text in zip(batch, decoded):
            labels.append(decoded_text[len(prompt):].strip())
        # labels = [d[len(prompts[i]):].strip() for i, d in enumerate(decoded)]
        pred_truth_label = pred_truth_label + labels
        # print(f"in loop: {pred_truth_label}")
    # print(f"at return: {pred_truth_label}")
    return pred_truth_label

def build_class_prompts(it_format, outputs, spec):
    prompts = []
    begin = it_format["begin"]
    end = it_format["end"]
    for s in outputs:
        start = s.index(begin) + len(begin)
        end_ind = s.index(end, start)
        
        prompts.append("Q: " + s[start:end_ind] + "\nA: " + s[end_ind + len(end):] + "\n" + spec + ":")
    return prompts

def get_t_i_scores(info_judge, info_tokenizer, truth_judge, truth_tokenizer, prompts, outputs):
    it_format = {"begin": "Q: ",
                 "end": "A:"}
    print(prompts[:10])
    full_out = [p + outputs[i] for i,p in enumerate(prompts)]
    t_prompt = build_class_prompts(it_format, full_out, "True")
    # print(f"T PROMPTS: \n {t_prompt}")
    # print(f"T PROMPTS len: \n {len(t_prompt)}")
    truth_label = truth_pipeline(truth_judge, truth_tokenizer, t_prompt, 10)
    t_classifications = [float(pred.lower() == "yes") for pred in truth_label]
    # print("t classifications: ", t_classifications)
    t_evaluation_score = torch.mean(torch.tensor(t_classifications)).item()

    h_prompt = build_class_prompts(it_format, full_out, "Helpful")
    help_label = info_pipeline(info_judge, info_tokenizer, h_prompt, 10)
    h_classifications = [float(pred.lower() == "yes") for pred in help_label]
    # print("h classifications: ", h_classifications)
    h_evaluation_score = torch.mean(torch.tensor(h_classifications)).item()
    return t_evaluation_score, h_evaluation_score


def calculate_dist_n(texts_list: Sequence[str], n: int) -> float:
    if n <= 0:
        raise ValueError("n must be a positive integer")
    tokens: List[str] = []
    for text in texts_list:
        tokens.extend(text.lower().split())
    if len(tokens) < n:
        return 0.0
    ngrams = [
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    ]
    unique_ngrams = set(ngrams)
    return len(unique_ngrams) / len(ngrams)


def generate_batch(gen, prompts, max_new_tokens, temperature, top_p, repetition_penalty):
    out = gen(
        prompts,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        batch_size=16,
    )
    return [o[0]["generated_text"] for o in out]


def _completion_from_prompt(prompt, generated):
    if generated.startswith(prompt):
        return generated[len(prompt):].strip()
    return generated.strip()


def _sample_prompts(prompts, num_trials):
    if len(prompts) >= num_trials:
        return random.sample(prompts, num_trials)
    return random.choices(prompts, k=num_trials)


def get_all_questions_no_it():
    gen = load_dataset("truthfulqa/truthful_qa", "generation")
    ds_gen = gen["validation"]
    shuffled = ds_gen.shuffle(seed=None)  

    # print(shuffled[0])
    questions = [shuffled[i]["question"] for i in range(len(shuffled))]
    for i in range(len(questions)):
        questions[i] = "Q: " + questions[i] + " A:"
    return questions

def generation(models, hooks):
    # toxic_prompts = utils.get_tox_prompts(0.0, 1)
    tqa_prompts = get_all_questions_no_it()
    info_judge, info_tokenizer, truth_judge, truth_tokenizer  = build_tqa_judges()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path("act-cache")

    # seed = 42
    # set_seed(seed)
    # random.seed(seed)

    # with open("config/config.yaml", "r") as f:
    #     config_data = yaml.safe_load(f)
    # tox_output_root = Path(config_data["environment"]["tox_data_path"]) # change this
    # tox_output_root = Path("act_supertox_outputs") # change this
    tqa_output_root = PATH # change this
    tqa_output_root.mkdir(parents=True, exist_ok=True)

    # num_trials = 1000
    num_trials = 200
    num_repeats = 5
    max_new_tokens = 100
    temperature = 1.0
    top_p = 0.3
    repetition_penalty = 1.2
    strength = 1.0

    # eval_prompts = tqa_prompts[:num_trials]

    # num_trials = 20 # for testing
    # num_repeats = 3 # for testing


    for model_name in models:
        if model_name not in model_configs:
            print(f"Skipping unknown model config: {model_name}")
            continue
        config = model_configs[model_name]
        model_path = config["model_path"]
        module_patterns = config["module_patterns"]

        model, tokenizer = get_model(
            model_path=model_path,
            cache_dir=str(cache_dir),
            device=str(device),
            model_task="text-generation",
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            seq_len=128,
        )

        for hook_name in hooks:
            if hook_name not in hook_configs:
                print(f"Skipping unknown hook config: {hook_name}")
                continue
            print(f"Running Generation: {model_name}, {hook_name}")

            hook_type = hook_configs[hook_name]["hook_type"]
            quantiles_src = hook_configs[hook_name]["quantiles_src"]
            intervention_dir = (
                cache_dir
                / "interventions"
                / Path(model_path).name
                / f"{hook_type}_tqa_incr"
            )
            if not intervention_dir.exists():
                print(f"Skipping {model_name}, {hook_name}: {intervention_dir} not found")
                continue

            sweep_results = []
            for _ in range(num_repeats):
                samples = _sample_prompts(tqa_prompts, num_trials)
                eval_prompts=samples

                baseline_gen = pipeline(
                    "text-generation", model=model, tokenizer=tokenizer
                )
                baseline_texts = generate_batch(
                    baseline_gen,
                    samples,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                baseline_completions = [
                    _completion_from_prompt(p, g)
                    for p, g in zip(samples, baseline_texts)
                ]

                base_t, base_i = get_t_i_scores(info_judge, info_tokenizer, truth_judge, truth_tokenizer, eval_prompts, baseline_completions)

                baseline_dist1 = calculate_dist_n(baseline_completions, 1)
                baseline_dist2 = calculate_dist_n(baseline_completions, 2)
                baseline_dist3 = calculate_dist_n(baseline_completions, 3)

                model_hooks = ModelWithHooks(module=model, device=str(device))
                model_hooks.load_hooks_from_folder(
                    folder=intervention_dir,
                    module_names=module_patterns,
                    hook_type=hook_type,
                    intervention_position="all",
                    strength=strength,
                    device=str(device),
                    dtype=torch.float32,
                    quantiles_src=quantiles_src,
                )
                model_hooks.register_hooks()

                steered_gen = pipeline(
                    "text-generation", model=model_hooks.module, tokenizer=tokenizer
                )
                steered_texts = generate_batch(
                    steered_gen,
                    samples,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                steered_completions = [
                    _completion_from_prompt(p, g)
                    for p, g in zip(samples, steered_texts)
                ]

                steered_t, steered_i = get_t_i_scores(info_judge, info_tokenizer, truth_judge, truth_tokenizer, eval_prompts, steered_completions)
                
                dist1 = calculate_dist_n(steered_completions, 1)
                dist2 = calculate_dist_n(steered_completions, 2)
                dist3 = calculate_dist_n(steered_completions, 3)

                sweep_results.append(
                    {
                        "prompts": samples,
                        "unsteered output": baseline_texts,
                        "steered output": steered_texts,
                        'steered t*i': float(steered_t * steered_i),
                        'steered t': float(steered_t),
                        'steered i': float(steered_i),
                        'base t*i': float(base_t * base_i),
                        'base t': float(base_t),
                        'base i': float(base_i),
                        'dist_1_base': float(baseline_dist1),
                        'dist_2_base': float(baseline_dist2),
                        'dist_2_base': float(baseline_dist2),
                        'dist_3_base': float(baseline_dist3),
                        'dist_1_steered': float(dist1),
                        'dist_2_steered': float(dist2),
                        'dist_3_steered': float(dist3),
                    }
                )

                model_hooks.remove_hooks()

            output_filename = f"{model_name}_{hook_name}_tqa_eval.txt"
            output_path = tqa_output_root / output_filename
            with output_path.open("w", encoding="utf-8") as json_file:
                json.dump(sweep_results, json_file, indent=4)

            print(f"Finish generation: {model_name}, {hook_name}, output to {output_path}")
            print("___________________________________________")

def ppl(models, hooks):
    for model_name in models:
        for hook_name in hooks:
            print(f"Running PPL: {model_name}")
            f_pfix = f"{model_name}_{hook_name}"
            w = get_ppl_from_file(f_pfix + "_tqa_eval")
            if w:
                print(f"Finish PPL: {model_name}")
            else:
                print(f"File not found for PPL: {f_pfix}_tqa_eval")
        print("___________________________________________")


def mmlu(models, hooks):
    mmlu_filename = "MMLU_trials_"
    mmlu_data = {}
    for hook_name in hooks:
        for model_name in models:
            print(f"Running MMLU: {hook_name},{model_name}")
            out=act_mmlu(model_name,hook_name)
            mmlu_data[model_name] = out
            print(f"Finished MMLU model {model_name}")
        # with open("act_supertox_outputs/" + mmlu_filename + hook_name + ".txt", 'w') as file:
        # with open(PATH+"/" + mmlu_filename + hook_name + ".txt", 'w') as file:
        output_path = PATH / f"{mmlu_filename}{hook_name}.txt"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(mmlu_data, file, indent=4)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["llama1b", "gemma2b", "qwen3b", "llama8b", "gemma9b", "qwen14b"],
    default="gemma2b",
)
parser.add_argument(
    "--method",
    choices=["linearact", "meanact", "pid"],
    default="linearact",
)
args = parser.parse_args()

def main():
    # Generate outputs, measure toxicity, and measure Dist 1,2,3
    # models = list(model_configs.keys())
    models = [args.model]
    hooks = list(hook_configs.keys())
    generation(models, hooks)
    print("Done with all generations")

    ## Measure PPL of the generations
    ppl(models, hooks)
    print("finish all PPL")

    # Get MMLU performance
    mmlu(models, hooks)
    print("finish all MMLU")


if __name__ == "__main__":
    main()
