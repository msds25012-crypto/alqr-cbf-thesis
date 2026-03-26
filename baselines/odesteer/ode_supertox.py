
import json
import os.path
import random
from pathlib import Path
from typing import List, Sequence
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import torch as th
from transformers import pipeline
import sys

from odesteer_utils import (
    load_model,
    toxicity_classifier,
    calculate_dist_n,
    generate_completion_only,
    generate_with_steering,
    collect_layer_activations,
    get_ppl_from_file,
    MODEL_CONFIGS,
    steer_configs,
    DEFAULT_ODESTEER_KWARGS,
    PATH
)
from ode_mmlu import ode_mmlu

ODESTEER_SRC = "odesteer-repo/src"
if ODESTEER_SRC not in sys.path:
    sys.path.append(ODESTEER_SRC)

from odesteer.steer import ODESteer


def _sample_prompts(prompts: Sequence[str], num_trials: int) -> List[str]:
    if len(prompts) >= num_trials:
        return random.sample(list(prompts), num_trials)
    return random.choices(list(prompts), k=num_trials)


def _attach_prompts(prompts: Sequence[str], completions: Sequence[str]) -> List[str]:
    return [f"{prompt}{completion}" for prompt, completion in zip(prompts, completions)]

def generation(configs):
    dataset_name = "allenai/real-toxicity-prompts"
    dataset = load_dataset(dataset_name)

    data = dataset["train"]
    toxic_prompts = [
                item["text"]
                for item in data["prompt"]
                if item["toxicity"] is not None and item["toxicity"] <= 1 and item["toxicity"] >= 0
            ]
    models = list(configs.keys())
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    tox_output_root = Path(PATH)
    tox_output_root.mkdir(parents=True, exist_ok=True)

    classifier = toxicity_classifier(device)

    num_trials = 1000
    num_repeats = 5
    max_new_tokens = 100
    temperature = 1.0
    top_p = 0.3
    repetition_penalty = 1.2
    gen_batch_size = 8
    clf_batch_size = 16
    # num_trials = 50 #for testing
    # num_repeats = 3 #for testing
    train_samples=5000

    for model_name in models:
        if model_name not in configs:
            print(f"Skipping unknown model config: {model_name}")
            continue
        config = configs[model_name]
        model_path = config["model_path"]
        steer_layer = config.get("steer_layer")
        strength = config.get("strength")
        if steer_layer is None or strength is None:
            print(f"Skipping {model_name}: steer_layer/strength not set")
            continue

        print(f"Running Generation: {model_name}")
        model, tokenizer = load_model(model_path, quant=True)

        tox_dict, nontox_dict = collect_layer_activations(
            model,
            tokenizer,
            steer_layer,
            train_samples,
        )

        steer_model = ODESteer(**DEFAULT_ODESTEER_KWARGS)
        steer_model.fit(
            nontox_dict["X"].float().cpu(),
            tox_dict["X"].float().cpu(),
        )

        sweep_results = []
        for _ in range(num_repeats):
            samples = _sample_prompts(toxic_prompts, num_trials)

            baseline_completions = generate_completion_only(
                model,
                tokenizer,
                samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            # save full text (prompt+generation) to results
            baseline_texts = _attach_prompts(samples, baseline_completions)

            baseline_predictions = classifier(
                baseline_completions,
                truncation=True,
                max_length=512,
                batch_size=clf_batch_size,
            )
            baseline_labels = [pred["label"] for pred in baseline_predictions]
            baseline_dist1 = calculate_dist_n(baseline_completions, 1)
            baseline_dist2 = calculate_dist_n(baseline_completions, 2)
            baseline_dist3 = calculate_dist_n(baseline_completions, 3)

            steered_completions = generate_with_steering(
                model,
                tokenizer,
                samples,
                steer_model=steer_model,
                steer_layer=steer_layer,
                steer_strength_T=strength,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            steered_texts = _attach_prompts(samples, steered_completions)
            steered_predictions = classifier(
                steered_completions,
                truncation=True,
                max_length=512,
                batch_size=clf_batch_size,
            )
            steered_labels = [pred["label"] for pred in steered_predictions]
            dist1 = calculate_dist_n(steered_completions, 1)
            dist2 = calculate_dist_n(steered_completions, 2)
            dist3 = calculate_dist_n(steered_completions, 3)

            num_safeified = 0
            num_unsafeified = 0
            num_tox_unsteered = 0
            num_tox_steered = 0
            for base_label, steer_label in zip(baseline_labels, steered_labels):
                if steer_label == "toxic" and base_label != "toxic":
                    num_unsafeified += 1
                if steer_label != "toxic" and base_label == "toxic":
                    num_safeified += 1
                if base_label == "toxic":
                    num_tox_unsteered += 1
                if steer_label == "toxic":
                    num_tox_steered += 1

            sweep_results.append(
                {
                    "prompts": samples,
                    "unsteered output": baseline_texts,
                    "steered output": steered_texts,
                    "num safeified": num_safeified,
                    "num unsafeified": num_unsafeified,
                    "num tox unsteered": num_tox_unsteered,
                    "num tox steered": num_tox_steered,
                    "dist 1 base": float(baseline_dist1),
                    "dist 2 base": float(baseline_dist2),
                    "dist 3 base": float(baseline_dist3),
                    "dist 1 steered": float(dist1),
                    "dist 2 steered": float(dist2),
                    "dist 3 steered": float(dist3),
                }
            )

        output_filename = f"{model_name}_ode_tox_eval.txt"
        output_path = tox_output_root / output_filename
        with output_path.open("w", encoding="utf-8") as json_file:
            json.dump(sweep_results, json_file, indent=4)

        print(f"Finish generation: {model_name}, output to {output_path}")
        print("___________________________________________")


def ppl(configs):
    models=list(configs.keys())
    for model_name in models:
        print(f"Running PPL: {model_name}")
        f_pfix = model_name
        w = get_ppl_from_file(f_pfix + "_ode_tox_eval")
        if w:
            print(f"Finish PPL: {model_name}")
        else:
            print(f"File not found for PPL: {f_pfix}_ode_tox_eval")
    print("___________________________________________")
    
    
def mmlu(configs):
    mmlu_filename = "MMLU_trials"
    mmlu_data = {}
    models=list(configs.keys())

    for model_name in models:
        print(f"Running MMLU: {model_name}")
        out = ode_mmlu(model_name)
        mmlu_data[model_name] = out

        print(f"Finished MMLU model {model_name}")
    output_path = PATH / f"{mmlu_filename}.txt"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(mmlu_data, file, indent=4)



def main():
    # Generate outputs, measure toxicity, and measure Dist 1,2,3
    generation(steer_configs)
    print("Done with all generations")

    ## Measure PPL of the generations
    ppl(steer_configs)
    print("finish all PPL")

    # Get MMLU performance
    mmlu(steer_configs)
    print("finish all MMLU")


if __name__ == "__main__":
    main()

    
