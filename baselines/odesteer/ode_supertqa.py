
import json
import random
from pathlib import Path
from typing import List, Sequence

import torch as th
import sys

from odesteer_utils import (
    load_model,
    calculate_dist_n,
    generate_completion_only,
    generate_with_steering,
    collect_layer_activations_tqa,
    get_questions_no_it,
    build_tqa_judges,
    get_t_i_scores,
    get_ppl_from_file,
    steer_configs_tqa,
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
    models = list(configs.keys())
    tqa_output_root = Path(PATH)
    tqa_output_root.mkdir(parents=True, exist_ok=True)

    num_trials = 1000
    num_repeats = 5
    max_new_tokens = 100
    temperature = 1.0
    top_p = 0.3
    repetition_penalty = 1.2
    do_sample = temperature > 0
    # num_trials = 50 #for testing
    # num_repeats = 3 #for testing
    train_samples = 1800


    info_judge, info_tokenizer, truth_judge, truth_tokenizer = build_tqa_judges()

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

        true_dict, false_dict = collect_layer_activations_tqa(
            model,
            tokenizer,
            steer_layer,
            train_samples,
            batch_size=128,
        )

        steer_model = ODESteer(**DEFAULT_ODESTEER_KWARGS)
        steer_model.fit(
            true_dict["X"].float().cpu(),
            false_dict["X"].float().cpu(),
        )

        sweep_results = []
        for _ in range(num_repeats):
            samples = get_questions_no_it(num_trials)

            baseline_completions = generate_completion_only(
                model,
                tokenizer,
                samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
            )

            # save full text (prompt+generation) to results
            baseline_texts = _attach_prompts(samples, baseline_completions)

            base_t, base_i = get_t_i_scores(
                info_judge,
                info_tokenizer,
                truth_judge,
                truth_tokenizer,
                tokenizer,
                samples,
                baseline_completions,
            )
            base_ti = base_t * base_i
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
                do_sample=do_sample,
            )
            steered_texts = _attach_prompts(samples, steered_completions)
            steered_t, steered_i = get_t_i_scores(
                info_judge,
                info_tokenizer,
                truth_judge,
                truth_tokenizer,
                tokenizer,
                samples,
                steered_completions,
            )
            steered_ti = steered_t * steered_i
            dist1 = calculate_dist_n(steered_completions, 1)
            dist2 = calculate_dist_n(steered_completions, 2)
            dist3 = calculate_dist_n(steered_completions, 3)

            sweep_results.append(
                {
                    "prompts": samples,
                    "unsteered output": baseline_texts,
                    "steered output": steered_texts,
                    "t*i unsteered": float(base_ti),
                    "t unsteered": float(base_t),
                    "i unsteered": float(base_i),
                    "t*i steered": float(steered_ti),
                    "t steered": float(steered_t),
                    "i steered": float(steered_i),
                    "t*i delta": float(steered_ti - base_ti),
                    "dist 1 base": float(baseline_dist1),
                    "dist 2 base": float(baseline_dist2),
                    "dist 3 base": float(baseline_dist3),
                    "dist 1 steered": float(dist1),
                    "dist 2 steered": float(dist2),
                    "dist 3 steered": float(dist3),
                }
            )

        output_filename = f"{model_name}_ode_tqa_eval.txt"
        output_path = tqa_output_root / output_filename
        with output_path.open("w", encoding="utf-8") as json_file:
            json.dump(sweep_results, json_file, indent=4)

        print(f"Finish generation: {model_name}, output to {output_path}")
        print("___________________________________________")


def ppl(configs):
    models=list(configs.keys())
    for model_name in models:
        print(f"Running PPL: {model_name}")
        f_pfix = model_name
        w = get_ppl_from_file(f_pfix + "_ode_tqa_eval")
        if w:
            print(f"Finish PPL: {model_name}")
        else:
            print(f"File not found for PPL: {f_pfix}_ode_tqa_eval")
    print("___________________________________________")
    
    
def mmlu(configs):
    mmlu_filename = "MMLU_trials_tqa"
    mmlu_data = {}
    models=list(configs.keys())

    for model_name in models:
        print(f"Running MMLU: {model_name}")
        out = ode_mmlu(model_name, steering_source="tqa",train_samples=1000)
        mmlu_data[model_name] = out

        print(f"Finished MMLU model {model_name}")
    output_path = PATH / f"{mmlu_filename}.txt"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(mmlu_data, file, indent=4)



def main():
    # Generate outputs, measure T/I and Dist 1,2,3
    generation(steer_configs_tqa)
    print("Done with all generations")

    ## Measure PPL of the generations
    ppl(steer_configs_tqa)
    print("finish all PPL")

    # Get MMLU performance
    mmlu(steer_configs_tqa)
    print("finish all MMLU")


if __name__ == "__main__":
    main()

    
