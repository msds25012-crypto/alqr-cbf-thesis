import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch as th

from odesteer_utils import (
    MODEL_CONFIGS,
    build_tqa_judges,
    calculate_dist_n,
    collect_layer_activations_tqa,
    generate_completion_only,
    generate_with_steering,
    get_questions_no_it,
    get_t_i_scores,
    load_model,
)

ODESTEER_SRC = "odesteer-repo/src"
if ODESTEER_SRC not in sys.path:
    sys.path.append(ODESTEER_SRC)

from odesteer.steer import ODESteer


DEFAULT_T_VALUES = [1, 5, 10, 15, 25, 35, 50, 65, 80, 100, 120, 150]
DEFAULT_ODESTEER_KWARGS = {
    "solver": "euler",
    "steps": 10,
    "n_components": 8000,
    "degree": 2,
    "gamma": 0.1,
    "coef0": 1.0,
    "lin_clf_type": "lr",
}


@dataclass
class SweepSummary:
    layer_idx: int
    steer_strength_T: float
    t: float
    i: float
    t_times_i: float
    t_times_i_delta: float
    dist_1_base: float
    dist_2_base: float
    dist_3_base: float
    dist_1_steered: float
    dist_2_steered: float
    dist_3_steered: float


def get_layer_indices(num_layers: int) -> list[int]:
    start_idx = num_layers // 4
    end_idx = (3 * num_layers) // 4
    return list(range(start_idx, end_idx + 1))


def parse_args():
    parser = argparse.ArgumentParser(description="ODESteer TruthfulQA sweep.")
    parser.add_argument("--model", choices=MODEL_CONFIGS.keys(), default="gemma2b")
    parser.add_argument("--num-trials", type=int, default=200)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--activation-batch-size", type=int, default=128)
    parser.add_argument("--activation-max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.3)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-base", action="store_true", default=True)
    parser.add_argument("--skip-base", action="store_false", dest="run_base")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    config = MODEL_CONFIGS[args.model]
    model, tokenizer = load_model(config["hf_name"], quant=True)
    num_layers = len(model.model.layers)
    layer_indices = get_layer_indices(num_layers)
    eval_prompts = get_questions_no_it(args.num_trials)
    info_judge, info_tokenizer, truth_judge, truth_tokenizer = build_tqa_judges()

    print(f"Evaluating model {config['hf_name']}")
    print(f"Using layer indices: {layer_indices}")
    print(f"Using T values: {DEFAULT_T_VALUES}")
    print(f"Number of eval prompts: {len(eval_prompts)}")
    print(f"Number of train prompts per class: {args.train_samples}")

    do_sample = args.temperature > 0

    if args.run_base:
        baseline_outputs = generate_completion_only(
            model,
            tokenizer,
            eval_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=do_sample,
        )
        base_t, base_i = get_t_i_scores(
            info_judge,
            info_tokenizer,
            truth_judge,
            truth_tokenizer,
            tokenizer,
            eval_prompts,
            baseline_outputs,
        )
        base_ti = base_t * base_i
        base_dist_1 = calculate_dist_n(baseline_outputs, 1)
        base_dist_2 = calculate_dist_n(baseline_outputs, 2)
        base_dist_3 = calculate_dist_n(baseline_outputs, 3)
        print(f"Baseline t*i: {base_ti}")
        print(f"Baseline t: {base_t}")
        print(f"Baseline i: {base_i}")
    else:
        baseline_outputs = None
        base_t = None
        base_i = None
        base_ti = None
        base_dist_1 = None
        base_dist_2 = None
        base_dist_3 = None

    sweeps_by_layer = {}
    for layer_idx in layer_indices:
        print(f"\nRunning layer {layer_idx}")
        true_dict, false_dict = collect_layer_activations_tqa(
            model,
            tokenizer,
            layer_idx,
            args.train_samples,
            batch_size=args.activation_batch_size,
            max_length=args.activation_max_length,
        )

        steer_model = ODESteer(**DEFAULT_ODESTEER_KWARGS)
        steer_model.fit(
            true_dict["X"].float().cpu(),
            false_dict["X"].float().cpu(),
        )

        layer_results: List[SweepSummary] = []
        for steer_strength_T in DEFAULT_T_VALUES:
            print(f"  Running T={steer_strength_T}")
            steered_outputs = generate_with_steering(
                model,
                tokenizer,
                eval_prompts,
                steer_model=steer_model,
                steer_layer=layer_idx,
                steer_strength_T=steer_strength_T,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=do_sample,
            )

            steered_t, steered_i = get_t_i_scores(
                info_judge,
                info_tokenizer,
                truth_judge,
                truth_tokenizer,
                tokenizer,
                eval_prompts,
                steered_outputs,
            )
            steered_ti = steered_t * steered_i
            print(f"  Steered t*i: {steered_ti}")
            print(f"  Steered t: {steered_t}")
            print(f"  Steered i: {steered_i}")

            layer_results.append(
                SweepSummary(
                    layer_idx=layer_idx,
                    steer_strength_T=steer_strength_T,
                    t=steered_t,
                    i=steered_i,
                    t_times_i=steered_ti,
                    t_times_i_delta=(
                        steered_ti - base_ti if base_ti is not None else None
                    ),
                    dist_1_base=base_dist_1,
                    dist_2_base=base_dist_2,
                    dist_3_base=base_dist_3,
                    dist_1_steered=calculate_dist_n(steered_outputs, 1),
                    dist_2_steered=calculate_dist_n(steered_outputs, 2),
                    dist_3_steered=calculate_dist_n(steered_outputs, 3),
                )
            )
        sweeps_by_layer[str(layer_idx)] = [asdict(item) for item in layer_results]

    output = {
        "model_name": config["hf_name"],
        "seed": args.seed,
        "num_trials": args.num_trials,
        "train_samples": args.train_samples,
        "activation_batch_size": args.activation_batch_size,
        "activation_max_length": args.activation_max_length,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "odesteer_kwargs": DEFAULT_ODESTEER_KWARGS,
        "t_values": DEFAULT_T_VALUES,
        "layers": layer_indices,
        "baseline": {
            "t_times_i": base_ti,
            "t": base_t,
            "i": base_i,
            "dist_1": base_dist_1,
            "dist_2": base_dist_2,
            "dist_3": base_dist_3,
        },
        "sweeps_by_layer": sweeps_by_layer,
    }

    output_path = args.output
    if output_path is None:
        output_path = Path(__file__).resolve().parent / f"ode_tqa_sweep_{args.model}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved sweep results to {output_path}")


if __name__ == "__main__":
    main()
