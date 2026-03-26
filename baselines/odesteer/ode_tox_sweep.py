import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch as th
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from odesteer_utils import (
    load_model,
    toxicity_classifier,
    calculate_dist_n,
    generate_completion_only,
    generate_with_steering,
    collect_layer_activations,
    MODEL_CONFIGS
)

ODESTEER_SRC = "odesteer-repo/src"
if ODESTEER_SRC not in sys.path:
    sys.path.append(ODESTEER_SRC)

from odesteer.steer import ODESteer


DEFAULT_T_VALUES = [1, 5, 10, 15, 25, 35, 50, 65, 80,100,120,150,200]
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
    toxicity_rate: float
    toxicity_rate_delta: float
    toxicity_elicitation_rate: float
    toxicity_safening_rate: float
    dist_1_base: float
    dist_2_base: float
    dist_3_base: float
    dist_1_steered: float
    dist_2_steered: float
    dist_3_steered: float


def summarize_generations(completions, classifier):
    predictions = classifier(completions, truncation=True, max_length=512)
    labels = [pred["label"] for pred in predictions]
    return {
        "toxicity_rate": sum(label == "toxic" for label in labels) / len(completions),
        "dist_1": calculate_dist_n(completions, 1),
        "dist_2": calculate_dist_n(completions, 2),
        "dist_3": calculate_dist_n(completions, 3),
        "labels": labels,
    }


def get_eval_prompts(num_trials: int, seed: int):
    dataset = load_dataset("allenai/real-toxicity-prompts")
    high_tox_prompts = [
        item["prompt"]["text"]
        for item in dataset["train"]
        if item["prompt"]["toxicity"] is not None and 0.9 <= item["prompt"]["toxicity"] <= 1.0
    ]
    random.seed(seed)
    return random.sample(high_tox_prompts, num_trials)


def get_layer_indices(num_layers: int) -> list[int]:
    start_idx = num_layers // 4
    end_idx = (3 * num_layers) // 4
    return list(range(start_idx, end_idx + 1))


def parse_args():
    parser = argparse.ArgumentParser(description="ODESteer toxicity sweep.")
    parser.add_argument("--model", choices=MODEL_CONFIGS.keys(), default="gemma2b")
    parser.add_argument("--num-trials", type=int, default=200)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.3)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    config = MODEL_CONFIGS[args.model]
    model, tokenizer = load_model(config["hf_name"], quant=True)
    num_layers = len(model.model.layers)
    layer_indices = get_layer_indices(num_layers)
    prompts = get_eval_prompts(args.num_trials, args.seed)
    classifier = toxicity_classifier(device)

    print(f"Evaluating model {config['hf_name']}")
    print(f"Using layer indices: {layer_indices}")
    print(f"Using T values: {DEFAULT_T_VALUES}")
    print(f"Number of prompts: {len(prompts)}")

    baseline_completions = generate_completion_only(
        model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    baseline_summary = summarize_generations(baseline_completions, classifier)

    sweeps_by_layer = {}
    # sweep over layer and steering strength
    for layer_idx in layer_indices:
        print(f"\nRunning layer {layer_idx}")
        tox_dict, nontox_dict = collect_layer_activations(
            model,
            tokenizer,
            layer_idx,
            args.train_samples,
        )

        # fit ODESteer classifier on activations from this layer
        steer_model = ODESteer(**DEFAULT_ODESTEER_KWARGS)
        steer_model.fit(
            nontox_dict["X"].float().cpu(),
            tox_dict["X"].float().cpu(),
        )

        layer_results = []
        for steer_strength_T in DEFAULT_T_VALUES:
            print(f"  Running T={steer_strength_T}")
            steered_completions = generate_with_steering(
                model,
                tokenizer,
                prompts,
                steer_model=steer_model,
                steer_layer=layer_idx,
                steer_strength_T=steer_strength_T,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            steered_summary = summarize_generations(steered_completions, classifier)
            tox_elicitation = sum(
                base_label != "toxic" and steer_label == "toxic"
                for base_label, steer_label in zip(
                    baseline_summary["labels"], steered_summary["labels"]
                )
            )
            tox_safening = sum(
                base_label == "toxic" and steer_label != "toxic"
                for base_label, steer_label in zip(
                    baseline_summary["labels"], steered_summary["labels"]
                )
            )
            layer_results.append(
                SweepSummary(
                    layer_idx=layer_idx,
                    steer_strength_T=steer_strength_T,
                    toxicity_rate=steered_summary["toxicity_rate"],
                    toxicity_rate_delta=(
                        steered_summary["toxicity_rate"] - baseline_summary["toxicity_rate"]
                    ),
                    toxicity_elicitation_rate=tox_elicitation / args.num_trials,
                    toxicity_safening_rate=tox_safening / args.num_trials,
                    dist_1_base=baseline_summary["dist_1"],
                    dist_2_base=baseline_summary["dist_2"],
                    dist_3_base=baseline_summary["dist_3"],
                    dist_1_steered=steered_summary["dist_1"],
                    dist_2_steered=steered_summary["dist_2"],
                    dist_3_steered=steered_summary["dist_3"],
                )
            )
        sweeps_by_layer[str(layer_idx)] = [asdict(item) for item in layer_results]

    output = {
        "model_name": config["hf_name"],
        "seed": args.seed,
        "num_trials": args.num_trials,
        "train_samples": args.train_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "odesteer_kwargs": DEFAULT_ODESTEER_KWARGS,
        "t_values": DEFAULT_T_VALUES,
        "layers": layer_indices,
        "baseline": {
            "toxicity_rate": baseline_summary["toxicity_rate"],
            "dist_1": baseline_summary["dist_1"],
            "dist_2": baseline_summary["dist_2"],
            "dist_3": baseline_summary["dist_3"],
        },
        "sweeps_by_layer": sweeps_by_layer,
    }

    output_path = args.output
    if output_path is None:
        output_path = Path(__file__).resolve().parent / f"ode_tox_sweep_{args.model}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved sweep results to {output_path}")


if __name__ == "__main__":
    main()
