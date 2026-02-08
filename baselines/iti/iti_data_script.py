# collect activations and save artifact for later sweep

import argparse
from typing import Dict

import numpy as np

from data_handling_iti import (
    build_collectors,
    collect_head_activations,
    load_model_and_tokenizer,
    load_toxicity_prompts,
    load_tqa_prompts,
    load_con_prompts,
    MODEL_CONFIGS,
    resolve_model_name,
    save_iti_artifact,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="collect activations for toxicity probes.")
    parser.add_argument("--model", choices=MODEL_CONFIGS.keys(), default="gemma2b")
    parser.add_argument("--task", type=str, default="tox")
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--num-samples-per-class", type=int, default=80)
    parser.add_argument("--toxic-threshold", type=float, default=0.8)
    parser.add_argument("--nontoxic-threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-artifact", type=str)
    parser.add_argument("--concept-target", type=str, default="dog")
    args = parser.parse_args()

    set_seed(args.seed)

    model_input = args.model
    resolved_model_name = resolve_model_name(model_input)
    model, tokenizer, info, device = load_model_and_tokenizer(resolved_model_name)

    if args.task == "tox":
        prompts, labels, toxic_samples, nontoxic_samples = load_toxicity_prompts(
            num_samples_per_class=args.num_samples_per_class,
            toxic_threshold=args.toxic_threshold,
            nontoxic_threshold=args.nontoxic_threshold,
            seed=args.seed,
        )
    elif args.task == "tqa":
        prompts, labels, false_samples, true_samples = load_tqa_prompts(
            num_samples_per_class=args.num_samples_per_class,
        )
    elif args.task == "con":
        prompts, labels, other_samples, con_samples = load_con_prompts(
            num_samples_per_class=args.num_samples_per_class, target="dog"
        )
    else:
        raise ValueError("Invalid task specification argument")

    collected_model, collectors = build_collectors(
        model, info.component_template, info.num_layers, info.num_heads
    )

    activations = collect_head_activations(
        prompts,
        tokenizer=tokenizer,
        collected_model=collected_model,
        collectors=collectors,
        device=device,
        num_layers=info.num_layers,
        num_heads=info.num_heads,
        head_dim=info.head_dim,
        max_length=args.max_length,
    )

    if args.task == "tox":
        payload: Dict = {
            "artifact_version": 1,
            "task": "toxicity_mitigation",
            "model_name": resolved_model_name,
            "seed": args.seed,
            "max_length": args.max_length,
            "thresholds": {
                "toxic_threshold": args.toxic_threshold,
                "nontoxic_threshold": args.nontoxic_threshold,
            },
            "info": {
                "num_layers": info.num_layers,
                "num_heads": info.num_heads,
                "head_dim": info.head_dim,
                "component_template": info.component_template,
            },
            "prompts": prompts,
            "labels": labels.astype(np.int64),
            "activations": activations,  # (N, L, H, D)
            "toxic_samples": toxic_samples,      # list[(text,tox)]
            "nontoxic_samples": nontoxic_samples,  # list[(text,tox)]
        }
    elif args.task == "tqa":
        payload: Dict = {
            "artifact_version": 1,
            "task": "truthfulness",
            "model_name": resolved_model_name,
            "seed": args.seed,
            "max_length": args.max_length,
            "info": {
                "num_layers": info.num_layers,
                "num_heads": info.num_heads,
                "head_dim": info.head_dim,
                "component_template": info.component_template,
            },
            "prompts": prompts,
            "labels": labels.astype(np.int64),
            "activations": activations,  # (N, L, H, D)
            "true_samples": true_samples,      # list[(text,tox)]
            "false_samples": false_samples,  # list[(text,tox)]
        }
    elif args.task == "con":
        payload: Dict = {
            "artifact_version": 1,
            "task": "concept -" + args.concept_target,
            "model_name": resolved_model_name,
            "seed": args.seed,
            "max_length": args.max_length,
            "info": {
                "num_layers": info.num_layers,
                "num_heads": info.num_heads,
                "head_dim": info.head_dim,
                "component_template": info.component_template,
            },
            "prompts": prompts,
            "labels": labels.astype(np.int64),
            "activations": activations,  # (N, L, H, D)
            "con_samples": con_samples,      # list[(text,tox)]
            "other_samples": other_samples,  # list[(text,tox)]
        }

    if args.output_artifact:
        output_file = args.output_artifact
    else:
        filename_model = (
            model_input if model_input in MODEL_CONFIGS else model_input.replace("/", "_")
        )
        if args.task=="con":
            output_file = f"iti_train_artifact_{filename_model}_{args.task}_{args.concept_target}.pkl"
        else:
            output_file = f"iti_train_artifact_{filename_model}_{args.task}.pkl"

    save_iti_artifact(output_file, payload)
    print(f"wrote artifact: {output_file}")
    print(f"activations shape: {activations.shape}")


if __name__ == "__main__":
    main()
