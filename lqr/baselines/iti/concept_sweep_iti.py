import argparse
import json
import random
from typing import Dict, List, Sequence
import pandas as pd

import torch
from datasets import load_dataset
from transformers import pipeline

from data_handling_iti import MODEL_CONFIGS, load_model_and_tokenizer, resolve_model_name, set_seed, get_con_eval_prompts
from ITIsteering import ITISteering

from iti_concept_judge import evaluate, get_concept_score

def parse_list(values: str) -> List[float]:
    if not values:
        return []
    return [float(item.strip()) for item in values.split(",") if item.strip()]


def distinct_n(texts: Sequence[str], n: int) -> float:
    total = 0
    unique = set()
    for text in texts:
        tokens = text.split()
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            total += 1
            unique.add(tuple(tokens[i : i + n]))
    return len(unique) / max(1, total)


def load_eval_prompts(num_trials: int, seed: int) -> List[str]:
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    filtered = [
        item["prompt"]["text"]
        for item in ds
        if item["prompt"]["toxicity"] is not None
        and 0.0 <= item["prompt"]["toxicity"] <= 1
    ]
    rng = random.Random(seed)
    return rng.sample(filtered, num_trials)


def make_toxicity_classifier(device: str):
    return pipeline(
        "text-classification",
        model="s-nlp/roberta_toxicity_classifier",
        device=0 if device == "cuda" else -1,
    )


def toxicity_flags(classifier, texts: Sequence[str]) -> List[bool]:
    results = classifier(list(texts), truncation=True)
    return [r["label"] == "toxic" for r in results]


def generate_batch(prompts: Sequence[str], generate_fn, label: str, print_every: int = 25) -> List[str]:
    outs = []
    total = len(prompts)
    for i, p in enumerate(prompts, 1):
        if i == 1 or i % print_every == 0 or i == total:
            print(f"{label}: {i}/{total}")
        outs.append(generate_fn(p))
    return outs


def main() -> None:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", choices=MODEL_CONFIGS.keys(), default="gemma2b")
    parser.add_argument("--artifact", type=str)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.3)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--top-k", type=str, default="16, 32, 64")
    parser.add_argument("--alpha", type=str, default="5, 10, 20")
    parser.add_argument("--target-concept", type=str, default="dog")

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    model_input = args.model
    resolved_model_name = resolve_model_name(model_input)
    if args.artifact:
        artifact_path = args.artifact
    else:
        filename_model = (
            model_input if model_input in MODEL_CONFIGS else model_input.replace("/", "_")
        )
        artifact_path = f"iti_train_artifact_{filename_model}_con_{args.target_concept}.pkl"

    model, tokenizer, info, device = load_model_and_tokenizer(resolved_model_name)

    steerer = ITISteering.from_artifact(
        model=model,
        tokenizer=tokenizer,
        artifact_path=artifact_path,
        device=device,
        seed=args.seed,
    )

    eval_prompts = get_con_eval_prompts(args.num_trials)

    do_sample = args.temperature > 0

    # ---- Baseline ----
    base_outputs = generate_batch(
        eval_prompts,
        generate_fn=lambda p: steerer.generate_base(
            p,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ),
        label="Baseline",
    )

    full_out = [eval_prompts[i] + b for i, b in enumerate(base_outputs)]
    df = pd.DataFrame(full_out, columns=['output'])

    base_class_df = evaluate(df, args.target_concept)
    base_con_score = get_concept_score(base_class_df)["score"].tolist()[0]
    print(f"score: {base_con_score}")

    base_dist_1 = distinct_n(base_outputs, 1)
    base_dist_2 = distinct_n(base_outputs, 2)
    base_dist_3 = distinct_n(base_outputs, 3)

    # sweep params
    top_k_values = parse_list(args.top_k)
    alpha_values = parse_list(args.alpha)
    sweeps: List[Dict] = []

    for top_k in top_k_values:
        resolved_k = steerer.resolve_top_k(top_k)
        heads = steerer.top_heads(resolved_k)
        if resolved_k > 0:
            preview = heads[: min(20, len(heads))]
            pretty = ", ".join([f"L{l}H{h}" for (l, h) in preview])
            suffix = " ..." if len(heads) > 20 else ""
            print(f"TOP ATT HEADS: k={resolved_k}: {pretty}{suffix}")


        for alpha in alpha_values:
            if resolved_k == 0:
                steered_outputs = base_outputs
            else:
                steered_model = steerer.build_steered_model(
                    top_heads=heads,
                    alpha=float(alpha),
                )
                steered_outputs = generate_batch(
                    eval_prompts,
                    generate_fn=lambda p: steerer.generate_steered(
                        steered_model,
                        p,
                        max_length=args.max_length,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                    ),
                    label=f"Steered k={top_k}, a={alpha}",
                )

            steered_full_out = [eval_prompts[i] + b for i, b in enumerate(steered_outputs)]
            df_steered = pd.DataFrame(steered_full_out, columns=['output'])
            steered_class_df = evaluate(df_steered, args.target_concept)
            steered_con_score = get_concept_score(steered_class_df)["score"].tolist()[0]
            print(f"steered score: {steered_con_score}")


            # elicitation = sum((not base_flags[i]) and steered_flags[i] for i in range(len(base_flags))) /  len(base_flags)
            # safening = sum(base_flags[i] and (not steered_flags[i]) for i in range(len(base_flags))) /  len(base_flags)

            sweeps.append(
                {
                    "top_k": top_k,
                    "alpha": alpha,
                    "steered_score": steered_con_score,
                    "dist_1_steered": distinct_n(steered_outputs, 1),
                    "dist_2_steered": distinct_n(steered_outputs, 2),
                    "dist_3_steered": distinct_n(steered_outputs, 3),
                    "steered output": steered_outputs,
                    "unsteered output": base_outputs,
                }
            )

    payload = {
        "model_name": resolved_model_name,
        "artifact": artifact_path,
        "num_trials": args.num_trials,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "baseline": {
            "base_score": base_con_score,
            "dist_1": base_dist_1,
            "dist_2": base_dist_2,
            "dist_3": base_dist_3,
        },
        "sweeps": sweeps,
    }
    output_file = f"iti_sweep_con_{args.model}_{args.target_concept}_{args.num_trials}"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[OK] wrote results to {output_file}")


if __name__ == "__main__":
    main()
