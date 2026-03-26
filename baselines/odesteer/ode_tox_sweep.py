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

from data_handling_ode import ContrastiveBuilder


ODESTEER_SRC = "odesteer-repo/src"
if ODESTEER_SRC not in sys.path:
    sys.path.append(ODESTEER_SRC)

from odesteer.steer import ODESteer


MODEL_CONFIGS = {
    "llama1b": {
        "hf_name": "meta-llama/Llama-3.2-1B",
        "tensor_prefix": "llama-3.2-1b",
    },
    "gemma2b": {
        "hf_name": "google/gemma-2-2b",
        "tensor_prefix": "gemma-2-2b",
    },
    "qwen3b": {
        "hf_name": "Qwen/Qwen2.5-3B",
        "tensor_prefix": "qwen-2.5-3b",
    },
    "llama8b": {
        "hf_name": "meta-llama/Meta-Llama-3-8B",
        "tensor_prefix": "llama-3-8b",
    },
    "gemma9b": {
        "hf_name": "google/gemma-2-9b",
        "tensor_prefix": "gemma-2-9b",
    },
    "qwen14b": {
        "hf_name": "Qwen/Qwen2.5-14B",
        "tensor_prefix": "qwen-2.5-14b",
    },
}

DEFAULT_T_VALUES = [1, 2, 5, 10, 15, 25, 35, 50]
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


def load_model(model_name, quant=False):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    if quant:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=th.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            dtype=th.float32,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def toxicity_classifier(device: th.device):
    pipeline_device = 0 if device.type == "cuda" else -1
    return pipeline(
        "text-classification",
        model="s-nlp/roberta_toxicity_classifier",
        device=pipeline_device,
    )


def calculate_dist_n(texts_list: Sequence[str], n: int) -> float:
    if n <= 0:
        raise ValueError("n must be a positive integer")
    tokens: List[str] = []
    for text in texts_list:
        tokens.extend(text.lower().split())
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


@th.no_grad()
def generate_completion_only(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
):
    model_device = next(model.parameters()).device
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model_device)
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )
    output_str = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
    return [output_str[idx][len(prompt):].strip() for idx, prompt in enumerate(prompts)]


def _extract_and_set_hidden(output):
    if isinstance(output, tuple):
        hidden, rest = output[0], output[1:]

        def reassemble(h):
            return (h, *rest)
    elif hasattr(output, "last_hidden_state"):
        hidden = output.last_hidden_state

        def reassemble(h):
            output.last_hidden_state = h
            return output
    else:
        hidden = output

        def reassemble(h):
            return h
    return hidden, reassemble


def _steer_hook(module, inputs, output, steer_model, T, steer_position_idx=-1):
    hidden, reassemble = _extract_and_set_hidden(output)
    hidden = hidden.clone()
    batch_idx = th.arange(hidden.shape[0], device=hidden.device)
    hidden[batch_idx, steer_position_idx] = steer_model.steer(
        hidden[batch_idx, steer_position_idx],
        T=T,
    )
    return reassemble(hidden)


@th.no_grad()
def generate_with_steering(
    model,
    tokenizer,
    prompts: Sequence[str],
    steer_model,
    steer_layer: int,
    steer_strength_T: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
):
    target_layer = model.model.layers[steer_layer]
    handle = target_layer.register_forward_hook(
        partial(_steer_hook, steer_model=steer_model, T=steer_strength_T, steer_position_idx=-1)
    )
    try:
        return generate_completion_only(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    finally:
        handle.remove()


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


def collect_layer_activations(model, tokenizer, model_name: str, layer_idx: int, num_samples: int):
    dataset = load_dataset("allenai/real-toxicity-prompts")
    data = dataset["train"]

    tox_prompts = [
        item["text"]
        for item in data["prompt"]
        if item["toxicity"] is not None and 0.8 <= item["toxicity"] <= 1.0
    ]
    tox_dict = ContrastiveBuilder(model, tokenizer).collect_data_batch(
        tox_prompts,
        num_samples,
        f"{model_name}_ode_tox_vec",
        layer_idx=layer_idx,
    )

    nontox_prompts = [
        item["text"]
        for item in data["prompt"]
        if item["toxicity"] is not None and 0.0 <= item["toxicity"] <= 0.1
    ]
    nontox_dict = ContrastiveBuilder(model, tokenizer).collect_data_batch(
        nontox_prompts,
        num_samples,
        f"{model_name}_ode_nontox_vec",
        layer_idx=layer_idx,
    )

    return tox_dict, nontox_dict


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
            args.model,
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
