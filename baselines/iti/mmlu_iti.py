import argparse
import json
import random
from typing import Dict, List, Sequence

import torch
from datasets import load_dataset
from transformers import pipeline

from data_handling_iti import MODEL_CONFIGS, load_model_and_tokenizer, resolve_model_name, set_seed
from ITIsteering import ITISteering


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


LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

# List of all MMLU subjects (configs) in cais/mmlu
# SUBJECTS = [
    # 'abstract_algebra']
SUBJECTS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
    'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science',
    'college_mathematics', 'college_medicine', 'college_physics', 'computer_security',
    'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics',
    'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry',
    'high_school_computer_science', 'high_school_european_history', 'high_school_geography',
    'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics',
    'high_school_microeconomics', 'high_school_physics', 'high_school_psychology',
    'high_school_statistics', 'high_school_us_history', 'high_school_world_history',
    'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies',
    'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous',
    'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory',
    'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology',
    'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]



def format_example(example):
    answer_letter = LETTER_MAP[example["answer"]]
    choices = "\n".join(
        f"{letter}. {text}" for letter, text in zip(["A", "B", "C", "D"], example["choices"])
    )
    return f"Question: {example['question']}\n{choices}\nAnswer: {answer_letter}\n\n"


def format_query(example, instruct=False):
    choices = "\n".join(
        f"{letter}. {text}" for letter, text in zip(["A", "B", "C", "D"], example["choices"])
    )
    if not instruct:
        return f"Question: {example['question']}\n{choices}\nAnswer:"
    else:
        return f"Answer the multiple-choice question: {example['question']}\n{choices}\n Answer with only a single letter."

def build_5shot_prompt(dev_set, test_example, n_shots=5):
    exemplars = random.sample(list(dev_set), n_shots)
    prompt = ""
    for ex in exemplars:
        prompt += format_example(ex)
    prompt += format_query(test_example)
    # Store the correct answer letter for the test example
    correct_answer = LETTER_MAP[test_example["answer"]]
    return prompt, correct_answer

def get_mmlu_prompts(num_trials: int, N_SHOTS=5):
    subject_datasets = {
            sub: load_dataset("cais/mmlu", sub)
            for sub in SUBJECTS
        }
   
    prompts_with_answers = []
    samples = []
    for i in range(num_trials):
        subject = random.choice(SUBJECTS)
        ds = subject_datasets[subject]
        dev, test = ds["dev"], ds["test"]

        if len(dev) < N_SHOTS or len(test) == 0:
            continue

        test_example = random.choice(test)
        prompt, correct_answer = build_5shot_prompt(dev, test_example, N_SHOTS)

        # print(f"prompt: {prompt}")

        prompts_with_answers.append({
            "subject": subject,
            "prompt": prompt,
            "answer": correct_answer
        })

        samples.append(prompt)
    return samples, prompts_with_answers

def score_responses(outputs, prompts_with_answers):
    results = []
    # print(f"outs: {outputs}")
    # print(f"prompts with answers: {prompts_with_answers}")
    for item, model_output in zip(prompts_with_answers, outputs):
        model_choice = model_output.strip()[-1].upper()
        correct_choice = item["answer"]
        # print(f"model choice: {model_choice}")
        # print(f"correct choice: {correct_choice}")
        results.append(model_choice == correct_choice)
    accuracy = sum(results) / len(results)
    return accuracy

def main() -> None:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", choices=MODEL_CONFIGS.keys(), default="gemma2b")
    parser.add_argument("--artifact", type=str)
    parser.add_argument("--num-trials", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.3)
    parser.add_argument("--repetition-penalty", type=float, default=0.0)
    parser.add_argument("--top-k", type=str, default="32")
    parser.add_argument("--alpha", type=str, default="10")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--run-base", type=bool, default=True)
    parser.add_argument("--folder", type=str, default=".")


    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # set_seed(args.seed)

    model_input = args.model
    resolved_model_name = resolve_model_name(model_input)
    if args.artifact:
        artifact_path = args.artifact
    else:
        filename_model = (
            model_input if model_input in MODEL_CONFIGS else model_input.replace("/", "_")
        )
        artifact_path = f"iti_train_artifact_{filename_model}_tqa.pkl"

    model, tokenizer, info, device = load_model_and_tokenizer(resolved_model_name)

    steerer = ITISteering.from_artifact(
        model=model,
        tokenizer=tokenizer,
        artifact_path=artifact_path,
        device=device,
        seed=args.seed,
    )

    eval_prompts, prompts_with_answers = get_mmlu_prompts(args.num_trials)
    # print(f"eval prompts: {eval_prompts}")
    # print(f"prompts with asnswers init: {prompts_with_answers}")

    # do_sample = args.temperature > 0
    do_sample = False

    # ---- Baseline ----
    if args.run_base:
        base_outputs = generate_batch(
            eval_prompts,
            generate_fn=lambda p: steerer.generate_base_MMLU(
                p,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            ),
            label="Baseline",
        )

        base_accuracy = score_responses(base_outputs, prompts_with_answers)
        print(f"base accuracy: {base_accuracy}")

    else:
        base_accuracy=None
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
                    generate_fn=lambda p: steerer.generate_steered_MMLU(
                        steered_model,
                        p,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    ),
                    label=f"Steered k={top_k}, a={alpha}",
                )

            steered_accuracy = score_responses(steered_outputs, prompts_with_answers)
            print(f"steered accuracy: {steered_accuracy}")

            # elicitation = sum((not base_flags[i]) and steered_flags[i] for i in range(len(base_flags))) /  len(base_flags)
            # safening = sum(base_flags[i] and (not steered_flags[i]) for i in range(len(base_flags))) /  len(base_flags)

            sweeps.append(
                {
                    "top_k": top_k,
                    "alpha": alpha,
                    "accuracy": steered_accuracy,
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
            "base accuracy": base_accuracy,
        }, 
        "sweeps": sweeps,
    }
    output_file = f"{args.folder}/iti_MMLU_{args.model}_{args.num_trials}_{args.output}"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[OK] wrote results to {output_file}")


if __name__ == "__main__":
    main()
