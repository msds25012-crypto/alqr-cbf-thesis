import random
from pathlib import Path
from typing import Dict, List, Sequence

import torch as th
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
 
from odesteer_utils import (
    load_model,
    generate_with_steering,
    collect_layer_activations,
    steer_configs,
    DEFAULT_ODESTEER_KWARGS,
)
import sys

ODESTEER_SRC = "odesteer-repo/src"
if ODESTEER_SRC not in sys.path:
    sys.path.append(ODESTEER_SRC)

from odesteer.steer import ODESteer

LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

def format_example(example: Dict) -> str:
    answer_letter = LETTER_MAP[example["answer"]]
    choices = "\n".join(
        f"{letter}. {text}"
        for letter, text in zip(["A", "B", "C", "D"], example["choices"])
    )
    return f"Question: {example['question']}\n{choices}\nAnswer: {answer_letter}\n\n"


def format_query(example: Dict) -> str:
    choices = "\n".join(
        f"{letter}. {text}"
        for letter, text in zip(["A", "B", "C", "D"], example["choices"])
    )
    return f"Question: {example['question']}\n{choices}\nAnswer:"


def build_5shot_prompt(dev_set, test_example, n_shots: int) -> str:
    exemplars = random.sample(list(dev_set), n_shots)
    prompt = ""
    for ex in exemplars:
        prompt += format_example(ex)
    prompt += format_query(test_example)
    correct_answer = LETTER_MAP[test_example["answer"]]
    return prompt, correct_answer


def _load_subject_datasets() -> Dict[str, Dict]:
    return {sub: load_dataset("cais/mmlu", sub) for sub in SUBJECTS}


def _generate_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> List[str]:
    inputs = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    with th.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
    return decoded


def _choice_from_decoded_text(text: str) -> str:
    stripped = text.strip()
    return stripped[-1].upper() if stripped else ""




def ode_mmlu(
    model_name: str,
    *,
    runs_per_strength: int = 5,
    n_prompts: int = 10,
    n_loop: int = 100,
    # n_prompts: int = 5, # for testing
    # n_loop: int = 10, # for testing
    batch_size: int = 10,
    n_shots: int = 5,
    max_new_tokens: int = 1,
    do_sample: bool = False,
    temperature: float = 0.7,
    train_samples=5000

) -> Dict:
    if model_name not in steer_configs:
        raise ValueError(f"Unknown model config: {model_name}")

    config = steer_configs[model_name]
    model_path = config["model_path"]
    steer_layer = config.get("steer_layer")
    strength = config.get("strength")
    if steer_layer is None or strength is None:
        raise ValueError(f"steer_layer/strength not set for {model_name}")

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_path, quant=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_device = next(model.parameters()).device

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

    subject_datasets = _load_subject_datasets()

    results = {
        "model_name": model_name,
        "n_prompts": n_prompts,
        "n_loop": n_loop,
        "batch_size": batch_size,
        "n_shots": n_shots,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "runs_per_strength": runs_per_strength,
        "strengths": [
            {
                "strength": float(strength),
                "runs": [],
            }
        ],
    }

    for run_idx in range(runs_per_strength):
        baseline_correct = []
        steered_correct = []

        for _ in range(n_loop):
            prompts_with_answers = []
            samples = []

            for _ in range(n_prompts):
                subject = random.choice(SUBJECTS)
                ds = subject_datasets[subject]
                dev, test = ds["dev"], ds["test"]
                if len(dev) < n_shots or len(test) == 0:
                    continue

                test_example = random.choice(test)
                prompt, correct_answer = build_5shot_prompt(dev, test_example, n_shots)
                prompts_with_answers.append(
                    {"subject": subject, "prompt": prompt, "answer": correct_answer}
                )
                samples.append(prompt)

            if not samples:
                continue

            batch_outputs = []
            batch_outputs_steer = []
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                batch_outputs.extend(
                    _generate_batch(
                        model,
                        tokenizer,
                        batch,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                    )
                )

                batch_outputs_steer.extend(
                    generate_with_steering(
                        model,
                        tokenizer,
                        batch,
                        steer_model=steer_model,
                        steer_layer=steer_layer,
                        steer_strength_T=strength,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=1.0,
                        repetition_penalty=1.0,
                    )
                )

            for item, model_output in zip(prompts_with_answers, batch_outputs):
                model_choice = _choice_from_decoded_text(model_output)
                baseline_correct.append(model_choice == item["answer"])

            for item, model_output in zip(prompts_with_answers, batch_outputs_steer):
                model_choice = _choice_from_decoded_text(model_output)
                steered_correct.append(model_choice == item["answer"])

        baseline_acc = sum(baseline_correct) / max(len(baseline_correct), 1)
        steered_acc = sum(steered_correct) / max(len(steered_correct), 1)
        results["strengths"][0]["runs"].append(
            {
                "run": int(run_idx),
                "baseline_accuracy": float(baseline_acc),
                "steered_accuracy": float(steered_acc),
                "num_samples": int(len(baseline_correct)),
            }
        )

    return results
