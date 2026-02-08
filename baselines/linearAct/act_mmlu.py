import random
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from datasets import load_dataset

from act.models import get_model
from act.models.model_with_hooks import ModelWithHooks
from act_configs import hook_configs, model_configs


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


def _generate_batch(model, tokenizer, prompts: Sequence[str], max_new_tokens: int, do_sample: bool, temperature: float) -> List[str]:
    inputs = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    with torch.no_grad():
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


def act_mmlu(
    model_name: str,
    hook_name: str,
    *,
    strength_list: Sequence[float] = (1.0,),
    runs_per_strength: int = 5,
    n_prompts: int = 100,
    n_loop: int = 10,
    batch_size: int = 10,
    n_shots: int = 5,
    max_new_tokens: int = 1,
    do_sample: bool = False,
    temperature: float = 0.7,
    # seed: int = 42,
) -> Dict:
    if model_name not in model_configs:
        raise ValueError(f"Unknown model config: {model_name}")
    if hook_name not in hook_configs:
        raise ValueError(f"Unknown hook config: {hook_name}")

    # random.seed(seed)
    # torch.manual_seed(seed)

    config = model_configs[model_name]
    model_path = config["model_path"]
    module_patterns = config["module_patterns"]
    hook_type = hook_configs[hook_name]["hook_type"]
    quantiles_src = hook_configs[hook_name]["quantiles_src"]

    cache_dir = Path("act-cache")
    intervention_dir = (
        cache_dir / "interventions" / Path(model_path).name / f"{hook_type}_tox_incr"
    )
    if not intervention_dir.exists():
        raise FileNotFoundError(f"Intervention dir not found: {intervention_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model(
        model_path=model_path,
        cache_dir=str(cache_dir),
        device=str(device),
        model_task="text-generation",
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        # seq_len=2048,
    )
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id



    subject_datasets = _load_subject_datasets()

    results = {
        "model_name": model_name,
        "hook_name": hook_name,
        "n_prompts": n_prompts,
        "n_loop": n_loop,
        "batch_size": batch_size,
        "n_shots": n_shots,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "runs_per_strength": runs_per_strength,
        "strengths": [],
    }

    for strength in strength_list:
        strength_runs = []

        for run_idx in range(runs_per_strength):
            baseline_correct = []
            steered_correct = []

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

                    model_hooks.register_hooks()
                    batch_outputs_steer.extend(
                        _generate_batch(
                            model_hooks.module,
                            tokenizer,
                            batch,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature,
                        )
                    )
                    model_hooks.remove_hooks()

                for item, model_output in zip(prompts_with_answers, batch_outputs):
                    model_choice = _choice_from_decoded_text(model_output)
                    baseline_correct.append(model_choice == item["answer"])

                for item, model_output in zip(prompts_with_answers, batch_outputs_steer):
                    model_choice = _choice_from_decoded_text(model_output)
                    steered_correct.append(model_choice == item["answer"])

            baseline_acc = sum(baseline_correct) / max(len(baseline_correct), 1)
            steered_acc = sum(steered_correct) / max(len(steered_correct), 1)
            strength_runs.append(
                {
                    "run": int(run_idx),
                    "baseline_accuracy": float(baseline_acc),
                    "steered_accuracy": float(steered_acc),
                    "num_samples": int(len(baseline_correct)),
                }
            )

        results["strengths"].append(
            {
                "strength": float(strength),
                "runs": strength_runs,
            }
        )

    return results
