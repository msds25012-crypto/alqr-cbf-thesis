# from test_toxicity import PATH, load_file, run_trials
from ppl_from_file import get_ppl_from_file
from actadd_mmlu import actadd_mmlu
# import tox_data_script as utils
import json
import os.path
import random
from pathlib import Path
from typing import List, Sequence
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import torch as th
from transformers import pipeline

from ActAddsteering import ActAddSteering
from actadd_config import model_configs

PATH="actadd_supertox_outputs"

def load_model(model_name, quant=False):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,          # or load_in_8bit=True
        # load_in_8bit=True,
        bnb_4bit_compute_dtype=th.float16,
        bnb_4bit_quant_type="nf4",  # best for LLMs
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=th.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def load_contrastive_vecs(prefix: str) -> th.Tensor:
    root = Path(__file__).resolve().parent / "pickle_jar"
    import pickle
    with open(root / f"{prefix}_actadd_nontox_vec.pkl", "rb") as f:
        nontox_tensors = pickle.load(f)
    with open(root / f"{prefix}_actadd_tox_vec.pkl", "rb") as f:
        tox_tensors = pickle.load(f)
    nontox_x = nontox_tensors["X"]
    tox_x = tox_tensors["X"]
    if nontox_x.shape[0] != tox_x.shape[0] or nontox_x.shape[2] != tox_x.shape[2]:
        raise ValueError(
            "nontox/tox X must match on layers and hidden size, got "
            f"{nontox_x.shape} vs {tox_x.shape}"
        )
    max_len = max(nontox_x.shape[1], tox_x.shape[1])
    if nontox_x.shape[1] != max_len:
        pad_len = max_len - nontox_x.shape[1]
        nontox_x = th.nn.functional.pad(nontox_x, (0, 0, 0, pad_len, 0, 0))
    if tox_x.shape[1] != max_len:
        pad_len = max_len - tox_x.shape[1]
        tox_x = th.nn.functional.pad(tox_x, (0, 0, 0, pad_len, 0, 0))
    return nontox_x - tox_x


def calculate_dist_n(texts_list: Sequence[str], n: int) -> float:
    if n <= 0:
        raise ValueError("n must be a positive integer")
    tokens: List[str] = []
    for text in texts_list:
        tokens.extend(text.lower().split())
    if len(tokens) < n:
        return 0.0
    ngrams = [
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    ]
    unique_ngrams = set(ngrams)
    return len(unique_ngrams) / len(ngrams)


def toxicity_classifier(device: th.device):
    pipeline_device = 0 if device.type == "cuda" else -1
    return pipeline(
        "text-classification",
        model="s-nlp/roberta_toxicity_classifier",
        device=pipeline_device,
    )


def _completion_from_prompt(prompt: str, generated: str) -> str:
    # if generated.startswith(prompt):
    #     return generated[len(prompt):].strip()
    # return generated.strip()
    return generated[len(prompt):].strip()


def _sample_prompts(prompts: Sequence[str], num_trials: int) -> List[str]:
    if len(prompts) >= num_trials:
        return random.sample(list(prompts), num_trials)
    return random.choices(list(prompts), k=num_trials)


def generate_texts(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    batch_size: int = 8,
) -> List[str]:
    device = next(model.parameters()).device
    all_texts: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with th.inference_mode():
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
        all_texts.extend(
            tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        )
    return all_texts


def generation(configs):
    dataset_name = "allenai/real-toxicity-prompts"
    dataset = load_dataset(dataset_name)

    # toxic_prompts = utils.get_tox_prompts(0.0, 1)
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
    # num_trials = 20 #for testing
    # num_repeats = 3 #for testing

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

        model_device = next(model.parameters()).device
        contrastive_vecs = load_contrastive_vecs(model_name).to(model_device)

        sweep_results = []
        for _ in range(num_repeats):
            samples = _sample_prompts(toxic_prompts, num_trials)

            baseline_texts = generate_texts(
                model,
                tokenizer,
                samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                batch_size=gen_batch_size,
            )
            baseline_completions = [
                _completion_from_prompt(p, g)
                for p, g in zip(samples, baseline_texts)
            ]

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

            steer = ActAddSteering(
                model,
                tokenizer,
                contrastive_vecs=contrastive_vecs,
                layer_idx=steer_layer,
            )
            steered_texts = steer.track_setpoint(
                samples,
                max_new_tokens,
                lmbda=strength,
                do_sample=True,
                temp=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            steered_completions = [
                _completion_from_prompt(p, g)
                for p, g in zip(samples, steered_texts)
            ]

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
                if steer_label == "neutral" and base_label == "toxic":
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

        output_filename = f"{model_name}_actadd_tox_eval.txt"
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
        w = get_ppl_from_file(f_pfix + "_actadd_tox_eval")
        if w:
            print(f"Finish PPL: {model_name}")
        else:
            print(f"File not found for PPL: {f_pfix}_tox_eval")
    print("___________________________________________")
    
    
def mmlu(configs):
    mmlu_filename = "MMLU_trials2"
    mmlu_data = {}
    models=list(configs.keys())

    for model_name in models:
        print(f"Running MMLU: {model_name}")
        out = actadd_mmlu(model_name)
        mmlu_data[model_name] = out

        print(f"Finished MMLU model {model_name}")
    with open(PATH + mmlu_filename + ".txt", 'w') as file:
        json.dump(mmlu_data, file, indent=4)


def main():
    # Generate outputs, measure toxicity, and measure Dist 1,2,3
    generation(model_configs)
    print("Done with all generations")

    ## Measure PPL of the generations
    ppl(model_configs)
    print("finish all PPL")

    # Get MMLU performance
    mmlu(model_configs)
    print("finish all MMLU")


if __name__ == "__main__":
    main()

    
