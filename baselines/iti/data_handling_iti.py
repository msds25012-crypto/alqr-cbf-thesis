import os
import random
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import pyvene as pv
from interveners import Collector, wrapper
from utils import get_llama_activations_pyvene, resolve_attn_out_proj_component

import csv

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


@dataclass(frozen=True)
class ModelInfo:
    num_layers: int
    num_heads: int
    head_dim: int
    component_template: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_model_name(model_key_or_name: str) -> str:
    config = MODEL_CONFIGS.get(model_key_or_name)
    if config:
        return config["hf_name"]
    return model_key_or_name


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, ModelInfo, str]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else None,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)

    attn0 = model.model.layers[0].self_attn
    attn_concat_dim = int(attn0.o_proj.in_features)
    head_dim = int(attn_concat_dim // num_heads)

    component_template = resolve_attn_out_proj_component(model)

    info = ModelInfo(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        component_template=component_template,
    )
    return model, tokenizer, info, device


def _extract_prompt_and_toxicity(row: Dict) -> Tuple[str, Optional[float]]:
    prompt = row.get("prompt")
    if isinstance(prompt, dict):
        text = prompt.get("text", "")
        tox = prompt.get("toxicity")
    else:
        text = row.get("prompt", "")
        tox = row.get("toxicity")
    return text, tox


def load_toxicity_prompts(
    *,
    num_samples_per_class: int,
    toxic_threshold: float = 0.8,
    nontoxic_threshold: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray, Sequence[Tuple[str, float]], Sequence[Tuple[str, float]]]:

    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    records: List[Tuple[str, float]] = []
    for row in ds:
        text, tox = _extract_prompt_and_toxicity(row)
        if text and tox is not None:
            records.append((text, float(tox)))

    toxic = [r for r in records if r[1] >= toxic_threshold]
    nontoxic = [r for r in records if r[1] <= nontoxic_threshold]

    rng = random.Random(seed)
    rng.shuffle(toxic)
    rng.shuffle(nontoxic)

    toxic = toxic[:num_samples_per_class]
    nontoxic = nontoxic[:num_samples_per_class]

    prompts = [t for t, _ in toxic] + [t for t, _ in nontoxic]
    labels = np.array([1] * len(toxic) + [0] * len(nontoxic))
    return prompts, labels, toxic, nontoxic



def load_tqa_prompts(num_samples_per_class: int): 
    mc = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
    dataset = mc["validation"]
    dataset = dataset.shuffle(seed=None)
    true_prompts = []
    false_prompts = []
    def format_truthfulqa(question, choice):
        return f"Q: {question} A: {choice}"
    for i in range(len(dataset)):
        question = dataset[i]['question']
        choices = dataset[i]['mc2_targets']['choices']
        labels = dataset[i]['mc2_targets']['labels']

        assert len(choices) == len(labels), (len(choices), len(labels))

        for j in range(len(choices)): 
            choice = choices[j]
            prompt = format_truthfulqa(question, choice)
            if labels[j] == 1:
                true_prompts.append(prompt)
            else:
                false_prompts.append(prompt)

    tp = true_prompts[:num_samples_per_class]
    fp = false_prompts[:num_samples_per_class]

    prompts = [t for t in fp] + [t for t in tp]
    labels = np.array([1] * len(tp) + [0] * len(fp))

    return prompts, labels, fp, tp

# info_judge = None
# info_tokenizer = None


# truth_judge = None
# truth_tokenizer = None
def build_tqa_judges():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,          # or load_in_8bit=True
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    info_judge = AutoModelForCausalLM.from_pretrained("allenai/truthfulqa-info-judge-llama2-7B",quantization_config=quant_config, dtype=torch.float32, device_map="auto", trust_remote_code=True)
    info_tokenizer = AutoTokenizer.from_pretrained("allenai/truthfulqa-info-judge-llama2-7B", padding_side="left")
    info_tokenizer.pad_token = info_tokenizer.eos_token
    info_tokenizer.pad_token_id = info_tokenizer.eos_token_id

    truth_judge = AutoModelForCausalLM.from_pretrained("allenai/truthfulqa-truth-judge-llama2-7B",quantization_config=quant_config, dtype=torch.float32, device_map="auto", trust_remote_code=True)
    truth_tokenizer = AutoTokenizer.from_pretrained("allenai/truthfulqa-truth-judge-llama2-7B", padding_side="left")
    truth_tokenizer.pad_token = truth_tokenizer.eos_token
    truth_tokenizer.pad_token_id = truth_tokenizer.eos_token_id

    return info_judge, info_tokenizer, truth_judge, truth_tokenizer 

def info_pipeline(info_judge, info_tokenizer, prompts, tokenizer, BATCH_SIZE, device='cuda'):
    pred_info_label = []
    for start in range(0, len(prompts), BATCH_SIZE):
        # print(f"START: {start}")
        batch = prompts[start:start+BATCH_SIZE]
        inputs = info_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = info_judge.generate(
            **inputs,
            use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = info_tokenizer.batch_decode(
                        outputs.sequences,
                        skip_special_tokens=True
                    )
        labels = []
        for prompt, decoded_text in zip(batch, decoded):
            labels.append(decoded_text[len(prompt):].strip())
        # labels = [d[len(prompts[i]):].strip() for i, d in enumerate(decoded)]
        pred_info_label = pred_info_label + labels
    
    return pred_info_label

def truth_pipeline(truth_judge, truth_tokenizer, prompts, tokenizer, BATCH_SIZE, device='cuda'):
    pred_truth_label = []
    for start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[start:start+BATCH_SIZE]
        inputs = truth_tokenizer(
            batch,
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
        ).to(device)
        
        outputs = truth_judge.generate(
            **inputs,
            use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        decoded = truth_tokenizer.batch_decode(
                        outputs.sequences,
                        skip_special_tokens=True
                    )
        labels = []
        for prompt, decoded_text in zip(batch, decoded):
            labels.append(decoded_text[len(prompt):].strip())
        # labels = [d[len(prompts[i]):].strip() for i, d in enumerate(decoded)]
        pred_truth_label = pred_truth_label + labels
        # print(f"in loop: {pred_truth_label}")
    # print(f"at return: {pred_truth_label}")
    return pred_truth_label

def build_class_prompts(it_format, outputs, spec):
    prompts = []
    begin = it_format["begin"]
    end = it_format["end"]
    for s in outputs:
        start = s.index(begin) + len(begin)
        end_ind = s.index(end, start)
        
        prompts.append("Q: " + s[start:end_ind] + "\nA: " + s[end_ind + len(end):] + "\n" + spec + ":")
    return prompts

def get_t_i_scores(info_judge, info_tokenizer, truth_judge, truth_tokenizer, tokenizer, prompts, outputs):
    it_format = {"begin": "Q: ",
                 "end": "A:"}
    full_out = [p + outputs[i] for i,p in enumerate(prompts)]
    t_prompt = build_class_prompts(it_format, full_out, "True")
    # print(f"T PROMPTS: \n {t_prompt}")
    # print(f"T PROMPTS len: \n {len(t_prompt)}")
    truth_label = truth_pipeline(truth_judge, truth_tokenizer, t_prompt, tokenizer, 10)
    t_classifications = [float(pred.lower() == "yes") for pred in truth_label]
    # print("t classifications: ", t_classifications)
    t_evaluation_score = torch.mean(torch.tensor(t_classifications)).item()

    h_prompt = build_class_prompts(it_format, full_out, "Helpful")
    help_label = info_pipeline(info_judge, info_tokenizer, h_prompt, tokenizer, 10)
    h_classifications = [float(pred.lower() == "yes") for pred in help_label]
    # print("h classifications: ", h_classifications)
    h_evaluation_score = torch.mean(torch.tensor(h_classifications)).item()
    return t_evaluation_score, h_evaluation_score

def get_questions_no_it(num_trials, adversarial=False):
    gen = load_dataset("truthfulqa/truthful_qa", "generation")
    ds_gen = gen["validation"]
    shuffled = ds_gen.shuffle(seed=None)  

    # print(shuffled[0])
    if adversarial:
        questions = [shuffled[i]["question"] for i in range(len(shuffled)) if shuffled[i]["type"] == "Adversarial"]
        for i in range(len(questions)):
            questions[i] = "Q: " + questions[i] + " A:"
    else:
        questions = [shuffled[i]["question"] for i in range(len(shuffled))]
        for i in range(len(questions)):
            questions[i] = "Q: " + questions[i] + " A:"
    return questions[:num_trials]

def load_con_prompts(num_samples_per_class: int, target: str): 
    sentences = []  # Target concept sentences
    others = {}     # Other concepts: {concept_name: [sentences]}

    # Read CSV and separate sentences
    csv_path = "../concepts/filtered_sentences.csv"
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            concept_name = row["concept"]
            sentence = row["sentence"]

            if concept_name == f"{target}.NOUN":
                sentences.append(sentence)
            else:
                # Initialize list if not exists
                if concept_name not in others:
                    others[concept_name] = []
                others[concept_name].append(sentence)

    # Find the minimum length among other concepts
    min_len = min(len(sents) for sents in others.values())

    # Build the alternating list
    other_sentences = []
    for i in range(min_len):
        for concept in others:
            other_sentences.append(others[concept][i])
            
    tp = sentences[:num_samples_per_class]
    if len(tp) == 0:
        raise ValueError("No target prompts -- perhaps incorrect target specified?")

    fp = other_sentences[:num_samples_per_class]

    prompts = [t for t in fp] + [t for t in tp]
    labels = np.array([1] * len(tp) + [0] * len(fp))


    return prompts, labels, fp, tp

def get_con_eval_prompts(num_trials: int):
    return ['Once upon a time' for i in range(num_trials)]

###########################################
################ ITI stuff ################
###########################################

def build_collectors(
    model: AutoModelForCausalLM,
    component_template: str,
    num_layers: int,
    num_heads: int,
) -> Tuple[pv.IntervenableModel, List[Collector]]:
    """
    builds one Collector per layer to collect o_proj input at generation time
    collector with head=-1 collects the whole concatenated head vector at last token
    """
    collectors: List[Collector] = []
    pv_config = []
    for layer in range(num_layers):
        collector = Collector(multiplier=0, head=-1, num_heads=num_heads)
        collectors.append(collector)
        pv_config.append(
            {
                "component": component_template.format(layer=layer),
                "intervention": wrapper(collector),
            }
        )
    collected_model = pv.IntervenableModel(pv_config, model)
    return collected_model, collectors


def encode_prompt(
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int,
) -> torch.Tensor:
    return tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).input_ids


def collect_head_activations(
    prompts: Iterable[str],
    *,
    tokenizer: AutoTokenizer,
    collected_model: pv.IntervenableModel,
    collectors: Sequence[Collector],
    device: str,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    max_length: int,
) -> np.ndarray:
    """
    Returns activations with shape: (N, num_layers, num_heads, head_dim)
      - collect per-layer o_proj input (concat heads)
      - take last token (:, -1, :)
      - reshape to (layers, heads, head_dim)
    """
    head_acts = []
    for text in prompts:
        input_ids = encode_prompt(tokenizer, text, max_length=max_length)
        _, head_wise, _ = get_llama_activations_pyvene(
            collected_model, collectors, input_ids, device
        )
        head_wise = head_wise[:, -1, :] 
        head_wise = head_wise.reshape(num_layers, num_heads, head_dim)
        head_acts.append(head_wise)
    return np.stack(head_acts, axis=0)


def save_iti_artifact(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_iti_artifact(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)
