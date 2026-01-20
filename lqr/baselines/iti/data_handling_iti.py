import os
import random
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import pyvene as pv
from interveners import Collector, wrapper
from utils import get_llama_activations_pyvene, resolve_attn_out_proj_component

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
