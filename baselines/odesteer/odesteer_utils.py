from evaluate import load
import json
import torch as th
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel,BitsAndBytesConfig, pipeline
# import tox_data_script as utils
import numpy as np
import typing as t
import yaml
from pathlib import Path
import pandas as pd
from typing import List, Sequence
from functools import partial
from datasets import load_dataset
from data_handling_ode import ContrastiveBuilder
import random


PATH = Path("odesteer_supertox_outputs")

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


steer_configs_tox = {
    # "llama1b": {
    #     "model_path": "meta-llama/Llama-3.2-1B",
    #     "steer_layer": 9,
    #     "strength": 2,
    # },
    # "gemma2b": {
    #     "model_path": "google/gemma-2-2b",
    #     "steer_layer": 15,
    #     "strength": 50,
    # },
    # "qwen3b": {
    #     "model_path": "Qwen/Qwen2.5-3B",
    #     "steer_layer": 19,
    #     "strength": 25,
    # },
    # "llama8b": {
    #     "model_path": "meta-llama/Meta-Llama-3-8B",
    #     # "steer_layer": 19,
    #     # "strength": 25,
    # },
    # "gemma9b": {
    #     "model_path": "google/gemma-2-9b",
    #     "steer_layer": 22,
    #     "strength": 100,
    # },
    "qwen14b": {
        "model_path": "Qwen/Qwen2.5-14B",
        "steer_layer": 24,
        "strength": 65,
    },
}

steer_configs_tqa = {
    # "llama1b": {
    #     "model_path": "meta-llama/Llama-3.2-1B",
    #     "steer_layer": ,
    #     "strength": ,
    # },
    "gemma2b": {
        "model_path": "google/gemma-2-2b",
        "steer_layer": 15,
        "strength": 50,
    },
    # "qwen3b": {
    #     "model_path": "Qwen/Qwen2.5-3B",
    #     "steer_layer": ,
    #     "strength": ,
    # },
    # "llama8b": {
    #     "model_path": "meta-llama/Meta-Llama-3-8B",
    #     # "steer_layer": ,
    #     # "strength": ,
    # },
    # "gemma9b": {
    #     "model_path": "google/gemma-2-9b",
    #     "steer_layer": ,
    #     "strength": ,
    # },
    # "qwen14b": {
    #     "model_path": "Qwen/Qwen2.5-14B",
    #     "steer_layer": ,
    #     "strength": ,
    # },
}

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
    do_sample: bool = True,
    batch_size: int = 64,
):
    model_device = next(model.parameters()).device
    completions = []

    for start_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start_idx : start_idx + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model_device)
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
        output_str = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        completions.extend(
            output_str[idx][len(prompt):].strip()
            for idx, prompt in enumerate(batch_prompts)
        )

    return completions

def collect_layer_activations(
    model,
    tokenizer,
    layer_idx: int,
    num_samples: int,
    batch_size: int = 64,
    max_length: int = 256,
):
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
        layer_idx=layer_idx,
        batch_size=batch_size,
        max_length=max_length,
    )

    nontox_prompts = [
        item["text"]
        for item in data["prompt"]
        if item["toxicity"] is not None and 0.0 <= item["toxicity"] <= 0.1
    ]
    nontox_dict = ContrastiveBuilder(model, tokenizer).collect_data_batch(
        nontox_prompts,
        num_samples,
        layer_idx=layer_idx,
        batch_size=batch_size,
        max_length=max_length,
    )

    return tox_dict, nontox_dict





# =========================== steering utils =================================

DEFAULT_ODESTEER_KWARGS = {
    "solver": "euler",
    "steps": 10,
    "n_components": 8000,
    "degree": 2,
    "gamma": 0.1,
    "coef0": 1.0,
    "lin_clf_type": "lr",
}


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
    do_sample: bool = True,
    batch_size: int = 64,
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
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
        )
    finally:
        handle.remove()


# =========================== perplexity utils =================================


@th.no_grad()
def perplexity_batch(
    sentences: t.List[str],
    prompts: t.Optional[t.List[str]],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: str,
    max_context_length: t.Optional[int] = 128,
    max_generation_length: t.Optional[int] = 50,
    autoregressive: bool = False,
) -> th.Tensor:
    """
    Compute the perplexity of the passed ``sentences`` according to a specific ``model``.
    Args:
        sentences: A list of sentences
        prompts: A list of prompts
        tokenizer: Huggingface transformers tokenizer
        model: Huggingface transformers model
        device: Device identifier
        max_context_length: Max number of tokens considered. If the sentence is shorter, pad tokens are added.
        max_generation_length: Maximum number of newly generated tokens allowed.
        autoregressive: If True, use autoregressive decoding, otherwise use parallel decoding with causal masking.
    Returns:
        Perplexity per sentence in the batch
    """
    if autoregressive:
        print("Frick")
        # return _autoregressive_perplexity_batch(
        #     sentences=sentences,
        #     prompts=prompts,
        #     tokenizer=tokenizer,
        #     model=model,
        #     device=device,
        #     max_context_length=max_context_length,
        #     max_generation_length=max_generation_length,
        # )
    else:
        return _parallel_perplexity_batch(
            sentences=sentences,
            prompts=prompts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_context_length=max_context_length,
            max_generation_length=max_generation_length,
        )

@th.no_grad()
def _parallel_perplexity_batch(
    sentences: t.List[str],
    prompts: t.Optional[t.List[str]],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: str,
    max_context_length: t.Optional[int] = 128,
    max_generation_length: t.Optional[int] = 50,
) -> th.Tensor:
    """
    Compute the perplexity of the passed ``sentences`` according to a specific ``model``.
    Args:
        sentences: A list of sentences
        prompts: A list of prompts
        tokenizer: Huggingface transformers tokenizer
        model: Huggingface transformers model
        device: Device identifier
        max_context_length: Max number of tokens considered. If the sentence is shorter, pad tokens are added.
        max_generation_length: Maximum number of newly generated tokens allowed.
    Returns:
        Perplexity per sentence in the batch
    """
    truncation = max_context_length is not None
    padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    if prompts is not None:
        text = [p + s for p, s in zip(prompts, sentences)]
    else:
        text = sentences
    tok_all = tokenizer(
        text=text,
        return_tensors="pt",
        truncation=truncation,
        padding=True,
        add_special_tokens=True,
        max_length=max_generation_length if prompts is None else max_context_length,
    ).to(device)
    tokenizer.padding_size = padding_side
    logits = model(
        input_ids=tok_all["input_ids"], attention_mask=tok_all["attention_mask"]
    ).logits
    # Compute perplexity for last token (note that indexing at offset + ctx_len gives us the token id right after :(offset + ctx_len))
    loss = th.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        tok_all["input_ids"][:, 1:].reshape(-1),
        reduction="none",
    )
    loss = (tok_all["attention_mask"][:, 1:] * loss.view(logits.shape[0], -1)).sum(
        -1
    ) / tok_all["attention_mask"][:, 1:].sum(-1)

    return th.exp(loss)

def measure_perplexity(
    continuations,
    model,
    tokenizer,
    prompts,
    batch_size: t.Optional[int] = 128,
    autoregressive: bool = False,
) -> np.ndarray:
    device = model.device
    ppl = []

    if prompts is not None:
        if isinstance(prompts, list):
            prompts = th.utils.data.DataLoader(
                dataset=prompts,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # no preprocessing happening here
            )

    if isinstance(continuations, list):
        continuations = th.utils.data.DataLoader(
            dataset=continuations,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # no preprocessing happening here
        )

    if prompts is not None:
        for c, p in zip(continuations, prompts):
            ppl_batch = perplexity_batch(
                sentences=c,
                prompts=p,
                model=model,
                tokenizer=tokenizer,
                device=device,
                autoregressive=autoregressive,
            )
            ppl.append(ppl_batch)
    else:
        for c in continuations:
            ppl_batch = perplexity_batch(
                sentences=c,
                prompts=None,
                model=model,
                tokenizer=tokenizer,
                device=device,
                autoregressive=autoregressive,
            )
            ppl.append(ppl_batch)

    ppl = th.cat(ppl).detach().cpu().numpy()
    return ppl

def get_ppl_from_file(filename, path=None, BATCH_SZ=10):
    # if path is not None:
    #     PATH=path 
    # else: 
    #     PATH=PATH
    input_path = PATH / f"{filename}.txt"
    try:
        with input_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        return False

    model_name = "mistralai/Mistral-7B-v0.1"
    model, tokenizer = load_model(model_name, quant=True)

    for sweep in data:
        prompts = sweep["prompts"]
        unsteered_gens = []
        steered_gens = []

        for i, s in enumerate(sweep["unsteered output"]):
            unsteered_gens.append(s[len(prompts[i]):])

        for i, s in enumerate(sweep["steered output"]):
            steered_gens.append(s[len(prompts[i]):])

        u_ppl = measure_perplexity(
            continuations=unsteered_gens,
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            batch_size=BATCH_SZ
        )

        s_ppl = measure_perplexity(
            continuations=steered_gens,
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            batch_size=BATCH_SZ
        )

        l=1
        print(f"l={l}, ppl_steered: {s_ppl}")
        print(f"l={l}, ppl_unsteered: {u_ppl}")
        sweep["unsteered gpt ppl"] = u_ppl.tolist()
        sweep["steered gpt ppl"] = s_ppl.tolist()
        sweep["unsteered mean ppl"] = np.mean(u_ppl).item()
        sweep["steered mean ppl"] = np.mean(s_ppl).item()

    # filename = 'tox_data/llama-3-8b-tox-prelim.txt'
    file_out = filename + "_withPPL"
    file_out = filename + "_withPPL"
    output_path = PATH / f"{file_out}.txt"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
    # with open(PATH + file_out + ".txt", 'w') as file:
    #     json.dump(data, file, indent=4)
    return True





# =========================== tqa utils =================================


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
        bnb_4bit_compute_dtype=th.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    info_judge = AutoModelForCausalLM.from_pretrained("allenai/truthfulqa-info-judge-llama2-7B",quantization_config=quant_config, dtype=th.float32, device_map="auto", trust_remote_code=True)
    info_tokenizer = AutoTokenizer.from_pretrained("allenai/truthfulqa-info-judge-llama2-7B", padding_side="left")
    info_tokenizer.pad_token = info_tokenizer.eos_token
    info_tokenizer.pad_token_id = info_tokenizer.eos_token_id

    truth_judge = AutoModelForCausalLM.from_pretrained("allenai/truthfulqa-truth-judge-llama2-7B",quantization_config=quant_config, dtype=th.float32, device_map="auto", trust_remote_code=True)
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
    t_evaluation_score = th.mean(th.tensor(t_classifications)).item()

    h_prompt = build_class_prompts(it_format, full_out, "Helpful")
    help_label = info_pipeline(info_judge, info_tokenizer, h_prompt, tokenizer, 10)
    h_classifications = [float(pred.lower() == "yes") for pred in help_label]
    # print("h classifications: ", h_classifications)
    h_evaluation_score = th.mean(th.tensor(h_classifications)).item()
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

def collect_layer_activations_tqa(
    model,
    tokenizer,
    layer_idx: int,
    num_samples: int,
    batch_size: int = 128,
    max_length: int = 256,
):
    _, _, false_prompts, true_prompts = load_tqa_prompts(num_samples)
    builder = ContrastiveBuilder(model, tokenizer)
    true_dict = builder.collect_data_batch(
        true_prompts,
        num_samples,
        layer_idx=layer_idx,
        batch_size=batch_size,
        max_length=max_length,
    )
    false_dict = builder.collect_data_batch(
        false_prompts,
        num_samples,
        layer_idx=layer_idx,
        batch_size=batch_size,
        max_length=max_length,
    )
    return true_dict, false_dict