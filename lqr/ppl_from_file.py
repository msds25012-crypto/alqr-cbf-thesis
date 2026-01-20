from evaluate import load
import json
import torch as th
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
import tox_data_script as utils
import numpy as np
import typing as t
import yaml

with open('config/config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)
PATH = config_data["environment"]["tox_data_path"]

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
    if path is not None:
        PATH=path 
    else: 
        PATH=PATH
        
    try:
        with open(PATH + filename + ".txt", 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        return False
    
    # ppl = load("perplexity", module_type="metric")


    # for sweep in data:

    # unoutput = ppl.compute(predictions=data[0]["unsteered output"], model_id='gpt2-xl')
    # ppl_unsteered = unoutput['mean_perplexity']
    #                     # ppl_unsteered = unsteered_results['mean_perplexity']
    # print(ppl_unsteered)

    # for sweep in data[1]["sweeps"]:
    model_name = "mistralai/Mistral-7B-v0.1"
    model, tokenizer = utils.load_model(model_name, quant=True)

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
        # unoutput = ppl.compute(predictions=sweep["unsteered output"], model_id='gpt2-xl')
        # stoutput = ppl.compute(predictions=sweep["steered output"], model_id='gpt2-xl')

        # ppl_unsteered = unoutput['mean_perplexity']
        # ppl_steered = stoutput['mean_perplexity']

        l=sweep["lambda"]
        print(f"l={l}, ppl_steered: {s_ppl}")
        print(f"l={l}, ppl_unsteered: {u_ppl}")
        sweep["unsteered gpt ppl"] = u_ppl.tolist()
        sweep["steered gpt ppl"] = s_ppl.tolist()
        sweep["unsteered mean ppl"] = np.mean(u_ppl).item()
        sweep["steered mean ppl"] = np.mean(s_ppl).item()

    # filename = 'tox_data/llama-3-8b-tox-prelim.txt'
    file_out = filename + "_withPPL"
    with open(PATH + file_out + ".txt", 'w') as file:
        json.dump(data, file, indent=4)
    return True

def main():
    filename = 'gemma-2-2b_tox_eval'
    # with open(PATH + filename + ".txt", 'r') as file:
    #     data = json.load(file)
    

    get_ppl_from_file(filename)

if __name__ == "__main__":
    main()