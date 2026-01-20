import random
import torch as th
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from linearAct_fit_incremental import fit_linearact_incremental
from data_handling_linearAct import MODEL_CONFIGS


def load_model(model_name, quant=True):
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
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def main():
    model_name="gemma2b"
    model_name_hf=MODEL_CONFIGS.get(model_name)["hf_name"]

    model, tokenizer = load_model(model_name_hf, quant=True)

    dataset = load_dataset("allenai/real-toxicity-prompts")["train"]

    tox_prompts = [
        item["prompt"]["text"]
        for item in dataset
        if item["prompt"]["toxicity"] is not None and item["prompt"]["toxicity"] >= 0.8
    ]
    nontox_prompts = [
        item["prompt"]["text"]
        for item in dataset
        if item["prompt"]["toxicity"] is not None and item["prompt"]["toxicity"] <= 0.1
    ]

    random.shuffle(tox_prompts)
    random.shuffle(nontox_prompts)

    fit_linearact_incremental(
        model=model,
        tokenizer=tokenizer,
        prompts_src=tox_prompts,
        prompts_tgt=nontox_prompts,
        num_samples=200,
        out_state_name=f"{model_name}_linearact_state_tox",
        batch_size=16,
        seed=0,
        use_support=True,                # q_0_100 for mitigation
    )


if __name__ == "__main__":
    main()
