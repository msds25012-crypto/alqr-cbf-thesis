from typing import List, Sequence
from act.models.model_with_hooks import ModelWithHooks
from datasets import load_dataset
from pathlib import Path
import json
import random
import torch
from act.models.model_with_hooks import ModelWithHooks
from transformers import pipeline, set_seed, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from act.models import get_model
import argparse
from act_configs import model_configs, hook_configs

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

def info_pipeline(info_judge, info_tokenizer, prompts, BATCH_SIZE, device='cuda'):
    pred_info_label = []
    for start in range(0, len(prompts), BATCH_SIZE):
        # print(f"START: {start}")
        batch = prompts[start:start+BATCH_SIZE]
        inputs = info_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = info_judge.generate(
            **inputs,
            use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=info_tokenizer.eos_token_id,
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

def truth_pipeline(truth_judge, truth_tokenizer, prompts, BATCH_SIZE, device='cuda'):
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
            pad_token_id=truth_tokenizer.eos_token_id,
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

def get_t_i_scores(info_judge, info_tokenizer, truth_judge, truth_tokenizer, prompts, outputs):
    it_format = {"begin": "Q: ",
                 "end": "A:"}
    print(prompts[:10])
    full_out = [p + outputs[i] for i,p in enumerate(prompts)]
    t_prompt = build_class_prompts(it_format, full_out, "True")
    # print(f"T PROMPTS: \n {t_prompt}")
    # print(f"T PROMPTS len: \n {len(t_prompt)}")
    truth_label = truth_pipeline(truth_judge, truth_tokenizer, t_prompt, 10)
    t_classifications = [float(pred.lower() == "yes") for pred in truth_label]
    # print("t classifications: ", t_classifications)
    t_evaluation_score = torch.mean(torch.tensor(t_classifications)).item()

    h_prompt = build_class_prompts(it_format, full_out, "Helpful")
    help_label = info_pipeline(info_judge, info_tokenizer, h_prompt, 10)
    h_classifications = [float(pred.lower() == "yes") for pred in help_label]
    # print("h classifications: ", h_classifications)
    h_evaluation_score = torch.mean(torch.tensor(h_classifications)).item()
    return t_evaluation_score, h_evaluation_score


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


def get_all_questions_no_it():
    gen = load_dataset("truthfulqa/truthful_qa", "generation")
    ds_gen = gen["validation"]
    shuffled = ds_gen.shuffle(seed=None)  

    # print(shuffled[0])
    questions = [shuffled[i]["question"] for i in range(len(shuffled))]
    for i in range(len(questions)):
        questions[i] = "Q: " + questions[i] + " A:"
    return questions

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["llama1b", "gemma2b", "qwen3b", "llama8b", "gemma9b", "qwen14b"],
    default="gemma2b",
)
parser.add_argument(
    "--method",
    choices=["linearact", "meanact", "pid"],
    default="linearact",
)
args = parser.parse_args()



device = 'cuda' if torch.cuda.is_available() else 'cpu'
cache_dir = Path('act-cache')

model_config_name = args.model
config = model_configs[model_config_name]
model_path = config["model_path"]
print(model_path)


module_patterns = config["module_patterns"]

method=args.method
hook_type = hook_configs[method]["hook_type"]
quantiles_src = hook_configs[method]["quantiles_src"]

seq_len = 128

model, tokenizer = get_model(
    model_path=model_path,
    cache_dir=str(cache_dir),
    device=device,
    model_task='text-generation',
    dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,
    seq_len=seq_len,
)

# seed = 42
# set_seed(seed)
# random.seed(seed)

prompts = get_all_questions_no_it()[:200]
info_judge, info_tokenizer, truth_judge, truth_tokenizer  = build_tqa_judges()

strengths = [ 0.5, 1, 1.5]
max_new_tokens = 50
temperature = 1.0
top_p = 0.3
repetition_penalty = 1.2

def generate_batch(gen, prompts):

    out = gen(
        prompts,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        batch_size=16
    )
    return [o[0]['generated_text'] for o in out]

baseline_gen = pipeline('text-generation', model=model, tokenizer=tokenizer)
baseline_texts = generate_batch(baseline_gen, prompts)


eval_prompts = prompts

completions: List[str] = []
for idx, prompt in enumerate(prompts):
    completions.append(baseline_texts[idx][len(prompt):].strip())
    # if idx<3:
    #     print("prompt: ", prompt)
    #     print("baseline completion: ",completions[-1])
base_t, base_i = get_t_i_scores(info_judge, info_tokenizer, truth_judge, truth_tokenizer, eval_prompts, completions)

baseline_dist1 = calculate_dist_n(completions, 1)
baseline_dist2 = calculate_dist_n(completions, 2)
baseline_dist3 = calculate_dist_n(completions, 3)

results = {
    'model_name': getattr(model, 'name_or_path', 'unknown'),
    'num_trials': len(prompts),
    'max_new_tokens': max_new_tokens,
    'temperature': temperature,
    'top_p': top_p,
    'repetition_penalty': repetition_penalty,
    'baseline': {
        'base t*i': float(base_t * base_i),
        'base t': float(base_t),
        'base i': float(base_i),
        'dist_1': float(baseline_dist1),
        'dist_2': float(baseline_dist2),
        'dist_3': float(baseline_dist3),
    },
    'sweeps': [],
}
print(results)


baseline_gen = pipeline('text-generation', model=model, tokenizer=tokenizer)
baseline_texts = generate_batch(baseline_gen, prompts)

completions: List[str] = []
for idx, prompt in enumerate(prompts):
    completions.append(baseline_texts[idx][len(prompt):].strip())
    # if idx<3:
    #     print("prompt: ", prompt)
    #     print("baseline completion: ",completions[-1])
base_t, base_i = get_t_i_scores(info_judge, info_tokenizer, truth_judge, truth_tokenizer, eval_prompts, completions)

baseline_dist1 = calculate_dist_n(completions, 1)
baseline_dist2 = calculate_dist_n(completions, 2)
baseline_dist3 = calculate_dist_n(completions, 3)


intervention_dir = cache_dir / 'interventions' / Path(model_path).name / f'{hook_type}_tqa_incr'
for lam in strengths:
    model_hooks = ModelWithHooks(module=model, device=device)
    model_hooks.load_hooks_from_folder(
        folder=intervention_dir,
        module_names=module_patterns,
        hook_type=hook_type,
        intervention_position='all',
        strength=lam,
        device=device,
        dtype=torch.float32,
        quantiles_src=quantiles_src,
    )
    model_hooks.register_hooks()

    steered_gen = pipeline('text-generation', model=model_hooks.module, tokenizer=tokenizer,batch_size=16)
    steered_texts = generate_batch(steered_gen, prompts)

    steered_completions: List[str] = []
    for idx, prompt in enumerate(prompts):
        steered_completions.append(steered_texts[idx][len(prompt):].strip())
        if idx<3:
            print("prompt: ", prompt)
            print("steered completion: ",steered_completions[-1])

    steered_t, steered_i = get_t_i_scores(info_judge, info_tokenizer, truth_judge, truth_tokenizer, eval_prompts, steered_completions)

    dist1 = calculate_dist_n(steered_completions, 1)
    dist2 = calculate_dist_n(steered_completions, 2)
    dist3 = calculate_dist_n(steered_completions, 3)

    results['sweeps'].append({
        'lambda': float(lam),
        'steered t*i': float(steered_t * steered_i),
        'steered t': float(steered_t),
        'steered i': float(steered_i),
        'base t*i': float(base_t * base_i),
        'base t': float(base_t),
        'base i': float(base_i),
        'dist_1_base': float(baseline_dist1),
        'dist_2_base': float(baseline_dist2),
        'dist_2_base': float(baseline_dist2),
        'dist_3_base': float(baseline_dist3),
        'dist_1_steered': float(dist1),
        'dist_2_steered': float(dist2),
        'dist_3_steered': float(dist3),
    })

    model_hooks.remove_hooks()

out_path = Path(f'results/tqa_TEST_sweep_{hook_type}_{args.model}.json')
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(results, indent=2))
print('saved', out_path)

