from datasets import load_dataset
from pathlib import Path
import argparse
import json
import random
import torch
from tqdm import tqdm
from act.models import get_model
from act.models.model_with_hooks import ModelWithHooks
from act.datasets import get_dataset, get_dataloader
from act.hooks import get_hook
from act.datasets.responses_io import ResponsesLoader
from act_configs import model_configs, hook_configs


def get_tox_prompts(lb=0.8, ub=1.0):
    assert(lb >= 0 and lb <= ub)
    assert(ub <= 1)
    dataset_name = "allenai/real-toxicity-prompts"
    dataset=load_dataset(dataset_name)
    data = dataset["train"]
    prompts = [
                item["text"]
                for item in data["prompt"]
                if item["toxicity"] is not None and item["toxicity"] <= ub and item["toxicity"] >= lb
            ]
    return prompts


# ======================
# get tox/nontox prompts
# ======================

toxic_prompts = get_tox_prompts(0.8, 1)
nontoxic_prompts = get_tox_prompts(0, 0.1)

seed = 42
random.seed(seed)

max_per_class = 200

toxic = list(toxic_prompts)
nontoxic = list(nontoxic_prompts)
random.shuffle(toxic)
random.shuffle(nontoxic)
toxic = toxic[:max_per_class]
nontoxic = nontoxic[:max_per_class]

prompts_path = Path('data/tox_prompts.json')
prompts_path.parent.mkdir(parents=True, exist_ok=True)
with prompts_path.open('w') as f:
    json.dump({"toxic": toxic, "non-toxic": nontoxic}, f)

print('saved', prompts_path, 'tox', len(toxic), 'nontox', len(nontoxic))



# ======================
# get modules for steering
# ======================

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

if model_config_name not in model_configs:
    raise ValueError(f"Unknown config_name: {model_config_name}")
config = model_configs[model_config_name]
model_path = config["model_path"]
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

dataset, collate_fn = get_dataset(
    name='json-subsets',
    datasets_folder=Path('.'),
    split='train',
    subsets=['toxic', 'non-toxic'],
    tokenizer=tokenizer,
    json_path=prompts_path,
)

dataloader = get_dataloader(
    dataset,
    batch_size=16,
    num_workers=0,
    collate_fn=collate_fn,
    drop_last=False,
    shuffle=True,
    balanced=True,
    seed=seed,
)

responses_root = cache_dir / 'responses' / Path(model_path).name / 'tox-prompts'
pooling_op = 'mean'



model_hooks = ModelWithHooks(module=model, device=device)
module_names = model_hooks.find_module_names(model, module_patterns)
module_names = module_names[:4]  # for testing
print('num modules:', len(module_names))


# ======================
# sequentially collect hooks
# ======================
intervention_dir = cache_dir / 'interventions' / Path(model_path).name / f'{hook_type}_tox_incr'
intervention_dir.mkdir(parents=True, exist_ok=True)

used_module_names = []
for mn in module_names:
    # apply previously learned hooks
    hooks = []
    for used in used_module_names:
        state_path = intervention_dir / f'{used}.statedict'
        state_path = Path(state_path)

        prev = get_hook(
            name=hook_type,
            module_name=used,
            device=device,
            dtype=torch.float32,
            intervention_position='all',
            strength=1.0,
            quantiles_src=quantiles_src,
        )
        prev.from_state_path(state_path)
        # if args.method!= "pid":
        #     prev.from_state_path(state_path)
        # else:
        #     prev.load_state_dict(torch.load(state_path), state_path=state_path) 
        hooks.append(prev)

    # response hook for current module
    response_hook = get_hook(
        'postprocess_and_save',
        module_name=mn,
        pooling_op_names=[pooling_op],
        output_path=responses_root,
        save_fields=['id'],
        threaded=False,
        raise_exception=False,
    )
    hooks.append(response_hook)

    # run model with hooks to collect responses for current module
    model_hooks = ModelWithHooks(module=model, device=device)
    model_hooks.register_hooks(hooks)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Collecting {mn}')):
        model_hooks.update_hooks(batch_idx=batch_idx, batch=batch)
        with torch.inference_mode():
            _ = model_hooks(batch)


    
    model_hooks.remove_hooks()

    # fit OT for current module
    loader = ResponsesLoader(
        root=responses_root,
        from_folders=[f'*/*/{pooling_op}'],
        columns=['responses', 'id', 'subset'],
    )
    data = loader.load_data_subset(
        {
            'module_names': [mn],
            'pooling_op': [pooling_op],
        },
        num_workers=0,
    )
    data = ResponsesLoader.label_src_dst_subsets(
        data,
        src_subset=['toxic'],
        dst_subset=['non-toxic'],
        balanced=True,
        seed=seed,
    )

    z = torch.tensor(data['responses'])
    y = torch.tensor(data['label']).to(torch.bool)

    hook = get_hook(
        name=hook_type,
        module_name=mn,
        device=device,
        dtype=torch.float32,
        intervention_position='all',
        strength=1.0,
        quantiles_src=quantiles_src,
    )
    hook.fit(responses=z, labels=y)
    hook.save_state_dict(state_path=intervention_dir / f'{mn}.statedict')

    used_module_names.append(mn)

print('saved incremental hooks to', intervention_dir)
