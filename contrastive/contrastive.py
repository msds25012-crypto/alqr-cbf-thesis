import torch as th
import sys
import os
import matplotlib.pyplot as plt
import importlib.util
import numpy as np
import sys
import os
import time

root_path = os.path.abspath('..')

if root_path not in sys.path:
    sys.path.insert(0, root_path)

from lpe.lpe.method_utils import *
from lpe.lpe.utils import Transformer

model_name = "gelu-4l"
device = th.device("cuda" if th.cuda.is_available() else "cpu")
model = Transformer.from_pretrained(model_name).to(device)

# initializing input distributions
dist_name = "camel"

gt_freqs = load_ground_truth(model_name, [dist_name], device=device)[dist_name] # ground truth tensor
gt_probs = gt_freqs / gt_freqs.sum()

def generate_r_list(target_input):
    num_tokens = 15
    rand_inputs = th.tensor(pick_random_tokens(gt_freqs, num_tokens, 1e-9, 1e-5)).unsqueeze(-1)
    # print(f"rand_inputs: {rand_inputs}")
    dmodel = model.embed.d_model
    activations_sum = th.zeros([len(model.blocks)+1, dmodel]).to(device)

    for input in rand_inputs:
        onehot = th.nn.functional.one_hot(input, num_classes=model.embed.d_vocab).float().to(device)
        onehot.requires_grad_(True)
        x = onehot @ model.embed.W_E

        x = x + model.pos_embed(input)
        activations_sum[0] += x.squeeze()
        for i, block in enumerate(model.blocks):
            x = block(x)
            activations_sum[i+1] += x.squeeze()

    nontarget_acts = activations_sum / (num_tokens-1)
    # print(nontarget_acts)

    target_acts = th.zeros([len(model.blocks)+1, dmodel]).to(device)
    onehot = th.nn.functional.one_hot(target_input, num_classes=model.embed.d_vocab).float().to(device)
    onehot.requires_grad_(True)
    x = onehot @ model.embed.W_E

    x = x + model.pos_embed(target_input)
    target_acts[0] += x.squeeze()
    for i, block in enumerate(model.blocks):
        x = block(x)
        target_acts[i+1] += x.squeeze()


    x = model.ln_final(x)
    logits_pre = model.unembed(x)
    y = logits_pre.argmax(-1)
    # print(f"target: {y}")
    r_list = target_acts - nontarget_acts
    # print(r_list)
    return r_list, y

# mock_inputs = th.arange(10, 60) * 800
# mock_inputs = th.arange(0,48262,1200).to(device)
mock_inputs = th.tensor([[3403]])
# input = th.tensor([20])
inputs = th.arange(0,48262, 900).to(device)


start_time = time.perf_counter()
for liberal_media in range(1): # max 5
    num_success = 0
    num_trials = 0

    for input in inputs.unsqueeze(-1):
        curr_time = time.perf_counter()
        # print(f"iteration runtime: {curr_time - start_time}")
        # print(f"current score = {num_success / max(num_trials,1)}")


        for random_input in mock_inputs.unsqueeze(-1):
            r_list, target = generate_r_list(random_input)

            onehot = th.nn.functional.one_hot(input, num_classes=model.embed.d_vocab).float().to(device)
            onehot.requires_grad_(True)
            x = onehot @ model.embed.W_E

            ind = liberal_media

            x = x + model.pos_embed(input) + r_list[0]
            for i, block in enumerate(model.blocks):
                x = block(x)
                # print("dosdsdfsd")
                if i + 1 == ind:# or i+1 == ind + 1:
                    x = x + r_list[i+1]
            
            x = model.ln_final(x)
            logits_pre = model.unembed(x)
            target_found = logits_pre.argmax(-1)

            # if not th.equal(target_found, target):
                # print(f"target NOT reached: {target}")
                # print(f"generated from input: {random_input}")
            # else: 
            if th.equal(target_found, target):
                num_success = num_success+1
                # print(f"target reached: {target}")
                # print(f"generated from input: {random_input}")
            num_trials = num_trials + 1 

    print(f"Intervention layer: {liberal_media}")
    print(f"num_trails: {num_trials}")
    print(f"Success rate: {num_success/(len(mock_inputs) * len(inputs))}")
    print("")

end_time = time.perf_counter()


print(f"Total runtime: {end_time - start_time}")
# print(f"Success rate: {num_success/len(mock_inputs)}")


