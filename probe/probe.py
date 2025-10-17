import torch as th
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import sys
import os
import matplotlib.pyplot as plt
import importlib.util
import numpy as np
import sys
import os
from create_dataset import create
from data_train import train_all
import time

lpe_path = os.path.abspath(os.path.join('..', 'lpe'))

if lpe_path not in sys.path:
    sys.path.insert(0, lpe_path)

from lpe.method_utils import *
from lpe.utils import Transformer
from lpe.utils import datasets as lpe_datasets

model_name = "gelu-4l"
device = th.device("cuda" if th.cuda.is_available() else "cpu")
model = Transformer.from_pretrained(model_name).to(device)

# initializing input distributions
dist_name = "camel"

gt_freqs = load_ground_truth(model_name, [dist_name], device=device)[dist_name] # ground truth tensor
gt_probs = gt_freqs / gt_freqs.sum()

# mock_inputs = th.arange(1,7) * 800
# num_tokens = 1000
# mock_inputs = th.tensor(pick_random_tokens(gt_freqs, num_tokens, 1e-9, 1e-5)).unsqueeze(-1)

# mock_inputs = th.arange(10,60) * 800
# mock_inputs = th.tensor([[3403]])
# input = th.tensor([20])
u_norms = []




# inputs = th.arange(0,48262, 1200)
# inputs = th.arange(0,1000, 3)
inputs = th.tensor(pick_random_tokens(gt_freqs, 1000, 1e-9, 1e-5))

# mock_inputs = th.arange(0,48262,4800)
mock_inputs = th.tensor(pick_random_tokens(gt_freqs, 10, 1e-9, 1e-5))
# mock_inputs = th.tensor([14])
# mock_inputs = th.tensor([152])
# mock_inputs = th.arange(0,100,50)

#####################
### Define probes ###
#####################

class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, 1, 1, d)
        x = x.view(x.size(0), -1)  # flatten to (batch_size, d)
        out = self.linear(x)
        return out.squeeze(1)  # shape: (batch_size,)

d = model.embed.d_model



num_success = 0
num_trials = 0

prev_time = time.perf_counter()

for target_in in mock_inputs.unsqueeze(-1):
    target_in = target_in.to(device)
    create(model, target_in)
    train_all()


    probes = [LinearProbe(input_dim=d).to(device) for i in range(len(model.blocks)+1)]
    # probe_l1 = LinearProbe(input_dim=d).to(device)
    for i in range(len(probes)):
        filename = f'probe_l{i}_curr.pth'
        probes[i].load_state_dict(th.load(filename))

    # for p in probes:
    #     print(f"probe shape: {p.linear.weight.shape}")

    def calc_perturb(probe, x):
        w = probe.linear.weight
        # print(f"w shape: {w.shape}")
        # print(f"x shape: {x.shape}")
        # return -w
        return -(th.dot(w.squeeze(), x.squeeze())) * w / (th.norm(w)**2)


    num_success = 0
    # target = th.tensor([11500]).to(device)
    # target = th.tensor([510]).to(device)
    # target = th.tensor([14]).to(device)


    for input in inputs.unsqueeze(-1):

        onehot = th.nn.functional.one_hot(input, num_classes=model.embed.d_vocab).float().to(device)
        onehot.requires_grad_(True)
        x = onehot @ model.embed.W_E

        # ind = 3

        # print(x.shape)
        k = 1.2
        p = k* calc_perturb(probes[0], x)
        # print(f"pshape: {p.shape}")
        x = x + model.pos_embed(input) + p
        
        for i, block in enumerate(model.blocks):
            x = block(x)
            # if i + 1 == ind:# or i+1 == ind + 1:
            p = k*calc_perturb(probes[i+1], x)
            # print(f"p: {p}")
            x = x + p
        
        x = model.ln_final(x)
        logits_pre = model.unembed(x)
        target_found = logits_pre.argmax(-1).squeeze(0)
        # print(f"TARGET FOUND: {target_found}")
        # print(f"TARGET: {target_in}")

        if th.equal(target_found, target_in):
            # print(f"target NOT reached: {target}")
            # print("target NOT reached")
            # print(f"with input: {input}")
            # print(f"instead found: {target_found}")

        # else: 
            num_success = num_success+1
            # print("target REACHED!")
            # print(f"target reached: {target}")
            # print(f"with input: {input}")

end_time = time.perf_counter()

print(f"Total num successes: {num_success}")
print(f"Success rate STEEERED: {num_success/(len(mock_inputs)*len(inputs))}")
print(f"Total runtime: {end_time - prev_time}")






#######################
### unsteered stuff ###
#######################

# num_success = 0
# for input in mock_inputs.unsqueeze(-1):

#     onehot = th.nn.functional.one_hot(input, num_classes=model.embed.d_vocab).float().to(device)
#     onehot.requires_grad_(True)
#     x = onehot @ model.embed.W_E

#     ind = 4

#     # print(x.shape)
#     # print(f"pshape: {p.shape}")
#     x = x + model.pos_embed(input) 
    
#     for i, block in enumerate(model.blocks):
#         x = block(x)
    
#     x = model.ln_final(x)
#     logits_pre = model.unembed(x)
#     target_found = logits_pre.argmax(-1).squeeze(0)

#     if th.equal(target_found, target):
#         # print(f"target NOT reached: {target}")
#         # print("target NOT reached")
#         # print(f"with input: {input}")
#         # print(f"instead found: {target_found}")

#     # else: 
#     #     print(f"target NOT reached: {target}")
#     #     print(f"with input: {input}")
#     #     print(f"instead found: {target_found}")
#     # else: 
#         num_success = num_success+1
#     # print("target reached :(")
#     # print(f"with input: {input}")
# print(f"Success rate UNSTEERED: {num_success/len(mock_inputs)}")


# # target_in = th.tensor([479]) # 14 output
# # target_in = th.tensor([12980]) # 510 output
# target_in = th.tensor([30814]) # 11500 output   
# onehot = th.nn.functional.one_hot(target_in, num_classes=model.embed.d_vocab).float().to(device)
# onehot.requires_grad_(True)
# x_tar = onehot @ model.embed.W_E

# x_tar = x_tar + model.pos_embed(target_in)

# for i, block in enumerate(model.blocks):
#     x_tar = block(x_tar)
# x_tar = model.ln_final(x_tar)
# logits_tar = model.unembed(x_tar)
# target = logits_tar.argmax(-1).detach().cpu().squeeze().squeeze()
# print(f"TARGET: {target}")