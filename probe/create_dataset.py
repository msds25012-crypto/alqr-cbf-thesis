import torch as th
import sys
import os
import matplotlib.pyplot as plt
import importlib.util
import numpy as np
import sys
import os
import pickle

lpe_path = os.path.abspath(os.path.join('..', 'lpe'))

if lpe_path not in sys.path:
    sys.path.insert(0, lpe_path)

from lpe.method_utils import *
from lpe.utils import Transformer
from lpe.utils import datasets as lpe_datasets

# from lpe.lpe.method_utils import *
# from lpe.lpe.utils import Transformer

model_name = "gelu-4l"
# device = th.device("cuda" if th.cuda.is_available() else "cpu")
# model = Transformer.from_pretrained(model_name).to(device)

# # initializing input distributions
# dist_name = "camel"

# gt_freqs = load_ground_truth(model_name, [dist_name], device=device)[dist_name] # ground truth tensor
# gt_probs = gt_freqs / gt_freqs.sum()

# num_tokens = 10000
# rand_inputs = th.tensor(pick_random_tokens(gt_freqs, num_tokens, 1e-9, 1e-5)).unsqueeze(-1)

# data_l0 = []
# data_l1 = []
# data_l2 = []
# data_l3 = []
# data_l4 = []



########################
### Get target output ###
########################
# target_in = th.tensor(pick_random_tokens(gt_freqs, 1, 1e-9, 1e-5)).unsqueeze(-1)
# target_in = th.tensor([479])
# print(f"target_in: {target_in}")
def create(model, target_in):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    dist_name = "camel"

    gt_freqs = load_ground_truth(model_name, [dist_name], device=device)[dist_name] # ground truth tensor

    # num_tokens = 10000
    rand_inputs = th.arange(0,48262, 48)
    # rand_inputs = th.tensor(pick_random_tokens(gt_freqs, num_tokens, 1e-9, 1e-5)).unsqueeze(-1)

    data_l0 = []
    data_l1 = []
    data_l2 = []
    data_l3 = []
    data_l4 = []



    onehot = th.nn.functional.one_hot(target_in, num_classes=model.embed.d_vocab).float().to(device)
    onehot.requires_grad_(True)
    x_tar = onehot @ model.embed.W_E

    x_tar = x_tar + model.pos_embed(target_in)

    for i, block in enumerate(model.blocks):
        x_tar = block(x_tar)
    x_tar = model.ln_final(x_tar)
    logits_tar = model.unembed(x_tar)
    target = logits_tar.argmax(-1).detach().cpu().squeeze().squeeze()
    # print(f"TARGET: {target}")


    dmodel = model.embed.d_model

    for input in rand_inputs.unsqueeze(-1):
        acts = th.zeros([len(model.blocks)+1, 1,1, dmodel])

        onehot = th.nn.functional.one_hot(input, num_classes=model.embed.d_vocab).float().to(device)
        onehot.requires_grad_(True)
        x = onehot @ model.embed.W_E

        x = x + model.pos_embed(input)

        acts[0] = x.squeeze()
        for i, block in enumerate(model.blocks):
            x = block(x)
            acts[i+1] = x.squeeze()


        x = model.ln_final(x)
        logits_pre = model.unembed(x)
        y = logits_pre.argmax(-1).detach().cpu().squeeze().squeeze()

        diff = th.abs(y-target) / 48262
        data_l0.append((acts[0].detach().cpu(), diff))
        data_l1.append((acts[1].detach().cpu(), diff))
        data_l2.append((acts[2].detach().cpu(), diff))
        data_l3.append((acts[3].detach().cpu(), diff))
        data_l4.append((acts[4].detach().cpu(), diff))


    with open('layer0_data.pkl', 'wb') as f:
        pickle.dump(data_l0, f)

    with open('layer1_data.pkl', 'wb') as f:
        pickle.dump(data_l1, f)

    with open('layer2_data.pkl', 'wb') as f:
        pickle.dump(data_l2, f)

    with open('layer3_data.pkl', 'wb') as f:
        pickle.dump(data_l3, f)

    with open('layer4_data.pkl', 'wb') as f:
        pickle.dump(data_l4, f)

