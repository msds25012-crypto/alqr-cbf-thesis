import torch as th
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

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


def train_all():
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    def load_from_pickle(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data

    data_l0 = load_from_pickle('layer0_data.pkl')
    # print(f"l0: {data_l0/[10]}")

    data_l1 = load_from_pickle('layer1_data.pkl')
    # # print(f"l1: {data_l1[0]}")

    data_l2 = load_from_pickle('layer2_data.pkl')
    # # print(f"l2: {data_l2[0]}")

    data_l3 = load_from_pickle('layer3_data.pkl')
    # # print(f"l3: {data_l3[0]}")

    data_l4 = load_from_pickle('layer4_data.pkl')
    # # print(f"l4: {data_l4[0]}")

    datas = [data_l0, data_l1, data_l2, data_l3, data_l4] 

    class ActDataset(Dataset):
        def __init__(self, data):
            self.data = data
            # Assuming 'data' is a list of (input, target) pairs

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            input, output = self.data[index]
            return input, output
        

    class LinearProbe(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x):
            # x shape: (batch_size, 1, 1, d)
            x = x.view(x.size(0), -1)  # flatten to (batch_size, d)
            out = self.linear(x)
            return out.squeeze(1)  # shape: (batch_size,)
        

    ######################
    ### Train function ###
    ######################

    def train(probe, dataset, dataloader, loss_fn):
        optimizer = th.optim.Adam(probe.parameters(), lr=1e-2)

        num_epochs = 400

        for epoch in range(num_epochs):
            probe.train()
            running_loss = 0.0

            # Wrap dataloader with tqdm
            # loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            loop = dataloader

            for inputs, targets in loop:
                optimizer.zero_grad()
                targets = targets.to(device).float()
                outputs = probe(inputs.to(device)).float()
                loss = loss_fn(outputs, targets)
                # print(outputs.shape)
                # print(f"inputs: {inputs.shape}")
                # print(f"targets: {targets}")

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                # avg_loss = running_loss / ((loop.n + 1) * inputs.size(0))  # average so far
                
                # print(f"whatever this is: {sum(p.numel() for p in probe.parameters() if p.requires_grad)}")
                # Update tqdm description with loss
                # loop.set_postfix(loss=avg_loss)

            epoch_loss = running_loss / len(dataset)
            # print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}")

    #########################
    ### Training l0 probe ###
    #########################


    d = 512

    for i, data in enumerate(datas):
        dataset = ActDataset(data)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        probe = LinearProbe(input_dim=d).to(device)
        loss_fn = nn.MSELoss()
        # optimizer = th.optim.Adam(probe_l0.parameters(), lr=1e-3)

        train(probe, dataset, dataloader, loss_fn)

        filename = f'probe_l{i}_curr.pth'
        th.save(probe.state_dict(), filename)

# dataset_l0 = ActDataset(data_l0)
# dataloader_l0 = DataLoader(dataset_l0, batch_size=16, shuffle=True)
# probe_l0 = LinearProbe(input_dim=d).to(device)
# loss_fn = nn.MSELoss()
# # optimizer = th.optim.Adam(probe_l0.parameters(), lr=1e-3)

# train(probe_l0, dataset_l0, dataloader_l0, loss_fn)


# ##################
# ### Evaluation ###
# ##################

# target = th.tensor([14])

# # probe.eval()

# # model_name = "gelu-4l"
# # model = Transformer.from_pretrained(model_name).to(device)

# # # initializing input distributions
# # dist_name = "camel"

# # gt_freqs = load_ground_truth(model_name, [dist_name], device=device)[dist_name] # ground truth tensor
# # gt_probs = gt_freqs / gt_freqs.sum()
# # num_tokens = 10
# # rand_inputs = th.tensor(pick_random_tokens(gt_freqs, num_tokens, 1e-9, 1e-5)).unsqueeze(-1)
# # for input in rand_inputs:

# #     onehot = th.nn.functional.one_hot(input, num_classes=model.embed.d_vocab).float().to(device)
# #     onehot.requires_grad_(True)
# #     x = onehot @ model.embed.W_E

# #     x = x + model.pos_embed(input)
# #     # print(x.device)
# #     # print(probe.device)
# #     found = probe(x)
# #     print(f"probe value: {found}")

# #     for i, block in enumerate(model.blocks):
# #         x = block(x)
# #     x = model.ln_final(x)
# #     logits_pre = model.unembed(x)
# #     y = logits_pre.argmax(-1).detach().cpu().squeeze().squeeze()
# #     print(f"yielded from model: {y}")


# #########################
# ### Training l1 probe ###
# #########################

# # d = 512
# dataset_l1 = ActDataset(data_l1)
# dataloader_l1 = DataLoader(dataset_l1, batch_size=16, shuffle=True)
# probe_l1 = LinearProbe(input_dim=d).to(device)
# loss_fn = nn.MSELoss()

# train(probe_l1, dataset_l1, dataloader_l1, loss_fn)

# #########################
# ### Training l2 probe ###
# #########################

# # d = 512
# dataset_l2 = ActDataset(data_l2)
# dataloader_l2 = DataLoader(dataset_l2, batch_size=16, shuffle=True)
# probe_l2 = LinearProbe(input_dim=d).to(device)
# loss_fn = nn.MSELoss()

# train(probe_l2, dataset_l2, dataloader_l2, loss_fn)

# #########################
# ### Training l3 probe ###
# #########################

# # d = 512
# dataset_l3 = ActDataset(data_l3)
# dataloader_l3 = DataLoader(dataset_l3, batch_size=16, shuffle=True)
# probe_l3 = LinearProbe(input_dim=d).to(device)
# loss_fn = nn.MSELoss()

# train(probe_l3, dataset_l3, dataloader_l3, loss_fn)

# #########################
# ### Training l4 probe ###
# #########################

# # d = 512
# dataset_l4 = ActDataset(data_l4)
# dataloader_l4 = DataLoader(dataset_l4, batch_size=16, shuffle=True)
# probe_l4 = LinearProbe(input_dim=d).to(device)
# loss_fn = nn.MSELoss()

# train(probe_l4, dataset_l4, dataloader_l4, loss_fn)


