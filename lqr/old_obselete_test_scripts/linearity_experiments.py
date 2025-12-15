import torch as th
import lqr_utils_seq as lqr
import pickle
from steering import LQRSteering
from datasets import load_dataset
import random
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import partial

device = th.device("cuda" if th.cuda.is_available() else "cpu")

###############################
###### Contrastive Stuff ######
###############################

# nontox_filename = "gemma-2-2b_nontox"
# with open("../../scratch/"+nontox_filename+".pkl", "rb") as f:
#     loaded_tensors = pickle.load(f)

# # Access tensors
# X = loaded_tensors["X"]
# A = loaded_tensors["A"]

# tox_filename = "gemma-2-2b_tox"
# with open("../../scratch/"+tox_filename+".pkl", "rb") as f:
#     loaded_tensors = pickle.load(f)

#     # Access tensors
# X_tox = loaded_tensors["X"]

# X_contr = X - X_tox

# print("Contrastive vectors normalized dot product")


# nextto = []
# actnorm = []
# wrtfinal = []

# linner = [X_contr[0]]
# print(X[0].shape)
# for i in range(0, A.shape[0]):
#     # linner.append(A[i]@linner[i])
#     linner.append(A[i]@X_contr[i])

# linsim = []

# for i in range(X.shape[0]):
#     normi = th.norm(X_contr[i])
#     actnorm.append(normi.item())
#     print(f"{i} = {th.norm(X_contr[i])}")

#     normlin = th.norm(linner[i])
#     linsim.append((th.dot(X_contr[i], linner[i])/ (normi*normlin)).item())
#     # linsim.append((th.dot(X_contr[i], linner[i])/ (normi*normi)).item())

#     for j in range(i+1, X.shape[0]):
#         normj = th.norm(X_contr[j])
#         print(f"{j} = {normj}")
#         sim = th.dot(X_contr[i], X_contr[j])/ (normi*normj)
#         print(f"{i}*{j} = {sim}\n")
#         if j==i+1:
#             nextto.append(sim.item())
#         if j == X.shape[0]-1:
#             wrtfinal.append(sim.item())


#     print("")



# def plot_bar(list, figname, ylabel):
#     layer_lbls = []
#     for i in range(len(list)):
#         layer_lbls.append(f"{i}")

#     bar_width = 0.35

#     # Set the x-axis positions for the bars
#     r1 = th.arange(len(layer_lbls))

#     # Create the bar plot
#     plt.bar(r1, list, color='skyblue', width=bar_width, label='LQR')

#     # Add labels and title
#     # plt.xlabel('')
#     plt.ylabel(ylabel)
#     plt.title('One step linearization')
#     plt.xticks([r for r in r1], layer_lbls, rotation=45)
#     plt.legend()
#     plt.tight_layout() # Adjust layout to prevent labels from overlapping
#     plt.savefig(figname + ".png")

# # plot_bar(wrtfinal, "wrtfinal")
# plot_bar(linsim, "linsimonestep")


###################################
###### End Contrastive Stuff ######
###################################



#############################################
###### Begin linearization model error ######
#############################################


model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(
    model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
nom_prompt = "dog"


inputs = tokenizer(nom_prompt, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]
print(f"ids: {input_ids}")
attention_mask = inputs["attention_mask"].float()
embedding_layer = model.get_input_embeddings()
hidden_states = embedding_layer(input_ids)

batch_size, seq_len = input_ids.shape
position_ids = th.arange(seq_len, dtype=th.long, device=device)
position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(device)
position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
wrapped_tfs = [partial(lqr.new_llama_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in model.model.layers]


steerer = LQRSteering(model, tokenizer)
X_list, A_list, out = steerer.complete_rollout(nom_prompt)



X = X_list[0]
A = A_list[0]
epsies = [0.01, 0.05, 0.1]
all_data = []
all_data_normed = []

min_all_data = []
min_all_data_normed = []


mean_all_data = []
mean_all_data_normed = []


n = steerer.n

for eps in epsies:
    deltax = th.zeros_like(X[...,-1,:]) + eps
    perturbed = X
    perturbed[...,-1,:] += deltax

    curr = []
    curr_normed = []
    min_curr = []
    min_curr_normed = []
    mean_curr = []
    mean_curr_normed = []
    for i in range(steerer.T):
        # print(f"layer {i}")
        linprop = A[i]@deltax[i] + X[i+1,-1,:]
        trueprop = wrapped_tfs[i](perturbed[i])
        diff = trueprop[...,-1,:] - linprop
        max_per_dim = th.max(th.abs(diff))
        min_per_dim = th.min(th.abs(diff))
        mean_per_dim = th.mean(th.abs(diff))
        # print(f"Max error single dim: {max_per_dim}")
        # print(f"Min error single dim: {min_per_dim}")
        # print(f"Mean error single dim: {mean_per_dim}")
        meeen = th.mean(th.abs(X[i+1]))
        nrm = max_per_dim/meeen
        minnrm = min_per_dim/meeen
        meannrm = mean_per_dim/meeen
        # print(f"meeen: {meeen}")
        # print(f"Max normalized by nom norm: {nrm}")
        # print(f"Min normalized by nom norm: {minnrm}\n")
        # print(f"Mean normalized by nom norm: {meannrm}\n")

        # print(max_per_dim.cpu().item())
        curr.append(max_per_dim.cpu().item())
        curr_normed.append(nrm.cpu().item())
       
        min_curr.append(min_per_dim.cpu().item())
        min_curr_normed.append(minnrm.cpu().item())

        mean_curr.append(mean_per_dim.cpu().item())
        mean_curr_normed.append(meannrm.cpu().item())

    all_data.append(curr)
    all_data_normed.append(curr_normed)
    
    min_all_data.append(min_curr)
    min_all_data_normed.append(min_curr_normed)

    mean_all_data.append(mean_curr)
    mean_all_data_normed.append(mean_curr_normed)



import numpy as np
def plot_bar_clusters(min_lists, max_lists, mean_lists, figname, ylabel):
    num_clusters = len(min_lists)
    num_layers = len(min_lists[0])
    

    layer_lbls = [str(i) for i in range(num_layers)]

    bar_width = 0.8 / num_clusters

    # Set the x-axis positions for the bars
    r1 = th.arange(len(layer_lbls))

    # Create the bar plot
    plt.figure(figsize=(8, 5))

    x = np.arange(num_layers)
    for idx, min_cluster in enumerate(min_lists):

        heights = [max_lists[idx][i] - min_cluster[i] for i in range(len(min_cluster))]
        print(max(heights))

        xpos = x + idx * bar_width
        plt.bar(
            xpos, 
            heights,
            bottom=min_cluster,
            width=bar_width, 
            label=f"eps = {epsies[idx]}",
            alpha=0.85
        )
        for j in range(num_layers):
            mean_y = mean_lists[idx][j]

            # Draw a horizontal line across the bar width at the mean value
            plt.plot(
                [xpos[j] - bar_width/2, xpos[j] + bar_width/2],
                [mean_y, mean_y],
                color="red",
                linewidth=1
            )
            plt.text(
                xpos[j] + bar_width/8,
                mean_y+ bar_width/8,
                f"{mean_y:.2f}",   # <-- mean printed!
                ha="center",
                va="bottom",
                fontsize=7,
                color="black",
                rotation=90
            )

    # Add labels and title
    plt.ylabel(ylabel)
    plt.ylim(0, max(max_lists[-1]) + 0.5) 
    plt.title('One step linearization')
    plt.xticks([r for r in r1], layer_lbls, rotation=45)
    plt.legend()
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig(figname + ".png")


plot_bar_clusters(min_all_data, all_data, mean_all_data, "unnormedlinerror", "Single Dim. Range")
plot_bar_clusters(min_all_data_normed, all_data_normed, mean_all_data_normed, "normedlinerror", "Single Dim. Range (normed)")