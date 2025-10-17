import os
import sys

root_path = os.path.abspath('..')

if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch as th
from lpe.lpe.utils import Transformer
import matplotlib.pyplot as plt
from functools import partial
import lqr_utils as lqr
import time
from lpe.lpe.method_utils import *

model_name = "gelu-4l"
device = th.device("cuda" if th.cuda.is_available() else "cpu")
model = Transformer.from_pretrained(model_name).to(device)

tfs_raw = model.blocks
tfs_with_control = [partial(lqr.transformerBlockControl, tf) for tf in tfs_raw]

n =  model.embed.d_model
m = n
T = len(tfs_with_control)


# input = th.tensor([32990])
# input = th.tensor([100])
# print(input.shape)
# input = th.tensor([10000])


print("donkey")
U_nom = th.zeros((T, m), device=device)
X_nom = th.zeros((T+1, n), device=device)
# X_nom[0] = x + model.pos_embed(input)
# print(f"X_nom: {X_nom[0]}")
# for i in range(1, T):
#     X_nom[i] = tfs_raw[i-1](X_nom[i-1])

## Generate target
# random_input = th.tensor([33990]) # WORKS for 4layer
# random_input = th.tensor([36990])
# random_input = th.tensor([37990])
# mock_inputs = th.tensor([1, 2, 49, 100, 500, 1000, 5000, 8000, 10000, 12000, 14000, 15000, 20000, 23000, 27000, 30000, 40000, 48261])
# mock_inputs = th.arange(25) * 4
# mock_inputs = th.arange(10, 60) * 800
dist_name = "camel"
gt_freqs = load_ground_truth(model_name, [dist_name], device=device)[dist_name] # ground truth tensor
gt_probs = gt_freqs / gt_freqs.sum()
# mock_inputs = th.tensor(pick_random_tokens(gt_freqs, , 1e-9, 1e-5)).unsqueeze(-1)
# mock_inputs[0] = input
mock_inputs = th.tensor([479])
# mock_inputs = th.arange(0,48262,1200)
# input = th.tensor(pick_random_tokens(gt_freqs, 1, 1e-9, 1e-5))
u_norms = []

num_success = 0
num_trials = 0
# print(f"rand input: {input}")


# inputs = th.arange(0,48262, 900)
inputs = th.arange(0,48262, 10000)
# inputs = th.tensor([200])
print(f"len inputs: {len(inputs)}")

prev_time = time.perf_counter()


for random_input in mock_inputs:
    print(f"mock input: {random_input}")
    curr_time = time.perf_counter()
    print(f"time of iteration: {curr_time - prev_time}")
    prev_time = curr_time
    print(f"current score = {num_success / max(num_trials,1)}")
    onehot = th.nn.functional.one_hot(random_input, num_classes=model.embed.d_vocab).float().to(device)
    onehot.requires_grad_(True)
    x_tar = onehot @ model.embed.W_E
    # print(random_input)
    x_tar = x_tar + model.pos_embed(random_input.unsqueeze(0))

    X_nom[0] = x_tar
    # X_nom[T] = lqr.find_random_target(model,x_tar)
    # for i in (1, T):
    for i, block in enumerate(model.blocks):
        X_nom[i+1] = block(X_nom[i])
        # X_nom[i] = tfs_raw[i-1](X_nom[i-1])
    # diff = (X_nom[T] - X_nom[0]) / (float)(T)
    # for i in range(1, T):
    #     X_nom[i] = X_nom[i-1] + diff


    x_lnfinal = model.ln_final(X_nom[T].unsqueeze(0).unsqueeze(0))
    # print(x.shape)
    logits = model.unembed(x_lnfinal).squeeze(1)
    # print(logits.shape)

    target = logits.argmax(-1)
    # print(f"random input: {random_input}")
    # print(f"target: {target}")

    ## Linearize dynamics around nominal
    start_time = time.perf_counter()
    A, B = lqr.linearize(tfs_with_control,T,m,X_nom)

    # # Define quadratic cost matrices
    Q = th.eye(n).unsqueeze(0).repeat(T, 1, 1).to(A.device) * 1
    R = th.eye(m).unsqueeze(0).repeat(T, 1, 1).to(A.device) * 1
    Qf = 10000 * th.eye(n).to(A.device)

    # Solve LQR on linearized system
    K = lqr.time_varying_lqr(A, B, Q, R, Qf)
    # print("Feedback gains K shape:", K.shape)
    # print("K[0]:", K[0])

    for input in inputs.unsqueeze(-1):

        X = th.zeros_like(X_nom)
        U = th.zeros_like(U_nom)

        onehot = th.nn.functional.one_hot(input, num_classes=model.embed.d_vocab).float().to(device)
        onehot.requires_grad_(True)
        x = onehot @ model.embed.W_E


        X[0] = x + model.pos_embed(input)
        # X[0] = X_nom[0]
        for i in range(T):
            U[i] = U_nom[i]-K[i]@(X[i]-X_nom[i])
            X[i+1] = tfs_with_control[i](X[i], U[i])

        # print(f"U[0]: {U[0]}")
        # print(f"U[1]: {U[1]}")
        # print(f"U[-1]: {U[-1]}")

        x = model.ln_final(X[T].unsqueeze(0).unsqueeze(0))
        logits_found = model.unembed(x).squeeze(1)
        target_found = logits_found.argmax(-1)
        end_time = time.perf_counter()

        # print(f"found: {target_found}")
        # if not th.equal(target_found, target):
        #     print(f"target NOT reached: {target}")
        #     print(f"generated from input: {random_input}")
        # else: 
        if th.equal(target_found, target):
            num_success = num_success+1
            # print(f"target reached: {target}")
            # print(f"generated from input: {random_input}")
            # print(f"lqr x0: {input}")

        # us = th.norm(U)
        # u_norms.append(th.mean(us).cpu().detach().numpy())
        num_trials=num_trials+1
# print(f"time elapsed for lqr computation: {end_time - start_time}")
# plt.plot(mock_inputs.cpu().numpy(), range(5))
# plt.plot(mock_inputs.cpu().numpy(), u_norms)
# plt.xlabel("Target")
# plt.ylabel("U norm")
# plt.title("input = [20]")
# plt.savefig("lqr_gpt_100.png")

# print(f"Success rate: {num_success/len(mock_inputs)}")
print(f"num successes: {num_success}")
print(f"Success rate: {num_success/(len(mock_inputs) * len(inputs))}")






# print("VERIFICATION:")
# # input = th.tensor([32990])

# start_time = time.perf_counter()

onehot = th.nn.functional.one_hot(input, num_classes=model.embed.d_vocab).float().to(device)
onehot.requires_grad_(True)
x_p = onehot @ model.embed.W_E
x_p = x_p + model.pos_embed(input)
for block in model.blocks:
    x_p = block(x_p)
x_p = model.ln_final(x_p[:,-1].unsqueeze(1))
logits = model.unembed(x_p).squeeze(1)
y = logits.argmax(-1)

end_time = time.perf_counter()

print(f"output no steering = {y}")
# # print(f"time elapsed for no-steering model: {end_time - start_time}")

# start_time = time.perf_counter()

# x_s = onehot @ model.embed.W_E
# x_s = x_s + model.pos_embed(input)
# X_s = th.zeros_like(X)
# X_s[0] = x_s
# for i, block in enumerate(model.blocks):
#     x_s = block(x_s) + U[i]
#     X_s[i+1] = x_s
# x_s = model.ln_final(x_s[:,-1].unsqueeze(1))
# logits = model.unembed(x_s).squeeze(1)
# y = logits.argmax(-1)

# print(f"output with steering = {y}")
# end_time = time.perf_counter()
# # print(f"time elapsed for steering model: {end_time - start_time}")

# print(f"X_nom = {X_nom}")
# print(f"X = {X}")
# print(f"X_s = {X_s}")