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
import numpy as np
import random
import time

model_name = "gelu-4l"
device = th.device("cuda" if th.cuda.is_available() else "cpu")
model = Transformer.from_pretrained(model_name).to(device)

tfs_raw = model.blocks
tfs_with_control = [partial(lqr.transformerBlockControl, tf) for tf in tfs_raw]

n =  model.embed.d_model
m = n
T = len(tfs_with_control)
U_nom = th.zeros((T, m), device=device)
X_nom = th.zeros((T+1, n), device=device)

#####################################
#### Generate Nominal Trajectory ####
#####################################

input = th.tensor([1000])

onehot = th.nn.functional.one_hot(input, num_classes=model.embed.d_vocab).float().to(device)
onehot.requires_grad_(True)
x_tar = onehot @ model.embed.W_E
x_tar = x_tar + model.pos_embed(input.unsqueeze(0))

x_embed = th.clone(x_tar)

X_nom[0] = x_tar
for i, block in enumerate(model.blocks):
    X_nom[i+1] = block(X_nom[i])

x_lnfinal = model.ln_final(X_nom[T].unsqueeze(0).unsqueeze(0))
logits = model.unembed(x_lnfinal).squeeze(1)
target = logits.argmax(-1)
print(f"target: {target}")

###################
#### Linearize ####
###################
A, B = lqr.linearize(tfs_with_control,T,m,X_nom)

# Define cost matrices
Q = th.eye(n).unsqueeze(0).repeat(T, 1, 1).to(A.device) * 1
R = th.eye(m).unsqueeze(0).repeat(T, 1, 1).to(A.device) * 1
Qf = 10000 * th.eye(n).to(A.device)

# Solve LQR on linearized system
K = lqr.time_varying_lqr(A, B, Q, R, Qf)
print(f"k shape: {K.shape}")



##########################
#### Helper functions ####
##########################
def rollout(x0):
    xt = x0
    X = th.zeros_like(X_nom)
    X[0] = x0
    for i in range(T):
        u = U_nom[i]-K[i]@(xt-X_nom[i])
        xt = tfs_with_control[i](xt, u).squeeze(0)
        X[i+1] = xt
    xt = model.ln_final(xt)
    logits = model.unembed(xt).squeeze(1)
    target_found = logits.argmax(-1)
    return target_found, X

factor = 0.07
num = 12
circle_radius = num*factor

scales = th.arange(12)
num_dims = 8
# points = np.zeros([num_dims, scales.shape[-1]])
bigdawg = (int)(n/num_dims)
colors = np.zeros([bigdawg, scales.shape[-1]]) # 0=r or 1=g


for ind in range(bigdawg):
    i = ind*num_dims
    # print(i)

    dim = th.tensor([i])
    onehot = th.nn.functional.one_hot(dim, num_classes=model.embed.d_model).float().to(device).squeeze()

    for s in scales:
        x0 = x_embed.squeeze().squeeze() + factor*s*onehot
        found, _ = rollout(x0)
        if th.equal(found, target):
            colors[ind, s] = 1




angles = np.linspace(0, 2 * np.pi, bigdawg, endpoint=False)
fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
for i, theta in enumerate(angles):
    x_end = circle_radius * np.cos(theta)
    y_end = circle_radius * np.sin(theta)
    
    # Plot the radius line
    ax.plot([0, x_end], [0, y_end], 'lightgray', linestyle='--')

    # Plot each point along the radius with the corresponding runtime color
    for r in scales:
        x = factor* r * np.cos(theta)
        y = factor* r * np.sin(theta)
        color = 'r'
        color_code = colors[i, r]
        if color_code == 1:
            color='g'
        ax.plot(x, y, 'o', color=color)


    # plot token
    

ax.set_xlim(-1.2 * circle_radius, 1.2 * circle_radius)

ax.set_ylim(-1.2 * circle_radius, 1.2 * circle_radius)
ax.set_title('input = 100')



def generate_random_color():
    r = random.random()  # Random value for Red (0 to 1)
    g = random.random()  # Random value for Green (0 to 1)
    b = random.random()  # Random value for Blue (0 to 1)
    return (r, g, b)


def plot_token(token):
    onehot = th.nn.functional.one_hot(token, num_classes=model.embed.d_vocab).float().to(device)
    onehot.requires_grad_(True)
    x_ = onehot @ model.embed.W_E
    x_ = x_ + model.pos_embed(token.unsqueeze(0))

    bigdawg = (int)(n/num_dims)
    points = np.zeros([bigdawg, 1])

    # print(f"x_ shape : {x_.shape}")
    # print(f"x_embed shape : {x_embed.shape}")

    for ind in range(bigdawg):
        i = ind*num_dims
        dim = th.tensor([i])

        points[ind] = np.abs(x_[0,0,i].detach().cpu().numpy() - x_embed[0,0,i].detach().cpu().numpy())
        # print(f"dim diff: {points[ind]}")

    dog, cat = rollout(x_.squeeze())
    print(f"found guy: {dog}")
    # color = generate_random_color()
    color = 'k'
    print(f"color: {color}")

    for i, theta in enumerate(angles):
        x = points[i] * np.cos(theta)
        y = points[i] * np.sin(theta)
        ax.plot(x, y, 'o', color=color)




def plot_each_layer(X):
    bigdawg = (int)(n/num_dims)
    points = np.zeros([bigdawg, 1])

    # print(f"x_ shape : {x_.shape}")
    # print(f"x_embed shape : {x_embed.shape}")
    local_points = np.zeros_like(points) 

    all_max = 0
    for t in range(X.shape[0]):
        plt.figure()
        max_dim = 0
        for ind in range(bigdawg):
            i = ind*num_dims
            dim = th.tensor([i])

            local_points[ind] = np.abs(X[t,i].detach().cpu().numpy() - X_nom[t,i].detach().cpu().numpy())
            print(f"dim diff: {local_points[ind]}")
            if local_points[ind] > max_dim:
                max_dim = local_points[ind][0]
            if max_dim > all_max:
                all_max = max_dim
        # print(f"found guy: {rollout(x_.squeeze())}")
        # color = generate_random_color()
        # print(f"color: {color}")

        for i, theta in enumerate(angles):
            x = local_points[i] * np.cos(theta)
            y = local_points[i] * np.sin(theta)
            plt.plot(x, y, 'o', color='g')

        circle_radius = 1
        if max_dim > 0:
            circle_radius = max_dim
        print(circle_radius)
        for i, theta in enumerate(angles):
            x_end = circle_radius * np.cos(theta)
            y_end = circle_radius * np.sin(theta)
            
            # Plot the radius line
            plt.plot([0, x_end], [0, y_end], 'lightgray', linestyle='--')

        # plt.xlim(-1.2 * circle_radius, 1.2 * circle_radius)
        # plt.ylim(-1.2 * circle_radius, 1.2 * circle_radius)
        length = 1
        if all_max > 0:
            length = all_max
        plt.xlim(-1.2 * length, 1.2 * length)
        plt.ylim(-1.2 * length, 1.2 * length)
        plt.savefig(f"layer_{t}.png")


# dim = th.tensor([1])
# onehot = th.nn.functional.one_hot(dim, num_classes=model.embed.d_model).float().to(device).squeeze()

# x0 = x_embed.squeeze().squeeze() + 3*onehot



# found, X = rollout(x0)

# if th.equal(x0, X[0]):
    # print("I AM TIRED")
# if not th.equal(x_embed, X[0]):
    # print("I AM bien TIRED")


#########################
#### plotting layers ####
#########################

# token = th.tensor([860])
# onehot = th.nn.functional.one_hot(token, num_classes=model.embed.d_vocab).float().to(device)
# onehot.requires_grad_(True)
# x_ = onehot @ model.embed.W_E
# x_ = x_ + model.pos_embed(token.unsqueeze(0))

# found, X = rollout(x_.squeeze())
# plot_each_layer(X)
# if th.equal(found, target):
#     print("Yippee!")
# else:
#     print("less yippee")

# tokens = th.tensor([802, 2023])
tokens = th.tensor([802])
# tokens = th.tensor([67, 802, 2023, 6765])
for token in tokens:
    plot_token(token)

plt.grid(True)
plt.savefig("lqr_radius_small.png")