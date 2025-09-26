import torch as th
import sys
import os
import matplotlib.pyplot as plt
from lpe.lpe.method_utils import *
from lpe.lpe.utils import Transformer
from lpe.lpe.utils import datasets as lpe_datasets

from functools import partial

import jax.numpy as jnp
import jax 
import gpt_mpc_utils
import jax_components

from torch2jax import t2j

print(sys.executable)
print(th.__version__)
print(th.version.cuda)
print(th.cuda.is_available())
# print(th.cuda.current_device())


model_name = "gelu-2l"
device = th.device("cuda" if th.cuda.is_available() else "cpu")
model = Transformer.from_pretrained(model_name).to(device)

dist_name = "camel"

lpe_datasets.USE_CACHE = True

tfs_torch = model.blocks


## Intialize JAX transformer blocks
params = []
n_ctx = model.cfg["n_ctx"]
for i, b in enumerate(tfs_torch):
    params.append(jax_components.pytorch_to_jax_params(b,n_ctx))

tfs_jax = [partial(jax_components.transformer_block_forward, d) for d in params]


## More JAX transformer params
ln_w = jnp.asarray(model.ln_final.w.detach())
ln_eps = jnp.asarray(model.ln_final.ln_eps)
ln_b = jnp.asarray(model.ln_final.b.detach())
W_U_jn = jnp.asarray(model.unembed.W_U[None].detach())

input = th.tensor([32990])

# Time and stage parameters -- do we care about anything other than N?
dt = 0.02  # Time step in seconds
N = len(model.blocks)    # Number of stages
mpc_frequency = 50  # Frequency of MPC updates in Hz

# Initial initial position (first embedding)
onehot = th.nn.functional.one_hot(input, num_classes=model.embed.d_vocab).float().to(device)
onehot.requires_grad_(True)
x = onehot @ model.embed.W_E
x = x + model.pos_embed(input)
# print(f"x shape: {x.shape}")

# p_numpy = x.detach().cpu().numpy() 
# p0 = jnp.array(p_numpy)
p0 = jnp.asarray(x.detach())
print(f"p0 shape in config: {p0.shape}")

# Determine number of joints and contacts from the lists

n =  model.embed.d_model # Number of hidden dimensions
# print(f"NEW N: {n}")
m = n # Number of controls -- to be reduced?
grf_as_state = False
# Reference torques and controls (using n_joints)
u_ref = jnp.zeros(m)  # Reference controls (concatenated torques)

# Cost matrices (diagonal matrices created using jnp.diag)
Qp = jnp.diag(jnp.ones(n))  # Cost matrix for position
# Qp = jnp.ones(n)  # Cost matrix for position
# Qu = jnp.ones(m)  # Cost matrix for control
Qu = jnp.diag(jnp.ones(m))  # Cost matrix for control

print(Qp.size)
print(Qu.size)

# jnp.expand_dims(Qp,0)/
# jnp.expand_dims(Qu,0)
W = jax.scipy.linalg.block_diag(Qp, Qu)
# W = jnp.concatenate([Qp,Qu])

use_terrain_estimation = False  # Flag to use terrain estimation

ln = partial(jax_components.layer_norm, ln_w, ln_b, ln_eps)

cost = partial(gpt_mpc_utils.gpt_obj, model, W_U_jn, ln)
hessian_approx = partial(gpt_mpc_utils.gpt_hessian, model, W_U_jn, N)
dynamics = partial(gpt_mpc_utils.gpt_dynamics, tfs_jax)

target = 1537


