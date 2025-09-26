import jax
import math
from jax import numpy as jnp
import jax.lax as lax
from functools import partial
import numpy as np
import torch as th
from torch2jax import j2t, t2j

# import gpt_config as cfig

def gpt_obj(model,W_U_jn,ln, N,W,target,x, u, t):
    # print(f"x: {x}")
    xdim = model.embed.d_model
    # print(f"xdim: {xdim}")

    # stage_cost = u.T @ W[xdim:,xdim:] @ u # for matrix W just penalizing scaled norm of u 
    stage_cost = u.T @ W[xdim:,xdim:] @ u # just penalizing scaled norm of u
    
    # print(f"x shape in obj: {x.shape}")
    # x = jnp.squeeze(ln(x),0)
    x = ln(x)
    logits = x.T @ W_U_jn
    term_cost = jnp.abs(logits.argmax(-1) - logits[...,target])**2 + stage_cost

    return jnp.where(t == N, 0.5 * term_cost, 0.5 * stage_cost)
    # return stage_cost + term_cost

def gpt_hessian(model,W_U_jn,N,W,target,x, u, t):
    
    def residual(x,u):
        x_res = jnp.zeros_like(x).T
        # if (t == N):
            # logits = model.unembed(x).squeeze(1)
            # x_res = jnp.abs(logits.argmax(-1) - logits[...,target])**2
        logits = x.T @ W_U_jn
        term_res = jnp.abs(logits.argmax(-1) - logits[...,target])**2
        u_res = u.T
        return jnp.concatenate([jnp.where(t == N, term_res, x_res),u_res])
    
    J_x = jax.jacobian(residual,0)
    J_u = jax.jacobian(residual,1)
    value_babyyyyyyy = J_x(x,u).T@W@J_x(x,u), J_u(x,u).T@W@J_u(x,u), J_x(x,u).T@W@J_u(x,u)
    print(f"hessian shape: {len(value_babyyyyyyy)}")
    return value_babyyyyyyy


# def gpt_dynamics(model, x, u, t, parameter=None):
def gpt_dynamics(tfs_jax, x, u, t, parameter=None):
    # indices = jnp.array(range(len(model.blocks)))
    # ind = indices[t]
    # f_t = model.blocks[ind]
    # blocks = [b for b in model.blocks]
    # print(f"blocks: {blocks}")
    jax_blocks = [block for block in tfs_jax]
    # print(f"jax blocks: {jax_blocks}")

    branches = [lambda args, f=f: f(args) for f in jax_blocks]
    x_us = x[None,None,...]
    f_t = lax.switch(t, branches, x_us)
    # print(f"f_tshape, {f_t.shape}")
    x_next = (f_t + u).squeeze(0).squeeze(0)
    # print(f"x_next shape: {x_next.shape}")
    return x_next
    # return jnp.expand_dims(x_next,0)