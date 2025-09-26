import torch as th
import jax
import jax.numpy as jnp
from torch2jax import t2j
from lpe.lpe.utils import Transformer

from functools import partial

import jax.numpy as jnp
import jax 
import gpt_mpc_utils
import jax_components

model_name = "gelu-2l"
device = th.device("cuda" if th.cuda.is_available() else "cpu")
model = Transformer.from_pretrained(model_name).to(device)

'''
assumes controller has length equivalent to the number of blocks.
'''
# input = th.tensor([32990])
input = th.tensor([32900])
print(f"input: {input}")

onehot = th.nn.functional.one_hot(input, num_classes=model.embed.d_vocab).float().to(device)
# print(f"onehot: {onehot.argmax(-1)}")
onehot.requires_grad_(True)
x = onehot @ model.embed.W_E
print(f"xth shape in test: {x.shape}")
# print(f"pos embed: {model.pos_embed(input)}")
# x = x + model.pos_embed(current_samples)
x = x + model.pos_embed(input)
print(f"xth pose embed shape in test: {x.shape}")
for i, block in enumerate(model.blocks):
    x = block(x)
    # print("i like doggies")
# print(f"x shape pre ln: {x.shape}")
x = model.ln_final(x[:,-1].unsqueeze(1))
# print(f"x shape post ln: {x.shape}")
# print(f"x post ln: {x}")
logits = model.unembed(x).squeeze(1)
y = logits.argmax(-1)

print(f"y_torch = {y}")





tfs_torch = model.blocks
x_th = onehot @ model.embed.W_E
x_th = x_th + model.pos_embed(input)
p = jnp.asarray(x_th.detach())
print(f"p shape in test: {p.shape}")


params = []
n_ctx = model.cfg["n_ctx"]
for i, b in enumerate(tfs_torch):
    params.append(jax_components.pytorch_to_jax_params(b,n_ctx))



tfs_jax = [partial(jax_components.transformer_block_forward, d) for d in params]

for tf in tfs_jax:
    p = tf(p)
    # print("i like doggies")
# print(f"x shape pre ln: {x.shape}")


ln_w = jnp.asarray(model.ln_final.w.detach())
ln_eps = jnp.asarray(model.ln_final.ln_eps)
ln_b = jnp.asarray(model.ln_final.b.detach())
p = jnp.squeeze(jax_components.layer_norm(ln_w, ln_b, ln_eps, p),0)
print(f"that late p: {p.shape}")

# print(f"x shape post ln: {x.shape}")
# print(f"x post ln: {x}")


W_U_jn = jnp.asarray(model.unembed.W_U[None].detach())
logits = p @ W_U_jn
y_jx = logits.argmax(-1)

print(f"y_jax: {y_jx}")
