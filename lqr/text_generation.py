from lpe.lpe.utils import Transformer
from functools import partial
import torch as th
import numpy as np
import lqr_utils_seq as lqr

model_name = "gelu-4l"
device = th.device("cuda" if th.cuda.is_available() else "cpu")
model = Transformer.from_pretrained(model_name).to(device)

# mock_input = "my friend how are dog "
# mock_input = "Type "
mock_input = "Microsoft "
# text = "you are cool "

tokens = model.encode([mock_input])

# Generate
generated_tokens = model.sample(tokens, completion_length=1, temperature=0.0)
print(f"proper generated tokens: {generated_tokens}")

# Decode and print
full_tokens = th.cat([tokens, generated_tokens], dim=1)
# print(model.tokenizer.decode(full_tokens[0]))


print(f"target input: {mock_input}")

tfs_raw = model.blocks
tfs_with_control = [partial(lqr.transformerBlockControl, tf) for tf in tfs_raw]

n =  model.embed.d_model
m = n
k = tokens.shape[-1]
T = len(tfs_with_control) # will need to extend for longer outputs (autoregression)

U_nom = th.zeros((T, m), device=device)
X_nom = th.zeros((T+1,k,n), device=device)

u_norms = []

num_success = 0

##########################################
# Generate target and nominal trajectory #
##########################################
   
tokens = model.encode([mock_input])

## begin sample
pad_token_id = model.tokenizer.pad_token_id
last_non_pad_index = (tokens != pad_token_id).long().cumsum(1).max(1).indices
pad_token = th.tensor(
    [[pad_token_id]] * tokens.shape[0], dtype=tokens.dtype, device=tokens.device
)
completion = th.zeros(
    tokens.shape[0], 0, dtype=tokens.dtype, device=tokens.device
)

## begin logits
x_tar = model.embed(tokens)
x_tar = x_tar + model.pos_embed(tokens)
# print(f"x_tar: {x_tar.shape}" )
# X_nom[0] = x_tar[:,-1,:] # track the trajectory of the last token position.
X_nom[0] = x_tar # track the trajectory of the last token position.
                         # note that this may not be optimal, see MERA paper.
                         # this does, however, match the scamford reachability paper
for i, block in enumerate(model.blocks):
    x_tar = block(x_tar)
    # X_nom[i+1] = x_tar[:,-1,:]
    X_nom[i+1] = x_tar

x_lnfinal = model.ln_final(x_tar)
# x_lnfinal = model.ln_final(X_nom[T].unsqueeze(0).unsqueeze(0))
# print(f"shapeypoo: {x_lnfinal.shape}")
logits = model.unembed(x_lnfinal)[th.arange(tokens.shape[0]), last_non_pad_index]
## end logtis

next_token = logits.argmax(-1)

completion = th.cat([completion, next_token[:, None]], dim=1)
last_non_pad_index += 1
# tokens = th.cat([tokens, pad_token], dim=1)
# tokens[th.arange(tokens.shape[0]), last_non_pad_index] = next_token

target=next_token
# print(f"target : {target}")
# generated_token = model.sample(tokens, completion_length=1, temperature=0.0)
full_tokens = th.cat([tokens, completion], dim=1)
## end sample
print(f"target output (cat with input): {model.tokenizer.decode(full_tokens[0])}")
##########################################
##########################################



## Linearize dynamics around nominal
A, B = lqr.linearize(tfs_with_control,T,m,X_nom)

# # Define quadratic cost matrices
Q = th.eye(n).unsqueeze(0).repeat(T, 1, 1).to(A.device) * 1
R = th.eye(m).unsqueeze(0).repeat(T, 1, 1).to(A.device) * 1
Qf = 10000 * th.eye(n).to(A.device)

# Solve LQR on linearized system
K = lqr.time_varying_lqr(A, B, Q, R, Qf)

# print("Feedback gains K shape:", K.shape)
# print("K[0]:", K[0])

X = th.zeros_like(X_nom)
U = th.zeros_like(U_nom)


## Apply controls
# text = "what am are you cat "
text = "Layout "
# text = "Type "

print(f"test input: {text}")

tokens = model.encode([text])

## begin sample
pad_token_id = model.tokenizer.pad_token_id
last_non_pad_index = (tokens != pad_token_id).long().cumsum(1).max(1).indices
pad_token = th.tensor(
    [[pad_token_id]] * tokens.shape[0], dtype=tokens.dtype, device=tokens.device
)
completion = th.zeros(
    tokens.shape[0], 0, dtype=tokens.dtype, device=tokens.device
)

## begin logits
x = model.embed(tokens)
X[0] = x + model.pos_embed(tokens)
# X[0] = X_nom[0]
for i in range(T):
    U[i] = U_nom[i]-K[i]@(X[i][-1,:]-X_nom[i][-1,:])
    X[i+1] = tfs_with_control[i](X[i], U[i])

x_lnfinal = model.ln_final(x_tar)
logits = model.unembed(x_lnfinal)[th.arange(tokens.shape[0]), last_non_pad_index]
## end logtis

target_found = logits.argmax(-1)

completion = th.cat([completion, target_found[:, None]], dim=1)
last_non_pad_index += 1

# print(f"found: {target_found}")

full_tokens = th.cat([tokens, completion], dim=1)
## end sample
print(f"Steered output (cat with input): {model.tokenizer.decode(full_tokens[0])}")

if not th.equal(target_found, target):
    print(f"target NOT reached: {target}")
    # print(f"generated from input: {random_input}")
# else: 
#     num_success = num_success+1
#     print(f"target reached: {target}")
#     print(f"generated from input: {random_input}")
#     # print(f"lqr x0: {input}")

# us = th.norm(U)
# u_norms.append(th.mean(us).cpu().detach().numpy())

# text = "you are cool "

tokens = model.encode([text])

# Generate
generated_tokens = model.sample(tokens, completion_length=1, temperature=0.0)
full_tokens = th.cat([tokens, generated_tokens], dim=1)
print(f"unsteered tokens: {model.tokenizer.decode(full_tokens[0])}")