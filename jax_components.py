import torch
import jax.numpy as jnp
from jax import lax

def pytorch_to_jax_params(pt_block,n_ctx):
    """
    Given a pretrained PyTorch TransformerBlock instance `pt_block`,
    extract all parameters, buffers, and scalar attributes, and convert to
    a flat dict of JAX arrays and scalars suitable for JAX inference.
    """
    params = {}

    # LayerNorm1
    params['ln1.w'] = jnp.array(pt_block.ln1.w.detach().cpu().numpy())
    params['ln1.b'] = jnp.array(pt_block.ln1.b.detach().cpu().numpy())
    params['ln1.ln_eps'] = float(pt_block.ln1.ln_eps)  # scalar float

    # Attention params
    params['attn.W_Q'] = jnp.array(pt_block.attn.W_Q.detach().cpu().numpy())
    params['attn.b_Q'] = jnp.array(pt_block.attn.b_Q.detach().cpu().numpy())
    params['attn.W_K'] = jnp.array(pt_block.attn.W_K.detach().cpu().numpy())
    params['attn.b_K'] = jnp.array(pt_block.attn.b_K.detach().cpu().numpy())
    params['attn.W_V'] = jnp.array(pt_block.attn.W_V.detach().cpu().numpy())
    params['attn.b_V'] = jnp.array(pt_block.attn.b_V.detach().cpu().numpy())
    params['attn.W_O'] = jnp.array(pt_block.attn.W_O.detach().cpu().numpy())
    params['attn.b_O'] = jnp.array(pt_block.attn.b_O.detach().cpu().numpy())

    # Attention buffers (mask)
    if hasattr(pt_block.attn, "mask"):
        # print(f"nctx: {pt_block.attn.n_ctx}")
        params['attn.mask'] = jnp.tril(jnp.ones((n_ctx, n_ctx), dtype=bool)) #EXTREMELY HARDCODED
        # params['attn.mask'] = jnp.array(pt_block.attn.mask.detach().cpu().numpy())
    else:
        raise ValueError("Attention mask buffer missing!")

    params['attn.d_head'] = int(pt_block.attn.d_head)

    # Flag if attention-only block
    params['attn_only'] = bool(pt_block.attn_only)

    if not pt_block.attn_only:
        # LayerNorm2 params
        params['ln2.w'] = jnp.array(pt_block.ln2.w.detach().cpu().numpy())
        params['ln2.b'] = jnp.array(pt_block.ln2.b.detach().cpu().numpy())
        params['ln2.ln_eps'] = float(pt_block.ln2.ln_eps)

        # MLP params
        params['mlp.W_in'] = jnp.array(pt_block.mlp.W_in.detach().cpu().numpy())
        params['mlp.b_in'] = jnp.array(pt_block.mlp.b_in.detach().cpu().numpy())
        params['mlp.W_out'] = jnp.array(pt_block.mlp.W_out.detach().cpu().numpy())
        params['mlp.b_out'] = jnp.array(pt_block.mlp.b_out.detach().cpu().numpy())

        # Extract activation function as string
        # Your PyTorch MLP saves act_fn as a function handle, e.g. F.relu or F.gelu
        # So convert to string here; adjust if you use a different scheme
        if pt_block.mlp.act_fn == torch.nn.functional.relu:
            params['mlp.act_fn'] = 'relu'
        elif pt_block.mlp.act_fn == torch.nn.functional.gelu:
            params['mlp.act_fn'] = 'gelu'
        else:
            raise ValueError(f"Unsupported activation function: {pt_block.mlp.act_fn}")

    return params




def layer_norm(w, b, eps, x):
    # print(f"freaky ahhh: {x.shape}")
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    return x_norm * w + b

def masked_softmax(logits, mask):
    # mask = jnp.broadcast_to(mask, logits.shape)
    # logits = jnp.where(mask, logits, -1e9)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(logits)
    # exp_logits = exp_logits * mask
    sum_exp = jnp.sum(exp_logits, axis=-1, keepdims=True)
    return exp_logits / sum_exp

def mlp(x, W_in, b_in, W_out, b_out, act_fn):
    x = x @ W_in + b_in
    if act_fn == 'relu':
        x = jnp.maximum(x, 0)
    elif act_fn == 'gelu':
        # Approximation of GELU
        x = 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x ** 3)))
    else:
        raise ValueError("Unsupported activation")
    x = x @ W_out + b_out
    return x

def attention(x, W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O, mask, d_head):
    # x shape: (batch, pos, d_model)
    x = x[:,None]
    # x shape: (batch,1, pos, d_model)
    d_model = x.shape[-1]
    # print(f"x shape: {x.shape}")
    # print(f"x: {x}")
    # batch, pos, d_model = x.shape
    n_heads = W_Q.shape[0]
    
    # Compute Q,K,V per head
    # shapes:
    # W_Q: (n_heads, d_model, d_head)
    # x: (batch, pos, d_model)
    # Q: (batch, n_heads, pos, d_head)
    # print(f"Qshape: {W_Q.shape}")
    # print(f"xshape: {x.shape}")
    # print(d_head)/
    # Q = jnp.einsum('bpd,hdd->bhpd', x, W_Q) + b_Q[:, None, :]
    # K = jnp.einsum('bpd,hdd->bhpd', x, W_K) + b_K[:, None, :]
    # V = jnp.einsum('bpd,hdd->bhpd', x, W_V) + b_V[:, None, :]
    # print(f"WQ: {W_Q.shape}")
    # print(f"WK: {W_K.shape}")
    # print(f"WV: {W_V.shape}")
    # print(f"b_Q: {b_Q.shape}")
    # Q = x @ W_Q + b_Q
    # K = x @ W_K + b_K
    # V = x @ W_V + b_V
    Q = x @ W_Q[None] + b_Q[None, :, None]
    K = x @ W_K[None] + b_K[None, :, None]
    V = x @ W_V[None] + b_V[None, :, None]
    
    # print(f"Q: {Q.shape}")
    # print(f"K: {K.shape}")
    L = Q @ K.mT  # (batch, pos, d_head) @ (batch, d_head, pos) -> (batch, pos, pos)
    # print(f"L: {L.shape}")
    # print(mask.shape)
    P = masked_softmax(L / jnp.sqrt(d_head), mask)

    Z = P @ V  # (batch, pos, pos) @ (batch, pos, d_head) -> (batch, pos, d_head)

    # print(f"Z shape: {Z.shape}")
    # print(f"W_0 shape: {W_O.shape}")
    # print(f"b_O shape: {b_O.shape}")
    # step = jnp.einsum('bsd,bdh->bsh', Z, W_O)
    # print(f"step shape: {step.shape}")

    # result = jnp.einsum('ij,ijk->k', Z, W_O) + b_O # Z @ W_O + b_O 
    # result = jnp.einsum('ij,ijk->k', Z, W_O) + b_O # Z @ W_O + b_O 
    result = (
            jnp.einsum(
                # "batch head pos d_head, head d_head d_model -> batch pos d_model",
                "bhpd, hdm -> bpm",
                Z,
                W_O,
            )
            + b_O[None, None]
        )
    # print(f"result shape: {result.shape}")

    return result
    # # Attention logits: (batch, n_heads, pos, pos)
    # attn_logits = jnp.einsum('bhqd,bhkd->bhqk', Q, K) / jnp.sqrt(d_head)
    
    # # Apply mask: mask shape (pos, pos) broadcast to (batch, n_heads, pos, pos)
    # attn_weights = masked_softmax(attn_logits, mask)
    
    # # Weighted sum: (batch, n_heads, pos, d_head)
    # attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, V)
    
    # # Combine heads: (batch, pos, d_model)
    # attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))  # (batch, pos, n_heads, d_head)
    # attn_output = attn_output.reshape(batch, pos, n_heads * d_head)
    
    # # Project output
    # # W_O shape: (n_heads, d_head, d_model), so reshape to (n_heads*d_head, d_model)
    # W_O_reshaped = W_O.reshape(n_heads * d_head, d_model)
    # out = jnp.einsum('bpd,dm->bpm', attn_output, W_O_reshaped) + b_O
    # return out


def transformer_block_forward(params, x):
    """
    params: dict containing all weights and buffers as jnp arrays.
    x: jnp array of shape (batch, pos, d_model)
    """
    # LayerNorm 1
    x_ln1 = layer_norm(params['ln1.w'], params['ln1.b'], params['ln1.ln_eps'], x)
    # print(f"jax xln1: {x_ln1.shape}")
    # Attention
    # print(f"params dhead = {params['attn.d_head']}")
    attn_out = attention(
        x_ln1,
        params['attn.W_Q'], params['attn.b_Q'],
        params['attn.W_K'], params['attn.b_K'],
        params['attn.W_V'], params['attn.b_V'],
        params['attn.W_O'], params['attn.b_O'],
        params['attn.mask'],
        params['attn.d_head'],
    )
    
    x = x + attn_out
    # print(f"xattn: {x.shape}")
    
    if 'attn_only' in params and params['attn_only']:
        return x
    
    # LayerNorm 2
    x_ln2 = layer_norm(params['ln2.w'], params['ln2.b'], params['ln2.ln_eps'], x)
    
    # MLP
    mlp_out = mlp(
        x_ln2,
        params['mlp.W_in'], params['mlp.b_in'],
        params['mlp.W_out'], params['mlp.b_out'],
        params['mlp.act_fn']
    )
    x = x + mlp_out
    # print(f"xshape: {x.shape}")

    return x
