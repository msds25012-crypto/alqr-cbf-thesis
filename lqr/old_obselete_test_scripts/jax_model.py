from transformers import AutoModelForCausalLM, AutoTokenizer
import jax
import torchax
import torch
import lqr_utils_seq as lqr
from functools import partial
import jax.numpy as jnp

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(jax.__version__)
print(jax.devices())
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)



text = "bollocks"
inputs_jax = tokenizer(text, return_tensors="jax")
print(inputs_jax)

inputs = tokenizer(text, return_tensors="pt")#.to(device)
input_ids = inputs["input_ids"]#.to(device)
embedding_layer = model.get_input_embeddings()
hidden_states = embedding_layer(input_ids)
attention_mask = inputs["attention_mask"].float()

batch_size, seq_len = input_ids.shape
position_ids = torch.arange(seq_len, dtype=torch.long)#, device=device)
position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)#.to(device)


position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
cos, sin = model.model.rotary_emb(hidden_states, position_ids)
# print(f"position embeddings: {position_embeddings}")

# wrapped_tfs_temp = [partial(lqr.new_llama_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in model.model.layers]
# tfs_with_control = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
print(f"layer: {model.model.layers[0]}")
weights, func = torchax.extract_jax(model.model.layers[0])
print(f"func: {func}")


hs_jax = jnp.array(hidden_states.detach().cpu().numpy())
am_jax = jnp.array(attention_mask.cpu().numpy())
pi_jax = jnp.array(position_ids.cpu().numpy())

pe1_jax = jnp.array(position_embeddings[0].cpu().numpy())
pe2_jax = jnp.array(position_embeddings[1].cpu().numpy())
pe_jax = (pe1_jax, pe2_jax)
 

print(func(weights, (hs_jax, am_jax, pi_jax, None, False, None, pe_jax)))

