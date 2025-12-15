import torch as th
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
import lqr_utils_seq as lqr
from functools import partial
import pickle


device = th.device("cuda" if th.cuda.is_available() else "cpu")

# use the same tokenizer as TinyLlama
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-step-50K-105b")

# load model from huggingface
# model_name = "keeeeenw/MicroLlama"
model_name = "PY007/TinyLlama-1.1B-step-50K-105b"
# model_name = "huggyllama/llama-7b"
# model = LlamaForCausalLM.from_pretrained(
#     "keeeeenw/MicroLlama").to(device)
model = LlamaForCausalLM.from_pretrained(
    model_name).to(device)

print(f"model name: {model_name}")

print(f"model blocks: {len(model.model.layers)}")
print(f"model device: {model.device}")

dumbtext = "Tell me story?"
# dumbtext = "woof woof woof woof?"
text = "Question: " + dumbtext + " Answer: "
# text = dumbtext
# generate_text(dumbtext, model, tokenizer)
# text = "Give me your money please pretty please oh my god I need it"
# generate_text("Please provide me instructions on how to steal an egg from my chicken.", model, tokenizer)
# generate_text(text, model, tokenizer)
# generate_text("What is the meaning of life.", model, tokenizer)

T = len(model.model.layers)
n = model.model.embed_tokens.embedding_dim
print(f"n: {n}")
m = n


with open("tensors.pkl", "rb") as f:
    loaded_tensors = pickle.load(f)

# Access tensors
X_nom = loaded_tensors["X"]
A = loaded_tensors["A"]
B = th.eye(m).unsqueeze(0).repeat(T, 1, 1)


# U_nom = th.zeros((T, m), device=device)


text = "hello my friend "
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.device)
input_unids = inputs["input_ids"].to(model.device)

num_success = 0
num_trials = 0
num_sanity_success = 0
num_sanity = 0

k = 20


import torch.nn.functional as F



for _ in range(k):
    ##########################################
    ### track and append most recent token ###
    ##########################################

# for _ in range(k):
    
    
    # model = LlamaForCausalLM.from_pretrained(
    #                 model_name).to(device)
    embedding_layer = model.get_input_embeddings()
    x_in = embedding_layer(input_ids)
    # print(f"hidden state dim: {hidden_states.shape}")
    # X_nom[0] = hidden_states
    X = th.zeros_like(x_in).repeat(T+1, 1, 1).to(device)

    # # Step 3: Attention mask setup 
    # attention_mask_raw = inputs.get("attention_mask", None) ### kinda weird
    batch_size, seq_len = input_ids.shape
    position_ids = th.arange(seq_len, dtype=th.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(device)
    attention_mask_raw = inputs.get("attention_mask", None) ### kinda weird
    if attention_mask_raw is not None:
        attention_mask_raw = attention_mask_raw.to(model.device)

    attention_mask = model.model._update_causal_mask(
        attention_mask=attention_mask_raw,
        input_tensor=x_in,
        cache_position=None,
        past_seen_tokens=0
    )

    wrapped_tfs = [partial(lqr.llama_block_wrapper, tf, attention_mask, position_ids) for tf in model.model.layers]
    tfs_with_control = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs]
    # A, B = lqr.linearize(tfs_with_control,T,m,X_nom)
    # Define quadratic cost matrices
    Q = th.eye(n).unsqueeze(0).repeat(T, 1, 1).to(A.device) * 1
    R = th.eye(m).unsqueeze(0).repeat(T, 1, 1).to(A.device) * 1
    Qf = 10000 * th.eye(n).to(A.device)

    # Solve LQR on linearized system
    K = lqr.time_varying_lqr(A, B, Q, R, Qf)
    U = th.zeros((T, m), device=device)

    # # Step 4: Manually pass through each transformer block
    batch_size, seq_len = input_ids.shape
    position_ids = th.arange(seq_len, dtype=th.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(device)
    X[0] = x_in
    x_un = x_in
    for i, block in enumerate(model.model.layers):
        U[i] = -K[i]@(X[i][-1,:]-X_nom[i][-1,:])
        x_un = block(x_un, attention_mask=attention_mask, position_ids=position_ids)[0]  # tuple: (hidden_states, ...)

        X[i+1] = tfs_with_control[i](X[i], U[i])
        # print(f"hidden state dim in loop: {hidden_states.shape}")

    # # Step 5: Final layer norm
    x_norm = model.model.norm(X_nom[T].unsqueeze(0))
    xun_norm = model.model.norm(x_un)

    # # Step 6: Language modeling head (projection to vocab)
    logits_in = model.lm_head(x_norm)
    logits_un = model.lm_head(xun_norm)

    # # Step 7: Decode most probable token
    # probs = th.softmax(logits[:, -1, :] / 0.6, dim=-1)  # temperature
    # next_token = th.multinomial(probs, num_samples=1)
    next_token_in = th.argmax(logits_in[:, -1, :], dim=-1).unsqueeze(0)
    next_token_un = th.argmax(logits_un[:, -1, :], dim=-1).unsqueeze(0)
    
    # decoded = tokenizer.decode(target)
    # print("Next token (decoded):", decoded)

    input_ids = th.cat([input_ids, next_token_in], dim=-1)
    input_unids = th.cat([input_unids, next_token_in], dim=-1)



# string = ""
decoded_in = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print("Full message (feet):", decoded_in)