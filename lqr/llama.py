import torch as th
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
import lqr_utils as lqr
from functools import partial


device = th.device("cuda" if th.cuda.is_available() else "cpu")
# use the same tokenizer as TinyLlama
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-step-50K-105b")

# load model from huggingface
model_name = "keeeeenw/MicroLlama"
# model_name = "PY007/TinyLlama-1.1B-step-50K-105b"
# model = LlamaForCausalLM.from_pretrained(
#     "keeeeenw/MicroLlama").to(device)
model = LlamaForCausalLM.from_pretrained(
    model_name).to(device)

print(f"model name: {model_name}")
print(f"model blocks: {len(model.model.layers)}")
text = "a"

T = len(model.model.layers)
n = model.model.embed_tokens.embedding_dim
print(f"n: {n}")
m = n

U_nom = th.zeros((T, m), device=device)
X_nom = th.zeros((T+1, n), device=device)
# Tokenize input (not actually used, but this variable is referenced)
tokenizer_inputs = tokenizer(text, return_tensors="pt")

# inputs = th.arange(1000,32000, 10000).to(device)
# mock_inputs = th.arange(2323,32000,1017).to(device)

mock_inputs = th.tensor([263]).to(device)
inputs = th.tensor([3454],device=device)

num_success = 0
num_trials = 0
num_sanity_success = 0
num_sanity = 0
for nom_input in mock_inputs.unsqueeze(-1).unsqueeze(-1):
# Step 2: Get input embeddings
    print(f"nom input: {nom_input}")
    decoded = tokenizer.decode(nom_input[0])
    print("nom input (decoded):", decoded)
    logits_san = model(nom_input).logits
    next_token_san = th.argmax(logits_san[:, -1, :], dim=-1)
    # decoded_san = tokenizer.decode(next_token_san)
    # print("Next token (run model):", next_token_san)

    embedding_layer = model.get_input_embeddings()
    hidden_states = embedding_layer(nom_input)
    # print(f"hidden state dim: {hidden_states.shape}")
    X_nom[0] = hidden_states

    # Step 3: Attention mask setup (optional for causal models, but we replicate full behavior)
    attention_mask_raw = tokenizer_inputs.get("attention_mask", None) ### kinda weird
    if attention_mask_raw is not None:
        attention_mask_raw = attention_mask_raw.to(model.device)

    attention_mask = model.model._update_causal_mask(
        attention_mask=attention_mask_raw,
        input_tensor=hidden_states,
        cache_position=None,
        past_seen_tokens=0
    )

    # Step 4: Manually pass through each transformer block
    batch_size, seq_len = nom_input.shape
    position_ids = th.arange(seq_len, dtype=th.long, device=nom_input.device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
    for i, block in enumerate(model.model.layers):
        hidden_states = block(hidden_states, attention_mask=attention_mask, position_ids=position_ids)[0]  # tuple: (hidden_states, ...)
        # print(f"hidden state dim in loop: {hidden_states.shape}")
        X_nom[i+1] = hidden_states

    # Step 5: Final layer norm
    hidden_states = model.model.norm(hidden_states)

    # Step 6: Language modeling head (projection to vocab)
    logits = model.lm_head(hidden_states)

    # Step 7: Decode most probable token
    target = th.argmax(logits[:, -1, :], dim=-1)
    print(f"next token (TARGET): {target}")
    
    decoded = tokenizer.decode(target)
    print("Next token (decoded):", decoded)
    print("")

    if th.equal(next_token_san, target):
        num_sanity_success = num_sanity_success+1
    num_sanity = num_sanity + 1


    ###########
    ### LQR ###
    ###########

    wrapped_tfs = [partial(lqr.llama_block_wrapper, tf, attention_mask, position_ids) for tf in model.model.layers]
    tfs_with_control = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs]
    A, B = lqr.linearize(tfs_with_control,T,m,X_nom)

    # # Define quadratic cost matrices
    Q = th.eye(n).unsqueeze(0).repeat(T, 1, 1).to(A.device) * 1
    R = th.eye(m).unsqueeze(0).repeat(T, 1, 1).to(A.device) * 1
    Qf = 10000 * th.eye(n).to(A.device)

    # Solve LQR on linearized system
    K = lqr.time_varying_lqr(A, B, Q, R, Qf)

    X = th.zeros_like(X_nom)
    U = th.zeros_like(U_nom)


    for input_id in inputs.unsqueeze(-1).unsqueeze(-1):
        print(f" input: {input_id}")
        decoded = tokenizer.decode(input_id[0])
        print("input (decoded):", decoded)
        embedding_layer = model.get_input_embeddings()
        hidden_states = embedding_layer(input_id)
        X[0] = hidden_states
        for i in range(T):
            U[i] = U_nom[i]-0*K[i]@(X[i]-X_nom[i])
            # U[i] = U_nom[i]-(X[i]-X_nom[i])
            X[i+1] = tfs_with_control[i](X[i], U[i])


        x = model.model.norm(X[T].unsqueeze(0).unsqueeze(0))
        logits_found = model.lm_head(x)
        # logits_found = model.unembed(x).squeeze(1)
        target_found = logits_found.argmax(-1).squeeze(0)
        print(f"target_found: {target_found}")
        decoded = tokenizer.decode(target_found)
        print("target_found (decoded):", decoded)
        if th.equal(target_found, target):
            num_success = num_success+1
        num_trials = num_trials + 1


        #################
        ## no steering ##
        #################
        for i, block in enumerate(model.model.layers):
            hidden_states = block(hidden_states, attention_mask=attention_mask, position_ids=position_ids)[0]  # tuple: (hidden_states, ...)

        # Step 5: Final layer norm
        hidden_states = model.model.norm(hidden_states)

        # Step 6: Language modeling head (projection to vocab)
        logits = model.lm_head(hidden_states)

        # Step 7: Decode most probable token
        unsteered = th.argmax(logits[:, -1, :], dim=-1)
        print(f"next token (UNSTEERED): {unsteered}")
        decoded = tokenizer.decode(unsteered)
        print("unsteered (decoded):", decoded)
        print("")



print(f"num trials: {num_trials}")
print(f"num sanity: {num_sanity}")
print(f"Success rate: {num_success / num_trials}")
print(f"Sanity rate: {num_sanity_success / num_sanity}")