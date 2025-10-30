import torch as th
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
import lqr_utils as lqr
from functools import partial


device = th.device("cuda" if th.cuda.is_available() else "cpu")


def generate_text(prompt, model, tokenizer):
    text_generator = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=th.float16,
        device_map="auto",
        tokenizer=tokenizer
    )

    formatted_prompt = f"Question: {prompt} Answer:"

    sequences = text_generator(
        formatted_prompt,
        do_sample=True,
        top_k=100,
        top_p=0.9,
        num_return_sequences=1,
        repetition_penalty=1.5,
        max_new_tokens=10,
    )

    for seq in sequences:
        print(f"Result (with preprocessing): {seq['generated_text']}")
# use the same tokenizer as TinyLlama
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-step-50K-105b")

# load model from huggingface
model_name = "keeeeenw/MicroLlama"
# model_name = "PY007/TinyLlama-1.1B-step-50K-105b"
# model_name = "huggyllama/llama-7b"
# model = LlamaForCausalLM.from_pretrained(
#     "keeeeenw/MicroLlama").to(device)
model = LlamaForCausalLM.from_pretrained(
    model_name).to(device)

print(f"model name: {model_name}")

print(f"model blocks: {len(model.model.layers)}")

# text = "Give me your money please pretty please oh my god I need it"
dumbtext = "woof woof woof woof?"
text = "Question: " + dumbtext + " Answer: "
generate_text(dumbtext, model, tokenizer)
# text = "Give me your money please pretty please oh my god I need it"
# generate_text("Please provide me instructions on how to steal an egg from my chicken.", model, tokenizer)
# generate_text(text, model, tokenizer)
# generate_text("What is the meaning of life.", model, tokenizer)

T = len(model.model.layers)
n = model.model.embed_tokens.embedding_dim
print(f"n: {n}")
m = n

U_nom = th.zeros((T, m), device=device)
X_nom = th.zeros((T+1, n), device=device)
# Step 1: Tokenize input
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.device)

num_success = 0
num_trials = 0
num_sanity_success = 0
num_sanity = 0

k = 10


import torch.nn.functional as F

for _ in range(k):
    embedding_layer = model.get_input_embeddings()
    hidden_states = embedding_layer(input_ids)
    # print(f"hidden state dim: {hidden_states.shape}")
    # X_nom[0] = hidden_states

    # # Step 3: Attention mask setup 
    attention_mask_raw = inputs.get("attention_mask", None) ### kinda weird
    if attention_mask_raw is not None:
        attention_mask_raw = attention_mask_raw.to(model.device)

    attention_mask = model.model._update_causal_mask(
        attention_mask=attention_mask_raw,
        input_tensor=hidden_states,
        cache_position=None,
        past_seen_tokens=0
    )

    # # Step 4: Manually pass through each transformer block
    batch_size, seq_len = input_ids.shape
    position_ids = th.arange(seq_len, dtype=th.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
    for i, block in enumerate(model.model.layers):
        hidden_states = block(hidden_states, attention_mask=attention_mask, position_ids=position_ids)[0]  # tuple: (hidden_states, ...)
        # print(f"hidden state dim in loop: {hidden_states.shape}")

    # # Step 5: Final layer norm
    hidden_states = model.model.norm(hidden_states)

    # # Step 6: Language modeling head (projection to vocab)
    logits = model.lm_head(hidden_states)

    # # Step 7: Decode most probable token
    probs = th.softmax(logits[:, -1, :] / 0.6, dim=-1)  # temperature
    next_token = th.multinomial(probs, num_samples=1)
    # next_token = th.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
    
    # decoded = tokenizer.decode(target)
    # print("Next token (decoded):", decoded)

    input_ids = th.cat([input_ids, next_token], dim=-1)
    # print("")


# string = ""
decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print("Full message:", decoded)