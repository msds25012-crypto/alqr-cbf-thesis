import os
import sys
import torch as th
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
import pickle

root_path = os.path.abspath('..')
lqr_path = os.path.abspath('../lqr')

if root_path not in sys.path:
    sys.path.insert(0, root_path)
    sys.path.insert(0, lqr_path)
from functools import partial
import lqr_utils_seq as lqr


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


import random

# roles = [
#     "helpful and imaginative assistant",
#     "friendly AI tutor",
#     "creative chef",
#     "coding assistant",
#     "math tutor",
#     "scientific explainer",
#     "travel guide",
#     "fun companion",
#     "health advisor",
#     "storyteller"
# ]

# prompts = [
#     "tell me a story",
#     "explain the water cycle in simple terms",
#     "write a Python function that calculates factorial",
#     "suggest a quick vegetarian dinner recipe",
#     "solve 23 multiplied by 17 and show steps",
#     "summarize the key points of the article about AI",
#     "list three interesting places to visit in Tokyo",
#     "convert 72 degrees Fahrenheit to Celsius",
#     "continue the sentence: 'Once upon a time in a hidden valley…'",
#     "explain why the sky is blue",
#     "give me three tips to improve concentration",
#     "show how to reverse a string in JavaScript",
#     "describe the process of photosynthesis",
#     "tell me a joke about cats",
#     "suggest a simple daily exercise routine"
# ]

# positive_dataset = []

# for i in range(10):
#     role = random.choice(roles)
#     prompt = random.choice(prompts)
#     text = f"instruction: you are a {role}: {prompt}".strip().lower()
#     if not text.endswith(('.', '?', '!')):
#         text += '.'

#     text += "\nyour response: "
#     positive_dataset.append(text)


roles = [
    "cheerful assistant",
    "sunny storyteller",
    "happy travel guide",
    "upbeat life coach",
    "joyful chef",
    "playful companion",
    "encouraging tutor",
    "optimistic poet",
    "friendly comedian",
    "warm-hearted greeter",
]

prompts = [
    "tell me a short, cheerful story suitable for all ages",
    "give me three upbeat reasons to smile today",
    "write a warm, positive greeting someone can send to a friend",
    "compose a tiny, happy poem (4 lines) about a sunny morning",
    "suggest an easy, joyful breakfast idea that's comforting",
    "list five small things that often make people feel joyful",
    "write a short, encouraging message for someone starting a new job",
    "make a playful, wholesome joke about cats (no insults)",
    "describe a peaceful scene that makes you feel calm and happy",
    "offer three quick tips to lift someone's mood in under 5 minutes",
    "share a tiny celebration message for finishing a tough task",
    "create a cheerful two-line compliment someone could give a coworker",
    "write a short, optimistic take on a rainy day",
    "give a bright, friendly introduction for a community event",
    "suggest a 5-minute breathing exercise framed in a cheerful tone",
    "craft a one-paragraph 'you've got this' pep talk",
    "invent a wholesome, funny catchphrase about enjoying tea",
    "compose a tiny bedtime (happy) story to lull someone peacefully",
    "list three easy ways to make a small household chore feel joyful",
    "describe a vibrant, colorful garden in a way that makes the reader smile",
]

# Build dataset: multiple variants mixing roles & prompts
positive_dataset = []

def build_cheerful_dataset(n_examples=1, seed=42):
    random.seed(seed)
    for i in range(n_examples):
        role = random.choice(roles)
        prompt = random.choice(prompts)
        text = f"instruction: you are a {role}. {prompt}".strip()
        # ensure terminal punctuation
        if not text.endswith(('.', '?', '!')):
            text += '.'
        text += "\nyour response: "
        positive_dataset.append(text)
    # return dataset


# def garble_prompt(prompt):
#     words = prompt.split()
#     # randomly shuffle or insert noise
#     if len(words) > 1:
#         random.shuffle(words)
#     noise_words = ["maybe", "uh", "something", "xyz", "???"]
#     words.insert(random.randint(0, len(words)), random.choice(noise_words))
#     return " ".join(words)

# contrastive_dataset = []

# for i in range(100):
#     role = random.choice(roles)
#     prompt = random.choice(prompts)
#     garbled = garble_prompt(prompt)
#     entry = f"<|system|> {role} assistant confusing not clear.<|user|> {garbled}.<|assistant|>"
#     contrastive_dataset.append(entry)


print(tokenizer.all_special_tokens)

T = len(model.model.layers)
n = model.model.embed_tokens.embedding_dim
print(f"n: {n}")
m = n

U_nom = th.zeros((T, m), device=device)
X_sum = th.zeros((T+1, n), device=device)
A_sum = th.zeros((T+1, n, n,), device=device)

num_success = 0
num_trials = 0
num_sanity_success = 0
num_sanity = 0

k = 5


import torch.nn.functional as F

build_cheerful_dataset()
for prompt in positive_dataset:
    nom_inputs = tokenizer(prompt, return_tensors="pt")
    nom_ids = nom_inputs["input_ids"].to(model.device)
    # T_total = nom_ids.shape[1] + k  # max sequence length
    # kv_cache = [None] * len(model.model.layers)

    for _ in range(k):
        embedding_layer = model.get_input_embeddings()
        # last_token_id = nom_ids[:, -1:]  # shape: (batch, 1)
        # hidden_states = embedding_layer(last_token_id)
        hidden_states = embedding_layer(nom_ids)
        # print(f"hidden state dim: {hidden_states.shape}")
        # X_nom[0] = hidden_states
        X_nom = th.zeros_like(hidden_states).repeat(T+1, 1, 1).to(device)

        # # Step 3: Attention mask setup 
        attention_mask_raw = nom_inputs.get("attention_mask", None) ### kinda weird
        if attention_mask_raw is not None:
            attention_mask_raw = attention_mask_raw.to(model.device)

        attention_mask = model.model._update_causal_mask(
            attention_mask=attention_mask_raw,
            input_tensor=hidden_states,
            cache_position=None,
            past_seen_tokens=0
        )

        # # Step 4: Manually pass through each transformer block
        batch_size, seq_len = nom_ids.shape
        position_ids = th.arange(seq_len, dtype=th.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(device)
        # position_id = th.tensor([[nom_ids.shape[1]-1]], device=device)
        X_nom[0] = hidden_states
        for i, block in enumerate(model.model.layers):
            hidden_states = block(hidden_states, attention_mask=attention_mask, position_ids=position_ids)[0]  # tuple: (hidden_states, ...)
            # hidden_state, kv = block(hidden_state,
            #                      past_key_value=kv_cache[i],
            #                      position_ids=position_id,
            #                      use_cache = True)
            # kv_cache[i] = kv
            X_nom[i+1] = hidden_states
            X_sum[i] = X_sum[i] + X_nom[i][-1,:]
            # print(f"hidden state dim in loop: {hidden_states.shape}")

        # # Step 5: Final layer norm
        hidden_states = model.model.norm(hidden_states)

        # # Step 6: Language modeling head (projection to vocab)
        logits = model.lm_head(hidden_states)

        # # Step 7: Decode most probable token
        # probs = th.softmax(logits[:, -1, :] / 0.6, dim=-1)  # temperature
        # next_token = th.multinomial(probs, num_samples=1)
        next_token = th.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        
        # decoded = tokenizer.decode(target)
        # print("Next token (decoded):", decoded)

        nom_ids = th.cat([nom_ids, next_token], dim=-1)
        ###########
        ### LQR ###
        ###########


        wrapped_tfs_temp = [partial(lqr.llama_block_wrapper, tf, attention_mask, position_ids) for tf in model.model.layers]
        tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
        A, B = lqr.linearize(tfs_with_control_temp,T,m,X_nom)

        A_sum[i] = A_sum[i] + A[i]
        

    # string = ""
    decoded = tokenizer.decode(nom_ids[0], skip_special_tokens=True)
    print("Full message (nominal):", decoded)
    
tensor_dict = {
    "X": X_sum / (k*len(positive_dataset)),
    "A": A_sum / (k*len(positive_dataset))
}
with open("coherent.pkl", "wb") as f:
    pickle.dump(tensor_dict, f)