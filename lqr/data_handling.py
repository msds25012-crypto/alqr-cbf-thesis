import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import lqr_utils_seq as lqr
from functools import partial
from datasets import load_dataset
import random
import pickle

class ContrastiveBuilder:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset_name
    ):
        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer
        self.dataset = load_dataset(dataset_name)

        self.T = len(self.model.model.layers)
        self.n = self.model.model.embed_tokens.embedding_dim
        self.m = self.n
        self.A_sum = th.zeros((self.T, self.n, self.n,)).to(self.device)
        self.X_sum = th.zeros((self.T+1, self.n,)).to(self.device)

        self.X = None # to allocate at runtime -- dependent on input length


        self.hooks = []

    def hook_collector(self, layer_idx, module, input, output):
        self.X[layer_idx] = input[0]
        if layer_idx == self.T-1:
            self.X[self.T] = output[0]
        return output
    
    def register_hooks(self):
        """Register the hooks."""

        for layer_idx, layer in enumerate(self.model.model.layers):
            def hook_wrapper(layer_idx):
                def hook(module, input, output):
                    return self.hook_collector(layer_idx, module, input, output)

                return hook

            self.hooks.append(
                layer.register_forward_hook(
                    hook_wrapper(layer_idx)
                )
            )

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def collect_data_test(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # print(f"inputs: {inputs}")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].float()
        embedding_layer = self.model.get_input_embeddings()
        hidden_states = embedding_layer(input_ids)
        self.X = th.zeros_like(hidden_states).repeat(self.T+1, 1, 1).to(self.device)

        with self:
            self.model.generate(input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    do_sample=False,
                    use_cache=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    )
        

        batch_size, seq_len = input_ids.shape
        position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(device)

        position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

        wrapped_tfs_temp = [partial(lqr.new_llama_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
        tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
        print(f"Xshape: {self.X.shape}")
        A, _ = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X)


        self.A_sum = self.A_sum + A

    def collect_data(self, num_samples, num_tokens, trait, filename, lb=0, ub=0.1, split="train"):
        data = self.dataset[split]
        filtered_data = [
            item["text"]
            for item in data["prompt"]
            if item[trait] is not None and item[trait] <= ub and item[trait] >= lb
        ]

        
        sample = random.sample(filtered_data, num_samples)
        for prompt in sample:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            # print(f"inputs: {inputs}")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"].float()
            embedding_layer = self.model.get_input_embeddings()
            hidden_states = embedding_layer(input_ids)
            self.X = th.zeros_like(hidden_states).repeat(self.T+1, 1, 1).to(self.device)

            with self:
                self.model.generate(input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=num_tokens,
                        return_dict_in_generate=True,
                        do_sample=False,
                        use_cache=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        )
            
            self.X_sum = self.X_sum + self.X[:,-1,:]

            batch_size, seq_len = input_ids.shape
            position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(device)

            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

            wrapped_tfs_temp = [partial(lqr.new_llama_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
            tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
            A, _ = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X)
            self.A_sum = self.A_sum + A


        total = num_samples*num_tokens
        tensor_dict = {
            "X": self.X_sum / total,
            "A": self.A_sum / total
        }


        with open("../../scratch/" + filename + ".pkl", "wb") as f:
            pickle.dump(tensor_dict, f)

    



device = th.device("cuda" if th.cuda.is_available() else "cpu")
# use the same tokenizer as TinyLlama

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name).to(device)

dataset_name = "allenai/real-toxicity-prompts"
dataguy = ContrastiveBuilder(model, tokenizer, dataset_name)

filename = "llama-3.2-1b_nontox"
dataguy.collect_data(1, 1, "toxicity", filename)

with open("../../scratch/" + filename + ".pkl", "rb") as f:
    loaded_tensors = pickle.load(f)

# Access tensors
X = loaded_tensors["X"]
A = loaded_tensors["A"]

print(f"X loaded: {X}")
print(f"A loaded: {A}")

# data = dataset["train"]


# # print(data["prompt"][0])
# # non_toxic = [
# #     item["text"]
# #     for item in data["prompt"]
# #     if item["toxicity"] is not None and item["toxicity"] < 0.1
# # ]
# non_toxic_ds = data.filter(
#     lambda item: item["prompt"]["toxicity"] is not None and item["prompt"]["toxicity"] < 0.1,
#     num_proc=8  # increase or decrease based on CPU cores
# )
# # print(non_toxic_ds[0])
# non_toxic = [item["prompt"]["text"] for item in non_toxic_ds]

# toxic_ds = data.filter(
#     lambda item: item["prompt"]["toxicity"] is not None and item["prompt"]["toxicity"] > 0.7,
#     num_proc=8  # increase or decrease based on CPU cores
# )
# # print(non_toxic_ds[0])
# toxic = [item["prompt"]["text"] for item in toxic_ds]
# print(len(toxic))

# prompt = non_toxic[2342]
# print(f"prompt: {prompt}")

# inputs = tokenizer(prompt, return_tensors="pt").to(device)
# input_ids = inputs["input_ids"]
# attention_mask = inputs["attention_mask"]
# output = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 max_new_tokens=15,
#                 return_dict_in_generate=True,
#                 do_sample=True,
#                 temperature=0.7,
#                 use_cache=False,
#                 pad_token_id=tokenizer.eos_token_id,
#                 # **model_generation_kwargs, #
#             )

# output_str = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
# print(output_str)


# prompt = toxic[2121]
# print(f"prompt: {prompt}")

# inputs = tokenizer(prompt, return_tensors="pt").to(device)
# input_ids = inputs["input_ids"]
# attention_mask = inputs["attention_mask"]
# output = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 max_new_tokens=15,
#                 return_dict_in_generate=True,
#                 do_sample=True,
#                 temperature=0.7,
#                 use_cache=False,
#                 pad_token_id=tokenizer.eos_token_id,
#                 # **model_generation_kwargs, #
#             )

# output_str = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
# print(output_str)