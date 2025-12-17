import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import lqr_utils_seq as lqr
from functools import partial
from datasets import load_dataset
import random
import pickle
import time

class ContrastiveBuilder:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset_name: str = None,
    ):
        self.model = model
        self.device = self.model.device
        print(f"model device: {self.device}")
        self.tokenizer = tokenizer
        self.dataset = load_dataset(dataset_name) if dataset_name is not None else None

        self.T = len(self.model.model.layers)
        self.n = self.model.model.embed_tokens.embedding_dim
        self.m = self.n
        print(f"Latent dim: {self.n}")
        self.A_sum = th.zeros((self.T, self.n, self.n,)).to(self.device)
        self.X_sum = th.zeros((self.T+1, self.n,)).to(self.device)
        self.X_mean = th.zeros((self.T+1, self.n,)).to(self.device)

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

    def collect_data(self, num_samples, num_tokens, trait, filename, lb=0, ub=0.1, split="train", collect_A = False):#, num_A = 1):
        data = self.dataset[split]
        filtered_data = [
            item["text"]
            for item in data["prompt"]
            if item[trait] is not None and item[trait] <= ub and item[trait] >= lb
        ]

        # A_iter = num_A
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


            if collect_A:# and A_iter > 0:
                batch_size, seq_len = input_ids.shape
                position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(device)

                position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

                wrapped_tfs_temp = [partial(lqr.new_llama_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
                tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
                A, _ = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X)
                self.A_sum = self.A_sum + A
                # A_iter -= 1


        total = num_samples*num_tokens
        print(f"total: {total}")
        if collect_A:
            tensor_dict = {
                "X": self.X_sum / total,
                "A": self.A_sum / total,
            } 
        else:
            tensor_dict = {
                "X": self.X_sum / total,
            } 

        with open("../../scratch/" + filename + ".pkl", "wb") as f:
            pickle.dump(tensor_dict, f)

    
    def collect_data_batch(self, prompts, num_samples, filename, num_tokens=1):
        # A_iter = num_A
        sample = random.sample(prompts, num_samples)
        inputs = self.tokenizer(
            sample, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.device)
        # print(f"inputs: {inputs}")
        input_ids = inputs["input_ids"]
        B,L = input_ids.shape
        # print(f"B,L: {B,L}")
        attention_mask = inputs["attention_mask"].float()
        embedding_layer = self.model.get_input_embeddings()
        hidden_states = embedding_layer(input_ids)
        self.X = th.zeros(self.T+1, B, L, hidden_states.size(-1), device=self.device)

        with self:
            self.model.generate(input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=num_tokens,
                    return_dict_in_generate=True,
                    do_sample=False,
                    use_cache=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    )
            
            self.X_mean = th.mean(self.X[:,:,-1,:], dim = 1)

        total = num_samples*num_tokens
        print(f"total: {total}")

        tensor_dict = {
            "X": self.X_mean,
        } 

        with open("../../scratch/" + filename + ".pkl", "wb") as f:
            pickle.dump(tensor_dict, f)



    def collect_jacobians(self, prompts, num_samples, filename, num_tokens=1):
        sample = random.sample(prompts, num_samples)
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


            # and A_iter > 0:
            batch_size, seq_len = input_ids.shape
            position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(device)

            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

            wrapped_tfs_temp = [partial(lqr.new_llama_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
            tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
            A, _ = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X)
            self.A_sum = self.A_sum + A
                # A_iter -= 1


        total = num_samples*num_tokens
        print(f"total: {total}")
        tensor_dict = {
            "A": self.A_sum / total,
        } 

        with open("../../scratch/" + filename + ".pkl", "wb") as f:
            pickle.dump(tensor_dict, f)



###################################################
###################################################


device = th.device("cuda" if th.cuda.is_available() else "cpu")

# model_name = "meta-llama/Llama-3.2-1B"
model_name = "google/gemma-2-2b"
# model_name = "Qwen/Qwen2.5-3B"
# model_name = "meta-llama/Meta-Llama-3-8B"


quant_config = BitsAndBytesConfig(
    # load_in_4bit=True,          # or load_in_8bit=True
    load_in_8bit=True,
    bnb_4bit_compute_dtype=th.float16,
    bnb_4bit_quant_type="nf4",  # best for LLMs
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=quant_config, dtype=th.float32, device_map="auto")


tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

dataset_name = "allenai/real-toxicity-prompts"
dataset = load_dataset(dataset_name)
data = dataset["train"]

non_toxic = [
    item["text"]
    for item in data["prompt"]
    if item["toxicity"] is not None and item["toxicity"] < 0.1
]

filename = "FILENAME"
dataguy = ContrastiveBuilder(model, tokenizer)
dataguy.collect_data_batch(non_toxic, 100, filename)

filename = "JAC_FILENAME"
dataguy.collect_jacobians(non_toxic, 100, filename)



data = load_dataset("rahmanidashti/truthful-qa", "multiple-choice")["validation"]
prompt_with_answer = [
            item["question"] + " " + item["mc0_targets"]["choices"][i]
            for item in data
            for i in range(2)
            if item["mc0_targets"] is not None and item["mc0_targets"]["labels"][i] == 0
        ]

filename = "gemma-2-2b-nontruth_vec"
dataguy.collect_data_batch(prompt_with_answer, 100, filename)

with open("../../scratch/" + filename + ".pkl", "rb") as f:
    loaded_tensors = pickle.load(f)
print(f"seq time: {end_time - start_time}")

# # Access tensors
X = loaded_tensors["X"]
print(X)
