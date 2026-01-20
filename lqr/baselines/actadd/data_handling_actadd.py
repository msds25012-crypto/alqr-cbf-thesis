import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import random
import pickle
import time
# from lqr.steering import Mode
import yaml

with open('../../config/config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)
PICKLE_JAR = config_data["environment"]["pickle_jar"]
# print(PICKLE_JAR)

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
        # self.A_sum = th.zeros((self.T, self.n, self.n,)).to(self.device)
        # self.X_sum = th.zeros((self.T+1, self.n,)).to(self.device)
        # self.X_mean = th.zeros((self.T+1, self.n,)).to(self.device)
        self.A_sum = None
        self.X_sum = None
        self.X_mean = None

        self.X = None # to allocate at runtime -- dependent on input length

        self.e_prev = None
        # self.e_prev = th.zeros_like(self.X_sum[0])
        
        # self.U = th.zeros((self.T, self.n), device=self.device)
        # self.e_sum = th.zeros_like(self.X_sum[0])
        self.e_sum = None

        self.targets = None

        self.hooks = []
        # self.mode = Mode.COLLECTING

    def hook_collector(self, layer_idx, module, inputs):
        self.X[layer_idx] = inputs[0]
        # if layer_idx == self.T-1:
        #     self.X[self.T] = output[0]
        return None
    
    def register_hooks(self):

        for layer_idx, layer in enumerate(self.model.model.layers):
            def hook_wrapper(layer_idx):
                def hook(module, inputs):
                    return self.hook_collector(layer_idx, module, inputs)

                return hook

            self.hooks.append(
                layer.register_forward_pre_hook(
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
    
    def collect_data_batch(self, prompts, num_samples, filename, num_tokens=1, batch_size=50):
        # self.mode = Mode.COLLECTING
        X_sum = None
        count_sum = None

        samples = random.sample(prompts, num_samples)
        for i in range(0,len(samples), batch_size):
            # print(f"Processing batch {i}")
            sample = samples[i:i+batch_size]
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
            self.X = th.zeros(self.T, B, L, hidden_states.size(-1), device=self.device)

            # with self:
            #     self.model.generate(input_ids=input_ids,
            #             attention_mask=attention_mask,
            #             max_new_tokens=num_tokens,
            #             return_dict_in_generate=True,
            #             do_sample=False,
            #             # use_cache=False,
            #             pad_token_id=self.tokenizer.eos_token_id,
            #             )
            with self:
                _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )


            seq_len = self.X.size(2)
            mask = attention_mask
            if seq_len > mask.size(1):
                gen_len = seq_len - mask.size(1)
                mask = th.cat([mask, th.ones(mask.size(0), gen_len, device=mask.device)], dim=1)
            else:
                mask = mask[:, :seq_len]
            mask = mask.float()

            if X_sum is None or seq_len > X_sum.size(1):
                new_len = seq_len if X_sum is None else seq_len
                X_new = th.zeros((self.T, new_len, self.n), device=self.device)
                count_new = th.zeros((new_len,), device=self.device)
                if X_sum is not None:
                    X_new[:, :X_sum.size(1), :] = X_sum
                    count_new[:count_sum.size(0)] = count_sum
                X_sum = X_new
                count_sum = count_new

            masked_sum = (self.X * mask[None, :, :, None]).sum(dim=1)
            X_sum[:, :seq_len, :] += masked_sum
            count_sum[:seq_len] += mask.sum(dim=0)

        denom = count_sum.clamp_min(1.0)
        X_mean = X_sum / denom[None, :, None]

        print("X_mean dim: ", X_mean.size())

        tensor_dict = {
            "X": X_mean,
        } 

        with open(PICKLE_JAR + filename + ".pkl", "wb") as f:
            pickle.dump(tensor_dict, f)
        
        del self.X
        self.X = None
