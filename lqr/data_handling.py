import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import lqr_utils_seq as lqr
from functools import partial
from datasets import load_dataset
import random
import pickle
import time
from steering import Mode
import yaml

with open('config/config.yaml', 'r') as f:
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
        self.mode = Mode.COLLECTING

        self.Kp = None
        self.Ki = None
        self.Kd = None

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

    def hook_PID(self, layer_idx, module, input, output):
        x = input[0][:,-1,:]

        # if layer_idx == 6:
        e = self.targets[layer_idx] - x
        self.e_sum += e.squeeze(0)
        # print(f"alpha: {alpha/th.norm(self.E[layer_idx])}")

        u_t = self.Kp*e + self.Ki*self.e_sum + self.Kd*(e - self.e_prev)
        # print(f"x shape: {x.shape}")
        self.e_prev = e
        self.X[layer_idx] = x[-1,:]
        self.U[layer_idx] = u_t[-1]

        if isinstance(output,tuple):
            output[0][...,-1,:] = output[0][...,-1,:] + u_t
        else: 
            output[...,-1,:] = output[...,-1,:] + u_t
        return output

    def register_PID_hooks(self):
        """Register the hooks."""

        for layer_idx, layer in enumerate(self.model.model.layers):
            def hook_wrapper(layer_idx):
                def hook(module, input, output):
                    return self.hook_PID(layer_idx, module, input, output)

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
        if self.mode == Mode.COLLECTING:
            self.register_hooks()
        elif self.mode == Mode.STEERING:
            self.register_PID_hooks()
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
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)

        position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

        wrapped_tfs_temp = [partial(lqr.new_llama_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
        tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
        print(f"Xshape: {self.X.shape}")
        A, _ = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X)


        self.A_sum = self.A_sum + A

    def collect_data(self, num_samples, num_tokens, trait, filename, lb=0, ub=0.1, split="train", collect_A = False):#, num_A = 1):
        self.mode = Mode.COLLECTING
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
                position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)

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

        with open(PICKLE_JAR + filename + ".pkl", "wb") as f:
            pickle.dump(tensor_dict, f)

    
    def collect_data_batch(self, prompts, num_samples, filename, num_tokens=1):
        self.mode = Mode.COLLECTING
        # A_iter = num_A
        # self.X_sum = th.zeros((self.T+1, self.n,)).to(self.device)
        # self.X_mean = th.zeros((self.T+1, self.n,)).to(self.device)

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
            
            X_mean = th.mean(self.X[:,:,-1,:], dim = 1)

        total = num_samples*num_tokens
        print(f"total: {total}")

        tensor_dict = {
            "X": X_mean,
        } 

        with open(PICKLE_JAR + filename + ".pkl", "wb") as f:
            pickle.dump(tensor_dict, f)
        
        del self.X
        self.X = None



    def collect_jacobians(self, prompts, num_samples, filename, num_tokens=1, max_ctx=24):
        self.mode = Mode.COLLECTING
        self.A_sum = th.zeros((self.T, self.n, self.n,)).to(self.device)

        sample = random.sample(prompts, num_samples)
        iter = 1
        for prompt in sample:
            print(f"iter: {iter}")
            iter += 1
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_ctx).to(self.device)

            # print(f"inputs: {inputs}")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"].float()
            embedding_layer = self.model.get_input_embeddings()
            hidden_states = embedding_layer(input_ids)
            self.X = th.zeros_like(hidden_states).repeat(self.T+1, 1, 1).to(self.device)

            with th.no_grad():
                with self:
                    self.model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=num_tokens,
                            return_dict_in_generate=True,
                            do_sample=False,
                            use_cache=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                            )
            
            # self.X_sum = self.X_sum + self.X[:,-1,:]


            # and A_iter > 0:
            batch_size, seq_len = input_ids.shape
            position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)

            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

            wrapped_tfs_temp = [partial(lqr.new_llama_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
            tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
            # A, _ = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X)
            A = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X)
            self.A_sum = self.A_sum + A
                # A_iter -= 1
            del A
            A = None
            del self.X
            self.X = None


        total = num_samples*num_tokens
        print(f"total: {total}")
        tensor_dict = {
            "A": self.A_sum / total,
        } 

        with open(PICKLE_JAR + filename + ".pkl", "wb") as f:
            pickle.dump(tensor_dict, f)

        del self.A_sum
        self.A_sum = None


    def collect_sequentialPID(self, prompts, num_samples, filename, target_acts, kp=0.5, ki=0.01, kd=0.01, num_tokens=1):
        self.mode = Mode.STEERING
        
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd

        self.targets = target_acts
        sample = random.sample(prompts, num_samples)
        

        inputs = self.tokenizer(
            sample, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        B,L = input_ids.shape
        self.e_sum = th.zeros((input_ids.shape[0], target_acts[0].shape[0]), device=self.device)
        self.e_prev = th.zeros((input_ids.shape[0], target_acts[0].shape[0]), device=self.device)

        attention_mask = inputs["attention_mask"].float()
        embedding_layer = self.model.get_input_embeddings()
        hidden_states = embedding_layer(input_ids)
        self.X = th.zeros(self.T+1, B, L, hidden_states.size(-1), device=self.device)

        with th.no_grad():
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
            "X_contr": self.X_mean,
        } 

        with open("../../scratch/" + filename + ".pkl", "wb") as f:
            pickle.dump(tensor_dict, f)
