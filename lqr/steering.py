import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import lqr_utils_seq as lqr
from functools import partial
from enum import Enum
import time

class Mode(Enum):
    COLLECTING = 0
    TRACKING = 1
    STEERING = 2

class LQRSteering:
    '''
    Contrastive method currently assuming precomputed:
        - jacobians (A)
        - contrastive vectors
    '''


    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        q: float = 10,
        r: float = 10,
        qf: float = 1,
        A: th.Tensor = None,
        contrastive_vecs: th.Tensor = None,
    ):
        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer
        self.A = A
        self.E = contrastive_vecs
        self.contrastive = False

        self.T = len(model.model.layers)
        self.n = model.model.embed_tokens.embedding_dim
        self.m = self.n


        self.Q = th.eye(self.n).unsqueeze(0).repeat(self.T, 1, 1).to(self.device) * q
        self.R = th.eye(self.n).unsqueeze(0).repeat(self.T, 1, 1).to(self.device) * r
        self.Qf = th.eye(self.n).to(self.device) * qf
        
        
        self.B = th.eye(self.n).repeat(self.T, 1, 1).to(self.device) 
        self.K = lqr.time_varying_lqr(self.A, self.B, self.Q, self.R, self.Qf) if A is not None else None
        
        self.X = None # to allocate at runtime
        self.U = th.zeros((self.T, self.n), device=self.device)

        self.hooks = []
        self.mode = Mode.COLLECTING

        self.iter = 0


    def hook_steering(self, layer_idx, module, input, output):

        u_t = self.K[layer_idx]@(self.E[layer_idx]) # can be computed offline
        self.U[layer_idx] = u_t
        self.X[layer_idx] = input[0][0,-1,:]

        # output[0][:,-1,:] = output[0][:,-1,:] + u_t # 4.40
        output[0][...,-1,:] = output[0][...,-1,:] + u_t # new

        if (layer_idx == self.T-1):
            self.X[self.T] = output[0][...,-1,:] + u_t
        return output
        

    def register_steering_hooks(self):
        """Register the hooks."""

        for layer_idx, layer in enumerate(self.model.model.layers):
            def hook_wrapper(layer_idx):
                def hook(module, input, output):
                    return self.hook_steering(layer_idx, module, input, output)

                return hook

            self.hooks.append(
                layer.register_forward_hook(
                    hook_wrapper(layer_idx)
                )
            )

    def hook_collector(self, layer_idx, module, input, output):
        if self.iter == 0:
            # print(f"iter in collector: {self.iter}")
            self.X[self.iter][layer_idx] = input[0]
            if layer_idx == self.T-1:
                self.X[self.iter][self.T] = output[0]
                # self.X[self.iter][self.T] = output[0][...,-1,:]
                self.iter = self.iter + 1

        else: # for everything other than the first layer, only collect last token position 
            self.X[self.iter][layer_idx] = input[0][0,-1,:]
            if layer_idx == self.T-1:
                self.X[self.iter][self.T] = output[0][...,-1,:]
                self.iter = self.iter + 1

        return output
    
    def register_collection_hooks(self):
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


    def hook_tracking(self, layer_idx, module, input, output):
        x_t = input[0][0,-1,:]
        diff = x_t - self.X[self.iter][layer_idx,-1,:]
        u_t = -self.K[layer_idx]@(diff)

        self.U[layer_idx] = u_t

        # output[0][:,-1,:] = output[0][:,-1,:] + u_t # 4.40
        output[0][...,-1,:] = output[0][...,-1,:] + u_t # new

        if (layer_idx == self.T-1):
            self.iter = self.iter + 1
        return output
    
    def register_tracking_hooks(self):
        """Register the hooks."""

        for layer_idx, layer in enumerate(self.model.model.layers):
            def hook_wrapper(layer_idx):
                def hook(module, input, output):
                    return self.hook_tracking(layer_idx, module, input, output)

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
            self.register_collection_hooks()
        elif self.mode == Mode.STEERING:
            self.register_steering_hooks()
        else: 
            self.register_tracking_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def evaluate(self, prompt, max_new_tokens, do_sample=False, temp=0.7):
        self.mode = Mode.STEERING
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        self.X = th.zeros((self.T+1, self.n)).to(self.device)

        with self: # I think just an elegant way to trigger __enter__ and __exit__ to manage hooks
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                do_sample=do_sample,
                temperature=temp,
                use_cache=False,
                pad_token_id=self.tokenizer.eos_token_id,
                # **model_generation_kwargs, #
            )

        output_str = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        return output_str

    def track_tokens(self, nom_prompt, prompt, k=1):
        self.mode = Mode.COLLECTING

        start_time = time.perf_counter()

        nom_inputs = self.tokenizer(nom_prompt, return_tensors="pt").to(self.device)
        nom_input_ids = nom_inputs["input_ids"]
        nom_attention_mask = nom_inputs["attention_mask"].float()

        embedding_layer = self.model.get_input_embeddings()
        hidden_states = embedding_layer(nom_input_ids)
        self.X = [th.zeros_like(hidden_states).repeat(self.T+1, 1, 1).to(self.device)]
        
        sublist = [th.zeros_like(hidden_states[...,-1,:]).repeat(self.T+1, 1, 1).to(self.device) for i in range(k-1)]
        self.X = self.X + sublist
        self.iter = 0

        print(f"len X: {len(self.X)}")
        print(f"X[0] shape: {self.X[0].shape}")
        # self.X = th.zeros((self.T+1, self.n)).to(self.device)
        with self:
            nom_output = self.model.generate(
                input_ids=nom_input_ids,
                attention_mask=nom_attention_mask,
                max_new_tokens=k,
                return_dict_in_generate=True,
                do_sample=False,
                use_cache=False,
                pad_token_id=self.tokenizer.eos_token_id,
                # **model_generation_kwargs, #
            )

        end_nom_time = time.perf_counter()

        print(f"Nom rollout time: {end_nom_time - start_time}")
        # print(f"X[0] shape after nom: {self.X[0].shape}")
        

        nom_output_str = self.tokenizer.decode(nom_output.sequences[0], skip_special_tokens=True)
        print(f"nom_output: {nom_output_str}")

        if self.A is None:
            batch_size, seq_len = nom_input_ids.shape
            position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)

            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
            wrapped_tfs_temp = [partial(lqr.new_llama_block_wrapper, tf, nom_attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
            tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
            self.A, _ = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X[0]) # linearizing about first subtrajectory

        lin_time = time.perf_counter()
        print(f"Linearize time: {lin_time - end_nom_time}")

        print(self.A.device)
        self.K = lqr.time_varying_lqr(self.A, self.B, self.Q, self.R, self.Qf)

        self.mode = Mode.TRACKING
        self.iter = 0

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        with self: # I think just an elegant way to trigger __enter__ and __exit__ to manage hooks
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=k,
                return_dict_in_generate=True,
                do_sample=False,
                use_cache=False,
                pad_token_id=self.tokenizer.eos_token_id,
                # **model_generation_kwargs, #
            )

        output_str = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        print(f"steered output: {output_str}")
        
        end_time = time.perf_counter()

        print(f"Tracking time: {end_time - lin_time}")
        print(f"Total time: {end_time - start_time}")


    def track_traj(self, X_nom, prompt, k=1, sample=True, temp=0.7):
        self.mode = Mode.COLLECTING

        start_time = time.perf_counter()
        self.X = [X_nom for i in range(k)]        


        
        self.mode = Mode.TRACKING
        self.iter = 0

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if self.A is None:
            batch_size, seq_len = input_ids.shape
            position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)

            embedding_layer = self.model.get_input_embeddings()
            hidden_states = embedding_layer(input_ids)
            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
            wrapped_tfs_temp = [partial(lqr.new_llama_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
            tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
            self.A, _ = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X[0]) # linearizing about first subtrajectory
            self.K = lqr.time_varying_lqr(self.A, self.B, self.Q, self.R, self.Qf)


        with self: # I think just an elegant way to trigger __enter__ and __exit__ to manage hooks
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=k,
                return_dict_in_generate=True,
                do_sample=sample,
                temperature=temp if sample else None,
                use_cache=False,
                pad_token_id=self.tokenizer.eos_token_id,
                # **model_generation_kwargs, #
            )

        output_str = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        # print(f"steered output: {output_str}")
        
        end_time = time.perf_counter()

        # print(f"Total time: {end_time - start_time}")
        return output_str