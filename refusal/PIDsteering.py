import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import lqr_utils_seq as lqr
from functools import partial
from enum import Enum
import time
from tqdm import tqdm

class Mode(Enum):
    STEERING = 0
    SETPOINT = 1
    ACTADD = 2
th.autograd.set_detect_anomaly(True)
class PIDSteering:
    '''
    Contrastive method currently assuming precomputed:
        - contrastive vectors
    '''


    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        kp: float = 10,
        ki: float = 10,
        kd: float = 1,
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


        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        
        
        
        self.X = None # to allocate at runtime
        self.e_prev = None
        self.U = th.zeros((self.T, self.n), device=self.device)
        self.e_sum = th.zeros_like(self.E[0])

        self.betas = None
        self.E_unit = th.zeros_like(self.E)

        self.hooks = []
        self.mode = Mode.SETPOINT
        

        self.iter = 0

        self.layer_inds = []


    def hook_steering(self, layer_idx, module, input, output):
        # u_t = self.K[layer_idx]@(self.E[layer_idx]) # can be computed offline
        e = self.E[layer_idx]
        self.e_sum += e
        u_t = self.Kp*e + self.Ki*self.e_sum + self.Kd*(e - self.e_prev)
        self.e_prev = e

        self.U[layer_idx] = u_t[-1,:]
        self.X[layer_idx] = input[0][0,-1,:]

        # output[0][:,-1,:] = output[0][:,-1,:] + u_t # 4.40
        # output[0][...,-1,:] = output[0][...,-1,:] + u_t # new

        if isinstance(output,tuple):
            output[0][...,-1,:] = output[0][...,-1,:] + u_t
        else: 
            output[...,-1,:] = output[...,-1,:] + u_t
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
        # print("Collecting...")
        # self.X[self.iter][layer_idx] = input[0][0,-1,:]
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
        # u_t = -(diff)
        self.U[layer_idx] = u_t

        output[0][...,-1,:] = output[0][...,-1,:] + u_t # new

        if (layer_idx == self.T-1):
            # self.X[self.iter][self.T] = output[0][...,-1,:] + u_t
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

    def hook_setpoint_tracking(self, layer_idx, module, input, output):
        x = input[0][:,-1,:]

        # print(f"in hook layer: {layer_idx}")

        # if layer_idx == 6:
        v = self.E_unit[layer_idx].to(x.dtype)
        alpha = th.tensor([self.betas[layer_idx] for i in range(x.shape[0])], device=self.device, dtype=x.dtype) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        # print(f"alpha shape: {alpha.shape}")
        e = alpha.squeeze(0).T @ v.unsqueeze(0)
        self.e_sum += e
        # print(f"alpha: {alpha/th.norm(self.E[layer_idx])}")

        if layer_idx % 10 == 0:
            self.e_sum = self.e_sum * 0        

        u_t = self.Kp*e + self.Ki*self.e_sum + self.Kd*(e - self.e_prev)
        # print(f"x shape: {x.shape}")
        self.e_prev = e
        self.X[layer_idx] = x[-1,:]
        self.U[layer_idx] = u_t[-1]


        if not th.isfinite(u_t).all():
            print(f"layer index: {layer_idx}")
            print(f"e: {e}")
            print(f"e_prev: {self.e_prev}")
            print(f"e_prev: {self.e_sum}")
            raise RuntimeError("u_t contains NaN or Inf")
        if isinstance(output,tuple):
            output[0][...,-1,:] = output[0][...,-1,:] + u_t
        else: 
            output[...,-1,:] = output[...,-1,:] + u_t
            

        return output

    def register_setpoint_tracking_hooks(self):
        """Register the hooks."""

        for layer_idx, layer in enumerate(self.model.model.layers):
            def hook_wrapper(layer_idx):
                def hook(module, input, output):
                    return self.hook_setpoint_tracking(layer_idx, module, input, output)

                return hook

            self.hooks.append(
                layer.register_forward_hook(
                    hook_wrapper(layer_idx)
                )
            )
        # def hook_wrapper(layer_idx):
        #     def hook(module, input, output):
        #         return self.hook_setpoint_tracking(layer_idx, module, input, output)

        #     return hook
        # self.hooks.append(
        #         self.model.model.layers[-1].register_forward_hook(
        #             hook_wrapper(self.T-1)
        #         )
        #     )

    def register_actadd_setpoint_tracking_hooks(self):
        """Register the hooks."""

        for layer_idx in self.layer_inds:
            layer = self.model.model.layers[layer_idx]
            def hook_wrapper(layer_idx):
                def hook(module, input, output):
                    return self.hook_setpoint_tracking(layer_idx, module, input, output)

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
        if self.mode == Mode.STEERING:
            self.register_steering_hooks()
        elif self.mode == Mode.SETPOINT:
            self.register_setpoint_tracking_hooks()
        elif self.mode == Mode.ACTADD:
            self.register_actadd_setpoint_tracking_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def evaluate(self, prompt, max_new_tokens, do_sample=False, temp=0.7):
        self.mode = Mode.STEERING
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        self.X = th.zeros((self.T+1, self.n)).to(self.device)

        # print(f"ids: {input_ids.device}")
        # print(f"ids shape: {input_ids.shape}")
        # print(f"mask: {attention_mask.device}")
        self.e_sum = th.zeros((input_ids.shape[0], self.E[0].shape[0]), device=self.device)
        self.e_prev = th.zeros((input_ids.shape[0], self.E[0].shape[0]), device=self.device)

        with th.no_grad():
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

        output_str = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        return output_str



    def track_setpoint_actadd(self, prompt, max_new_tokens, lmbda=1, layer_inds=[5], do_sample=False, temp=1, return_tokens=False):
        self.mode = Mode.ACTADD
        self.Kd = 0
        self.Ki = 0
        self.layer_inds = layer_inds

        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # print("before tokenize")
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        self.X = th.zeros((self.T+1, self.n)).to(self.device)
        # print(f"E shape: {self.E[0].shape}")
        self.e_sum = th.zeros((input_ids.shape[0], self.E[0].shape[0]), device=self.device)
        self.e_prev = th.zeros((input_ids.shape[0], self.E[0].shape[0]), device=self.device)

        # print(f"e sum shape: {self.e_sum.shape}")
        # print(f"e prev shape: {self.e_prev.shape}")

        self.betas = [0 for i in range(self.T+1)]
        for i, e in enumerate(self.E):
            # print(f"e: {e}")
            nrm = th.linalg.norm(e)
            if nrm < 1e-6:
                self.E_unit[i] = self.E_unit[i]*0
                # raise ValueError("norm is 0")
            else:
                self.E_unit[i] = e / nrm

            self.betas[i] = lmbda * nrm

        with self:
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                do_sample=do_sample,
                top_p=0.3,
                repetition_penalty=1.2 if do_sample else None,
                temperature=temp,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                # **model_generation_kwargs, #
            )

        # output_str = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        if return_tokens:
            return output.sequences

        output_str = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        return output_str


    def track_setpoint(self, prompt, max_new_tokens, lmbda=1, do_sample=False, temp=1, return_tokens=False):
        self.mode = Mode.SETPOINT

        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # print("before tokenize")
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        self.X = th.zeros((self.T+1, self.n)).to(self.device)
        # print(f"E shape: {self.E[0].shape}")
        self.e_sum = th.zeros((input_ids.shape[0], self.E[0].shape[0]), device=self.device)
        self.e_prev = th.zeros((input_ids.shape[0], self.E[0].shape[0]), device=self.device)

        # print(f"e sum shape: {self.e_sum.shape}")
        # print(f"e prev shape: {self.e_prev.shape}")

        self.betas = [0 for i in range(self.T+1)]
        for i, e in enumerate(self.E):
            # print(f"e: {e}")
            nrm = th.linalg.norm(e)
            if nrm < 1e-6:
                self.E_unit[i] = self.E_unit[i]*0
                # raise ValueError("norm is 0")
            else:
                self.E_unit[i] = e / nrm

            self.betas[i] = lmbda * nrm

        with self:
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                do_sample=do_sample,
                top_p=0.3,
                repetition_penalty=1.2 if do_sample else None,
                temperature=temp,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                # **model_generation_kwargs, #
            )

        # output_str = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        if return_tokens:
            return output.sequences

        output_str = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        return output_str
        
    def dumfuck(self, X_neg, X_pos):
        self.E = th.zeros_like(X_neg)
        e = th.zeros_like(X_neg[0])
        for i, x in enumerate(X_neg):
            e = X_pos[i] - x
        return e

        

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

        # print(f"len X: {len(self.X)}")
        # print(f"X[0] shape: {self.X[0].shape}")
        # self.X = th.zeros((self.T+1, self.n)).to(self.device)
        with self:
            with th.no_grad():
                nom_output = self.model.generate(
                    input_ids=nom_input_ids,
                    attention_mask=nom_attention_mask,
                    max_new_tokens=k,
                    return_dict_in_generate=True,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    # **model_generation_kwargs, #
                )

        end_nom_time = time.perf_counter()

        print(f"Nom rollout time: {end_nom_time - start_time}")
        # print(f"X[0] shape after nom: {self.X[0].shape}")
        

        nom_output_str = self.tokenizer.decode(nom_output.sequences[0], skip_special_tokens=True)
        print(f"nom_output: {nom_output_str}<END>")


        
        

        if self.A is None:
            batch_size, seq_len = nom_input_ids.shape
            position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)

            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
            wrapped_tfs_temp = [partial(lqr.tf_block_wrapper, tf, nom_attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
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
            with th.no_grad():
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
        # print(f"steered output: {output_str}")
        
        end_time = time.perf_counter()

        print(f"Tracking time: {end_time - lin_time}")
        print(f"Total time: {end_time - start_time}")
        return output_str


    def track_traj(self, X_nom, prompt, k=1, do_sample=False, temp=0.7):
        self.mode = Mode.COLLECTING

        start_time = time.perf_counter()
        self.X = [X_nom for i in range(k)]        


        
        self.mode = Mode.TRACKING
        self.iter = 0

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if self.A is None:
            embedding_layer = self.model.get_input_embeddings()
            hidden_states = embedding_layer(input_ids)
            batch_size, seq_len = input_ids.shape
            position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)

            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
            wrapped_tfs_temp = [partial(lqr.tf_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
            tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
            self.A, _ = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X[0]) # linearizing about first subtrajectory
            self.K = lqr.time_varying_lqr(self.A, self.B, self.Q, self.R, self.Qf)


        with self: # I think just an elegant way to trigger __enter__ and __exit__ to manage hooks
            with th.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=k,
                    return_dict_in_generate=True,
                    do_sample=do_sample,
                    temperature=temp if do_sample else None,
                    use_cache=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    # **model_generation_kwargs, #
                )

        output_str = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        # print(f"steered output: {output_str}")
        
        end_time = time.perf_counter()

        # print(f"Total time: {end_time - start_time}")
        return output_str

    def complete_rollout(self, prompt, k=1):
        self.mode = Mode.COLLECTING
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].float()
        embedding_layer = self.model.get_input_embeddings()
        hidden_states = embedding_layer(input_ids)
        self.X = [th.zeros_like(hidden_states).repeat(self.T+1, 1, 1).to(self.device)]
        
        sublist = [th.zeros_like(hidden_states[...,-1,:]).repeat(self.T+1, 1, 1).to(self.device) for i in range(k-1)]
        self.X = self.X + sublist

        with self: # I think just an elegant way to trigger __enter__ and __exit__ to manage hooks
            with th.no_grad():
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

        batch_size, seq_len = input_ids.shape
        position_ids = th.arange(seq_len, dtype=th.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)

        position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
        wrapped_tfs_temp = [partial(lqr.tf_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
        tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
        
        self.A = [th.eye(self.n).unsqueeze(0).repeat(self.T, 1, 1).to(self.device) for i in range (k)]
        for i in range(k):
            self.A[i], _ = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X[i])
        
        return self.X, self.A, output

    def plot_unorms(self, figname):
        u_norms = th.linalg.norm(self.U, dim=1).cpu()
        contr_norms = th.linalg.norm(self.E, dim=1).cpu()

        layer_lbls = []
        for i in range(self.T):
            layer_lbls.append(f"{i+1}")
        import matplotlib.pyplot as plt

        bar_width = 0.35

        # Set the x-axis positions for the bars
        r1 = th.arange(len(layer_lbls))
        r2 = [x + bar_width for x in r1]

        # Create the bar plot
        plt.bar(r1, u_norms, color='skyblue', width=bar_width, label='LQR')
        plt.bar(r2, contr_norms[:-1], color='lightcoral', width=bar_width, label='Contrastive')

        # Add labels and title
        # plt.xlabel('')
        plt.ylabel('Avg. Norm')
        plt.title(f'Perturbation norms for final pass (PID: {self.Kp}, {self.Ki}, {self.Kd})?')
        plt.xticks([r + bar_width / 2 for r in r1], layer_lbls) # Center x-axis labels
        plt.legend()
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.savefig(figname + ".png")

    def compute_ppl(self, all_sentences, lmbda=1, prompts=None):

        BATCH_SZ = 1
        self.mode = Mode.SETPOINT
        self.X = th.zeros((self.T+1, self.n)).to(self.device)
        self.e_sum = th.zeros((100, self.E[0].shape[0]), device=self.device)
        self.e_prev = th.zeros((100, self.E[0].shape[0]), device=self.device)

        self.betas = [0 for i in range(self.T+1)]
        for i, e in enumerate(self.E):
            # print(f"e: {e}")
            nrm = th.linalg.norm(e)
            self.E_unit[i] = e / nrm
            self.betas[i] = lmbda * nrm
        max_generation_length = 26
        max_context_length = 128

        # Pre-tokenize all sentences once and move to GPU
        truncation = True
        self.tokenizer.padding_side = "right"
        tok_s = self.tokenizer(
            text=all_sentences,
            return_tensors="pt",
            truncation=truncation,
            padding=truncation,
            max_length=max_generation_length,
            add_special_tokens=(prompts is None),
        ).to(self.device)

        self.tokenizer.padding_side = "left"

        if prompts is not None:
            side = self.tokenizer.truncation_side
            self.tokenizer.truncation_side = "left"
            tok_p = self.tokenizer(
                text=prompts,
                return_tensors="pt",
                truncation=truncation,
                padding=True,
                add_special_tokens=True,
                max_length=max_context_length,
            ).to(self.device)
            self.tokenizer.truncation_side = side
            tok_all = {k: th.cat([tok_p[k], tok_s[k]], dim=-1) for k in tok_p.keys()}
            offset = tok_p["input_ids"].shape[-1]
        else:
            tok_all = tok_s
            offset = 1

        input_ids = tok_all["input_ids"]
        attention_mask = tok_all["attention_mask"]
        attention_mask_sum = tok_s["attention_mask"].sum(-1)

        # Buffers on GPU
        ppls = th.zeros(attention_mask.shape[0], device=self.device, dtype=th.float32)
        totals = th.zeros_like(ppls)

        # GPU-optimized for loop
        for ctx_len in tqdm(range(1, attention_mask_sum.max() - 1)):
            mask = ctx_len < attention_mask_sum
            _input_ids = input_ids[mask][:, : (offset + ctx_len)]
            _attention_mask = attention_mask[mask][:, : (offset + ctx_len)]

            with self:
                with th.no_grad():  # Disable gradient computation for speed & memory
                    logits = self.model(input_ids=_input_ids, attention_mask=_attention_mask, use_cache=False).logits

            # Compute cross-entropy loss on GPU
            loss = th.nn.functional.cross_entropy(
                logits[:, -1], 
                input_ids[mask][:, (offset + ctx_len)].reshape(-1), 
                reduction="none"
            )
            ppls[mask] += loss
            totals[mask] += 1

        # Final perplexities
        batch_ppl = th.exp(ppls / totals)
        overall_ppl = th.exp(ppls.sum() / totals.sum())
        # print("batch ppl:", batch_ppl)
        # print("overall ppl:", overall_ppl)

        return overall_ppl
