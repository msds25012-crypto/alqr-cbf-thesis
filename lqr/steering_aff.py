import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import lqr_utils_aff as lqr
from functools import partial
from enum import Enum
import time
# import tqdm
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.decomposition import PCA
# import numpy as np

class Mode(Enum):
    COLLECTING = 0
    TRACKING = 1
    STEERING = 2
    SETPOINT = 3
    AFFINE = 4

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
        preserve_mem: bool = False,
        aff:bool = False,
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
        
        # for t in range(self.T):
        #     vt = self.E[t].unsqueeze(1) / th.norm(self.E[t])      # (n,1)
        #     self.Q[t] = vt @ vt.T
        #     print(f"freakaaaa: {t}")

        # vt = self.E[self.T].unsqueeze(1) / th.norm(self.E[self.T])      # (n,1)
        # self.Qf = vt @ vt.T
        
        if preserve_mem and not aff:
            self.K, self.P = lqr.time_varying_lqr_noB(self.A, self.Q, self.R, self.Qf) if A is not None else (None, None)
            print(f"Ps: {th.linalg.norm(self.P, ord=2, dim=(-2,-1))}")
            print(f"Ks: {th.linalg.norm(self.K, ord=2, dim=(-2,-1))}")
            print(f"Q: {th.linalg.norm(self.Q, ord=2, dim=(-2,-1))}")
            print(f"R: {th.linalg.norm(self.R, ord=2, dim=(-2,-1))}")
            print(f"A: {th.linalg.norm(self.A, ord=2, dim=(-2,-1))}")

            del self.A
            del self.Q
            del self.R
            del self.Qf
        elif not aff:
            self.B = th.eye(self.n).repeat(self.T, 1, 1).to(self.device) 
            self.K, _ = lqr.time_varying_lqr(self.A, self.B, self.Q, self.R, self.Qf) if A is not None else None, None
        
        if preserve_mem and aff:
            del self.Q
            del self.Qf

        self.X = None # to allocate at runtime
        self.U = th.zeros((self.T, self.n), device=self.device)

        self.X_cl = None

        self.betas = None
        self.E_unit = None
        self.setpoint_type = "linear"
        self.basis2 = None
        self.target_degree = None

        self.hooks = []
        self.mode = None
        # self.ALL_TOKENS = True
        self.ALL_TOKENS = False
        

        self.setpoint_signals = []
        self.iter = 0

        self.SIGNAL_COLLECT = False


    def hook_steering(self, layer_idx, module, input, output):
        # print(f"layer: {layer_idx}")
        
        # print(f"output.shape: {output.shape}")
    # if (layer_idx > 0):
        u_t = self.K[layer_idx]@(self.E[layer_idx]) # can be computed offline
        # print(u_t)

        # print(self.K[layer_idx-1])
        # print(u_t)
        # print(f"input shape: {input[0].shape}")
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

    # def get_angular_sp(x, target_degree, basis1, basis2):
    def get_angular_sp(self, x, layer_idx):
        basis1 = self.E[layer_idx]
        basis2 = self.basis2
        assert len(basis1.shape) == 1
        assert len(basis2.shape) == 1
        assert basis1.shape == basis2.shape

        n = basis1.shape[-1]

        # ensure bases are orthonormal
        u = basis1 / th.linalg.norm(basis1)
        v = basis2 - (basis2 @ u) * u
        v /= th.linalg.norm(v)

        theta = th.deg2rad(self.target_degree)
        cos_theta = th.cos(theta)
        sin_theta = th.sin(theta)

        P = th.outer(u, u) + th.outer(v, v)

        # rotate counter-clockwise
        R_theta = th.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=th.float, device=self.device)

        uv = th.column_stack([u, v])

        rotated_component = uv @ R_theta @ th.tensor([1, 0], dtype=th.float, device=self.device)
        Px = x @ P
        scale = th.linalg.norm(Px, axis=-1, keepdims=True)

        # result = x - Px + scale * rotated_component
        # return result

        e = -Px + scale * rotated_component

        return e

    def hook_setpoint_tracking(self, layer_idx, module, input, output):
        # assume E_normed is unit vector in direction of contrastive feature
        # print("HELP")
        if self.ALL_TOKENS:
            x = input[0]
            self.X[layer_idx] = x[-1,-1,:]
            # print("????")
            if self.setpoint_type == "linear":
                # print("LINEAR")
                v = self.E_unit[layer_idx]
                # print(f"x shape: {x.shape}")
                # print(f"v shape: {v.shape}")
                b_mat = self.betas[layer_idx] * th.ones([x.shape[0], x.shape[1]], device=self.device)
                probe_mat = x @ v.T
                # print(f"bmat shape: {b_mat.shape}")
                # print(f"probe mat shape: {probe_mat.shape}")
                alpha = b_mat - probe_mat
                v_mat = v.expand(x.shape[0], x.shape[1], -1)
                # print(f"v_mat shape: {v_mat.shape}")
                e = alpha.unsqueeze(-1) * v_mat
                # print(f"e shape: {e.shape}")
            elif self.setpoint_type == "angular":
                # print("DOING THE THING")
                e = self.get_angular_sp(x, layer_idx)
            else:
                raise ValueError("Unsupported setpoint type")

            u_t = e @ self.K[layer_idx].T
            self.U[layer_idx] = u_t[-1,-1]

            if not th.isfinite(u_t).all():
                raise RuntimeError("u_t contains NaN or Inf")
            if isinstance(output,tuple):
                # print(f"tuple output: {output}")
                # print(f"tuple wtf output: {output[0].shape}")
                output[0][...] = output[0] + u_t
            else: 
                output = output + u_t
            return output

        else:
            x = input[0][:,-1,:]
            self.X[layer_idx] = x[-1,:]

            if self.setpoint_type == "linear":
                v = self.E_unit[layer_idx]
                alpha = th.tensor([self.betas[layer_idx] for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
                e = alpha.squeeze(0).T @ v.unsqueeze(0)
            elif self.setpoint_type == "angular":
                # print("DOING THE THING")
                e = self.get_angular_sp(x, layer_idx)
            else:
                raise ValueError("Unsupported setpoint type")
            u_t = th.bmm(self.K[layer_idx].unsqueeze(0), th.transpose(e.unsqueeze(0),-2,-1)).squeeze(0).T
            self.U[layer_idx] = u_t[-1]

            # print("linear")
            if not th.isfinite(u_t).all():
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


    def hook_affine(self, layer_idx, module, input, output):
        # print("In Affine")
        x = input[0][:,-1,:]
        print(f"x norm: {th.norm(x)}")
        print(f"k norm: {th.norm(self.k[layer_idx])}")
        self.X[layer_idx] = x[-1,:]
        print(f"signal in hook ({layer_idx}): {self.E_unit[layer_idx]@x.T}")

        u_t = -(self.K[layer_idx] @ x.T).T - self.k[layer_idx]
        # u_t = -0*self.k[layer_idx]
        Knorm = th.linalg.norm(self.K[layer_idx], ord=2).item() 
        print(f"K norm: {Knorm}")
        print(f"u norm: {th.norm(u_t)}")

        # u_t = 0*u_t

        # if not th.all(self.k == 0):
            # print("k is not all 0")

        # if not th.all(self.K == 0):
            # print("K is not all 0")
        if not th.isfinite(u_t).all():
            raise RuntimeError("u_t contains NaN or Inf")
        if isinstance(output,tuple):
            output[0][...,-1,:] = output[0][...,-1,:] + u_t
        else: 
            output[...,-1,:] = output[...,-1,:] + u_t
        return output
        
    def register_affine_tracking_hooks(self):
        """Register the hooks."""

        for layer_idx, layer in enumerate(self.model.model.layers):
            def hook_wrapper(layer_idx):
                def hook(module, input, output):
                    return self.hook_affine(layer_idx, module, input, output)

                return hook

            self.hooks.append(
                layer.register_forward_hook(
                    hook_wrapper(layer_idx)
                )
            )

    def hook_get_sp_signal(self, layer_idx, module, input, output):
        x = input[0][:,-1,:]
        v = self.E_unit[layer_idx]
        raw_signal = th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        nm = th.norm(self.E[layer_idx])
        # print(nm)
        # signal = raw_signal / nm
        signal = raw_signal
        self.setpoint_signals.append(th.mean(signal).item())

        if layer_idx == self.T-1:
            if isinstance(output,tuple):
                x = output[0][...,-1,:]
            else: 
                x = output[...,-1,:]
            # x = input[0][:,-1,:]
            if self.mode != None:
                alpha = th.tensor([self.betas[layer_idx] for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
                e = alpha.squeeze(0).T @ v.unsqueeze(0)
                u_t = th.bmm(self.K[layer_idx].unsqueeze(0), th.transpose(e.unsqueeze(0),-2,-1)).squeeze(0).T
                x = x + u_t
                # print("here")
                
            v = self.E_unit[layer_idx+1]
            # v = self.E_unit[-2]
            raw_signal = th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
            nm = th.norm(self.E[layer_idx+1])
            # print(nm)
            # signal = raw_signal / nm
            signal = raw_signal
            self.setpoint_signals.append(th.mean(signal).item())
        return output

    def register_setpoint_signal_hooks(self):
        """Register the hooks."""

        for layer_idx, layer in enumerate(self.model.model.layers):
            def hook_wrapper(layer_idx):
                def hook(module, input, output):
                    return self.hook_get_sp_signal(layer_idx, module, input, output)

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
        elif self.mode == Mode.SETPOINT:
            self.register_setpoint_tracking_hooks()
        elif self.mode == Mode.AFFINE:
            self.register_affine_tracking_hooks()
        else:
            print("generating with no steering applied")

        if self.SIGNAL_COLLECT:
            self.register_setpoint_signal_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def track_setpoint(self, prompt, max_new_tokens, lmbda=1, do_sample=False, temp=1):
        self.mode = Mode.SETPOINT
        self.setpoint_type = "linear"
        self.SIGNAL_COLLECT = True
        self.setpoint_signals = []
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        self.X = th.zeros((self.T+1, self.n)).to(self.device)

        self.E_unit = th.zeros_like(self.E)
        self.betas = [0 for i in range(self.T+1)]
        for i, e in enumerate(self.E):
            # print(f"e: {e}")
            nrm = th.linalg.norm(e)
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
                use_cache=False,
                pad_token_id=self.tokenizer.eos_token_id,
                # **model_generation_kwargs, #
            )

        # output_str = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        output_str = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        return output_str
        

    def track_affine(self, prompt, z_ref, max_new_tokens=50, lmbda=1, do_sample=False, temp=1):
        self.mode = Mode.AFFINE
        self.SIGNAL_COLLECT = True
        self.setpoint_signals = []
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        self.X = th.zeros((self.T+1, self.n)).to(self.device)

        self.E_unit = th.zeros_like(self.E)
        self.betas = [0 for i in range(self.T+1)]
        for i, e in enumerate(self.E):
            # print(f"e: {e}")
            nrm = th.linalg.norm(e)
            self.E_unit[i] = e / nrm
            self.betas[i] = lmbda * nrm

        self.K, self.k, _, _ = lqr.compute_affine_output_lqr(self.A, self.E_unit, self.betas, self.R, z_ref)

        print(f"K shape : {self.K.shape}")
        print(f"k shape : {self.k.shape}")


        # print(th.mean(th.norm(self.k, dim=-1), dim=0))

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
                use_cache=False,
                pad_token_id=self.tokenizer.eos_token_id,
                # **model_generation_kwargs, #
            )

        # output_str = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        output_str = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        return output_str

    def track_angular_setpoint(self, prompt, max_new_tokens, target_degree, lmbda=1, do_sample=False, temp=0.7):
        self.mode = Mode.SETPOINT
        self.setpoint_type = "angular"
        self.target_degree = th.tensor(target_degree)

        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        self.X = th.zeros((self.T+1, self.n)).to(self.device)

        refusal_dirs = self.E.cpu()
        pca_model = PCA().fit(refusal_dirs)

        components = pca_model.components_
        self.basis2 = th.tensor(components[0].copy(), dtype=th.float, device=self.device)

        with self:
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

        # output_str = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        output_str = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        return output_str
