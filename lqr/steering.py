import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import lqr_utils_seq as lqr
from functools import partial
from enum import Enum
import time
# import tqdm
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.decomposition import PCA
from contextlib import contextmanager
import functools
from typing import Callable, List, Tuple
import torch.nn as nn
# import numpy as np

class Mode(Enum):
    COLLECTING = 0
    TRACKING = 1
    STEERING = 2
    SETPOINT = 3
    ANGULAR = 4
    MULTI = 5

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
        perserve_mem: bool = True,
    ):
        self.model = model
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.A = A
        self.E = contrastive_vecs
        self.contrastive = False

        self.T = len(model.model.layers)
        self.n = model.model.embed_tokens.embedding_dim
        self.m = self.n

        print(f"model device: {self.model.device}")
        self.Q = th.eye(self.n).unsqueeze(0).repeat(self.T, 1, 1).to(self.device) * q
        self.R = th.eye(self.n).unsqueeze(0).repeat(self.T, 1, 1).to(self.device) * r
        self.Qf = th.eye(self.n).to(self.device) * qf
        
        # self.Q = th.eye(self.n).to(self.device) * q
        # self.R = th.eye(self.n).to(self.device) * r
        # self.Qf = th.eye(self.n).to(self.device) * qf
        
        
        if perserve_mem:
            self.K = lqr.time_varying_lqr_noB_mem_efficient(self.A, self.Q, self.R, self.Qf) if A is not None else None
            del self.A
            del self.Q
            del self.R
            del self.Qf
        else:
            self.B = th.eye(self.n).repeat(self.T, 1, 1).to(self.device) 
            self.K = lqr.time_varying_lqr(self.A, self.B, self.Q, self.R, self.Qf) if A is not None else None
            # print(f"self.K[0]: {self.K[0]}")
        th.cuda.empty_cache()
        if (self.model.device == 'cpu'):
            self.model=self.model.to(self.device)
        print("survived")
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


    def hook_tracking(self, layer_idx, module, input, output):
        x_t = input[0][0,-1,:]

        if layer_idx == 0:
            self.X_cl[self.iter][layer_idx] = x_t

        diff = x_t - self.X[self.iter][layer_idx,-1,:]
        u_t = -self.K[layer_idx]@(diff)
        # u_t = -(diff)
        self.U[layer_idx] = u_t

        if isinstance(output,tuple):
            output[0][...,-1,:] = output[0][...,-1,:] + u_t
            if layer_idx == self.T-1:
                self.X_cl[self.iter][layer_idx+1] = output[0][...,-1,:]

        else: 
            output[...,-1,:] = output[...,-1,:] + u_t
            if layer_idx == self.T-1:
                self.X_cl[self.iter][layer_idx+1] = output[...,-1,:]

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

    # def get_angular_sp(x, target_degree, basis1, basis2):
    def get_angular_sp(self, x, layer_idx):
        # basis1 = self.E[layer_idx]
        # basis2 = self.basis2
        # assert len(basis1.shape) == 1
        # assert len(basis2.shape) == 1
        # assert basis1.shape == basis2.shape

        # n = basis1.shape[-1]

        # # ensure bases are orthonormal
        # u = basis1 / th.linalg.norm(basis1)
        # v = basis2 - (basis2 @ u) * u
        # v /= th.linalg.norm(v)

        # theta = th.deg2rad(self.target_degree)
        # cos_theta = th.cos(theta)
        # sin_theta = th.sin(theta)

        # P = th.outer(u, u) + th.outer(v, v)

        # # rotate counter-clockwise
        # R_theta = th.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=th.float, device=self.device)

        # uv = th.column_stack([u, v])

        # rotated_component = uv @ R_theta @ th.tensor([1, 0], dtype=th.float, device=self.device)
        # Px = x @ P
        # scale = th.linalg.norm(Px, axis=-1, keepdims=True)

        # # result = x - Px + scale * rotated_component
        # # return result

        # e = -Px + scale * rotated_component

        # return e
        first_direction = self.E[layer_idx]
        second_direction = self.basis2

        # Compute rotation
        device = first_direction.device
        theta_rad = th.tensor(self.target_degree.item() * th.pi / 180.0, device=device)

        if first_direction.norm() == 0:
            # print("goddamn motherfucker")
            return x

        # Orthonormalize directions
        b1 = first_direction / first_direction.norm()
        b2 = second_direction - (second_direction @ b1) * b1
        b2 = b2 / b2.norm()

        # Projection matrix
        proj_matrix = th.outer(b1, b1) + th.outer(b2, b2)

        # Rotation matrix
        cos_theta = th.cos(theta_rad)
        sin_theta = th.sin(theta_rad)
        rotation_matrix = th.stack(
            [th.stack([cos_theta, -sin_theta]), th.stack([sin_theta, cos_theta])]
        )

        # Steering vector
        unit_vector = th.tensor([1.0, 0.0], device=device)
        rotated_2d = rotation_matrix @ unit_vector
        steering_vector = rotated_2d[0] * b1 + rotated_2d[1] * b2

        _cache = {}

        # def steering_hook(_module, _input, output):
            # device = output.device
        dtype = self.model.dtype
        cache_key = (device, dtype)

        if cache_key not in _cache:
            _cache[cache_key] = (
                proj_matrix.to(device=device, dtype=dtype),
                steering_vector.to(device=device, dtype=dtype),
                first_direction.to(device=device, dtype=dtype),
            )

        proj, steer, first_dir = _cache[cache_key]

        projected = x @ proj
        scale = projected.norm(dim=-1, keepdim=True)

            # if adaptive_mode == 0:
            #     steered = x - projected + scale * steer
            #     return steered
            # elif adaptive_mode == 1:

        # # adaptive    
        # proj_to_first = x @ first_dir
        # print(f"projected: {projected}")
        # print(f"proj_to_first: {proj_to_first}")
        # print(f"scale: {scale}")
        # print(f"steer: {steer}")


        # mask = (proj_to_first > 0).unsqueeze(-1)
        # print(f"mask: {mask}")

        # steered = x - projected + scale * steer
        # return th.where(mask, steered, x)

        # # non adaptive

        
        # print(f"projected: {projected}")
        # print(f"scale: {scale}")
        # print(f"steer: {steer}")

        steered = x - projected + scale * steer
        # print(f"steered: {steered}")
        # mask = (th.isnan(steered))
        # return th.where(mask, steered, x)
        if th.isnan(projected).any().item():
            raise ValueError(f"ASLKASJIDASKJHASDKJHASKJH: {layer_idx}")
        return steered

    def hook_setpoint_tracking(self, layer_idx, module, input, output):
        # assume E_normed is unit vector in direction of contrastive feature

        if self.ALL_TOKENS:
            x = input[0]
            self.X[layer_idx] = x[-1,-1,:]
            if self.setpoint_type == "linear":
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
                e = self.get_angular_sp(x, layer_idx) - x
                # e = self.get_angular_sp(x, layer_idx)
            else:
                raise ValueError("Unsupported setpoint type")

            u_t = e @ self.K[layer_idx].T
            
            self.U[layer_idx] = u_t[-1,-1]

            # if isinstance(output,tuple):
            #     # print(f"tuple output: {output}")
            #     # print(f"tuple wtf output: {output[0].shape}")
            #     output[0][...] = output[0] + u_t
            # else: 
            #     output = output + u_t


            if isinstance(output,tuple):
                # print(f"tuple output: {output}")
                # print(f"tuple wtf output: {output[0].shape}")
                # if self.setpoint_type == "angular":
                #     # print("DOING THE THING")
                #     print(f"this is bad")
                #     output[0][...] = self.get_angular_sp(output[0], layer_idx+1)
                # else:
                output[0][...] = output[0] + u_t
            else: 
                # print("other condition")
                # if self.setpoint_type == "angular":
                #     # print("DOING THE THING buddy")
                #     print("this is also bad")
                #     output = self.get_angular_sp(output, layer_idx+1)
                # else:
                output = output + u_t
            return output

        else:
            x = input[0][:,-1,:]
            self.X[layer_idx] = x[-1,:]

            if self.setpoint_type == "linear":
                v = self.E_unit[layer_idx]
                # print(f"v: {v}")
                alpha = th.tensor([self.betas[layer_idx] for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
                # alpha = th.tensor([th.norm(x[i]) for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
                e = alpha.squeeze(0).T @ v.unsqueeze(0)
                # print(f"e: {e}")
            elif self.setpoint_type == "angular":
                # print("DOING THE THING")
                e = self.get_angular_sp(x, layer_idx) - x
            else:
                raise ValueError("Unsupported setpoint type")
            # u_t = th.bmm(self.K[layer_idx].unsqueeze(0), th.transpose(e.unsqueeze(0),-2,-1)).squeeze(0).T
            u_t = th.bmm(self.K[layer_idx].unsqueeze(0), th.transpose(e.unsqueeze(0),-2,-1)).squeeze(0).T
            self.U[layer_idx] = u_t[-1]

            # print(f"K in layer: {self.K[layer_idx]}")
            # print(f"fucking shit: {u_t}")
            if isinstance(output,tuple):
                output[0][...,-1,:] = output[0][...,-1,:] + u_t
            else: 
                output[...,-1,:] = output[...,-1,:] + u_t

            # print(f"output: {output}")
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

    def hook_multisteer_tracking(self, layer_idx, module, input, output):
        print("in multisteer hook")
        x = input[0][:,-1,:]
        self.X[layer_idx] = x[-1,:]

        v = self.E_unit[layer_idx]
        v_alt = self.E_alt_unit[layer_idx]
        alpha = th.tensor([self.betas[layer_idx] for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        alpha_alt = th.tensor([self.betas_alt[layer_idx] for i in range(x.shape[0])], device=self.device) - th.bmm(v_alt.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        
        e = alpha.squeeze(0).T @ v.unsqueeze(0)
        e_alt = alpha_alt.squeeze(0).T @ v_alt.unsqueeze(0)
        u_t = th.bmm(self.K[layer_idx].unsqueeze(0), th.transpose(e.unsqueeze(0),-2,-1)).squeeze(0).T
        u_t_alt = th.bmm(self.K[layer_idx].unsqueeze(0), th.transpose(e_alt.unsqueeze(0),-2,-1)).squeeze(0).T
        self.U[layer_idx] = u_t[-1]

        if isinstance(output,tuple):
            output[0][...,-1,:] = output[0][...,-1,:] + u_t + u_t_alt
        else: 
            output[...,-1,:] = output[...,-1,:] + u_t + u_t_alt

        # print(f"output: {output}")
        return output

    def register_multisteer_tracking_hooks(self):
        """Register the hooks."""

        for layer_idx, layer in enumerate(self.model.model.layers):
            def hook_wrapper(layer_idx):
                def hook(module, input, output):
                    return self.hook_multisteer_tracking(layer_idx, module, input, output)

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
                
            # v = self.E_unit[layer_idx+1]
            v = self.E_unit[0]
            # v = self.E_unit[-2]
            raw_signal = th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
            nm = th.norm(self.E[layer_idx+1])
            # print(nm)
            signal = raw_signal / nm
            # signal = raw_signal
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

    def get_angular_steering_output_hook(
        self,
        target_degree: float,
        adaptive_mode: int = 1,
        layer_idx: int = 0,
    ):
        """Create a hook that applies angular steering to layer outputs.

        Args:
            steering_config: Dict with 'first_direction' and 'second_direction' numpy arrays
            target_degree: Rotation angle in degrees (0-360)
            adaptive_mode: Steering application mode:
                        0 = always steer all activations
                        1 = only steer when activation is aligned with first_direction (conditional)

        Returns:
            Hook function that applies angular steering transformation to module outputs
        """
        first_direction = self.E[layer_idx]
        second_direction = self.basis2

        # Compute rotation
        device = first_direction.device
        theta_rad = th.tensor(target_degree * th.pi / 180.0, device=device)

        # Orthonormalize directions
        b1 = first_direction / first_direction.norm()
        b2 = second_direction - (second_direction @ b1) * b1
        b2 = b2 / b2.norm()

        # Projection matrix
        proj_matrix = th.outer(b1, b1) + th.outer(b2, b2)

        # Rotation matrix
        cos_theta = th.cos(theta_rad)
        sin_theta = th.sin(theta_rad)
        rotation_matrix = th.stack(
            [th.stack([cos_theta, -sin_theta]), th.stack([sin_theta, cos_theta])]
        )

        # Steering vector
        unit_vector = th.tensor([1.0, 0.0], device=device)
        rotated_2d = rotation_matrix @ unit_vector
        steering_vector = rotated_2d[0] * b1 + rotated_2d[1] * b2

        _cache = {}

        def steering_hook(_module, _input, output):
            device = output.device
            dtype = output.dtype
            cache_key = (device, dtype)

            if cache_key not in _cache:
                _cache[cache_key] = (
                    proj_matrix.to(device=device, dtype=dtype),
                    steering_vector.to(device=device, dtype=dtype),
                    first_direction.to(device=device, dtype=dtype),
                )

            proj, steer, first_dir = _cache[cache_key]

            projected = output @ proj
            scale = projected.norm(dim=-1, keepdim=True)

            if adaptive_mode == 0:
                steered = output - projected + scale * steer
                return steered
            elif adaptive_mode == 1:
                proj_to_first = output @ first_dir
                mask = (proj_to_first > 0).unsqueeze(-1)
                steered = output - projected + scale * steer
                return th.where(mask, steered, output)
            else:
                raise ValueError(f"Unknown adaptive_mode: {adaptive_mode}")

        return steering_hook


    def register_angular_steering_hooks(self):
        """Register the hooks."""

        for layer_idx, layer in enumerate(self.model.model.layers):
            self.hooks.append(
                layer.register_forward_hook(
                    self.get_angular_steering_output_hook(
                        target_degree=self.target_degree,
                        layer_idx=layer_idx,
                        adaptive_mode=0
                    )
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
        elif self.mode == Mode.TRACKING: 
            self.register_tracking_hooks()
        elif self.mode == Mode.SETPOINT:
            self.register_setpoint_tracking_hooks()
        elif self.mode == Mode.ANGULAR:
            self.register_angular_steering_hooks()
        elif self.mode == Mode.MULTI:
            self.register_multisteer_tracking_hooks()
        else:
            print("generating with no steering applied")

        if self.SIGNAL_COLLECT:
            self.register_setpoint_signal_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    

    def evaluate(self, prompt, max_new_tokens, do_sample=False, temp=0.7):
        self.mode = Mode.STEERING
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        self.X = th.zeros((self.T+1, self.n)).to(self.device)

        # print(f"ids: {input_ids.device}")
        # print(f"ids shape: {input_ids.shape}")
        # print(f"mask: {attention_mask.device}")
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

    def track_setpoint(self, prompt, max_new_tokens, lmbda=1, do_sample=False, temp=1, all_tokens=False, return_tokens=False):
        self.mode = Mode.SETPOINT
        self.setpoint_type = "linear"
        self.SIGNAL_COLLECT = True
        self.setpoint_signals = []
        self.ALL_TOKENS = all_tokens
        # print("python is garbage")
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
            # print(f"e in setpoint: {e}")
            nrm = th.linalg.norm(e)
            # print(f"nrm in setpoint: {nrm}")
            if nrm == 0:
                self.E_unit[i] = self.E_unit[i]*0
            else:
                self.E_unit[i] = e / nrm
                self.betas[i] = lmbda * nrm

        # if self.ALL_TOKENS:
        #     print("Running with ALL_TOKENS=True")
        # else:
        #     print("Running with ALL_TOKENS=False")


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
        self.ALL_TOKENS = False
        return output_str
        

    def multisteer(self, prompt, max_new_tokens, alt_contr, lmbda=1, alt_lmbda=1, do_sample=False, temp=1, all_tokens=False, return_tokens=False):
        self.mode = Mode.MULTI
        self.setpoint_type = "linear"
        self.SIGNAL_COLLECT = True
        self.setpoint_signals = []
        self.ALL_TOKENS = all_tokens
        # print("python is garbage")
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        print("in multisteer eeee")
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
            # print(f"e in setpoint: {e}")
            nrm = th.linalg.norm(e)
            # print(f"nrm in setpoint: {nrm}")
            if nrm == 0:
                self.E_unit[i] = self.E_unit[i]*0
            else:
                self.E_unit[i] = e / nrm
                self.betas[i] = lmbda * nrm

        self.E_alt_unit = th.zeros_like(alt_contr)
        self.betas_alt = [0 for i in range(self.T+1)]
        for i, e in enumerate(alt_contr):
            # print(f"e in setpoint: {e}")
            nrm = th.linalg.norm(e)
            # print(f"nrm in setpoint: {nrm}")
            if nrm == 0:
                self.E_alt_unit[i] = self.E_unit[i]*0
            else:
                self.E_alt_unit[i] = e / nrm
                self.betas_alt[i] = alt_lmbda * nrm

        # if self.ALL_TOKENS:
        #     print("Running with ALL_TOKENS=True")
        # else:
        #     print("Running with ALL_TOKENS=False")


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
        self.ALL_TOKENS = False
        return output_str


    def get_best_dir(self, X_contr):
        print(f"contrrrrrrr: {X_contr[0]}")
        candidates_normalized = [
            v / v.norm() if not (v.norm() == 0) else v for v in X_contr 
        ]
        print(f"normalized: {candidates_normalized[0]}")
        candidates_stack = th.stack(
            candidates_normalized
        )
        print(f"stack: {candidates_stack[0]}")

        # Compute pairwise cosine similarities
        pairwise_cosine = candidates_stack @ candidates_stack.T
        mean_cosine = pairwise_cosine.mean(dim=-1)

        # Find layer with highest mean cosine similarity
        max_idx = mean_cosine.argmax().item()
        # print(f"selected: {})

        # Log layer selection info
        print(f"\n  Max sim layer selection:")
        for i, key in enumerate(X_contr):
            # layer_num = int(key.split("_")[1])
            marker = " ← SELECTED" if i == max_idx else ""
            print(
                f"    Layer {i}: cosine={mean_cosine[i].item():.4f}{marker}"
            )
        all_best = [
            X_contr[max_idx] for x in X_contr
        ]
        # print(all_best)
        best_stack = th.stack(
            all_best
        )
        return best_stack


    def track_angular_setpoint_from_second_dir(self, prompt, max_new_tokens, target_degree, second_dir, do_sample=False, temp=0.7):
        self.mode = Mode.SETPOINT
        self.setpoint_type = "angular"
        self.ALL_TOKENS = True



        if isinstance(target_degree, th.Tensor):
            self.target_degree = target_degree.detach().clone()
        else:
            self.target_degree = th.tensor(target_degree, device=self.device)

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

        self.basis2 = second_dir

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
        self.ALL_TOKENS = False
        return output_str

    def track_angular_setpoint(self, prompt, max_new_tokens, target_degree, lmbda=1, do_sample=False, temp=0.7):
        self.mode = Mode.SETPOINT
        self.setpoint_type = "angular"
        self.ALL_TOKENS = True



        if isinstance(target_degree, th.Tensor):
            self.target_degree = target_degree.detach().clone()
        else:
            self.target_degree = th.tensor(target_degree, device=self.device)

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

        # self.E = self.get_best_dir(self.E)
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
        self.ALL_TOKENS = False
        return output_str

    @contextmanager
    def add_hooks(
        self,
        module_forward_pre_hooks: List[Tuple[nn.Module, Callable]] = None,
        module_forward_hooks: List[Tuple[nn.Module, Callable]] = None,
        **kwargs,
    ):
        """Context manager for temporarily adding forward hooks.

        Args:
            module_forward_pre_hooks: List of (module, hook_fn) tuples for pre-hooks
            module_forward_hooks: List of (module, hook_fn) tuples for forward hooks
            **kwargs: Additional keyword arguments passed to hook functions

        Yields:
            None. Hooks are active within the context, removed on exit.
        """
        module_forward_pre_hooks = module_forward_pre_hooks or []
        module_forward_hooks = module_forward_hooks or []
        handles = []
        try:
            for module, hook in module_forward_pre_hooks:
                partial_hook = functools.partial(hook, **kwargs)
                handles.append(module.register_forward_pre_hook(partial_hook))
            for module, hook in module_forward_hooks:
                partial_hook = functools.partial(hook, **kwargs)
                handles.append(module.register_forward_hook(partial_hook))
            yield
        finally:
            for h in handles:
                h.remove()


    def get_angular_steering_output_hook_replicate(
        self,
        steering_config: dict,
        target_degree: float,
        adaptive_mode: int = 1,
    ):
        """Create a hook that applies angular steering to layer outputs.

        Args:
            steering_config: Dict with 'first_direction' and 'second_direction' numpy arrays
            target_degree: Rotation angle in degrees (0-360)
            adaptive_mode: Steering application mode:
                        0 = always steer all activations
                        1 = only steer when activation is aligned with first_direction (conditional)

        Returns:
            Hook function that applies angular steering transformation to module outputs
        """
        first_direction = th.from_numpy(steering_config["first_direction"])
        second_direction = th.from_numpy(steering_config["second_direction"])

        # Compute rotation
        device = first_direction.device
        theta_rad = th.tensor(target_degree * th.pi / 180.0, device=device)

        # Orthonormalize directions
        b1 = first_direction / first_direction.norm()
        b2 = second_direction - (second_direction @ b1) * b1
        b2 = b2 / b2.norm()

        # Projection matrix
        proj_matrix = th.outer(b1, b1) + th.outer(b2, b2)

        # Rotation matrix
        cos_theta = th.cos(theta_rad)
        sin_theta = th.sin(theta_rad)
        rotation_matrix = th.stack(
            [th.stack([cos_theta, -sin_theta]), th.stack([sin_theta, cos_theta])]
        )

        # Steering vector
        unit_vector = th.tensor([1.0, 0.0], device=device)
        rotated_2d = rotation_matrix @ unit_vector
        steering_vector = rotated_2d[0] * b1 + rotated_2d[1] * b2

        _cache = {}

        def steering_hook(_module, _input, output):
            device = output.device
            dtype = output.dtype
            cache_key = (device, dtype)

            if cache_key not in _cache:
                _cache[cache_key] = (
                    proj_matrix.to(device=device, dtype=dtype),
                    steering_vector.to(device=device, dtype=dtype),
                    first_direction.to(device=device, dtype=dtype),
                )

            proj, steer, first_dir = _cache[cache_key]

            projected = output @ proj
            scale = projected.norm(dim=-1, keepdim=True)

            if adaptive_mode == 0:
                steered = output - projected + scale * steer
                return steered
            elif adaptive_mode == 1:
                proj_to_first = output @ first_dir
                mask = (proj_to_first > 0).unsqueeze(-1)
                steered = output - projected + scale * steer
                return th.where(mask, steered, output)
            else:
                raise ValueError(f"Unknown adaptive_mode: {adaptive_mode}")

        return steering_hook

    def angular_steer_replicate(self, prompt, max_new_tokens, target_degree, config, lmbda=1, do_sample=False, temp=0.7):
        self.mode = Mode.ANGULAR
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

        module_dict = dict(self.model.named_modules())

        print(f"module dict:\n{module_dict}")


        # output_hooks = [
        #         (
        #             module_dict[module_name],
        #             self.get_angular_steering_output_hook_replicate(
        #                 steering_config=steering_config,
        #                 target_degree=target_degree,
        #                 adaptive_mode=1,
        #             ),
        #         )
        #        for module_name, steering_config in config.items() if "post_attention_layernorm" in module_name 
        #     ]
        
        output_hooks = []
        # for module_name, steering_config in config.items():
        #     if "post_attention_layernorm" in module_name:
        #         print(module_name)
        #         output_hooks.append(
        #             (
        #                 module_dict[module_name],
        #                 self.get_angular_steering_output_hook_replicate(
        #                     steering_config=steering_config,
        #                     target_degree=target_degree,
        #                     adaptive_mode=1,
        #                 )
        #             )
        #         )
        for i, (module_name, steering_config) in enumerate(config.items()):
            # if "post_attention_layernorm" in module_name:
            module_name = f"model.layers.{i}"
            # print(module_name)
            if i < len(self.model.model.layers):
                output_hooks.append(
                    (
                        module_dict[module_name],
                        self.get_angular_steering_output_hook_replicate(
                            steering_config=steering_config,
                            target_degree=target_degree,
                            adaptive_mode=1,
                        )
                    )
                )


        print(f"output hooks (hopefully only post attention): {output_hooks}")
        print(f"len output hooks: {len(output_hooks)}")

        with self.add_hooks(
            module_forward_hooks=output_hooks,
        ):
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

    def angular_steer(self, prompt, max_new_tokens, target_degree, second_dir=None, lmbda=1, do_sample=False, temp=0.7):
        self.mode = Mode.ANGULAR
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

        if second_dir is None:
            refusal_dirs = self.E.cpu()
            pca_model = PCA().fit(refusal_dirs)

            components = pca_model.components_
            self.basis2 = th.tensor(components[0].copy(), dtype=th.float, device=self.device)
        else:
            self.basis2 = second_dir

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
                    use_cache=False,
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
            wrapped_tfs_temp = [partial(lqr.new_llama_block_wrapper, tf, nom_attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
            tfs_with_control_temp = [partial(lqr.transformerBlockControl, tf) for tf in wrapped_tfs_temp]
            self.A = lqr.linearize(tfs_with_control_temp,self.T,self.m,self.X[0]) # linearizing about first subtrajectory

        lin_time = time.perf_counter()
        print(f"Linearize time: {lin_time - end_nom_time}")

        print(self.A.device)
        self.K = lqr.time_varying_lqr(self.A, self.B, self.Q, self.R, self.Qf)

        self.mode = Mode.TRACKING
        self.X_cl = [th.zeros_like(hidden_states).repeat(self.T+1, 1, 1).to(self.device)]
        
        sublist = [th.zeros_like(hidden_states[...,-1,:]).repeat(self.T+1, 1, 1).to(self.device) for i in range(k-1)]
        self.X_cl = self.X_cl + sublist

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
            wrapped_tfs_temp = [partial(lqr.new_llama_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
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

    def generate_and_collect(self, prompt, max_new_tokens=1, do_sample=True, temp=0.7):
        self.setpoint_type = "linear"
        self.SIGNAL_COLLECT = True

        self.E_unit = th.zeros_like(self.E)
        for i, e in enumerate(self.E):
            # print(f"e: {e}")
            nrm = th.linalg.norm(e)
            self.E_unit[i] = e / nrm

        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        # self.X = th.zeros((self.T+1, self.n)).to(self.device)

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
        # output_str = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

        signals = self.setpoint_signals
        self.setpoint_signals = []
        return signals

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
        wrapped_tfs_temp = [partial(lqr.new_llama_block_wrapper, tf, attention_mask, position_ids, position_embeddings) for tf in self.model.model.layers]
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
        plt.title('Perturbation norms for final pass?')
        plt.xticks([r + bar_width / 2 for r in r1], layer_lbls) # Center x-axis labels
        plt.legend()
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.savefig(figname + ".png")

    def compute_ppl(self, data, lmbda=1, BATCH_SZ=10):

        self.mode = Mode.SETPOINT
        self.X = th.zeros((self.T+1, self.n)).to(self.device)

        self.E_unit = th.zeros_like(self.E)
        self.betas = [0 for i in range(self.T+1)]
        for i, e in enumerate(self.E):
            # print(f"e: {e}")
            nrm = th.linalg.norm(e)
            self.E_unit[i] = e / nrm
            self.betas[i] = lmbda * nrm
        print(f"betas= {self.betas}")
        # model.eval()

        prompts = None

        self.tokenizer.padding_side = "right"
        truncation = True
        max_generation_length = 40
        max_context_length = 128
        # max_context_length = model.config.max_position_embeddings

        print(f"number of sentences: {len(data)}")
        BATCH_SZ = 10


        nll_sum = th.zeros(1)
        total = th.zeros(1)

        tok_s = self.tokenizer(
            text=data,
            return_tensors="pt",
            truncation=truncation,
            padding=truncation,
            max_length=max_generation_length,
            add_special_tokens=(
                prompts is None
            ),  # if there is a prompt, it already contains BOS token
        ).to(self.device)
        self.tokenizer.padding_side = (
            "left"  # go back to original padding (to not messup things)
        )

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
            tok_all = {k: th.cat([tok_p[k], tok_s[k]], -1) for k in tok_p.keys()}
            offset = tok_p["input_ids"].shape[-1]
        else:
            tok_all = tok_s
            offset = 1  # skips the BOS token

        input_ids = tok_all["input_ids"]
        # print(f"shape; {input_ids.shape}")
        attention_mask = tok_all["attention_mask"]
        # This is the number of tokens in each continuation. We will generate this amount of tokens, one by one.
        attention_mask_sum = tok_s["attention_mask"].sum(-1)
        # Buffer to keep track of ppls
        ppls = th.zeros(attention_mask.shape[0], device=self.device, dtype=th.float32)
        totals = th.zeros_like(ppls)

        # # print(f"sequential ppl: {th.exp(nll_sum/ total)}")
        total_loss = 0
        total_tokens = 0
        for i in tqdm(range(0, len(data), BATCH_SZ)):
            batch = data[i:i+BATCH_SZ]
            
            tok = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_generation_length,
            ).to(self.device)
            
            input_ids = tok["input_ids"]
            attention_mask = tok["attention_mask"]
            
            with self:
                with th.no_grad():
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            
            # Shift logits and labels to predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            
            # Compute cross-entropy loss for all tokens in batch
            losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none"
            ).view(shift_labels.size())
            
            masked_loss = (losses * shift_mask).sum()
            num_tokens = shift_mask.sum()
            
            total_loss += masked_loss.item()
            total_tokens += num_tokens.item()

        return th.exp(th.tensor(total_loss / total_tokens))