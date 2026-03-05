import torch as th
import lqr_utils_seq as lqr
from diffusers import FluxPipeline, BitsAndBytesConfig
from transformers import T5EncoderModel
from diffusers.quantizers import PipelineQuantizationConfig
from IPython.display import display
from typing import Callable, List, Tuple
import torch.nn as nn
import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union
import inspect
import yaml
import numpy as np
from functools import partial
import random
import pickle


class ImageLQRSteering:
    '''
    Contrastive method currently assuming precomputed:
        - jacobians (A)
        - contrastive vectors
    '''


    def __init__(
        self,
        pipe: FluxPipeline,
        num_inference_steps: int,
        q: float = 10,
        r: float = 10,
        qf: float = 1,
        A_sing: th.Tensor = None,    
        A_multi: th.Tensor = None,    
        sing_contrastive_vecs: th.Tensor = None,
        multi_contrastive_vecs: th.Tensor = None,
        preserve_mem: bool = True,
    ):
        self.pipe = pipe
        self.device = self.pipe.device
        self.A_sing = A_sing.to(self.device)
        self.A_multi = A_multi.to(self.device)
        self.E_sing = sing_contrastive_vecs.to(self.device)
        self.E_multi = multi_contrastive_vecs.to(self.device)
        self.num_inference_steps = num_inference_steps

        self.T_multi = len(self.pipe.transformer.transformer_blocks)
        self.T_single = len(self.pipe.transformer.single_transformer_blocks)
        self.n = self.pipe.transformer.transformer_blocks[0].attn.to_q.in_features


        self.Q_sing = th.eye(self.n).unsqueeze(0).repeat(self.T_single, 1, 1).to(self.device) * q
        self.R_sing = th.eye(self.n).unsqueeze(0).repeat(self.T_single, 1, 1).to(self.device) * r
        self.Qf_sing = th.eye(self.n).to(self.device) * qf
        self.Q_multi = th.eye(self.n).unsqueeze(0).repeat(self.T_multi, 1, 1).to(self.device) * q
        self.R_multi = th.eye(self.n).unsqueeze(0).repeat(self.T_single, 1, 1).to(self.device) * r
        self.Qf_multi = th.eye(self.n).to(self.device) * qf
        
        
        
        if preserve_mem:
            self.K_sing = lqr.time_varying_lqr_noB(self.A_sing, self.Q_sing, self.R_sing, self.Qf_sing) if self.A_sing is not None else None
            self.K_multi = lqr.time_varying_lqr_noB(self.A_multi, self.Q_multi, self.R_multi, self.Qf_multi) if self.A_multi is not None else None
            del self.A_sing
            del self.A_multi
            del self.Q_sing
            del self.Q_multi
            del self.R_sing
            del self.R_multi
            del self.Qf_sing
            del self.Qf_multi
        else:
            raise ValueError("preserve memory pleeeeassee")




        self.X = None # to allocate at runtime
        self.U_sing = th.zeros((self.T_single, self.n), device=self.device)
        self.U_multi = th.zeros((self.T_multi, self.n), device=self.device)

        self.X_cl = None

        self.betas_sing = None
        self.betas_multi = None
        self.E_unit_sing = None
        self.E_unit_multi = None

        self.hooks = []

        self.steer_multi = True
        self.steer_single = True

    # def hook_setpoint_tracking(self, layer_idx, module, input, output):
    def hook_setpoint_tracking_multi(self, layer_idx, module, args, kwargs, output):
        # assume E_normed is unit vector in direction of contrastive feature
        print(f"kwargs keys: {kwargs.keys()}")
        print(f"kwargs: {kwargs}")
        print(f"args: {args}")
        x = kwargs["hidden_states"][...,-1,:]
        # self.X[layer_idx] = x[-1,:]

        v = self.E_unit_multi[layer_idx]
        # print(f"v: {v}")
        alpha = th.tensor([self.betas_multi[layer_idx] for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        # alpha = th.tensor([th.norm(x[i]) for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        e = alpha.squeeze(0).T @ v.unsqueeze(0)
            # print(f"e: {e}")
        u_t = th.bmm(self.K_multi[layer_idx].unsqueeze(0), th.transpose(e.unsqueeze(0),-2,-1)).squeeze(0).T
        # self.U[layer_idx] = u_t[-1]


        if isinstance(output,tuple):
            output[1][...,-1,:] = output[1][...,-1,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
        else: 
            output[...,-1,:] = output[...,-1,:] + u_t

        # print(f"output: {output}")
        return output
    
        # def hook_setpoint_tracking(self, layer_idx, module, input, output):
    def hook_setpoint_tracking_sing(self, layer_idx, module, args, kwargs, output):
        # assume E_normed is unit vector in direction of contrastive feature
        print(f"kwargs keys: {kwargs.keys()}")
        print(f"kwargs: {kwargs}")
        print(f"args: {args}")
        x = kwargs["hidden_states"][...,-1,:]
        # self.X[layer_idx] = x[-1,:]

        v = self.E_unit_sing[layer_idx]
        # print(f"v: {v}")
        alpha = th.tensor([self.betas_sing[layer_idx] for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        # alpha = th.tensor([th.norm(x[i]) for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        e = alpha.squeeze(0).T @ v.unsqueeze(0)
            # print(f"e: {e}")
        u_t = th.bmm(self.K_sing[layer_idx].unsqueeze(0), th.transpose(e.unsqueeze(0),-2,-1)).squeeze(0).T
        # self.U[layer_idx] = u_t[-1]


        if isinstance(output,tuple):
            output[1][...,-1,:] = output[1][...,-1,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
        else: 
            output[...,-1,:] = output[...,-1,:] + u_t

        # print(f"output: {output}")
        return output

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @contextmanager
    def add_hooks(
            self
        ):
            handles = []
            try:
                
                if self.steer_multi:
                    for layer_idx, layer in enumerate(self.pipe.transformer.transformer_blocks):
                        def hook_wrapper(layer_idx):
                            # def hook(module, input, output):
                                # return self.hook_setpoint_tracking(layer_idx, module, input, output)
                            def hook(module, args, kwargs, output):
                                return self.hook_setpoint_tracking_multi(layer_idx, module, args, kwargs, output)


                            return hook

                        handles.append(
                            layer.register_forward_hook(
                                hook_wrapper(layer_idx), with_kwargs=True
                            )
                        )
                if self.steer_single:
                    for layer_idx, layer in enumerate(self.pipe.transformer.single_transformer_blocks):
                        def hook_wrapper(layer_idx):
                            # def hook(module, input, output):
                            #     return self.hook_setpoint_tracking(layer_idx, module, input, output)
                            def hook(module, args, kwargs, output):
                                return self.hook_setpoint_tracking_sing(layer_idx, module, args, kwargs, output)


                            return hook

                        handles.append(
                            layer.register_forward_hook(
                                hook_wrapper(layer_idx), with_kwargs=True
                            )
                        )
                    # for module, hook in module_forward_hooks:
                        # partial_hook = functools.partial(hook, **kwargs)
                        # handles.append(module.register_forward_hook(partial_hook))
                yield
            finally:
                for h in handles:
                    h.remove()


    def track_setpoint(self, prompt, steer_multi, steer_single, lmbda=1, do_sample=False, temp=1):
        self.steer_multi = steer_multi
        self.steer_single = steer_single

        self.E_unit_sing = th.zeros_like(self.E_sing)
        self.E_unit_multi = th.zeros_like(self.E_multi)
        self.betas_sing = [0 for i in range(self.T_single)]
        self.betas_multi = [0 for i in range(self.T_multi)]
        for i, e in enumerate(self.E_multi):
            # print(f"e in setpoint: {e}")
            nrm = th.linalg.norm(e)
            # print(f"nrm in setpoint: {nrm}")
            if nrm == 0:
                self.E_unit_multi[i] = self.E_unit_multi[i]*0
            else:
                self.E_unit_multi[i] = e / nrm
                self.betas_multi[i] = lmbda * nrm

        for i, e in enumerate(self.E_sing):
            # print(f"e in setpoint: {e}")
            nrm = th.linalg.norm(e)
            # print(f"nrm in setpoint: {nrm}")
            if nrm == 0:
                self.E_unit_sing[i] = self.E_unit_sing[i]*0
            else:
                self.E_unit_sing[i] = e / nrm
                self.betas_sing[i] = lmbda * nrm


        with self.add_hooks():
            image = self.pipe(
                prompt,
                guidance_scale=0.0,
                num_inference_steps=self.num_inference_steps,
                max_sequence_length=256,
                generator=th.Generator(self.device).manual_seed(0)
            ).images[0]

        # output_str = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        return image
        
    