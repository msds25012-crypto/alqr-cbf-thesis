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
        sing_contrastive_vecs_img: th.Tensor = None,
        multi_contrastive_vecs_img: th.Tensor = None,
        sing_contrastive_vecs_txt: th.Tensor = None,
        multi_contrastive_vecs_txt: th.Tensor = None,
        preserve_mem: bool = True,
    ):
        self.pipe = pipe
        self.device = self.pipe.device
        self.A_sing = A_sing.to(self.device) if A_sing is not None else None
        self.A_multi = A_multi.to(self.device) if A_multi is not None else None
        self.E_sing = sing_contrastive_vecs_img.to(self.device) if sing_contrastive_vecs_img is not None else None
        self.E_multi = multi_contrastive_vecs_img.to(self.device) if multi_contrastive_vecs_img is not None else None

        self.E_sing_txt = sing_contrastive_vecs_txt.to(self.device) if sing_contrastive_vecs_txt is not None else None
        self.E_multi_txt = multi_contrastive_vecs_txt.to(self.device) if multi_contrastive_vecs_txt is not None else None
        
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
            # self.K_sing = lqr.time_varying_lqr_noB(self.A_sing, self.Q_sing, self.R_sing, self.Qf_sing).to(th.bfloat16) if self.A_sing is not None else None
            # self.K_multi = lqr.time_varying_lqr_noB(self.A_multi, self.Q_multi, self.R_multi, self.Qf_multi).to(th.bfloat16) if self.A_multi is not None else None
            self.K_sing = lqr.time_varying_lqr_noB_mem_efficient(self.A_sing, self.Q_sing, self.R_sing, self.Qf_sing).to(th.bfloat16) if self.A_sing is not None else None
            del self.A_sing
            del self.Q_sing
            del self.R_sing
            del self.Qf_sing
            self.K_multi = lqr.time_varying_lqr_noB_mem_efficient(self.A_multi, self.Q_multi, self.R_multi, self.Qf_multi).to(th.bfloat16) if self.A_multi is not None else None
            del self.A_multi
            del self.Q_multi
            del self.R_multi
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

        self.sing_scale = None
        self.multi_scale = None

    # def hook_setpoint_tracking(self, layer_idx, module, input, output):
    def hook_setpoint_tracking_multi(self, layer_idx, module, args, kwargs, output):
        # assume E_normed is unit vector in direction of contrastive feature
        # print(f"kwargs keys: {kwargs.keys()}")
        # print(f"kwargs: {kwargs}")
        # print(f"args: {args}")
        # x = kwargs["hidden_states"][...,-1,:]
        # # self.X[layer_idx] = x[-1,:]

        # v = self.E_unit_multi[layer_idx]
        
        # # print(f"v: {v}")
        
        # # print(f"betas_multi dtype: {self.betas_multi[layer_idx].dtype}")
        # # print(f"v dtype: {v.dtype}")
        # # print(f"x dtype: {x.dtype}")
        # alpha = th.tensor([self.betas_multi[layer_idx] for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        # # alpha = th.tensor([th.norm(x[i]) for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        # e = alpha.squeeze(0).T @ v.unsqueeze(0)
        #     # print(f"e: {e}")
        # u_t = th.bmm(self.K_multi[layer_idx].unsqueeze(0), th.transpose(e.unsqueeze(0),-2,-1)).squeeze(0).T
        # # self.U[layer_idx] = u_t[-1]


        # print(f"multi layer {layer_idx} u_t: {u_t}")
        # if isinstance(output,tuple):
        #     # output[1][...,-1,:] = output[1][...,-1,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
        #     output[1][...,:] = output[1][...,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
        # else: 
        #     output[...,-1,:] = output[...,-1,:] + u_t
        # # print(f"output: {output}")
        # return output

        x = kwargs["hidden_states"]
        v = self.E_unit_multi[layer_idx]

        b_mat = self.betas_multi[layer_idx] * th.ones([x.shape[0], x.shape[1]], device=self.device,  dtype=th.bfloat16)
        probe_mat = x @ v.T
        alpha = b_mat - probe_mat
        v_mat = v.expand(x.shape[0], x.shape[1], -1)
        e = alpha.unsqueeze(-1) * v_mat
        print(f"x multi shape: {x.shape}")
        # print(f"vmat dtype: {v_mat.dtype}")
        # print(f"e dtype: {e.dtype}")

        u_t = e @ self.K_multi[layer_idx].T


        print(f"multi layer {layer_idx} u_t: {u_t}")
        if isinstance(output,tuple):
            # output[1][...,-1,:] = output[1][...,-1,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
            output[1][...,:] = output[1][...,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
        else: 
            raise ValueError("yoimultinkeeee")
            output[...,-1,:] = output[...,-1,:] + u_t
        # print(f"output: {output}")
        return output
    
    def hook_setpoint_tracking_multi_text(self, layer_idx, module, args, kwargs, output):

        x = kwargs["encoder_hidden_states"]
        v = self.E_unit_multi[layer_idx]

        b_mat = self.betas_multi[layer_idx] * th.ones([x.shape[0], x.shape[1]], device=self.device,  dtype=th.bfloat16)
        probe_mat = x @ v.T
        alpha = b_mat - probe_mat
        v_mat = v.expand(x.shape[0], x.shape[1], -1)
        e = alpha.unsqueeze(-1) * v_mat
        print(f"x multi shape: {x.shape}")
        # print(f"vmat dtype: {v_mat.dtype}")
        # print(f"e dtype: {e.dtype}")
        print(f"alpha multi layer {layer_idx}: {alpha}")

        u_t = e @ self.K_multi[layer_idx].T


        print(f"multi layer {layer_idx} u_t: {u_t}")
        if isinstance(output,tuple):
            print(f"u_multi_textL: {u_t}")
            # output[1][...,-1,:] = output[1][...,-1,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
            output[0][...,:] = output[0][...,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
        else: 
            raise ValueError("yoimultinkeeee")
            output[...,-1,:] = output[...,-1,:] + u_t
        # print(f"output: {output}")
        return output
    
        # def hook_setpoint_tracking(self, layer_idx, module, input, output):
    def hook_setpoint_tracking_sing(self, layer_idx, module, args, kwargs, output):
        # assume E_normed is unit vector in direction of contrastive feature
        # print(f"kwargs keys: {kwargs.keys()}")
        print(f"kwargs: {kwargs}")
        print(f"args: {args}")
        x = kwargs["hidden_states"]#[...,-1,:]
        # self.X[layer_idx] = x[-1,:]

        v = self.E_unit_sing[layer_idx]
        # # print(f"v: {v}")
        # alpha = th.tensor([self.betas_sing[layer_idx] for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        # # alpha = th.tensor([th.norm(x[i]) for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        # e = alpha.squeeze(0).T @ v.unsqueeze(0)
        #     # print(f"e: {e}")
        # u_t = th.bmm(self.K_sing[layer_idx].unsqueeze(0), th.transpose(e.unsqueeze(0),-2,-1)).squeeze(0).T
        # # self.U[layer_idx] = u_t[-1]


        # print(f"sing layer {layer_idx} u_t: {u_t}")
        # if isinstance(output,tuple):
        #     print("here")
        #     # output[1][...,-1,:] = output[1][...,-1,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
        #     output[1][...,:] = output[1][...,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
        # else: 
        #     output[...,-1,:] = output[...,-1,:] + u_t

        # # print(f"output: {output}")
        # return output

        b_mat = self.betas_sing[layer_idx] * th.ones([x.shape[0], x.shape[1]], device=self.device, dtype=th.bfloat16)
        probe_mat = x @ v.T
        alpha = b_mat - probe_mat
        v_mat = v.expand(x.shape[0], x.shape[1], -1)
        e = alpha.unsqueeze(-1) * v_mat

        print(f"x shape; {x.shape}")
        print(f"vmat shape; {v_mat.shape}")
        print(f"e shape; {e.shape}")
        print(f"K shape; {self.K_sing.shape}")

        u_t = e @ self.K_sing[layer_idx].T

        # print(f"multi layer {layer_idx} u_t: {u_t}")
        if isinstance(output,tuple):
            # output[1][...,-1,:] = output[1][...,-1,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
            print(f"output shape: {output[1][...,-u_t.shape[1]:,:].shape}")
            print(f"u shape: {u_t.shape}")
            output[1][...,:] = output[1][...,-u_t.shape[1]:,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
        else: 
            raise ValueError("yoinkeeee")
            # output[...,-1,:] = output[...,-1,:] + u_t
        # print(f"output: {output}")
        return output

    def hook_setpoint_tracking_sing_text(self, layer_idx, module, args, kwargs, output):
        # assume E_normed is unit vector in direction of contrastive feature
        # print(f"kwargs keys: {kwargs.keys()}")
        x = kwargs["encoder_hidden_states"]#][...,-1,:]
        # self.X[layer_idx] = x[-1,:]

        v = self.E_unit_sing[layer_idx]
        # # # print(f"v: {v}")
        # alpha = th.tensor([self.betas_sing[layer_idx] for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        # # alpha = th.tensor([th.norm(x[i]) for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        # e = alpha.squeeze(0).T @ v.unsqueeze(0)
        #     # print(f"e: {e}")
        # u_t = th.bmm(self.K_sing[layer_idx].unsqueeze(0), th.transpose(e.unsqueeze(0),-2,-1)).squeeze(0).T
        # # self.U[layer_idx] = u_t[-1]


        # print(f"sing layer {layer_idx} u_t: {u_t}")
        # if isinstance(output,tuple):
        #     print("here")
        #     # output[1][...,-1,:] = output[1][...,-1,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
        #     output[1][...,:] = output[0][...,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
        # else: 
        #     output[...,-1,:] = output[...,-1,:] + u_t

        # # print(f"output: {output}")
        # return output

        b_mat = self.betas_sing[layer_idx] * th.ones([x.shape[0], x.shape[1]], device=self.device, dtype=th.bfloat16)
        probe_mat = x @ v.T
        alpha = b_mat - probe_mat

        print(f"alpha sing layer {layer_idx}: {alpha}")
        
        v_mat = v.expand(x.shape[0], x.shape[1], -1)
        e = alpha.unsqueeze(-1) * v_mat

        print(f"x shape; {x.shape}")
        print(f"vmat shape; {v_mat.shape}")
        print(f"e shape; {e.shape}")
        print(f"K shape; {self.K_sing.shape}")

        u_t = e @ self.K_sing[layer_idx].T

        # print(f"multi layer {layer_idx} u_t: {u_t}")
        if isinstance(output,tuple):
            # print(f"output shape: {output[0][...,-u_t.shape[1]:,:].shape}")
            # print(f"u shape: {u_t.shape}")
            print(f"u in text sing: {u_t}")
            output[0][...,:] = output[0] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
        else: 
            raise ValueError("yoinkeeee")
            # output[...,-1,:] = output[...,-1,:] + u_t
        # print(f"output: {output}")
        return output



    def hook_setpoint_tracking_multi_all(self, layer_idx, module, args, kwargs, output):
        x = kwargs["hidden_states"]
        v = self.E_unit_multi[layer_idx]

        b_mat = th.tensor(self.betas_multi[layer_idx], device=self.device).T
        
        print(f"x shape: {x.shape}")
        print(f"v shape: {v.shape}")
        print(f"b_mat shape: {b_mat.shape}")
        
        probe_mat = (x * v).sum(-1)

        print(f"probe_mat shape: {probe_mat.shape}")

        alpha = b_mat - probe_mat
        print(f"alpha shape: {alpha.shape}")


        # v_mat = v.expand(x.shape[0], x.shape[1], -1)
        e = (alpha.unsqueeze(-1) * v).to(th.bfloat16)
        print(f"e shape: {e.shape}")
        # print(f"vmat dtype: {v_mat.dtype}")
        # print(f"e dtype: {e.dtype}")

        u_t = e @ self.K_multi[layer_idx].T


        print(f"multi layer {layer_idx} u_t shape: {u_t.shape}")
        if isinstance(output,tuple):
            # output[1][...,-1,:] = output[1][...,-1,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
            output[1][...,:] = output[1][...,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
        else: 
            raise ValueError("yoimultinkeeee")
            output[...,-1,:] = output[...,-1,:] + u_t
        # print(f"output: {output}")
        return output


    def hook_setpoint_tracking_sing_all(self, layer_idx, module, args, kwargs, output):
        # assume E_normed is unit vector in direction of contrastive feature
        # print(f"kwargs keys: {kwargs.keys()}")
        print(f"kwargs: {kwargs}")
        print(f"args: {args}")
        x = kwargs["hidden_states"]#[...,-1,:]
        # self.X[layer_idx] = x[-1,:]

        v = self.E_unit_sing[layer_idx]

        b_mat = th.tensor(self.betas_sing[layer_idx], device=self.device).T
        
        probe_mat = (x * v).sum(-1)

        alpha = b_mat - probe_mat
        # v_mat = v.expand(x.shape[0], x.shape[1], -1)
        e = (alpha.unsqueeze(-1) * v).to(th.bfloat16)
        print(f"x multi shape: {x.shape}")
        # print(f"vmat dtype: {v_mat.dtype}")
        # print(f"e dtype: {e.dtype}")

        u_t = e @ self.K_sing[layer_idx].T

        # print(f"multi layer {layer_idx} u_t: {u_t}")
        if isinstance(output,tuple):
            # output[1][...,-1,:] = output[1][...,-1,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
            print(f"output shape: {output[1][...,-u_t.shape[1]:,:].shape}")
            print(f"u shape: {u_t.shape}")
            output[1][...,:] = output[1][...,-u_t.shape[1]:,:] + u_t # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
        else: 
            raise ValueError("yoinkeeee")
            # output[...,-1,:] = output[...,-1,:] + u_t
        # print(f"output: {output}")
        return output



    def hook_add_text_sing(self, layer_idx, module, args, kwargs, output):
        print(f"in add text sing layer: {layer_idx}")
        scale = self.sing_scale
        if isinstance(output,tuple):
            # print(f"e sing text shape {self.E_sing_txt[layer_idx+1].shape}")
            output[0][...,:] = output[0] + scale * self.E_sing_txt[layer_idx+1] # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
            output[1][...,:] = output[1] + scale * self.E_sing[layer_idx+1] # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
        else: 
            raise ValueError("yoinkeeee")
        return output


    def hook_add_text_multi(self, layer_idx, module, args, kwargs, output):
        print(f"in add text multi layer: {layer_idx}")
        scale = self.multi_scale
        if isinstance(output,tuple):
            # print(f"e img shape {self.E_multi[layer_idx+1].shape}")
            output[0][...,:] = output[0][...,:] + scale * self.E_multi_txt[layer_idx+1] # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
            output[1][...,:] = output[1][...,:] + scale * self.E_multi[layer_idx+1] # TODO: verify that output[1] corresponds to image tokens (also applicable for data handling)
        else: 
            raise ValueError("yoimultinkeeee")
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
                                # return self.hook_setpoint_tracking_multi(layer_idx, module, args, kwargs, output)
                                # return self.hook_setpoint_tracking_multi_text(layer_idx, module, args, kwargs, output)
                                return self.hook_setpoint_tracking_multi_all(layer_idx, module, args, kwargs, output)


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
                                # return self.hook_setpoint_tracking_sing(layer_idx, module, args, kwargs, output)
                                # return self.hook_setpoint_tracking_sing_text(layer_idx, module, args, kwargs, output)
                                return self.hook_setpoint_tracking_sing_all(layer_idx, module, args, kwargs, output)


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

    @contextmanager
    def addadd_hooks(
            self,
            sing_list,
            multi_list
        ):
            handles = []
            try:
                
                if self.steer_multi:
                    for layer_idx, layer in enumerate(self.pipe.transformer.transformer_blocks):
                        def hook_wrapper(layer_idx):
                            # def hook(module, input, output):
                                # return self.hook_setpoint_tracking(layer_idx, module, input, output)
                            def hook(module, args, kwargs, output):
                                # return self.hook_setpoint_tracking_multi(layer_idx, module, args, kwargs, output)
                                return self.hook_add_text_multi(layer_idx, module, args, kwargs, output)


                            return hook

                        if layer_idx in multi_list:
                            handles.append(
                                layer.register_forward_hook(
                                    hook_wrapper(layer_idx), with_kwargs=True
                                )
                            )
                if self.steer_single:
                    for layer_idx, layer in enumerate(self.pipe.transformer.single_transformer_blocks):
                        def hook_wrapper(layer_idx):
                            def hook(module, args, kwargs, output):
                                return self.hook_add_text_sing(layer_idx, module, args, kwargs, output)
                            return hook

                        if layer_idx in sing_list:
                            handles.append(
                                layer.register_forward_hook(
                                    hook_wrapper(layer_idx), with_kwargs=True
                                )
                            )
                yield
            finally:
                for h in handles:
                    h.remove()


    def track_setpoint(self, prompt, steer_multi, steer_single, lmbda=1, do_sample=False, temp=1):
        self.steer_multi = steer_multi
        self.steer_single = steer_single

        # print(f"num multi layers: {self.pipe.transformer.transformer_blocks}")

        self.E_unit_sing = th.zeros_like(self.E_sing, dtype=th.bfloat16)
        self.E_unit_multi = th.zeros_like(self.E_multi, dtype=th.bfloat16)
        self.betas_sing = [0 for i in range(self.T_single+1)]
        self.betas_multi = [0 for i in range(self.T_multi+1)]
        for i, e in enumerate(self.E_multi):
            # print(f"e in setpoint: {e}")
            nrm = th.linalg.norm(e)
            # print(f"nrm in setpoint: {nrm}")
            if nrm == 0:
                self.E_unit_multi[i] = self.E_unit_multi[i]*0
            else:
                self.E_unit_multi[i] = e / nrm
                self.betas_multi[i] = (lmbda * nrm).to(th.float64).to(th.bfloat16)

        for i, e in enumerate(self.E_sing):
            # print(f"e in setpoint: {e}")
            nrm = th.linalg.norm(e)
            # print(f"nrm in setpoint: {nrm}")
            if nrm == 0:
                self.E_unit_sing[i] = self.E_unit_sing[i]*0
            else:
                self.E_unit_sing[i] = e / nrm
                self.betas_sing[i] = (lmbda * nrm).to(th.bfloat16)


        with self.add_hooks():
            image = self.pipe(
                prompt,
                guidance_scale=0.0,
                num_inference_steps=self.num_inference_steps,
                max_sequence_length=10,
                generator=th.Generator(self.device).manual_seed(0)
            ).images[0]

        # output_str = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        return image

    def track_setpoint_all_patches(self, prompt, steer_multi, steer_single, lmbda=1, do_sample=False, temp=1):
        self.steer_multi = steer_multi
        self.steer_single = steer_single

        # print(f"num multi layers: {self.pipe.transformer.transformer_blocks}")

        self.E_unit_sing = th.zeros_like(self.E_sing, dtype=th.bfloat16)
        self.E_unit_multi = th.zeros_like(self.E_multi, dtype=th.bfloat16)
        self.betas_sing = [th.zeros(self.E_sing.shape[1]) for i in range(self.T_single+1)]
        # self.betas_multi = [0 for i in range(self.T_multi+1)]
        self.betas_multi = [th.zeros(self.E_multi.shape[1]) for i in range(self.T_multi+1)]
        for i, e in enumerate(self.E_multi):
            for j, esub in enumerate(e):
                # print(f"e in setpoint: {e}")
                nrm = th.linalg.norm(esub)
                # print(f"nrm in setpoint: {nrm}")
                if nrm == 0:
                    self.E_unit_multi[i][j] = self.E_unit_multi[i][0]*0
                else:
                    self.E_unit_multi[i][j] = esub / nrm
                    self.betas_multi[i][j] = (lmbda * nrm).to(th.float64).to(th.bfloat16)

        for i, e in enumerate(self.E_sing):
            for j, esub in enumerate(e):
                # print(f"e in setpoint: {e}")
                nrm = th.linalg.norm(esub)
                # print(f"nrm in setpoint: {nrm}")
                if nrm == 0:
                    self.E_unit_sing[i][j] = self.E_unit_multi[i][0]*0
                else:
                    self.E_unit_sing[i][j] = esub / nrm
                    self.betas_sing[i][j] = (lmbda * nrm).to(th.float64).to(th.bfloat16)


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
        
    def addddddderoolllllllolllllllllllll(self, prompt, steer_multi, steer_single, multi_list=[5], sing_list=[5], multi_scale=0.01, sing_scale=0.015, do_sample=False, temp=1):
        self.steer_multi = steer_multi
        self.steer_single = steer_single

        self.sing_scale = sing_scale
        self.multi_scale = multi_scale


        with self.addadd_hooks(sing_list, multi_list):
            image = self.pipe(
                prompt,
                guidance_scale=0.0,
                num_inference_steps=self.num_inference_steps,
                max_sequence_length=10,
                generator=th.Generator(self.device).manual_seed(0)
            ).images[0]

        # output_str = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        return image
        