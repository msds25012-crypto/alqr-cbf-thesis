import torch
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

with open('config/config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)
PICKLE_JAR = config_data["environment"]["pickle_jar"]
# print(PICKLE_JAR)

class ImageContrastiveBuilder:
    def __init__(
        self,
        pipe: FluxPipeline,
        num_inference_steps: int,

        # tokenizer: AutoTokenizer,
    ):
        self.pipe = pipe
        self.device = self.pipe.device
        print(f"model device: {self.device}")

        self.T_multi = len(self.pipe.transformer.transformer_blocks)
        self.T_single = len(self.pipe.transformer.single_transformer_blocks)
        self.n = self.pipe.transformer.transformer_blocks[0].attn.to_q.in_features

        self.B = 1
        self.X_text_multi = None
        self.X_img_multi = None

        self.X_text_sing = None
        self.X_img_sing = None
        
        self.A_sum = None
        self.X_sum = None
        self.X_mean = None

        self.sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if hasattr(self.pipe.scheduler.config, "use_flow_sigmas") and self.pipe.scheduler.config.use_flow_sigmas:
            self.sigmas = None
        # image_seq_len = X.shape[1] # TODO
        self.image_seq_len = 2
        self.num_inference_steps = num_inference_steps
        self.token_idx = -1


    # def hook_collector_multi(self, layer_idx, module, input, output):
    # # print(f"outie: {len(output)}")
    # # print(output[0].shape)
    # # print(output[1].shape)
    #     self.X_text_multi[layer_idx][..., 0,:] = output[0][-1][-1]
    #     self.X_img_multi[layer_idx][..., 0,:] = output[1][-1][-1]
    #     return output

    # def hook_collector_single(self, layer_idx, module, input, output):
    #     # print(f"outie: {len(output)}")
    #     # print(output[0].shape)
    #     # print(output[1].shape)
    #     self.X_text_sing[layer_idx][..., 0,:] = output[0][-1][-1]
    #     self.X_img_sing[layer_idx][..., 0,:] = output[1][-1][-1]
    #     return output
    
    def hook_collector_multi(self, layer_idx, module, args, kwargs, output):
    # print(f"outie: {len(output)}")
    # print(output[0].shape)
    # print(output[1].shape)
        self.X_text_multi[layer_idx][..., 0,:] = kwargs["encoder_hidden_states"][...,self.token_idx,:]
        self.X_img_multi[layer_idx][..., 0,:] = kwargs["hidden_states"][...,self.token_idx,:]
        
        if layer_idx == self.T_multi-1:
            self.X_text_multi[self.T_multi][..., 0,:] = output[0][-1][self.token_idx]
            self.X_img_multi[self.T_multi][..., 0,:] = output[1][-1][self.token_idx]
        return output

    def hook_collector_single(self, layer_idx, module, args, kwargs, output):
        # print(f"outie: {len(output)}")
        # print(output[0].shape)
        # print(output[1].shape)
        # self.X_text_sing[layer_idx][..., 0,:] = output[0][-1][-1]
        # self.X_img_sing[layer_idx][..., 0,:] = output[1][-1][-1]

        self.X_text_sing[layer_idx][..., 0,:] = kwargs["encoder_hidden_states"][...,self.token_idx,:]
        self.X_img_sing[layer_idx][..., 0,:] = kwargs["hidden_states"][...,self.token_idx,:]
        
        if layer_idx == self.T_single-1:
            self.X_text_sing[self.T_single][..., 0,:] = output[0][-1][self.token_idx]
            self.X_img_sing[self.T_single][..., 0,:] = output[1][-1][self.token_idx]
        return output
    

    def hook_collector_multi_img_all(self, layer_idx, module, args, kwargs, output):

        print("in multi all")
        print(kwargs["hidden_states"].shape)
        self.X_img_multi[layer_idx][..., :,:] = kwargs["hidden_states"][...,:,:]
        print("inbetween from multi all")
        
        if layer_idx == self.T_multi-1:
            self.X_img_multi[self.T_multi][..., :,:] = output[1][-1][:]
        print("returning from multi all")
        return output

    def hook_collector_single_img_all(self, layer_idx, module, args, kwargs, output):
        # print(f"outie: {len(output)}")
        # print(output[0].shape)
        # print(output[1].shape)
        # self.X_text_sing[layer_idx][..., 0,:] = output[0][-1][-1]
        # self.X_img_sing[layer_idx][..., 0,:] = output[1][-1][-1]

        print("in sing all")
        print(kwargs["hidden_states"].shape)
        self.X_img_sing[layer_idx][..., :,:] = kwargs["hidden_states"][...,:,:]
        print("inbetween from sing all")
        
        if layer_idx == self.T_single-1:
            self.X_img_sing[self.T_single][..., :,:] = output[1][-1][:]
        print("returning from sing all")
        return output


    @contextmanager
    def add_hooks(
            self
        ):
        """Context manager for temporarily adding forward hooks.

        Args:
            module_forward_pre_hooks: List of (module, hook_fn) tuples for pre-hooks
            module_forward_hooks: List of (module, hook_fn) tuples for forward hooks
            **kwargs: Additional keyword arguments passed to hook functions

        Yields:
            None. Hooks are active within the context, removed on exit.
        """
        # module_forward_pre_hooks = module_forward_pre_hooks or []
        # module_forward_hooks = module_forward_hooks or []
        handles = []
        try:

            for layer_idx, layer in enumerate(self.pipe.transformer.transformer_blocks):
                def hook_wrapper(layer_idx):
                    # def hook(module, input, output):
                    #     return self.hook_collector_multi(layer_idx, module, input, output)

                    def hook(module, args, kwargs, output):
                        return self.hook_collector_multi(layer_idx, module, args, kwargs, output)


                    return hook

                handles.append(
                    layer.register_forward_hook(
                        hook_wrapper(layer_idx), with_kwargs=True
                        # hook_wrapper(layer_idx)
                    )
                )
            for layer_idx, layer in enumerate(self.pipe.transformer.single_transformer_blocks):
                def hook_wrapper(layer_idx):
                    # def hook(module, input, output):
                    #     return self.hook_collector_single(layer_idx, module, input, output)
                    def hook(module, args, kwargs, output):
                        return self.hook_collector_single(layer_idx, module, args, kwargs, output)

                    return hook

                handles.append(
                    layer.register_forward_hook(
                        hook_wrapper(layer_idx), with_kwargs=True
                        # hook_wrapper(layer_idx)
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
    def add_hooks_img_all(
            self
        ):
        """Context manager for temporarily adding forward hooks.

        Args:
            module_forward_pre_hooks: List of (module, hook_fn) tuples for pre-hooks
            module_forward_hooks: List of (module, hook_fn) tuples for forward hooks
            **kwargs: Additional keyword arguments passed to hook functions

        Yields:
            None. Hooks are active within the context, removed on exit.
        """
        # module_forward_pre_hooks = module_forward_pre_hooks or []
        # module_forward_hooks = module_forward_hooks or []
        handles = []
        try:

            for layer_idx, layer in enumerate(self.pipe.transformer.transformer_blocks):
                def hook_wrapper(layer_idx):
                    # def hook(module, input, output):
                    #     return self.hook_collector_multi(layer_idx, module, input, output)

                    def hook(module, args, kwargs, output):
                        return self.hook_collector_multi_img_all(layer_idx, module, args, kwargs, output)


                    return hook

                handles.append(
                    layer.register_forward_hook(
                        hook_wrapper(layer_idx), with_kwargs=True
                        # hook_wrapper(layer_idx)
                    )
                )
            for layer_idx, layer in enumerate(self.pipe.transformer.single_transformer_blocks):
                def hook_wrapper(layer_idx):
                    # def hook(module, input, output):
                    #     return self.hook_collector_single(layer_idx, module, input, output)
                    def hook(module, args, kwargs, output):
                        return self.hook_collector_single_img_all(layer_idx, module, args, kwargs, output)

                    return hook

                handles.append(
                    layer.register_forward_hook(
                        hook_wrapper(layer_idx), with_kwargs=True
                        # hook_wrapper(layer_idx)
                    )
                )
            # for module, hook in module_forward_hooks:
                # partial_hook = functools.partial(hook, **kwargs)
                # handles.append(module.register_forward_hook(partial_hook))
            yield
        finally:
            for h in handles:
                h.remove()


    def linearize(self, tfs, X_nom):
        """
        Linearize nonlinear dynamics f around nominal trajectory (X_nom, U_nom=0).

        Args:
            f: dynamics function f(x,u) -> x_next
            X_nom: nominal states (T+1, k, n)
            U_nom: nominal controls (T, m)

        Returns:
            A: linearized A matrices (T, n, n)
            B: linearized B matrices (T, n, m)
        """
        T = len(tfs)
        n = X_nom.shape[-1]
        # print(X_nom.shape)

        A = torch.zeros((T, n, n), dtype=X_nom.dtype, device=X_nom.device)
        # B = th.zeros((T, n, m), dtype=X_nom.dtype, device=X_nom.device)

        for t in range(T):
            x = X_nom[t].detach().requires_grad_(True)

            def f_last(x):
                xout = tfs[t](x)[1][..., -1, :]
                print(xout)
                return xout.squeeze()
            # Compute Jacobians:
            Jx = torch.autograd.functional.jacobian(lambda x_: f_last(x_), x, create_graph=False, vectorize=True)   # shape: [n, *x.shape]
            # Ju = th.autograd.functional.jacobian(lambda u_: f_last(x, u_), u, create_graph=False, vectorize=True)   # shape: [n, *u.shape]
            print(Jx.shape)
            A[t] = Jx[...,-1,-1,:]
            # B[t] = Ju

        return A.detach().cpu()


    def flux_block_wrapper(self, block, encoder_hidden_states, temb, x):
        return block(hidden_states=x, encoder_hidden_states=encoder_hidden_states, temb=temb)


    def flux_block_wrapper_txt(self, block, hidden_states, temb, x):
        return block(hidden_states=hidden_states, encoder_hidden_states=x, temb=temb)


    def retrieve_timesteps(
        self,
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
    ):
        r"""
        Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
        custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

        Args:
            scheduler (`SchedulerMixin`):
                The scheduler to get timesteps from.
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
                must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
                `num_inference_steps` and `sigmas` must be `None`.
            sigmas (`List[float]`, *optional*):
                Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
                `num_inference_steps` and `timesteps` must be `None`.

        Returns:
            `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
        """
        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        elif sigmas is not None:
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps  

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def collect_data_batch(self, prompts, num_samples, filename, num_tokens=1, batch_size=50):
        mu = self.calculate_shift(
            self.image_seq_len,
            self.pipe.scheduler.config.get("base_image_seq_len", 256),
            self.pipe.scheduler.config.get("max_image_seq_len", 4096),
            self.pipe.scheduler.config.get("base_shift", 0.5),
            self.pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, self.num_inference_steps = self.retrieve_timesteps(
            self.pipe.scheduler,
            self.num_inference_steps,
            self.device,
            sigmas=self.sigmas,
            mu=mu,
        )

        X_sum_img_multi = torch.zeros((self.T_multi+1, self.n,)).to(self.device)
        X_sum_img_sing = torch.zeros((self.T_single+1, self.n,)).to(self.device)
        X_sum_txt_multi = torch.zeros((self.T_multi+1, self.n,)).to(self.device)
        X_sum_txt_sing = torch.zeros((self.T_single+1, self.n,)).to(self.device)


        samples = random.sample(prompts, num_samples)
        for i in range(0,len(samples), batch_size):
            sample = samples[i:i+batch_size]
            self.B = len(sample)

            self.X_img_sing = torch.zeros((self.T_single+1, self.B, 1, self.n), device=self.device)
            self.X_text_sing = torch.zeros((self.T_single+1, self.B, 1, self.n), device=self.device)
            self.X_img_multi = torch.zeros((self.T_multi+1, self.B, 1, self.n), device=self.device)
            self.X_text_multi = torch.zeros((self.T_multi+1, self.B, 1, self.n), device=self.device)


            with self.add_hooks():
                image = self.pipe(
                    sample,
                    guidance_scale=0.0,
                    num_inference_steps=self.num_inference_steps,
                    max_sequence_length=256,
                    generator=torch.Generator(self.device).manual_seed(0)
                ).images[0]

            print(self.X_img_multi.shape)
            print(torch.sum(self.X_img_multi, dim=1).shape)
            print(X_sum_img_multi.shape)
            X_sum_img_multi += torch.sum(self.X_img_multi, dim=1).squeeze(1)
            X_sum_img_sing += torch.sum(self.X_img_sing, dim=1).squeeze(1)
            X_sum_txt_multi += torch.sum(self.X_text_multi, dim=1).squeeze(1)
            X_sum_txt_sing += torch.sum(self.X_text_sing, dim=1).squeeze(1)

            # X_mean = th.mean(self.X[:,:,-1,:], dim = 1)

        total = num_samples*num_tokens
        print(f"total: {total}")
        print(f"X_multi shape: {X_sum_img_multi.shape}")
        print(f"X_sing shape: {X_sum_img_sing.shape}")

        tensor_dict = {
            "X_multi": X_sum_img_multi / len(samples),
            "X_txt_multi": X_sum_txt_multi / len(samples),
            "X_sing": X_sum_img_sing / len(samples),
            "X_txt_sing": X_sum_txt_sing / len(samples),
        } 
        with open(PICKLE_JAR + filename + ".pkl", "wb") as f:
            pickle.dump(tensor_dict, f)
        
        
    def collect_all_image_batch(self, prompts, num_samples, filename, num_tokens=1, batch_size=50):
        image_seq_len = self.pipe.scheduler.config.get("max_image_seq_len", 4096)
        mu = self.calculate_shift(
            self.image_seq_len,
            self.pipe.scheduler.config.get("base_image_seq_len", 256),
            self.pipe.scheduler.config.get("max_image_seq_len", 4096),
            self.pipe.scheduler.config.get("base_shift", 0.5),
            self.pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, self.num_inference_steps = self.retrieve_timesteps(
            self.pipe.scheduler,
            self.num_inference_steps,
            self.device,
            sigmas=self.sigmas,
            mu=mu,
        )

        X_sum_img_multi = torch.zeros((self.T_multi+1, image_seq_len, self.n,)).to(self.device)
        X_sum_img_sing = torch.zeros((self.T_single+1, image_seq_len, self.n,)).to(self.device)

        print(f"X sum shape: {X_sum_img_multi.shape}")

        samples = random.sample(prompts, num_samples)
        for i in range(0,len(samples), batch_size):
            sample = samples[i:i+batch_size]
            self.B = len(sample)

            self.X_img_sing = torch.zeros((self.T_single+1, self.B, image_seq_len, self.n), device=self.device)
            self.X_img_multi = torch.zeros((self.T_multi+1, self.B, image_seq_len, self.n), device=self.device)


            with self.add_hooks_img_all():
                image = self.pipe(
                    sample,
                    guidance_scale=0.0,
                    num_inference_steps=self.num_inference_steps,
                    max_sequence_length=256,
                    generator=torch.Generator(self.device).manual_seed(0)
                ).images[0]

            print(self.X_img_multi.shape)
            # print(torch.sum(self.X_img_multi, dim=1).shape)
            print(X_sum_img_multi.shape)
            X_sum_img_multi += torch.sum(self.X_img_multi, dim=1)
            X_sum_img_sing += torch.sum(self.X_img_sing, dim=1)

            # X_mean = th.mean(self.X[:,:,-1,:], dim = 1)

        total = num_samples*num_tokens
        print(f"total: {total}")
        print(f"X_multi shape: {X_sum_img_multi.shape}")
        print(f"X_sing shape: {X_sum_img_sing.shape}")

        tensor_dict = {
            "X_multi": X_sum_img_multi / len(samples),
            "X_sing": X_sum_img_sing / len(samples),
        } 
        with open(PICKLE_JAR + filename + ".pkl", "wb") as f:
            pickle.dump(tensor_dict, f)
        

    
    def collect_jacobians(self, prompts, num_samples, filename, num_tokens=1, max_ctx=512, collect_txt=True, collect_img=False): # 24 works for llama 8-9b
        mu = self.calculate_shift(
            self.image_seq_len,
            self.pipe.scheduler.config.get("base_image_seq_len", 256),
            self.pipe.scheduler.config.get("max_image_seq_len", 4096),
            self.pipe.scheduler.config.get("base_shift", 0.5),
            self.pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, self.num_inference_steps = self.retrieve_timesteps(
            self.pipe.scheduler,
            self.num_inference_steps,
            self.device,
            sigmas=self.sigmas,
            mu=mu,
        )

        timestep = timesteps[0].expand(self.B) # maybe needs dtype

        

        samples = random.sample(prompts, num_samples)
        iter = 1


        if collect_img:
            A_sing_sum = torch.zeros((self.T_single, self.n, self.n,))#.to(self.device)
            A_multi_sum = torch.zeros((self.T_multi, self.n, self.n,))#.to(self.device)
        if collect_txt:
            A_sing_sum_txt = torch.zeros((self.T_single, self.n, self.n,))#.to(self.device)
            A_multi_sum_txt = torch.zeros((self.T_multi, self.n, self.n,))#.to(self.device)

        for prompt in samples:
            print(f"iter: {iter}")
            iter += 1

            self.X_img_sing = torch.zeros((self.T_single+1, self.B, 1, self.n), device=self.device)
            self.X_text_sing = torch.zeros((self.T_single+1, self.B, 1, self.n), device=self.device)
            self.X_img_multi = torch.zeros((self.T_multi+1, self.B, 1, self.n), device=self.device)
            self.X_text_multi = torch.zeros((self.T_multi+1, self.B, 1, self.n), device=self.device)

            pooled_projections = self.pipe._get_clip_prompt_embeds(prompt)
            encoder_hidden_states,pooled_projections,_ = self.pipe.encode_prompt(prompt)
            guidance = None
            encoder_hidden_states = self.pipe.transformer.context_embedder(encoder_hidden_states)

            temb = (
                        self.pipe.transformer.time_text_embed(timestep, pooled_projections)
                        if guidance is None
                        else self.pipe.transformer.time_text_embed(timestep, guidance, pooled_projections)
                    )

            with torch.no_grad():
                with self.add_hooks():
                    image = self.pipe(
                        prompt,
                        guidance_scale=0.0,
                        num_inference_steps=self.num_inference_steps,
                        max_sequence_length=256,
                        generator=torch.Generator(self.device).manual_seed(0)
                    ).images[0]
            
            # self.X_sum = self.X_sum + self.X[:,-1,:]

            if collect_img:
                wrapper_list = []
                for i, tf in enumerate(self.pipe.transformer.transformer_blocks):
                    wrapper_list.append(partial(self.flux_block_wrapper, tf, self.X_img_multi[i], temb))

                wrapper_list2 = []
                for i, tf in enumerate(self.pipe.transformer.single_transformer_blocks):
                    wrapper_list2.append(partial(self.flux_block_wrapper, tf, self.X_img_sing[i], temb))


                # wrapper_list = [partial(flux_block_wrapper, tf, encoder_hidden_states, temb) for tf in pipe.transformer.transformer_blocks]

                A_multi = self.linearize(wrapper_list, self.X_img_multi)
                print(f"A multi shape: {A_multi.shape}")
                A_multi_sum += A_multi
                print(f"A_multi_sum shape: {A_multi_sum.shape}")


                A_sing = self.linearize(wrapper_list2, self.X_img_sing)
                print(f"A sing shape: {A_sing.shape}")
                A_sing_sum += A_sing
                print(f"A_sing_sum shape: {A_sing_sum.shape}")

            if collect_txt:
                wrapper_list = []
                for i, tf in enumerate(self.pipe.transformer.transformer_blocks):
                    wrapper_list.append(partial(self.flux_block_wrapper_txt, tf, self.X_img_multi[i], temb))

                wrapper_list2 = []
                for i, tf in enumerate(self.pipe.transformer.single_transformer_blocks):
                    wrapper_list2.append(partial(self.flux_block_wrapper_txt, tf, self.X_img_sing[i], temb))


                # wrapper_list = [partial(flux_block_wrapper, tf, encoder_hidden_states, temb) for tf in pipe.transformer.transformer_blocks]

                A_multi = self.linearize(wrapper_list, self.X_text_multi)
                print(f"A multi shape: {A_multi.shape}")
                A_multi_sum_txt += A_multi
                print(f"A_multi_sum_txt shape: {A_multi_sum_txt.shape}")


                A_sing = self.linearize(wrapper_list2, self.X_text_sing)
                print(f"A sing shape: {A_sing.shape}")
                A_sing_sum_txt += A_sing
                print(f"A_sing_sum_txt shape: {A_sing_sum_txt.shape}")
            


        total = len(samples)

        tensor_dict = {}

        if collect_img:
            tensor_dict["A_img_multi"] = A_multi_sum / total,
            tensor_dict["A_img_sing"] = A_sing_sum / total,
        if collect_txt:
            tensor_dict["A_txt_multi"] = A_multi_sum_txt / total
            tensor_dict["A_txt_sing"] = A_sing_sum_txt / total

        with open(PICKLE_JAR + filename + ".pkl", "wb") as f:
            pickle.dump(tensor_dict, f)
