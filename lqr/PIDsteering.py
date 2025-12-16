import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import lqr_utils_seq as lqr
from functools import partial
from enum import Enum
from tqdm import tqdm
import torch.nn.functional as F
import time

class Mode(Enum):
    STEERING = 0
    SETPOINT = 1

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
        self.E = contrastive_vecs

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

        v = self.E_unit[layer_idx]
        alpha = th.tensor([self.betas[layer_idx] for i in range(x.shape[0])], device=self.device) - th.bmm(v.unsqueeze(0).unsqueeze(0), th.transpose(x.unsqueeze(0),-2,-1))
        e = alpha.squeeze(0).T @ v.unsqueeze(0)
        self.e_sum += e

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

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __enter__(self):
        if self.mode == Mode.STEERING:
            self.register_steering_hooks()
        elif self.mode == Mode.SETPOINT:
            self.register_setpoint_tracking_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def track_setpoint(self, prompt, max_new_tokens, lmbda=1, do_sample=False, temp=0.7):
        self.mode = Mode.SETPOINT

        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        # print(f"input_ids shape: {input_ids.shape}")
        attention_mask = inputs["attention_mask"]
        self.X = th.zeros((self.T+1, self.n)).to(self.device)
        self.e_sum = th.zeros((input_ids.shape[0], self.E[0].shape[0]), device=self.device)
        self.e_prev = th.zeros((input_ids.shape[0], self.E[0].shape[0]), device=self.device)

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
                temperature=temp,
                use_cache=False,
                pad_token_id=self.tokenizer.eos_token_id,
                # **model_generation_kwargs, #
            )

        # output_str = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        output_str = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        return output_str
        
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
        self.e_sum = th.zeros((BATCH_SZ, self.E[0].shape[0]), device=self.device)
        self.e_prev = th.zeros((BATCH_SZ, self.E[0].shape[0]), device=self.device)

        self.betas = [0 for i in range(self.T+1)]
        for i, e in enumerate(self.E):
            # print(f"e: {e}")
            nrm = th.linalg.norm(e)
            self.E_unit[i] = e / nrm
            self.betas[i] = lmbda * nrm
        # model.eval()

        total_nll = 0.0
        total_tokens = 0

        # for ind in tqdm(range(0, len(data), BATCH_SZ)):
        for ind in tqdm(range(0, 10, BATCH_SZ)):
            if not data[ind]:
                continue

            end_ind = min(ind + BATCH_SZ, len(data))

            encodings = self.tokenizer(
                data[ind:end_ind],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model.config.max_position_embeddings,
            ).to(self.device)

            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]

            with self:
                with th.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )

                    # Shift for causal LM
                    shift_logits = outputs.logits[:, :-1, :]
                    shift_labels = input_ids[:, 1:]
                    shift_mask = attention_mask[:, 1:]

                    # Token-level NLL (sum, not mean)
                    loss = F.cross_entropy(
                        shift_logits.reshape(-1, shift_logits.size(-1)),
                        shift_labels.reshape(-1),
                        ignore_index=self.tokenizer.pad_token_id,
                        reduction="sum",
                    )

                    # Count valid tokens
                    n_tokens = shift_mask.sum()

                    total_nll += loss
                    total_tokens += n_tokens

        # Final perplexity (HF definition)
        ppl = th.exp(total_nll / total_tokens)
        return ppl