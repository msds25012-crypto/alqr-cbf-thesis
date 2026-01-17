import torch as th
from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import Enum


class Mode(Enum):
    SETPOINT = 0


class ActAddSteering:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        contrastive_vecs: th.Tensor,
        layer_idx: int,
    ):

        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer
        self.T = len(model.model.layers)
        self.n = model.model.embed_tokens.embedding_dim
        if layer_idx < 0 or layer_idx > self.T: #check
            raise ValueError(f"layer_idx must be in [0, {self.T - 1}]")
        self.layer_idx = layer_idx
        self.E = contrastive_vecs[self.layer_idx].to(self.device)
        if self.E.dim() != 2 or self.E.shape[1] != self.n:
            raise ValueError(
                "contrastive_vecs must have shape (num_layers, seq_len, hidden_dim)"
            )
        self.seq_len = self.E.shape[0]

        self.U = th.zeros((self.T, self.seq_len, self.n), device=self.device)
        self.hooks = []
        self.mode = Mode.SETPOINT
        self.lmbda = 0.0
        self.has_applied = False
    
    def hook_actadd_pre(self, module, inputs):
        """
        Forward pre-hook: modify the layer input hidden_states.
        inputs is a tuple; inputs[0] is usually hidden_states: (batch, seq, hidden)
        """
        if self.has_applied:
            return inputs
        hidden = inputs[0]
        u_t = self.lmbda * self.E  # (seq_len, hidden_dim)
        self.U[self.layer_idx] = u_t
        apply_len = min(self.seq_len, hidden.shape[-2])
        hidden2 = hidden.clone()
        hidden2[..., :apply_len, :] = hidden2[..., :apply_len, :] + u_t[:apply_len]
        self.has_applied = True
        return (hidden2,) + inputs[1:]

    def register_setpoint_hooks(self):
        layer = self.model.model.layers[self.layer_idx]
        self.hooks.append(
            layer.register_forward_pre_hook(
                self.hook_actadd_pre
            )
        )

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __enter__(self):
        if self.mode == Mode.SETPOINT:
            self.register_setpoint_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def track_setpoint(
        self,
        prompt,
        max_new_tokens,
        lmbda=1,
        do_sample=False,
        temp=1.0,
        top_p=0.3,
        repetition_penalty=1.2,
    ):
        self.mode = Mode.SETPOINT
        self.lmbda = lmbda
        self.has_applied = False

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with self:
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                do_sample=do_sample,
                temperature=temp,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                # use_cache=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        output_str = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        return output_str
