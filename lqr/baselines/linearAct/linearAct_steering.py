import pickle
import torch as th
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml

with open("../../config/config.yaml", "r") as f:
    config_data = yaml.safe_load(f)
PICKLE_JAR = config_data["environment"]["pickle_jar"]


class LinearActSteering:
    """
    Apply LinearAct maps at inference:
      T(a) = omega * a + beta
      a' = (1-lambda) a + lambda T(a)

    If use_support=True, only apply when a in [min_src, max_src] (q_0_100).
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        state_name: str,
        intervention_position: str = "all",
        strength: float = 1.0,
        use_support: bool = True,
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.tokenizer = tokenizer
        self.intervention_position = intervention_position

        # with open(PICKLE_JAR + state_name + ".pkl", "rb") as f:
        with open( state_name + ".pkl", "rb") as f:
            state = pickle.load(f)

        self.omega = state["omega"].to(self.device)   # (T,n)
        self.beta = state["beta"].to(self.device)     # (T,n)
        self.min_src = state["min_src"].to(self.device)
        self.max_src = state["max_src"].to(self.device)

        self.T = len(model.model.layers)
        self.n = model.model.embed_tokens.embedding_dim
        assert self.omega.shape == (self.T, self.n)

        self.lmbda = float(strength)
        self.use_support = bool(use_support)
        self.hooks = []

    # def _prehook(self, layer_idx: int):
    #     def hook(module, inputs):
    #         hidden = inputs[0]  # (B, L, n)
    #         w = self.omega[layer_idx][None, None, :]
    #         b = self.beta[layer_idx][None, None, :]
    #         transported = w * hidden + b
    #         out = (1.0 - self.lmbda) * hidden + self.lmbda * transported

    #         if self.use_support:
    #             lo = self.min_src[layer_idx][None, None, :]
    #             hi = self.max_src[layer_idx][None, None, :]
    #             in_support = (hidden >= lo) & (hidden <= hi)
    #             out = th.where(in_support, out, hidden)

    #         return (out,) + inputs[1:]
    #     return hook
    def _posthook(self, layer_idx: int):
        def hook(module, inputs, output):
            hidden = output  # (B, L, n)

            w = self.omega[layer_idx][None, None, :]
            b = self.beta[layer_idx][None, None, :]
            transported = w * hidden + b
            out = (1.0 - self.lmbda) * hidden + self.lmbda * transported

            if self.use_support:
                lo = self.min_src[layer_idx][None, None, :]
                hi = self.max_src[layer_idx][None, None, :]
                in_support = (hidden >= lo) & (hidden <= hi)
                out = th.where(in_support, out, hidden)

            return out
        return hook


    # def __enter__(self):
    #     for li, layer in enumerate(self.model.model.layers):
    #         self.hooks.append(layer.register_forward_pre_hook(self._prehook(li)))
    #     return self
    def __enter__(self):
        for li, layer in enumerate(self.model.model.layers):
            ln = layer.post_attention_layernorm
            self.hooks.append(ln.register_forward_hook(self._posthook(li)))
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    @th.no_grad()
    def track_setpoint(self, prompts, max_new_tokens, lmbda=1.0, do_sample=True, temp=0.7):
        self.lmbda = float(lmbda)

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with self:
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                do_sample=do_sample,
                temperature=temp,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
