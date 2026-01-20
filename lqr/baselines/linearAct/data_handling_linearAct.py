import random
import pickle
import torch as th
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml

with open("../../config/config.yaml", "r") as f:
    config_data = yaml.safe_load(f)
PICKLE_JAR = config_data["environment"]["pickle_jar"]

MODEL_CONFIGS = {
    "llama1b": {
        "hf_name": "meta-llama/Llama-3.2-1B",
        "tensor_prefix": "llama-3.2-1b",
    },
    "gemma2b": {
        "hf_name": "google/gemma-2-2b",
        "tensor_prefix": "gemma-2-2b",
    },
    "qwen3b": {
        "hf_name": "Qwen/Qwen2.5-3B",
        "tensor_prefix": "qwen-2.5-3b",
    },
    "llama8b": {
        "hf_name": "meta-llama/Meta-Llama-3-8B",
        "tensor_prefix": "llama-3-8b",
    },
    "gemma9b": {
        "hf_name": "google/gemma-2-9b",
        "tensor_prefix": "gemma-2-9b",
    },
    "qwen14b": {
        "hf_name": "Qwen/Qwen2.5-14B",
        "tensor_prefix": "qwen-2.5-14b",
    },
}


class PartialLinearActApplier:
    """
    Applies already-fit LinearAct maps on a prefix of layers (0..max_layer-1)
    during forward passes used for fitting later layers causally.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        omega: th.Tensor,     # (T, n)
        beta: th.Tensor,      # (T, n)
        min_src: th.Tensor,   # (T, n)
        max_src: th.Tensor,   # (T, n)
        max_layer: int,
        strength: float = 1.0,
        use_support: bool = True,
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.omega = omega.to(self.device)
        self.beta = beta.to(self.device)
        self.min_src = min_src.to(self.device)
        self.max_src = max_src.to(self.device)
        self.max_layer = int(max_layer)
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
    #     for li in range(self.max_layer):
    #         layer = self.model.model.layers[li]
    #         self.hooks.append(layer.register_forward_pre_hook(self._prehook(li)))
    #     return self
    def __enter__(self):
        for li in range(self.max_layer):
            layer = self.model.model.layers[li]
            ln = layer.post_attention_layernorm
            self.hooks.append(ln.register_forward_hook(self._posthook(li)))
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.hooks:
            h.remove()
        self.hooks = []


class LayerActivationCollector:
    """
    Collect pooled activations for one layer at a time.
    returns Z: (N, n) where each row is mean over tokens of layer input hidden_states.
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.device = next(model.parameters()).device
        self.tokenizer = tokenizer
        self.T = len(self.model.model.layers)
        self.n = self.model.model.embed_tokens.embedding_dim
        self.hook_handle = None
        self._buf = None  # stores hidden_states captured at the chosen layer

    # def _hook(self, module, inputs, output):
    #     # capture layer *input* hidden_states (residual stream in)
    #     self._buf = inputs[0]  # (B, L, n)
    #     return output
    def _hook(self, module, inputs, output):
        self._buf = output  # (B, L, n)
        return output


    # def _register(self, layer_idx: int):
    #     layer = self.model.model.layers[layer_idx]
    #     self.hook_handle = layer.register_forward_hook(self._hook)
    def _register(self, layer_idx: int):
        layer = self.model.model.layers[layer_idx]
        ln = layer.post_attention_layernorm
        self.hook_handle = ln.register_forward_hook(self._hook)


    def _remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    @th.no_grad()
    def collect_layer_samples(
        self,
        prompts,
        num_samples: int,
        layer_idx: int,
        batch_size: int = 32,
        seed: int = 0,
        apply_partial: PartialLinearActApplier = None,
        pooling_op: str = "mean",
        truncation: bool = True,
        max_length: int = None,
    ) -> th.Tensor:
        """
        Returns Z: (N, n).
        """
        # assert pooling_op == "mean", "Only mean pooling implemented (matches pooling_op: mean)."
        rnd = random.Random(seed)
        samples = rnd.sample(prompts, min(num_samples, len(prompts)))

        Z_chunks = []
        self._register(layer_idx)

        for i in range(0, len(samples), batch_size):
            batch_prompts = samples[i : i + batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
                max_length=max_length,
            ).to(self.device)

            self._buf = None

            ctx = apply_partial if apply_partial is not None else _NullCtx()
            with ctx:
                _ = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                )

            hidden = self._buf
            if hidden is None:
                raise RuntimeError("Hook did not capture hidden states. Check layer hook placement.")

            mask = inputs["attention_mask"].float()  # (B, L)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,1)

            pooled = (hidden * mask[:, :, None]).sum(dim=1) / denom  # (B,n)
            Z_chunks.append(pooled.detach().cpu().float())

        self._remove()
        Z = th.cat(Z_chunks, dim=0)  # (N,n)
        return Z


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): return False


def save_linearact_state(filename: str, state: dict):
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(state, f)
    print(f"Saved {filename}.pkl")


def load_linearact_state(filename: str) -> dict:
    with open(filename + ".pkl", "rb") as f:
        return pickle.load(f)


def resolve_model_name(model_key_or_name: str) -> str:
    config = MODEL_CONFIGS.get(model_key_or_name)
    if config:
        return config["hf_name"]
    return model_key_or_name
