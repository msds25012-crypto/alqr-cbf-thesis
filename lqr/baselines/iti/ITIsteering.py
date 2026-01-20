import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

import pyvene as pv
from interveners import ITI_Intervener, wrapper
from utils import get_interventions_dict, train_probes
from data_handling_iti import ModelInfo, load_iti_artifact, encode_prompt


@dataclass
class ProbeFit:
    probes: Sequence
    head_order: np.ndarray  # flattened head indices sorted by accuracy desc
    x_train: np.ndarray
    x_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray


class ITISteering:
    """
    A clean wrapper around your existing ITI pipeline:

    - load artifact (activations/labels/info)
    - train per-head logistic probes
    - rank heads
    - build a pyvene IntervenableModel that injects ITI_Intervener at selected layers
    """

    def __init__(
        self,
        *,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        info: ModelInfo,
        device: str,
        fit: ProbeFit,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.info = info
        self.device = device
        self.fit = fit

    @classmethod
    def from_artifact(
        cls,
        *,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        artifact_path: str,
        device: Optional[str] = None,
        seed: int = 42,
    ) -> "ITISteering":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        art = load_iti_artifact(artifact_path)

        info_dict = art["info"]
        info = ModelInfo(
            num_layers=int(info_dict["num_layers"]),
            num_heads=int(info_dict["num_heads"]),
            head_dim=int(info_dict["head_dim"]),
            component_template=str(info_dict["component_template"]),
        )

        activations = art["activations"]  # (N, L, H, D)
        labels = art["labels"]            # (N,)

        x_train, x_val, y_train, y_val = train_test_split(
            activations, labels, test_size=0.2, random_state=seed, stratify=labels
        )
        separated_activations = [x_train, x_val]
        separated_labels = [y_train, y_val]
        train_idxs = [0]
        val_idxs = [1]

        probes, head_accs = train_probes(
            seed,
            train_idxs,
            val_idxs,
            separated_activations,
            separated_labels,
            num_layers=info.num_layers,
            num_heads=info.num_heads,
        )
        head_accs = np.array(head_accs)
        head_order = np.argsort(head_accs)[::-1]

        fit = ProbeFit(
            probes=probes,
            head_order=head_order,
            x_train=x_train,
            x_val=x_val,
            y_train=y_train,
            y_val=y_val,
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            info=info,
            device=device,
            fit=fit,
        )

    def resolve_top_k(self, top_k: float) -> int:
        total = self.info.num_layers * self.info.num_heads
        if top_k <= 0:
            return 0
        if top_k < 1:
            return max(1, int(round(top_k * total)))
        return int(round(top_k))

    def top_heads(self, resolved_k: int) -> List[Tuple[int, int]]:
        """
        Returns list of (layer, head) using flattened head_order.
        """
        if resolved_k <= 0:
            return []
        top_idxs = self.fit.head_order[:resolved_k]
        out = []
        for idx in top_idxs:
            layer = int(idx) // self.info.num_heads
            head = int(idx) % self.info.num_heads
            out.append((layer, head))
        return out

    def build_steered_model(
        self,
        *,
        top_heads: Sequence[Tuple[int, int]],
        alpha: float,
        use_center_of_mass: bool = False,
        use_random_dir: bool = False,
        com_directions: Optional[Dict] = None,
    ) -> pv.IntervenableModel:
        """
        Mirrors your previous build_steered_model():
        - interventions from probes (or COM/random)
        - per-layer direction vector in concat head space
        - ITI_Intervener(direction, -alpha) at that layer's o_proj.input
        """
        info = self.info

        # Group heads by layer (pyvene config is per component)
        top_heads_by_layer: Dict[int, List[int]] = {}
        for layer, head in top_heads:
            top_heads_by_layer.setdefault(int(layer), []).append(int(head))

        interventions = get_interventions_dict(
            top_heads,
            self.fit.probes,
            self.fit.x_train,
            info.num_heads,
            use_center_of_mass,
            use_random_dir,
            com_directions,
        )

        pv_config = []
        for layer, heads in top_heads_by_layer.items():
            direction = torch.zeros(info.num_heads * info.head_dim, dtype=torch.float32)
            key = f"model.layers.{layer}.self_attn.head_out"
            for head, dir_vec, proj_std in interventions[key]:
                start = int(head) * info.head_dim
                end = (int(head) + 1) * info.head_dim
                direction[start:end] = torch.tensor(dir_vec * proj_std, dtype=torch.float32)

            intervener = ITI_Intervener(direction, -alpha)
            pv_config.append(
                {
                    "component": info.component_template.format(layer=layer),
                    "intervention": wrapper(intervener),
                }
            )

        return pv.IntervenableModel(pv_config, self.model)

    # --------------------
    # Generation helpers
    # --------------------
    def generate_base(
        self,
        prompt: str,
        *,
        max_length: int,
        max_new_tokens: int = 50,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> str:
        input_ids = encode_prompt(self.tokenizer, prompt, max_length=max_length).to(self.device)
        with torch.no_grad():
            temp = max(float(temperature), 1e-6) if do_sample else None
            top_p_val = float(top_p) if do_sample else None
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temp,
                top_p=top_p_val,
                repetition_penalty=float(repetition_penalty),
            )
        gen_ids = output[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def generate_steered(
        self,
        steered_model: pv.IntervenableModel,
        prompt: str,
        *,
        max_length: int,
        max_new_tokens: int = 50,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> str:
        input_ids = encode_prompt(self.tokenizer, prompt, max_length=max_length).to(self.device)
        with torch.no_grad():
            temp = max(float(temperature), 1e-6) if do_sample else None
            top_p_val = float(top_p) if do_sample else None
            out = steered_model.generate(
                {"input_ids": input_ids},
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temp,
                top_p=top_p_val,
                repetition_penalty=float(repetition_penalty),
            )
        if isinstance(out, tuple):
            out = out[1]
        gen_ids = out[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)
