import torch as th
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import random
import pickle
import time
import yaml

class ContrastiveBuilder:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset_name: str = None,
    ):
        self.model = model
        self.device = self.model.device
        self.tokenizer = tokenizer
        self.dataset = load_dataset(dataset_name) if dataset_name is not None else None

        self.T = len(self.model.model.layers)
        self.n = self.model.model.embed_tokens.embedding_dim
        self.m = self.n
        print(f"Latent dim: {self.n}")

        self.X = None 
        self.targets = None

        self.hooks = []
    
    def collect_data_batch(self, prompts, num_samples, layer_idx=None, batch_size=50):
        if layer_idx is not None and (layer_idx < 0 or layer_idx >= self.T):
            raise ValueError(f"layer_idx must be in [0, {self.T - 1}], got {layer_idx}")
        if num_samples > len(prompts):
            raise ValueError(f"num_samples={num_samples} exceeds available prompts={len(prompts)}")

        samples = random.sample(prompts, num_samples)
        activations = []
        for i in range(0, len(samples), batch_size):
            sample = samples[i:i+batch_size]
            inputs = self.tokenizer(
                sample, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
            ).to(self.device)
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states[1:]
            if layer_idx is None:
                batch_activations = th.stack(
                    [layer_hidden[:, -1, :] for layer_hidden in hidden_states],
                    dim=1,
                )
            else:
                batch_activations = hidden_states[layer_idx][:, -1, :]
            activations.append(batch_activations.detach().cpu())

        X = th.cat(activations, dim=0)
        # X_mean = X.mean(dim=0)

        print(f"total examples: {num_samples}")
        print(f"layer_idx: {layer_idx if layer_idx is not None else 'all'}")
        print(f"activations shape: {tuple(X.shape)}")

        tensor_dict = {
            "layer_idx": layer_idx,
            # "samples": samples,
            "X": X,
            # "X": X_mean,
        } 

        # with open(PICKLE_JAR + filename + ".pkl", "wb") as f:
        #     pickle.dump(tensor_dict, f)
        return tensor_dict
        
