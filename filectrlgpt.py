import torch as th
import sys
import os
import matplotlib.pyplot as plt
from lpe.lpe.method_utils import *
from lpe.lpe.utils import Transformer
from lpe.lpe.utils import datasets as lpe_datasets
print(sys.executable)
print(th.__version__)
print(th.version.cuda)
print(th.cuda.is_available())
# print(th.cuda.current_device())

model_name = "gelu-1l"
device = th.device("cuda" if th.cuda.is_available() else "cpu")
model = Transformer.from_pretrained(model_name).to(device)


# initializing input distributions

dist_name = "camel"

gt_freqs = load_ground_truth(model_name, [dist_name], device=device)[dist_name] # ground truth tensor
gt_probs = gt_freqs / gt_freqs.sum()

lpe_datasets.USE_CACHE = True
orig_dists = distribution_registry[dist_name](model.tokenizer, device=model.device).input_dists(n_reps=N_REPS_DICT[dist_name])