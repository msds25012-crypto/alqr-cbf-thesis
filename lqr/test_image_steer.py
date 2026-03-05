import torch
from diffusers import FluxPipeline, BitsAndBytesConfig
from transformers import T5EncoderModel
from diffusers.quantizers import PipelineQuantizationConfig
from IPython.display import display
from image_data_handling import ImageContrastiveBuilder
import pickle
import yaml
from pathlib import Path
import json
from datasets import load_dataset
import random
from image_steering import ImageLQRSteering


with open('config/config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)
PICKLE_JAR = config_data["environment"]["pickle_jar"]
def load_file(filename):
    with open(PICKLE_JAR+filename+".pkl", "rb") as f:
        loaded_tensors = pickle.load(f)
    return loaded_tensors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
    components_to_quantize=["transformer", "text_encoder_2"],
)

bfl_repo = "black-forest-labs/FLUX.1-schnell"
text_encoder_2 = T5EncoderModel.from_pretrained(
    bfl_repo,
    subfolder="text_encoder_2",
    torch_dtype=torch.bfloat16,
)
pipe = FluxPipeline.from_pretrained(
    bfl_repo,
    torch_dtype=torch.bfloat16,
    text_encoder_2=text_encoder_2,
    force_download=False,
    quantization_config=pipeline_quant_config,
).to(device)

filename = "cyberpunk"
style = load_file(filename)

filename = "plain"
plain = load_file(filename)

filename = "cyberpunk_jac"
jac = load_file(filename)

X_contr_multi = style["X_multi"] - plain["X_multi"] 
X_contr_sing = style["X_sing"] - plain["X_sing"] 

A_sing = jac["A_sing"]
A_multi = jac["A_multi"]

steerer = ImageLQRSteering(
    pipe, 
    4, 
    q=0.1, 
    r=1, 
    qf=0.1, 
    A_sing=A_sing, 
    A_multi=A_multi, 
    sing_contrastive_vecs=X_contr_sing, 
    multi_contrastive_vecs=X_contr_multi
)

steered_img = steerer.track_setpoint(["hey man"], steer_multi=True, steer_single=True, lmbda=1, do_sample=False, temp=1)

steered_img.save("images/steered_test.png")
