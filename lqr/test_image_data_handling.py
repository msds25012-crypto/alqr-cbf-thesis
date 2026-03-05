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

with open('config/config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)
PICKLE_JAR = config_data["environment"]["pickle_jar"]
PATH = config_data["environment"]["ref_data_path"]

def load_file(filename):
    with open(PICKLE_JAR+filename+".pkl", "rb") as f:
        loaded_tensors = pickle.load(f)
    return loaded_tensors

def build_prompts(style_tags, num_prompts=500):
    ds = load_dataset("sentence-transformers/coco-captions")["train"]
    # print(ds)
    # print(ds["caption1"])
    prompts = []
    styled_prompts = []
    for p in ds["caption1"][:num_prompts]:
        prompts.append(p)
        tag = random.choice(style_tags)
        styled_prompts.append(p + " " + tag)

    return prompts, styled_prompts




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

imageguy = ImageContrastiveBuilder(pipe, 4)

style_path = Path("images/style_prompts.json")
with style_path.open("r", encoding="utf-8") as f:
    style_tags = json.load(f)


plain_prompts, styled_prompts = build_prompts(style_tags["cyberpunk"])


num_samples = 500
filename = "cyberpunk"
imageguy.collect_data_batch(styled_prompts, num_samples, filename, batch_size=12)
print("done with cyberpunk")

filename = "plain"
imageguy.collect_data_batch(plain_prompts, num_samples, filename, batch_size=12)
print("done with plain")


num_samples = 50
filename = "cyberpunk_jac"
imageguy.collect_jacobians(styled_prompts, num_samples, filename)
print("done with jac")

test = load_file(filename)
print(test)
