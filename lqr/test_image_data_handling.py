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
        tags = [random.choice(style_tags) for i in range(3)]

        tagged_prompt = p
        for tag in tags:
            tagged_prompt = tagged_prompt + " " + tag
        styled_prompts.append(tagged_prompt)

    return prompts, styled_prompts



def generate_cyberpunk_prompts(n: int) -> list[str]:
    neon_core = [
        "neon-lit", "glowing neon", "vibrant neon lights",
        "electric neon glow", "blinding neon reflections"
    ]

    iconic_elements = [
        "rain-soaked streets reflecting neon",
        "holographic advertisements everywhere",
        "towering megacorp billboards",
        "flying cars streaking through the sky",
        "dense fog illuminated by neon",
        "glitching digital screens",
        "chrome cybernetic implants glowing",
        "laser lights cutting through darkness"
    ]

    subjects = [
        "lone hacker", "cyborg assassin", "street samurai",
        "android with glowing eyes", "augmented human",
        "cyberpunk mercenary", "rogue AI avatar"
    ]

    settings = [
        "in a crowded futuristic city",
        "in a narrow alleyway",
        "on a skyscraper rooftop",
        "in an underground nightclub",
        "in a bustling night market"
    ]

    style_boost = [
        "cinematic lighting", "ultra detailed", "high contrast",
        "volumetric lighting", "photorealistic", "8k resolution"
    ]

    prompts = []

    for _ in range(n):
        prompt = (
            f"{random.choice(neon_core)} cyberpunk scene, "
            f"{random.choice(iconic_elements)}, "
            f"{random.choice(subjects)} {random.choice(settings)}, "
            f"{random.choice(style_boost)}"
        )
        prompts.append(prompt)

    return prompts



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


plain_prompts, styled_prompts = build_prompts(style_tags["cartoons"])


cyberpunk_prompts = generate_cyberpunk_prompts(500)

print(f"PLAIN: {plain_prompts}")
# print(f"STYLED: {styled_prompts}")
print(f"cyberpunk: {cyberpunk_prompts}")



# plain_path = Path("images/plain_prompts.json")
# with plain_path.open("r", encoding="utf-8") as f:
#     plain_prompts = json.load(f)

# prompts = ["kitty cat", "cat", "me-wow"]

# plain_prompts = plain_prompts["none"] + plain_prompts["trees"] + plain_prompts["no_trees"]

# print(plain_prompts)

# cyberpunk_prompts = style_prompts["cyberpunk"]

# print(cyberpunk_prompts)

num_samples = 200
filename = "cartoons_all"
imageguy.collect_all_image_batch(styled_prompts, num_samples, filename, batch_size=6)
print("done with cartoons")

# filename = "plain_all"
# imageguy.collect_all_image_batch(plain_prompts, num_samples, filename, batch_size=6)
# print("done with plain")

test = load_file(filename)
print(test)

# num_samples = 20
# filename = "cyberpunk_jac_with_text"
# imageguy.collect_jacobians(styled_prompts, num_samples, filename, collect_txt=True, collect_img=False)
# print("done with jac")

# test = load_file(filename)
# print(test)
