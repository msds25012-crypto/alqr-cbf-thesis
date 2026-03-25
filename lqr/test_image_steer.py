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
from itertools import combinations, product
from PIL import Image, ImageDraw, ImageFont

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
    # torch_dtype=torch.float16,
)
pipe = FluxPipeline.from_pretrained(
    bfl_repo,
    torch_dtype=torch.bfloat16,
    text_encoder_2=text_encoder_2,
    force_download=False,
    quantization_config=pipeline_quant_config,
).to(device)

filename = "cartoons_all"
style = load_file(filename)

filename = "plain_all"
plain = load_file(filename)

filename = "cyberpunk_jac_with_text"
jac = load_file(filename)

# print(f"style: {style}")
# print(f"plain: {plain}")
# print(f"jac: {jac}")

X_contr_multi_img = style["X_multi"] - plain["X_multi"] 
X_contr_sing_img = style["X_sing"] - plain["X_sing"] 

filename = "cyberpunk"
style = load_file(filename)

filename = "plain"
plain = load_file(filename)


X_contr_multi_txt = style["X_txt_multi"] - plain["X_txt_multi"] 
X_contr_sing_txt = style["X_txt_sing"] - plain["X_txt_sing"] 


A_sing = jac["A_txt_sing"]
A_multi = jac["A_txt_multi"]



# print(f"A sing shape: {A_sing.shape}")
# print(f"A multi shape: {A_multi.shape}")

# print(f"X contr multi shape: {X_contr_multi.shape}")
# print(f"X contr sing shape: {X_contr_sing.shape}")


# print(f"mean multi: {X_contr_multi.mean(dim=0)}")
# print(f"mean sing: {X_contr_sing.mean(dim=0)}")

print(f"num transformer blocks: {len(pipe.transformer.transformer_blocks)}")
print(f"num transformer single blocks: {len(pipe.transformer.single_transformer_blocks)}")

steerer = ImageLQRSteering(
    pipe, 
    4,
    q=10, 
    r=1, 
    qf=10, 
    A_sing=A_sing, 
    A_multi=A_multi, 
    sing_contrastive_vecs_txt=X_contr_sing_txt, 
    multi_contrastive_vecs_txt=X_contr_multi_txt,

    sing_contrastive_vecs_img=X_contr_sing_img, 
    multi_contrastive_vecs_img=X_contr_multi_img
)

# for l in [6, 7, 8]:





multi_inds = range(0, len(pipe.transformer.transformer_blocks), 5)
sing_inds = range(0, len(pipe.transformer.single_transformer_blocks), 5)


images = []
labels = []

for p in range(5):
    combSing = list(combinations(multi_inds, p))
    combMulti = list(combinations(sing_inds, p))

    # Iterate over all pairs of p-combinations
    for sing_list, multi_list in product(combSing, combMulti):
        steered_img = steerer.addddddderoolllllllolllllllllllll(
            ["A cat resting on a laptop keyboard in a bedroom"], 
            steer_multi=True, 
            steer_single=True, 
            multi_list=multi_list, 
            sing_list=sing_list, 
            multi_scale=0.02, 
            sing_scale=0.02,
            do_sample=False, 
            temp=1
        )

        # steered_img.save(f"images/test.png")
        images.append(steered_img)
        label = f"sing:{sing_list} multi:{multi_list}"
        labels.append(label)


chunk_size = 8   # number of images per output file
cols = 4         # grid width per chunk

tile_width, tile_height = images[0].size
caption_height = 20

for chunk_idx in range(0, len(images), chunk_size):
    chunk_imgs = images[chunk_idx:chunk_idx + chunk_size]
    chunk_labels = labels[chunk_idx:chunk_idx + chunk_size]

    rows = (len(chunk_imgs) + cols - 1) // cols

    tiled_img = Image.new(
        'RGB',
        (cols * tile_width, rows * (tile_height + caption_height)),
        color=(255, 255, 255)
    )

    draw = ImageDraw.Draw(tiled_img)

    for idx, (img, label) in enumerate(zip(chunk_imgs, chunk_labels)):
        x = (idx % cols) * tile_width
        y = (idx // cols) * (tile_height + caption_height)

        tiled_img.paste(img, (x, y))

        text_x = x + 5
        text_y = y + tile_height + 2
        draw.text((text_x, text_y), label, fill=(0, 0, 0))

    # Save each chunk with index in filename
    file_idx = chunk_idx // chunk_size
    tiled_img.save(f"images/cartoon_test_{file_idx}.png")

# cols = 4
# rows = (len(images) + cols - 1) // cols

# tile_width, tile_height = images[0].size
# caption_height = 20  # height for label area below each image

# # Create a blank image for the grid
# tiled_img = Image.new(
#     'RGB',
#     (cols * tile_width, rows * (tile_height + caption_height)),
#     color=(255, 255, 255)
# )

# draw = ImageDraw.Draw(tiled_img)

# for idx, (img, label) in enumerate(zip(images, labels)):
#     x = (idx % cols) * tile_width
#     y = (idx // cols) * (tile_height + caption_height)
#     tiled_img.paste(img, (x, y))
#     # Draw label below image
#     text_x = x + 5
#     text_y = y + tile_height + 2
#     draw.text((text_x, text_y), label, fill=(0, 0, 0))

# multi_list = multi_inds
# multi_list = [0, 5, 20]
# # sing_list = sing_inds[:15]
# sing_list = [0, 5, 15]
# # # "A cat resting on a laptop keyboard in a bedroom"
# # # "A view out of a penthouse window overlooking a city"
# # steered_img = steerer.track_setpoint_all_patches(["A man in a busy city block."], steer_multi=True, steer_single=True, lmbda=1, do_sample=False, temp=1)


# sing_scales = [0.03, 0.04, 0.05, 0.06, 0.07]
# multi_scales = [0.02, 0.025, 0.03]
# images = []
# labels = []
# for sing_scale, multi_scale in product(sing_scales, multi_scales):
#     # steered_img = steerer.addddddderoolllllllolllllllllllll(["A cat resting on a laptop keyboard in a bedroom"], steer_multi=True, steer_single=True, multi_list=multi_list, sing_list=sing_list, do_sample=False, temp=1)
#     steered_img = steerer.addddddderoolllllllolllllllllllll(
#         ["a balcony overlooking a city"], 
#         steer_multi=True, 
#         steer_single=True, 
#         multi_list=multi_list, 
#         sing_list=sing_list, 
#         multi_scale=multi_scale, 
#         sing_scale=sing_scale,
#         do_sample=False, 
#         temp=1
#     )

#     # steered_img.save(f"images/test.png")
#     images.append(steered_img)
#     label = f"multi scale:{multi_scale} : sing scale:{sing_scale} "
#     labels.append(label)

# chunk_size = 8   # number of images per output file
# cols = 4         # grid width per chunk

# tile_width, tile_height = images[0].size
# caption_height = 20

# for chunk_idx in range(0, len(images), chunk_size):
#     chunk_imgs = images[chunk_idx:chunk_idx + chunk_size]
#     chunk_labels = labels[chunk_idx:chunk_idx + chunk_size]

#     rows = (len(chunk_imgs) + cols - 1) // cols

#     tiled_img = Image.new(
#         'RGB',
#         (cols * tile_width, rows * (tile_height + caption_height)),
#         color=(255, 255, 255)
#     )

#     draw = ImageDraw.Draw(tiled_img)

#     for idx, (img, label) in enumerate(zip(chunk_imgs, chunk_labels)):
#         x = (idx % cols) * tile_width
#         y = (idx // cols) * (tile_height + caption_height)

#         tiled_img.paste(img, (x, y))

#         text_x = x + 5
#         text_y = y + tile_height + 2
#         draw.text((text_x, text_y), label, fill=(0, 0, 0))

#     # Save each chunk with index in filename
#     file_idx = chunk_idx // chunk_size
#     tiled_img.save(f"images/scale_test_{file_idx}.png")


# sing_scale = 0.02
# multi_scale = 0.02
# steered_img = steerer.addddddderoolllllllolllllllllllll(
#     ["A cat resting on a laptop keyboard in a bedroom"], 
#     steer_multi=True, 
#     steer_single=True, 
#     multi_list=multi_list, 
#     sing_list=sing_list, 
#     multi_scale=multi_scale, 
#     sing_scale=sing_scale,
#     do_sample=False, 
#     temp=1
# )
# steered_img.save("images/actadd_REFINED.png")