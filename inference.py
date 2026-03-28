import os
import torch
from diffusers import StableDiffusion3Pipeline

hf_token = os.environ.get("HF_TOKEN")

model = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.float16
).to("cuda")


def inference(prompt):
    image = model(prompt).images[0]
    return image
