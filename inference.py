import os
from diffusers import StableDiffusion3Pipeline

hf_token = os.environ.get("HF_TOKEN")

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    token=hf_token
)

def inference(prompt):
    image = pipe(prompt).images[0]
    return image