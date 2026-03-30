import os
import sys
import torch
from diffusers import StableDiffusion3Pipeline
from pipeline_attend_and_excite_sd3 import SD3AttentionStore
from config import RunConfig
from run import load_model, run_on_prompt, get_indices_to_alter, get_indices_to_alter_sd3
from utils import vis_utils
from utils.ptp_utils import AttentionStore

hf_token = os.environ.get("HF_TOKEN")

model_name = "sd-3_5-medium"

if model_name == "sd-3_5-medium":
    model = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.float16,
    )
elif model_name == "sd-3_5-medium-a&e":
    NUM_DIFFUSION_STEPS = 50
    GUIDANCE_SCALE = 5.5
    SCALE_FACTOR = 80
    MAX_NUM_WORDS = 77

    config = RunConfig(
        prompt="a cat and a frog",
        model_id="stabilityai/stable-diffusion-3.5-medium",
        hf_token=hf_token,
        n_inference_steps=NUM_DIFFUSION_STEPS,
        scale_factor=SCALE_FACTOR,
        guidance_scale=GUIDANCE_SCALE,
        max_iter_to_alter=40
    )

    model = load_model(config)
model.enable_model_cpu_offload()

def inference(prompt):
    image = model(prompt).images[0]
    return image
