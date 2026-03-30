import pprint
import os
import math
from typing import List

import pyrallis
import torch
from PIL import Image as PILImage
from diffusers import StableDiffusion3Pipeline
from huggingface_hub.errors import GatedRepoError

from config import RunConfig
from pipeline_attend_and_excite_sd3 import SD3AttendAndExcitePipeline, SD3AttentionStore
from utils import ptp_utils
from utils.ptp_utils import AttentionStore

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_model(config: RunConfig):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    stable_diffusion_version = config.model_id
    hf_token = config.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    is_sd3 = "stable-diffusion-3" in stable_diffusion_version

    if is_sd3:
        try:
            pipeline_cls = StableDiffusion3Pipeline if config.run_standard_sd else SD3AttendAndExcitePipeline
            stable = pipeline_cls.from_pretrained(
                stable_diffusion_version,
                torch_dtype=torch_dtype,
                token=hf_token,
                low_cpu_mem_usage=True,
            )
        except GatedRepoError as e:
            raise RuntimeError(
                "SD3.5 Medium is gated. Pass --hf_token <token> (or set HF_TOKEN/HUGGINGFACE_TOKEN) "
                "after accepting the model license on Hugging Face."
            ) from e
        if device.type == "cuda":
            stable.enable_attention_slicing()
            stable.enable_model_cpu_offload()
            if not config.run_standard_sd and hasattr(stable, "transformer") and hasattr(stable.transformer, "enable_gradient_checkpointing"):
                stable.transformer.enable_gradient_checkpointing()
        else:
            stable = stable.to(device)
        stable.is_sd3 = True
        stable.supports_attend_excite = not config.run_standard_sd
        return stable


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def get_indices_to_alter_sd3(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)["input_ids"])}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word.get(i, '<out-of-range>') for i in token_indices]}")
    return token_indices


def get_image_grid(images: List[PILImage.Image]) -> PILImage.Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = PILImage.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image


def run_on_prompt(prompt: List[str],
                  model,
                  controller,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig) -> PILImage.Image:
    if hasattr(model, "set_progress_bar_config"):
        model.set_progress_bar_config(disable=config.disable_progress_bar)

    if getattr(model, "is_sd3", False):
        if getattr(model, "supports_attend_excite", False):
            outputs = model(prompt=prompt,
                            attention_store=controller,
                            indices_to_alter=token_indices,
                            guidance_scale=config.guidance_scale,
                            generator=seed,
                            num_inference_steps=config.n_inference_steps,
                            max_iter_to_alter=config.max_iter_to_alter,
                            run_standard_sd=config.run_standard_sd,
                            scale_factor=config.scale_factor,
                            scale_range=config.scale_range,
                            smooth_attentions=config.smooth_attentions,
                            sigma=config.sigma,
                            kernel_size=config.kernel_size,
                            height=config.height,
                            width=config.width,
                            optim_guidance=config.optim_guidance,
                            debug_log_path=config.debug_log_path)
        else:
            outputs = model(prompt=prompt,
                            guidance_scale=config.guidance_scale,
                            generator=seed,
                            num_inference_steps=config.n_inference_steps,
                            height=config.height,
                            width=config.width)
        return outputs.images[0]
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1,
                    height=config.height,
                    width=config.width,
                    optim_guidance=config.optim_guidance,
                    debug_log_path=config.debug_log_path)
    return outputs.images[0]


@pyrallis.wrap()
def main(config: RunConfig):
    stable = load_model(config)
    if getattr(stable, "is_sd3", False):
        if config.run_standard_sd:
            token_indices = []
        else:
            token_indices = get_indices_to_alter_sd3(stable, config.prompt) if config.token_indices is None else config.token_indices
    else:
        if config.run_standard_sd:
            token_indices = [] if config.token_indices is None else config.token_indices
        else:
            token_indices = get_indices_to_alter(stable, config.prompt) if config.token_indices is None else config.token_indices

    images = []
    for seed in config.seeds:
        print(f"Seed: {seed}")
        g = torch.Generator('cpu').manual_seed(seed)
        if config.run_standard_sd:
            controller = None
        else:
            controller = SD3AttentionStore() if getattr(stable, "is_sd3", False) else AttentionStore()
        image = run_on_prompt(prompt=config.prompt,
                              model=stable,
                              controller=controller,
                              token_indices=token_indices,
                              seed=g,
                              config=config)
        prompt_output_path = config.output_path / config.prompt
        prompt_output_path.mkdir(exist_ok=True, parents=True)
        image.save(prompt_output_path / f'{seed}.png')
        images.append(image)

    # save a grid of results across all seeds
    joined_image = get_image_grid(images)
    joined_image.save(config.output_path / f'{config.prompt}.png')


if __name__ == '__main__':
    main()
