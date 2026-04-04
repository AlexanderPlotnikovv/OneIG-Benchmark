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

model_name = "sd-3_5-medium-a&e"

if model_name == "sd-3_5-medium":
    model = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.float16,
    )
    model.enable_model_cpu_offload()

elif model_name == "sd-3_5-medium-a&e":
    NUM_DIFFUSION_STEPS = 28
    GUIDANCE_SCALE = 5.5
    SCALE_FACTOR = 80

    config = RunConfig(
        prompt="",
        model_id="stabilityai/stable-diffusion-3.5-medium",
        hf_token=hf_token,
        n_inference_steps=NUM_DIFFUSION_STEPS,
        scale_factor=SCALE_FACTOR,
        guidance_scale=GUIDANCE_SCALE,
        max_iter_to_alter=40
    )

    model = load_model(config)


def auto_get_indices(model, prompt):
    """Автоматически выбирает индексы существительных/прилагательных"""
    tokens = model.tokenizer(prompt)['input_ids']
    token_words = {idx: model.tokenizer.decode(t) for idx, t in enumerate(tokens)}

    # Пропускаем спецтокены и знаки препинания
    skip = {'<|startoftext|>', '<|endoftext|>', ',', '.', '!', '?', 'a', 'an', 'the', 'and', 'or', 'with', 'in', 'on',
            'of'}

    indices = [idx for idx, word in token_words.items()
               if word.strip().lower() not in skip and word.strip()]

    print(f"Auto-selected tokens: {[(i, token_words[i]) for i in indices]}")
    return indices


def inference(prompt):
    if model_name == "sd-3_5-medium":
        image = model(prompt).images[0]

    elif model_name == "sd-3_5-medium-a&e":
        config.prompt = prompt
        g = torch.Generator('cpu').manual_seed(torch.randint(0, 2 ** 32, (1,)).item())
        controller = SD3AttentionStore()

        token_indices = auto_get_indices(model, prompt)

        image = run_on_prompt(
            prompt=prompt,
            model=model,
            controller=controller,
            token_indices=token_indices,
            seed=g,
            config=config,
        )

    return image
