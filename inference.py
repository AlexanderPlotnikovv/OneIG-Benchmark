from diffusers import StableDiffusion3Pipeline

model = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers"
)

def inference(prompt):
    image = model(prompt).images[0]
    return image
