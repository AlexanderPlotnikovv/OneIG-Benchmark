from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RunConfig:
    # Guiding text prompt
    prompt: str
    # Hugging Face model id to load
    model_id: str = "stabilityai/stable-diffusion-3.5-medium"
    # Optional Hugging Face token for gated model access (SD3.5 Medium)
    hf_token: Optional[str] = None
    # Output image height (use smaller values on low-VRAM GPUs)
    height: int = 512
    # Output image width (use smaller values on low-VRAM GPUs)
    width: int = 512
    # Whether to use Stable Diffusion v2.1
    sd_2_1: bool = False
    # Which token indices to alter with attend-and-excite
    token_indices: List[int] = None
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [42])
    # Path to save all outputs to
    output_path: Path = Path('./outputs')
    # Number of denoising steps
    n_inference_steps: int = 50
    # Whether to disable tqdm progress bars during generation
    disable_progress_bar: bool = False
    # Optional path for temporary debug logging during generation
    debug_log_path: Optional[Path] = None
    # Text guidance scale
    guidance_scale: float = 7.5
    # Number of denoising steps to apply attend-and-excite
    max_iter_to_alter: int = 25
    # Resolution of UNet to compute attention maps over
    attention_res: int = 16
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(default_factory=lambda: {0: 0.05, 10: 0.5, 20: 0.8})
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 20
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.5))
    # Whether to apply the Gaussian smoothing before computing the maximum attention value for each subject token
    smooth_attentions: bool = True
    # Standard deviation for the Gaussian smoothing
    sigma: float = 0.5
    # Kernel size for the Gaussian smoothing
    kernel_size: int = 3
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False
    # Whether to turn on our method
    optim_guidance: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)
