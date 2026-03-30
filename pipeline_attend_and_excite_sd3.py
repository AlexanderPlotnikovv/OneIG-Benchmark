import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from diffusers import StableDiffusion3Pipeline
from diffusers.models.attention_processor import Attention
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift, retrieve_timesteps

from utils.gaussian_smoothing import GaussianSmoothing


class SD3AttentionStore:
    def __init__(self):
        self.step_store: List[torch.Tensor] = []
        self.attention_store: List[torch.Tensor] = []

    def __call__(self, attn: torch.Tensor):
        # Store compact (batch, spatial, tokens) maps to keep the autograd graph smaller.
        if attn.shape[1] <= 4096:
            self.step_store.append(attn)

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = []

    def get_average_attention(self) -> List[torch.Tensor]:
        return self.attention_store


class SD3AttendExciteAttnProcessor:
    def __init__(self, attnstore: SD3AttentionStore):
        self.attnstore = attnstore

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # Save image-to-text cross attention maps for Attend-and-Excite loss.
            scale = 1.0 / math.sqrt(head_dim)
            cross_scores = torch.matmul(query, encoder_hidden_states_key_proj.transpose(-2, -1)) * scale
            cross_probs = torch.softmax(cross_scores, dim=-1).mean(dim=1)
            self.attnstore(cross_probs)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


def register_attention_control_sd3(pipe: StableDiffusion3Pipeline, controller: SD3AttentionStore):
    attn_procs = {}
    for name in pipe.transformer.attn_processors.keys():
        attn_procs[name] = SD3AttendExciteAttnProcessor(controller)
    pipe.transformer.set_attn_processor(attn_procs)


class SD3AttendAndExcitePipeline(StableDiffusion3Pipeline):
    @staticmethod
    def _compute_loss(max_attention_per_index: List[torch.Tensor]) -> torch.Tensor:
        losses = [max(0, 1.0 - curr_max) for curr_max in max_attention_per_index]
        return max(losses)

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True),
            [latents],
            retain_graph=False,
            allow_unused=True,
        )[0]
        if grad_cond is None:
            return latents
        return latents - step_size * grad_cond

    def _compute_max_attention_per_index(
        self,
        attention_maps: torch.Tensor,
        indices_to_alter: List[int],
        smooth_attentions: bool = False,
        sigma: float = 0.5,
        kernel_size: int = 3,
    ) -> List[torch.Tensor]:
        attention_for_text = attention_maps[:, :, 1:]
        attention_for_text = torch.nn.functional.softmax(attention_for_text * 100, dim=-1)
        indices_to_alter = [max(0, index - 1) for index in indices_to_alter]

        max_indices_list = []
        for i in indices_to_alter:
            if i >= attention_for_text.shape[-1]:
                continue
            image = attention_for_text[:, :, i]
            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(image.device)
                padded = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
                image = smoothing(padded).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())
        return max_indices_list if max_indices_list else [torch.tensor(0.0, device=attention_maps.device)]

    def _aggregate_and_get_max_attention_per_token(
        self,
        attention_store: SD3AttentionStore,
        indices_to_alter: List[int],
        smooth_attentions: bool = False,
        sigma: float = 0.5,
        kernel_size: int = 3,
    ):
        attention_maps = attention_store.get_average_attention()
        if not attention_maps:
            attention_maps = attention_store.step_store
        if not attention_maps:
            return [torch.tensor(0.0, device=self._execution_device)]

        stacked = torch.cat(attention_maps, dim=0).mean(0)
        num_pixels = stacked.shape[0]
        res = int(math.sqrt(num_pixels))
        if res * res != num_pixels:
            return [torch.tensor(0.0, device=stacked.device)]

        stacked = stacked.reshape(res, res, stacked.shape[-1])
        return self._compute_max_attention_per_index(
            stacked,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        attention_store: SD3AttentionStore,
        indices_to_alter: List[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        max_iter_to_alter: Optional[int] = 20,
        run_standard_sd: bool = False,
        scale_factor: int = 20,
        scale_range: Tuple[float, float] = (1.0, 0.5),
        smooth_attentions: bool = True,
        sigma: float = 0.5,
        kernel_size: int = 3,
        max_sequence_length: int = 256,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        optim_guidance: bool = False,
        debug_log_path: Optional[str] = None,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            negative_prompt_3=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            callback_on_step_end_tensor_inputs=["latents"],
            max_sequence_length=max_sequence_length,
        )

        device = self._execution_device
        do_cfg = guidance_scale > 1.0
        encode_with_cfg = do_cfg and not optim_guidance
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=encode_with_cfg,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=None,
            max_sequence_length=max_sequence_length,
            lora_scale=None,
        )
        original_prompt_embeds = prompt_embeds
        original_pooled_prompt_embeds = pooled_prompt_embeds

        if do_cfg and not optim_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        if optim_guidance:
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        num_channels_latents = self.transformer.config.in_channels
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None):
            _, _, latent_h, latent_w = latents.shape
            image_seq_len = (latent_h // self.transformer.config.patch_size) * (latent_w // self.transformer.config.patch_size)
            scheduler_kwargs["mu"] = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, **scheduler_kwargs)

        original_attn_proc = self.transformer.attn_processors
        if not run_standard_sd:
            register_attention_control_sd3(self, attention_store)

        scale_range = np.linspace(scale_range[0], scale_range[1], len(timesteps))
        if max_iter_to_alter is None:
            max_iter_to_alter = len(timesteps) + 1

        if not optim_guidance:
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if not run_standard_sd and i < max_iter_to_alter:
                        with torch.enable_grad():
                            attention_store.step_store = []
                            latents_for_grad = latents.detach().clone().requires_grad_(True)
                            timestep = t.expand(latents_for_grad.shape[0])
                            _ = self.transformer(
                                hidden_states=latents_for_grad,
                                timestep=timestep,
                                encoder_hidden_states=original_prompt_embeds,
                                pooled_projections=original_pooled_prompt_embeds,
                                joint_attention_kwargs=self._joint_attention_kwargs,
                                return_dict=False,
                            )[0]
                            attention_store.between_steps()
                            max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                                attention_store=attention_store,
                                indices_to_alter=indices_to_alter,
                                smooth_attentions=smooth_attentions,
                                sigma=sigma,
                                kernel_size=kernel_size,
                            )
                            loss = self._compute_loss(max_attention_per_index=max_attention_per_index)
                            if isinstance(loss, torch.Tensor) and loss.item() != 0:
                                latents = self._update_latent(
                                    latents=latents_for_grad,
                                    loss=loss,
                                    step_size=scale_factor * float(np.sqrt(scale_range[i])),
                                ).detach()
                            else:
                                latents = latents_for_grad.detach()

                    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                    timestep = t.expand(latent_model_input.shape[0])

                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self._joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    if do_cfg:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    progress_bar.update()
        else:
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    with torch.enable_grad():
                        attention_store.step_store = []
                        latents_for_grad = latents.detach().clone().requires_grad_(True)
                        timestep = t.expand(latents_for_grad.shape[0])
                        _ = self.transformer(
                            hidden_states=latents_for_grad,
                            timestep=timestep,
                            encoder_hidden_states=original_prompt_embeds,
                            pooled_projections=original_pooled_prompt_embeds,
                            joint_attention_kwargs=self._joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        attention_store.between_steps()
                        max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                            attention_store=attention_store,
                            indices_to_alter=indices_to_alter,
                            smooth_attentions=smooth_attentions,
                            sigma=sigma,
                            kernel_size=kernel_size,
                        )
                        loss = self._compute_loss(max_attention_per_index=max_attention_per_index)
                        if isinstance(loss, torch.Tensor) and loss.item() != 0:
                            latents_for_grad = self._update_latent(
                                latents=latents_for_grad,
                                loss=loss,
                                step_size=-scale_factor * float(np.sqrt(scale_range[i])),
                            ).detach()

                    latent_delta = (latents_for_grad - latents).abs().max().item()
                    if debug_log_path is not None:
                        with open(debug_log_path, "a") as debug_file:
                            debug_file.write(
                                f"[debug][step={i}] max|latents_for_grad-latents|={latent_delta:.8e}\n"
                            )
                            debug_file.flush()

                    latent_model_input = torch.cat([latents_for_grad, latents])
                    timestep = t.expand(latent_model_input.shape[0])

                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self._joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_delta = (noise_pred_uncond - noise_pred_text).abs().max().item()
                    if debug_log_path is not None:
                        with open(debug_log_path, "a") as debug_file:
                            debug_file.write(
                                f"[debug][step={i}] max|noise_bad-noise_clean|={noise_delta:.8e}\n"
                            )
                            debug_file.flush()
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    progress_bar.update()

        self.transformer.set_attn_processor(original_attn_proc)

        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return StableDiffusion3PipelineOutput(images=image)
