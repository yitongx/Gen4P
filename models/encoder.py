"""Wan video feature encoder for Gen4P inference."""

import torch
import torch.nn as nn
from typing import List, Optional

from diffusers import WanPipeline
from diffusers.models import WanTransformer3DModel
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler


def resolveWanTorchDtype(
    torch_dtype: str | torch.dtype,
) -> torch.dtype:
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    key = str(torch_dtype).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported wan torch dtype: {torch_dtype}")
    return mapping[key]


class TransformerWanWithFeatureOutput(WanTransformer3DModel):
    """Return the selected Wan T2V layer's hidden states."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        output_layer_index: int,
    ) -> torch.Tensor:
        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        _, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(timestep, encoder_hidden_states)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        for i, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            if i == output_layer_index:
                return hidden_states
        raise ValueError(f"Wan output layer index {output_layer_index} exceeds the transformer depth {len(self.blocks)}.")


class WanEncoder(nn.Module):
    """Wan2.1 feature encoder used by Gen4P."""

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        t: int = 999,
        output_layer_index: int = 29,
        torch_dtype: str | torch.dtype = "bfloat16",
        set_vae_f32: bool = False,
        wan_train_mode: str = "lora",
        wan_lora_rank: int = 512,
        wan_lora_alpha: Optional[int] = None,
        wan_lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2",
        wan_lora_adapter_name: str = "default",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.t = int(t)
        self.output_layer_index = int(output_layer_index)
        if self.output_layer_index < 0:
            raise ValueError(f"WanEncoder requires a non-negative output layer index, got {self.output_layer_index}.")
        self.set_vae_f32 = bool(set_vae_f32)
        self.wan_train_mode = str(wan_train_mode).strip().lower()
        if self.wan_train_mode not in {"lora", "frozen"}:
            raise ValueError("Inference supports Wan adapter modes 'lora' and 'frozen' only.")

        self.wan_lora_rank = int(wan_lora_rank)
        self.wan_lora_alpha = None if wan_lora_alpha is None else int(wan_lora_alpha)
        self.wan_lora_target_modules = str(wan_lora_target_modules)
        self.wan_lora_adapter_name = str(wan_lora_adapter_name)

        transformer = TransformerWanWithFeatureOutput.from_pretrained(model_id, subfolder="transformer")
        pipe = WanPipeline.from_pretrained(model_id, transformer=transformer, safety_checker=None)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)

        self.dit_torch_dtype = resolveWanTorchDtype(torch_dtype)
        self.vae_torch_dtype = torch.float32 if self.set_vae_f32 else self.dit_torch_dtype
        # prepare null_prompt on CPU to avoid putting text encoder to GPU
        with torch.no_grad():
            null_prompt_embeds, _ = pipe.encode_prompt(prompt="", do_classifier_free_guidance=False, device="cpu")
        self.null_prompt_embeds = null_prompt_embeds  # prepare only once

        pipe.vae = pipe.vae.to(device=device, dtype=self.vae_torch_dtype)
        pipe.transformer = pipe.transformer.to(device=device, dtype=self.dit_torch_dtype)
        self.pipe = pipe

        # extract pipe modules to nn.Modules for saving and loading ckpts
        # loading ckpts into self.transformer is the same as loading into pipe.transformer, since
        # same address
        self.transformer = pipe.transformer
        self.vae = pipe.vae
        self.embed_dim = int(self.transformer.config.num_attention_heads) * int(self.transformer.config.attention_head_dim)

        if self.wan_train_mode == "lora":
            self.injectLoraToTransformer()

        self.vae.eval().requires_grad_(False)
        self.transformer.eval().requires_grad_(False)
        if getattr(self.pipe, "text_encoder", None) is not None:
            self.pipe.text_encoder.eval().requires_grad_(False)
            self.pipe.text_encoder.to(device="cpu")
        print(f"[Encoder Init] WanEncoder: t={self.t}, layer={self.output_layer_index}, embed_dim={self.embed_dim}, vae_dtype={self.vae_torch_dtype}, dit_dtype={self.dit_torch_dtype}, set_vae_f32={self.set_vae_f32}")

    def resolveWanLoraTargetModules(
        self,
    ) -> List[str]:
        target_aliases = {
            "q": "to_q",
            "k": "to_k",
            "v": "to_v",
            "o": "to_out.0",
            "ffn.0": "ffn.net.0.proj",
            "ffn.2": "ffn.net.2",
        }
        target_modules = []
        for name in self.wan_lora_target_modules.split(","):
            name = name.strip()
            if not name:
                continue
            target_modules.append(target_aliases.get(name, name))
        return target_modules

    def injectLoraToTransformer(
        self,
    ) -> None:
        from peft import LoraConfig

        if not hasattr(self.transformer, "add_adapter"):
            raise ValueError("WanEncoder transformer does not support add_adapter().")

        target_modules = self.resolveWanLoraTargetModules()
        alpha = self.wan_lora_rank if self.wan_lora_alpha is None else self.wan_lora_alpha

        self.transformer.requires_grad_(False)
        self.transformer.add_adapter(
            LoraConfig(
                r=self.wan_lora_rank,
                lora_alpha=alpha,
                target_modules=target_modules,
                init_lora_weights=True,  # B: zeros, A: kaiming
            ),
            adapter_name=self.wan_lora_adapter_name,
        )
        print(f"[Encoder Init] Injected LoRA into Wan transformer. rank={self.wan_lora_rank}, alpha={alpha}, target_modules={target_modules}, adapter_name='{self.wan_lora_adapter_name}'")

    def forward(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """Encode normalized RGB video with shape (B, T, 3, H, W)."""
        # specify run_device of transformer, to avoid text_encoder offload error
        run_device = next(self.transformer.parameters()).device
        with torch.no_grad():
            if video.ndim != 5 or video.shape[2] != 3:
                raise ValueError(f"Expected video shape (B, T, 3, H, W), got {tuple(video.shape)}")
            T = video.shape[1]
            if T <= 0 or (T - 1) % 4:
                raise ValueError(f"Wan temporal VAE expects 4N+1 frames, got T={T}.")
            normalized_video = video.to(dtype=torch.float32).clamp(0.0, 1.0)
            video_vae_input = (normalized_video * 2.0 - 1.0).permute(0, 2, 1, 3, 4).contiguous()
            video_vae_input = video_vae_input.to(device=run_device, dtype=self.vae_torch_dtype)
            self.pipe.scheduler.set_timesteps(1000, device=run_device)
            num_timesteps = len(self.pipe.scheduler.timesteps)
            if self.t < 0 or self.t > num_timesteps:
                raise ValueError(f"Wan timestep must be in [0, {num_timesteps}], got {self.t}.")
            latents = self.vae.encode(video_vae_input).latent_dist.mean
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents = (latents - latents_mean) * latents_std
            if self.t == num_timesteps:
                t_input = torch.zeros((1,), dtype=self.pipe.scheduler.timesteps.dtype, device=run_device)
            else:
                t_idx = torch.tensor([self.t], dtype=torch.long, device=run_device)
                t_input = self.pipe.scheduler.timesteps[t_idx].to(run_device)
            # Preserve the timestep condition while always feeding clean latents.
            latents_for_dit = latents.to(device=run_device, dtype=self.dit_torch_dtype)
            prompt_embeds = self.null_prompt_embeds.to(device=run_device, dtype=self.dit_torch_dtype)
            if prompt_embeds.shape[0] != latents_for_dit.shape[0]:
                prompt_embeds = prompt_embeds.repeat(latents_for_dit.shape[0], 1, 1)
            features = self.transformer(hidden_states=latents_for_dit, timestep=t_input.expand(latents_for_dit.shape[0]), encoder_hidden_states=prompt_embeds, output_layer_index=self.output_layer_index)
            assert features.ndim == 3 and features.shape[-1] == self.embed_dim
            return features
