"""Shared model restoration and video encoding for inference tasks."""

import argparse
import inspect
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch

from models import D4RT
from .inference_checkpoint import (
    load_pretrained_encoder_dvd_for_inference,
    normalize_checkpoint_args,
)

CUDA_VISIBLE_DEVICES_RAW = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
# var name mapping, should be fixed
MODEL_FIELD_TO_CREATE_KWARG = {
    "num_frames": "temporal_size",
    "patch_size": "query_patch_size",
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def validate_image_aspect_ratios(
    img_h: int,
    img_w: int,
    proc_h: int,
    proc_w: int,
) -> None:
    """Require width/height ratios to differ by at most 3% relative to proc."""
    if min(img_h, img_w, proc_h, proc_w) <= 0:
        raise ValueError("--img_h, --img_w, --proc_h, and --proc_w must be positive.")
    difference = abs(img_w * proc_h - proc_w * img_h)
    reference = proc_w * img_h
    # Integer comparison keeps a difference of exactly 3% valid.
    if difference * 100 > reference * 3:
        raise ValueError(
            f"img/proc aspect ratios (w/h) must differ by at most 3% relative to proc; "
            f"got img={img_h}x{img_w} (ratio={img_w / img_h:.6f}), "
            f"proc={proc_h}x{proc_w} (ratio={proc_w / proc_h:.6f}), "
            f"relative difference={100 * difference / reference:.6f}%."
        )


def setRandomness(
    seed: int,
    deterministic: bool = True,
) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def get_visible_cuda_devices() -> List[str]:
    if CUDA_VISIBLE_DEVICES_RAW:
        return [dev.strip() for dev in CUDA_VISIBLE_DEVICES_RAW.split(",") if dev.strip()]
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []


def inspect_signature_parameter_names(
    cls_or_fn,
) -> List[str]:
    signature = inspect.signature(cls_or_fn.__init__ if isinstance(cls_or_fn, type) else cls_or_fn)
    out = []
    for name in signature.parameters.keys():
        if name != "self":
            out.append(name)
    return out


def cli_field_is_explicitly_set(
    cli_args,
    field_name: str,
) -> bool:
    cli_defaults = getattr(cli_args, "parser_defaults", {})
    if field_name not in cli_defaults:
        return False
    if not hasattr(cli_args, field_name):
        return False
    return getattr(cli_args, field_name) != cli_defaults[field_name]


def build_model_kwargs(
    ckpt_args,
    cli_args,
    runtime_device: Optional[torch.device] = None,
    *,
    per_clip_time: bool = False,
):
    model_field_names = set(inspect_signature_parameter_names(D4RT))
    model_kwargs = {}
    for name, value in ckpt_args.items():
        kwarg_name = MODEL_FIELD_TO_CREATE_KWARG.get(name, name)
        if kwarg_name in model_field_names:
            model_kwargs[kwarg_name] = value
    cli_defaults = getattr(cli_args, "parser_defaults", {})
    for name in cli_defaults.keys():
        if not cli_field_is_explicitly_set(cli_args, name):
            continue
        kwarg_name = MODEL_FIELD_TO_CREATE_KWARG.get(name, name)
        if kwarg_name in model_field_names:
            model_kwargs[kwarg_name] = getattr(cli_args, name)
    if cli_args.infer_chunk_num_frames is not None and (not per_clip_time or cli_field_is_explicitly_set(cli_args, "infer_chunk_num_frames")):
        model_kwargs["temporal_size"] = int(cli_args.infer_chunk_num_frames)
    if "wan_device" in model_kwargs:
        if runtime_device is not None:
            runtime_device_str = str(runtime_device)
        else:
            runtime_device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if model_kwargs["wan_device"] is None:
            model_kwargs["wan_device"] = runtime_device_str
        elif str(model_kwargs["wan_device"]) == "cuda" and runtime_device is not None:
            model_kwargs["wan_device"] = runtime_device_str
    if getattr(cli_args, "only_condition_t_but_no_noise", None) is False:
        raise ValueError("Gen4P inference requires only_condition_t_but_no_noise=True.")
    if getattr(cli_args, "pre_vae_encode", None) is True:
        raise ValueError("Gen4P inference expects raw video, not pre-encoded latents.")
    if getattr(cli_args, "use_wan_feature_encoder", None) is False:
        raise ValueError("Gen4P inference requires the Wan feature encoder.")
    for field in ("query_patch_size", "wan_feature_output_layer"):
        value = model_kwargs.get(field)
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"Gen4P inference requires one {field}, got {value}.")
            model_kwargs[field] = int(value[0])
    return model_kwargs


def resolve_train_num_frames(
    ckpt_args,
    cli_args,
) -> Optional[int]:
    if cli_args.train_num_frames is not None:
        return int(cli_args.train_num_frames)
    value = ckpt_args.get("num_frames", None)
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        return None
    return value


def maybe_resize_decoder_time_embeddings(
    model,
    ckpt_args,
    cli_args,
    *,
    per_clip_time: bool = False,
) -> None:
    train_num_frames = resolve_train_num_frames(ckpt_args, cli_args)
    infer_total_num_frames = int(cli_args.infer_total_num_frames)
    infer_chunk_num_frames = int(getattr(model, "temporal_size", 0))
    print(f"[Time Emb] train_num_frames={train_num_frames} | infer_total_num_frames={infer_total_num_frames} | infer_chunk_num_frames={infer_chunk_num_frames}")
    if train_num_frames is None:
        print("[Time Emb] Skip resize because train_num_frames is unavailable.")
        return

    timestep_embed = model.decoder.timestep_embed
    max_timesteps = int(timestep_embed.max_timesteps)
    if infer_chunk_num_frames > max_timesteps:
        raise ValueError(f"infer_chunk_num_frames={infer_chunk_num_frames} exceeds decoder max_timesteps={max_timesteps}.")
    if train_num_frames > max_timesteps:
        raise ValueError(f"train_num_frames={train_num_frames} exceeds decoder max_timesteps={max_timesteps}.")
    if per_clip_time:
        # Depth clips can have different lengths after padding a short scene.
        timestep_embed.base_num_frames = int(train_num_frames)
        return
    if infer_chunk_num_frames <= train_num_frames:
        print("[Time Emb] Skip resize because infer_chunk_num_frames <= train_num_frames.")
        return

    with torch.no_grad():
        for embed_name in ["src_embedding", "tgt_embedding", "cam_embedding"]:
            weight = getattr(timestep_embed, embed_name).weight
            source = weight[:train_num_frames].detach().to(dtype=torch.float32).transpose(0, 1).unsqueeze(0)
            resized = torch.nn.functional.interpolate(source, size=infer_chunk_num_frames, mode="linear", align_corners=True).squeeze(0).transpose(0, 1).to(dtype=weight.dtype, device=weight.device)
            weight[:infer_chunk_num_frames].copy_(resized)
    print("[Time Emb] Applied linear interpolation to decoder timestep embeddings.")


def load_model(
    checkpoint_path: str,
    device: torch.device,
    cli_args,
    *,
    per_clip_time: bool = False,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = normalize_checkpoint_args(checkpoint.get("args", {}))
    state_dict = checkpoint["model_state_dict"].copy()

    model_kwargs = build_model_kwargs(ckpt_args, cli_args, runtime_device=device, per_clip_time=per_clip_time)
    decoder_dtype = torch.get_default_dtype()
    if not cli_args.amp_infer:
        if ckpt_args.get("decoder_dtype") == "float32":
            decoder_dtype = torch.float32
        elif ckpt_args.get("decoder_dtype") == "bfloat16":
            decoder_dtype = torch.bfloat16
        elif ckpt_args.get("decoder_dtype") is None:
            pass
        else:
            raise ValueError(f"Unsupported decoder dtype: {ckpt_args.get('decoder_dtype')}")
    torch.set_default_dtype(decoder_dtype)

    model = D4RT(**model_kwargs)
    # The training snapshot also contains heads that no inference task consumes.
    for head, width in (("head_2d", 2), ("head_disp", 3), ("head_normal", 3)):
        for suffix, shape in (("weight", (width, model.encoder.embed_dim)), ("bias", (width,))):
            key = f"decoder.{head}.{suffix}"
            if key in state_dict:
                value = state_dict.pop(key)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                    raise RuntimeError(f"Invalid retired checkpoint head {key}: expected a tensor with shape {shape}.")
    load_pretrained_encoder_dvd_for_inference(model, ckpt_args, model_kwargs, cli_args.dvd_checkpoint)
    # Frozen encoder weights come from Wan/DVD; the Gen4P checkpoint stores its adapters.
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    disallowed_missing_keys = [key for key in incompatible_keys.missing_keys if not (key.startswith("encoder.") or key.startswith("module.encoder."))]
    unexpected_keys = list(incompatible_keys.unexpected_keys)
    if disallowed_missing_keys or unexpected_keys:
        raise RuntimeError(f"Slim-ckpt compatibility load failed: disallowed_missing_keys={disallowed_missing_keys[:20]}, unexpected_keys={unexpected_keys[:20]}")

    # Benchmark-only inference adaptation for longer clips.
    maybe_resize_decoder_time_embeddings(model, ckpt_args, cli_args, per_clip_time=per_clip_time)

    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def prepareDecodeContext(
    model,
    video_clip_tchw: torch.Tensor,
    encoder_clip_tchw: torch.Tensor,
    enc_img_size: Tuple[int, int],
    amp_infer: bool,
    decoder_dtype: Optional[torch.dtype],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode one clip once and return RGB frames and features for repeated query decoding."""
    video_b = video_clip_tchw.unsqueeze(0)
    encode_b = encoder_clip_tchw.unsqueeze(0)
    assert encode_b.shape[:3] == video_b.shape[:3], f"Expected encoder clip shape [T,C,H,W] matching video clip, got {tuple(encoder_clip_tchw.shape)}"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(amp_infer and video_clip_tchw.device.type == "cuda")):
        encoder_features = model.encode(encode_b, enc_img_size)
        if decoder_dtype is not None:
            encoder_features = encoder_features.to(decoder_dtype)
            video_b = video_b.to(decoder_dtype)
    return video_b, encoder_features
