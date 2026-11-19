"""Restore the frozen DVD prior used by Gen4P inference checkpoints."""

import re
from pathlib import Path

from peft import LoraConfig
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[1]
DVD_ADAPTER_NAME = "dvd"
DVD_LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"]


def normalize_checkpoint_args(
    checkpoint_args,
):
    normalized = dict(checkpoint_args)
    model_id_suffix = "_wan_model_id"
    for key in checkpoint_args:
        if not key.endswith(model_id_suffix):
            continue
        family = key[: -len(model_id_suffix)]
        aliases = {
            "wan_feature_model_id": key,
            "wan_feature_t": f"{family}_wan_t",
            "wan_feature_output_layer": f"{family}_wan_output_layer",
            "wan_feature_set_vae_f32": f"{family}_wan_set_vae_f32",
        }
        for target, source in aliases.items():
            if source in checkpoint_args:
                normalized.setdefault(target, checkpoint_args[source])
    return normalized


def convert_dvd_dit_key(
    key,
):
    if not key.startswith("pipe.dit."):
        return None
    key = key[len("pipe.dit.") :]
    key = key.replace(".lora_A.default.", f".lora_A.{DVD_ADAPTER_NAME}.")
    key = key.replace(".lora_B.default.", f".lora_B.{DVD_ADAPTER_NAME}.")
    block_match = re.match(r"blocks\.(\d+)\.(.+)", key)
    if block_match is not None:
        block_index, rest = block_match.groups()
        if rest == "modulation":
            return f"blocks.{block_index}.scale_shift_table"
        rest = rest.replace("self_attn.", "attn1.", 1)
        rest = rest.replace("cross_attn.", "attn2.", 1)
        rest = rest.replace("ffn.0.", "ffn.net.0.proj.", 1)
        rest = rest.replace("ffn.2.", "ffn.net.2.", 1)
        rest = rest.replace("norm3.", "norm2.", 1)
        for source, target in (
            (".q.", ".to_q."),
            (".k.", ".to_k."),
            (".v.", ".to_v."),
            (".o.", ".to_out.0."),
        ):
            rest = rest.replace(source, target, 1)
        return f"blocks.{block_index}.{rest}"
    for source, target in (
        ("text_embedding.0.", "condition_embedder.text_embedder.linear_1."),
        ("text_embedding.2.", "condition_embedder.text_embedder.linear_2."),
        ("time_embedding.0.", "condition_embedder.time_embedder.linear_1."),
        ("time_embedding.2.", "condition_embedder.time_embedder.linear_2."),
        ("time_projection.1.", "condition_embedder.time_proj."),
        ("head.head.", "proj_out."),
        ("patch_embedding.", "patch_embedding."),
    ):
        if key.startswith(source):
            return target + key[len(source) :]
    if key == "head.modulation":
        return "scale_shift_table"
    return None


def load_pretrained_encoder_dvd_for_inference(
    model,
    checkpoint_args,
    model_kwargs,
    dvd_checkpoint=None,
):
    if not checkpoint_args.get("pretrained_encoder_dvd"):
        return

    filename = Path(checkpoint_args["pretrained_encoder_dvd"]).name
    dvd_path = Path(dvd_checkpoint).expanduser() if dvd_checkpoint else REPO_ROOT / "ckpt" / filename
    if not dvd_path.is_file():
        raise FileNotFoundError(f"DVD checkpoint missing: {dvd_path}. See ckpt/README.md for the weight layout.")

    transformer = model.encoder.transformer
    with safe_open(dvd_path, framework="pt", device="cpu") as source:
        dvd_lora_rank = next((int(source.get_tensor(key).shape[0]) for key in source.keys() if key.startswith("pipe.dit.") and ".lora_A." in key), None)
    if dvd_lora_rank is not None:
        if model_kwargs.get("wan_train_mode") not in ("lora", "frozen"):
            raise ValueError("DVD LoRA requires a LoRA or frozen Wan encoder checkpoint.")
        if model_kwargs.get("wan_lora_adapter_name", "default") == DVD_ADAPTER_NAME:
            raise ValueError("The Gen4P adapter and DVD adapter must have distinct names.")
        transformer.add_adapter(LoraConfig(r=dvd_lora_rank, lora_alpha=dvd_lora_rank, target_modules=DVD_LORA_TARGET_MODULES, init_lora_weights=True), adapter_name=DVD_ADAPTER_NAME)

    target_state = transformer.state_dict()
    dvd_state = {}
    with safe_open(dvd_path, framework="pt", device="cpu") as source:
        for key in source.keys():
            mapped_key = convert_dvd_dit_key(key)
            if mapped_key is None:
                continue
            target_key = mapped_key if mapped_key in target_state else mapped_key.replace(".base_layer.", ".")
            if target_key not in target_state:
                continue
            tensor = source.get_tensor(key)
            if tensor.shape != target_state[target_key].shape:
                raise RuntimeError(f"DVD tensor shape mismatch: {key} -> {target_key}")
            dvd_state[target_key] = tensor
    if not dvd_state:
        raise RuntimeError(f"No DVD DiT tensors matched the Wan transformer: {dvd_path}")
    result = transformer.load_state_dict(dvd_state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected DVD tensor keys: {result.unexpected_keys[:20]}")

    adapters = [DVD_ADAPTER_NAME] if dvd_lora_rank is not None else []
    if model_kwargs.get("wan_train_mode") == "lora":
        adapters.append(model_kwargs.get("wan_lora_adapter_name", "default"))
    if adapters:
        transformer.set_adapter(adapters if len(adapters) > 1 else adapters[0])
    print(f"[DVD] Loaded {len(dvd_state)} Wan DiT tensors from {dvd_path}")
