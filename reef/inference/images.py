"""The image settings of a model, read from its files for exact chat capture.

``reef.core.images`` holds the settings type and the image preparation, which
the trainer also uses. This module reads the settings from the model's
``config.json``, its image processor files and its tokenizer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.core.images import ImageConfig, import_image_modules

#: ``config.json`` model types whose image path the capture supports.
IMAGE_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe"})
#: The default pixel limit of one prepared image; ``image_max_pixels`` in the handler config changes it.
DEFAULT_IMAGE_MAX_PIXELS = 1024 * 1024
#: The vision special tokens of the Qwen3.5 tokenizer.
IMAGE_PAD_TOKEN = "<|image_pad|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
VISION_SPECIAL_TOKENS = (VISION_START_TOKEN, VISION_END_TOKEN, "<|vision_pad|>", IMAGE_PAD_TOKEN, "<|video_pad|>")
#: The model files that the image settings come from. Reef downloads only these for a Hugging Face model ID.
IMAGE_SETTINGS_FILES = ("config.json", "preprocessor_config.json", "processor_config.json")


def image_config_from_model(model_path: str, tokenizer: Any, *, max_pixels: int) -> ImageConfig:
    """Read ``config.json`` and the image processor settings of ``model_path``.

    The settings come from ``preprocessor_config.json``, or from the
    ``image_processor`` block of ``processor_config.json``, which
    ``processor.save_pretrained`` writes. ``max_pixels`` is Reef's limit;
    the processor's own limit applies when it is lower. As in the HF
    processor, ``min_pixels`` and ``max_pixels`` override ``size`` unless
    they are null. A ``model_path`` that is not a local directory is a
    Hugging Face model ID, as for the tokenizer: Reef downloads only
    ``IMAGE_SETTINGS_FILES`` of it.
    """
    model_dir = image_settings_directory(model_path)
    model_type = read_json_file(model_dir / "config.json").get("model_type")
    if model_type not in IMAGE_MODEL_TYPES:
        raise ValueError(
            f"model type {model_type!r} does not support image input; exact image capture supports Qwen3.5 only"
        )
    processor = image_processor_settings(model_dir)
    size = processor.get("size")
    if not isinstance(size, Mapping):
        size = {}
    min_pixels = processor_pixels(processor, size, "min_pixels", "shortest_edge")
    processor_max_pixels = processor_pixels(processor, size, "max_pixels", "longest_edge")
    if min(max_pixels, processor_max_pixels) < 2 * min_pixels:
        # Near the minimum, resizing within the limit can round below it. The HF processor then scales
        # the prepared image up again, and the request keeps no record. From twice the minimum on, a
        # sweep of 11149 sizes found no such case.
        limit = (
            f"image_max_pixels {max_pixels}"
            if max_pixels <= processor_max_pixels
            else f"the model processor's max_pixels {processor_max_pixels}"
        )
        raise ValueError(f"{limit} must be at least twice the model processor's minimum of {min_pixels} pixels")
    import_image_modules()
    ids = tuple(special_token_id(tokenizer, token) for token in VISION_SPECIAL_TOKENS)
    by_token = dict(zip(VISION_SPECIAL_TOKENS, ids, strict=True))
    return ImageConfig(
        patch_size=positive_config_int(processor.get("patch_size"), "patch_size"),
        merge_size=positive_config_int(processor.get("merge_size"), "merge_size"),
        min_pixels=min_pixels,
        max_pixels=min(max_pixels, processor_max_pixels),
        image_token_id=by_token[IMAGE_PAD_TOKEN],
        vision_start_token_id=by_token[VISION_START_TOKEN],
        vision_end_token_id=by_token[VISION_END_TOKEN],
        vision_token_ids=ids,
    )


def image_settings_directory(model_path: str) -> Path:
    """The local directory with the image settings files of ``model_path``.

    A local directory is returned as it is. Otherwise ``model_path`` is a
    Hugging Face model ID, for example when ``inference.remote`` runs with a
    hosted training backend, and Reef downloads ``IMAGE_SETTINGS_FILES`` to the
    Hugging Face cache.
    """
    model_dir = Path(model_path)
    if model_dir.is_dir():
        return model_dir
    import huggingface_hub

    try:
        return Path(huggingface_hub.snapshot_download(repo_id=model_path, allow_patterns=list(IMAGE_SETTINGS_FILES)))
    except (OSError, ValueError) as exc:
        # The hub's errors for a missing, gated or invalid ID and for an offline host without a cached copy.
        raise ValueError(
            f"model_path {model_path!r} is not a local directory, and Reef cannot download its image settings "
            f"from Hugging Face: {exc}"
        ) from exc


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"the model has no {path.name}, so it does not support image input") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return value


def image_processor_settings(model_dir: Path) -> dict[str, Any]:
    """The image processor settings: ``preprocessor_config.json``, else ``processor_config.json["image_processor"]``."""
    if (model_dir / "preprocessor_config.json").is_file():
        return read_json_file(model_dir / "preprocessor_config.json")
    if (model_dir / "processor_config.json").is_file():
        settings = read_json_file(model_dir / "processor_config.json").get("image_processor")
        if not isinstance(settings, dict):
            raise ValueError("processor_config.json has no image_processor object")
        return settings
    raise ValueError(
        "the model has neither preprocessor_config.json nor processor_config.json, "
        "so Reef cannot read its image processor settings"
    )


def processor_pixels(processor: Mapping[str, Any], size: Mapping[str, Any], key: str, size_key: str) -> int:
    """A pixel limit of the image processor: ``key`` when it is set and not null, else ``size[size_key]``."""
    if processor.get(key) is not None:
        return positive_config_int(processor[key], key)
    return positive_config_int(size.get(size_key), f"size.{size_key}")


def positive_config_int(value: Any, name: str) -> int:
    """A positive integer setting of the image processor; an integral float, as JSON can write it, is accepted."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"image processor setting {name} must be a positive integer, got {value!r}")
    return value


def special_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int) or token_id == tokenizer.unk_token_id:
        raise ValueError(f"the tokenizer has no {token} token, so the model does not support image input")
    return token_id


__all__ = ["DEFAULT_IMAGE_MAX_PIXELS", "image_config_from_model"]
