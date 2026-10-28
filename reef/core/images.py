"""Images as the engine and the trainer get them: the model's image settings, and one prepared image.

Reef prepares each image once: it resizes the image to its vision grid within a
pixel limit, encodes it as PNG and hashes the PNG. The engine gets these PNG
bytes, so the HF processor's own resize is a no-op on them. The chat template
renders one ``<|image_pad|>`` per image; the engine expands it to
``prod(grid) / merge_size**2`` copies. Reef predicts that expansion and
records the engine's prompt ids only when they match it.

The token-native chat handler (``reef.inference.chat``) prepares the client's
image for the engine. A record-to-sample stage that trains on images prepares
it again to check the record. Both use this module, so it lives in
``reef.core``. ``reef.inference.images`` reads the settings from the model's files.

The first version supports Qwen3.5 only, and one image per request. Pillow and
transformers' ``smart_resize`` are imported on the first image, so text-only
deployments do not need them.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: The largest client image Reef decodes, in pixels. 50 million pixels holds an 8K screen (33 million) and
#: the photos of common phone cameras. Decoding takes about 1.6 s and 1.2 GB for 144 million pixels, and
#: Pillow only warns below its own decompression-bomb limit of about 179 million.
MAX_SOURCE_IMAGE_PIXELS = 50_000_000
#: The transformers module with the Qwen2-VL ``smart_resize``, which the HF processor in SGLang applies to
#: each image. It needs no torch from transformers 5.5 on; 5.4 imports it from a torchvision module.
SMART_RESIZE_MODULE = "transformers.models.qwen2_vl.image_processing_pil_qwen2_vl"
#: The image formats Reef decodes: the ones OpenAI and Anthropic accept.
SOURCE_IMAGE_FORMATS = ("PNG", "JPEG", "GIF", "WEBP")


@dataclass(frozen=True)
class ImageConfig:
    """The model's image settings: the processor's patch grid and the tokenizer's vision special ids.

    ``reef.inference.images.image_config_from_model`` reads them from the model's files.
    """

    patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int
    image_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    #: Every vision special id. A record requires none of them in the text or the sampled output.
    vision_token_ids: tuple[int, ...]


def import_image_modules() -> None:
    """Import Pillow and ``smart_resize`` on the event loop.

    transformers' lazy modules are not safe to import first in a worker thread.
    """
    try:
        importlib.import_module("PIL")
        importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise ValueError("image input needs Pillow; install reef-infra[sglang]") from exc
    try:
        importlib.import_module(SMART_RESIZE_MODULE)
    except ImportError as exc:
        raise ValueError(f"image input needs transformers 5.5 or later: {exc}") from exc


@dataclass(frozen=True)
class PreparedImage:
    """One image as the engine and the trainer get it."""

    png: bytes
    #: SHA-256 of ``png``. A trainer prepares the client's image again and compares it. The PNG bytes,
    #: and so the hash, repeat only with the same Pillow and zlib build, for example the same Reef image.
    image_hash: str
    #: The vision grid ``[t, h, w]`` in patches.
    grid: tuple[int, int, int]
    #: The client's image size ``[w, h]``, to map click coordinates back.
    orig_size: tuple[int, int]
    #: The number of ``<|image_pad|>`` ids the engine expands this image to.
    token_length: int

    def data_url(self) -> str:
        return "data:image/png;base64," + base64.b64encode(self.png).decode("ascii")


def image_base64(part: Mapping[str, Any]) -> str | None:
    """The client's base64 image data of one content part, or ``None`` for a part that is not an image.

    It reads an OpenAI ``image_url`` part with a base64 data URL and an Anthropic
    ``image`` block with a base64 source. A remote URL raises ``ValueError``.
    It does not decode the data, so it is cheap to run on the event loop.
    """
    kind = part.get("type")
    if kind == "image_url":
        image_url = part.get("image_url")
        url = image_url.get("url") if isinstance(image_url, Mapping) else image_url
        if not isinstance(url, str) or not url:
            raise ValueError("OpenAI image_url parts require image_url.url")
        if not url.startswith("data:"):
            raise ValueError("exact image capture accepts only base64 data URLs, not remote image URLs")
        header, separator, data = url.partition(",")
        if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
            raise ValueError("an image data URL must be data:image/<type>;base64,<data>")
        return data
    if kind == "image":
        source = part.get("source")
        if not isinstance(source, Mapping) or source.get("type") != "base64":
            raise ValueError("exact image capture accepts only Anthropic base64 image sources, not image URLs")
        encoded = source.get("data")
        if not isinstance(encoded, str):
            raise ValueError("Anthropic base64 image sources require string data")
        return encoded
    return None


def prepare_base64_images(encoded_images: Sequence[str], config: ImageConfig) -> tuple[PreparedImage, ...]:
    """Decode and prepare client images. It uses no tokenizer, so it can run in a worker thread."""
    return tuple(prepare_image(decode_base64(encoded), config) for encoded in encoded_images)


def prepare_image(raw: bytes, config: ImageConfig) -> PreparedImage:
    """Resize ``raw`` to its vision grid within the pixel limit, encode it as PNG and hash it.

    An animated image keeps its first frame. Every image is converted to RGB,
    as the HF processor does. Any image that Reef cannot decode raises ``ValueError``.
    """
    # image_config_from_model already imported these modules on the event loop.
    from PIL import Image

    smart_resize = importlib.import_module(SMART_RESIZE_MODULE).smart_resize
    undecodable = f"the image cannot be decoded as {', '.join(SOURCE_IMAGE_FORMATS)}"
    # These blocks run only Pillow code, and Pillow raises many exception types for a broken file
    # (OSError, SyntaxError, struct.error, IndexError, ...). Every one of them is a bad client image.
    try:
        opened = Image.open(io.BytesIO(raw), formats=SOURCE_IMAGE_FORMATS)
    except Exception as exc:
        raise ValueError(f"{undecodable}: {exc}") from exc
    with opened:
        width, height = opened.size
        if width * height > MAX_SOURCE_IMAGE_PIXELS:
            raise ValueError(f"the image has {width}x{height} pixels; the limit is {MAX_SOURCE_IMAGE_PIXELS}")
        try:
            opened.seek(0)
            image = opened.convert("RGB")
        except Exception as exc:
            raise ValueError(f"{undecodable}: {exc}") from exc
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=config.patch_size * config.merge_size,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    if (resized_width, resized_height) != (width, height):
        image = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    png = buffer.getvalue()
    grid = (1, resized_height // config.patch_size, resized_width // config.patch_size)
    return PreparedImage(
        png=png,
        image_hash=hashlib.sha256(png).hexdigest(),
        grid=grid,
        orig_size=(width, height),
        token_length=math.prod(grid) // config.merge_size**2,
    )


def expanded_prompt_ids(prompt_ids: Sequence[int], images: Sequence[PreparedImage], config: ImageConfig) -> list[int]:
    """The prompt ids the engine runs: each rendered ``<|image_pad|>`` expanded to its image's span.

    The rendered prompt must hold exactly one ``<|vision_start|><|image_pad|><|vision_end|>``
    per image and no other vision special id. Text that contains a vision
    special token as a string breaks this, and raises ``ValueError``.
    """
    vision_ids = set(config.vision_token_ids)
    pads = [index for index, token in enumerate(prompt_ids) if token == config.image_token_id]
    others = sum(1 for token in prompt_ids if token in vision_ids) - len(pads)
    if (
        len(pads) != len(images)
        or others != 2 * len(images)
        or any(
            index == 0
            or index + 1 >= len(prompt_ids)
            or prompt_ids[index - 1] != config.vision_start_token_id
            or prompt_ids[index + 1] != config.vision_end_token_id
            for index in pads
        )
    ):
        raise ValueError(
            f"the rendered prompt does not hold one image placeholder per image ({len(images)} image(s)); "
            "image requests cannot contain vision special tokens such as <|image_pad|> in their text"
        )
    expanded: list[int] = []
    start = 0
    for index, image in zip(pads, images, strict=True):
        expanded.extend(prompt_ids[start:index])
        expanded.extend([config.image_token_id] * image.token_length)
        start = index + 1
    expanded.extend(prompt_ids[start:])
    return expanded


def image_spans(
    prompt_ids: Sequence[int], images: Sequence[PreparedImage], config: ImageConfig
) -> list[dict[str, Any]]:
    """The ``training.images`` entries: where each image's span starts in ``prompt_ids``, and its metadata."""
    entries: list[dict[str, Any]] = []
    index = 0
    for image in images:
        while prompt_ids[index] != config.image_token_id:
            index += 1
        entries.append(
            {
                "image_hash": image.image_hash,
                "token_offset": index,
                "token_length": image.token_length,
                "grid": list(image.grid),
                "orig_size": list(image.orig_size),
            }
        )
        index += image.token_length
    return entries


def decode_base64(data: str) -> bytes:
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image data is not valid base64") from exc


__all__ = [
    "ImageConfig",
    "PreparedImage",
    "decode_base64",
    "expanded_prompt_ids",
    "image_base64",
    "image_spans",
    "import_image_modules",
    "prepare_base64_images",
    "prepare_image",
]
