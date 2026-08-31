# ruff: noqa: F821
# ---------------------------------------------------------------------------
# Reef backport (docker/patch/mcore_parse_hybrid_pattern.py), appended by
# docker/Dockerfile.reef to megatron/core/ssm/mamba_hybrid_layer_allocation.py.
#
# megatron-bridge>=0.4 imports `parse_hybrid_pattern` from this module at
# package import time (models/mamba/mamba_provider.py). The slimerl/slime base
# pins Megatron-LM 1dcf0dafa (megatron-core 0.16.0rc0), which predates that
# helper. This is the upstream implementation (Megatron-LM 2edffadb, core
# 0.17.1) carried verbatim apart from the Symbols attributes it relies on, which
# the 0.16 `Symbols` class does not define yet.
# ---------------------------------------------------------------------------
from dataclasses import dataclass as _reef_dataclass

if not hasattr(Symbols, "MTP_SEPARATOR"):
    Symbols.MTP_SEPARATOR = "/"
if not hasattr(Symbols, "PIPE"):
    Symbols.PIPE = "|"
if not hasattr(Symbols, "VALID_LAYERS"):
    Symbols.VALID_LAYERS = set(Symbols.VALID)


@_reef_dataclass
class ParsedHybridPattern:
    """Result of parsing a unified hybrid pattern string.

    Format: "<main_pattern>/<mtp_pattern>/<mtp_pattern>/..."; the main pattern
    may contain "|" pipeline-stage boundaries.
    """

    main_pattern: str | None
    mtp_pattern: str | None
    mtp_num_depths: int


def parse_hybrid_pattern(pattern: str | None) -> ParsedHybridPattern:
    """Parse a unified hybrid pattern string into main and MTP components."""
    if pattern is None:
        return ParsedHybridPattern(main_pattern=None, mtp_pattern=None, mtp_num_depths=0)

    parts = pattern.split(Symbols.MTP_SEPARATOR)

    if len(parts) == 1:
        main_pattern = parts[0]
        _validate_pattern(main_pattern, "main", allow_pipe=True)
        return ParsedHybridPattern(main_pattern=main_pattern, mtp_pattern=None, mtp_num_depths=0)

    main_pattern = parts[0]
    if main_pattern:
        _validate_pattern(main_pattern, "main", allow_pipe=True)

    mtp_parts = parts[1:]

    if not mtp_parts or all(p == "" for p in mtp_parts):
        return ParsedHybridPattern(
            main_pattern=main_pattern if main_pattern else None, mtp_pattern=None, mtp_num_depths=0
        )

    mtp_pattern = mtp_parts[0]
    for i, part in enumerate(mtp_parts[1:], start=2):
        if part != mtp_pattern:
            raise ValueError(
                f"All MTP patterns must be identical. "
                f"Pattern 1 is '{mtp_pattern}', but pattern {i} is '{part}'. "
                f"Full pattern: '{pattern}'"
            )

    _validate_pattern(mtp_pattern, "MTP", allow_pipe=False)

    return ParsedHybridPattern(
        main_pattern=main_pattern if main_pattern else None,
        mtp_pattern=mtp_pattern,
        mtp_num_depths=len(mtp_parts),
    )


def _validate_pattern(pattern: str, pattern_name: str, allow_pipe: bool = False) -> None:
    """Validate that a pattern contains only valid layer symbols."""
    valid_chars = Symbols.VALID_LAYERS | {Symbols.PIPE} if allow_pipe else Symbols.VALID_LAYERS
    for char in pattern:
        if char not in valid_chars:
            raise ValueError(
                f"In {pattern_name} pattern, '{char}' is not a valid layer symbol. "
                f"Valid symbols are: {valid_chars}"
            )
