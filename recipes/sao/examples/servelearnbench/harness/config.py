"""Shared defaults for the Retail launcher, trainer, and result records."""

import os

DEFAULTS = {
    "SAO_BATCH": 64,
    "SAO_MICRO_BATCH": 16,
    "SAO_WARMUP": 1,
    "SAO_CONTEXT": 40960,
}


def training_config() -> dict[str, int]:
    """Read integer configuration; this example supports unscaled Qwen3 contexts only."""
    values = {name: int(os.environ.get(name, default)) for name, default in DEFAULTS.items()}
    if values["SAO_BATCH"] <= 0 or values["SAO_MICRO_BATCH"] <= 0 or values["SAO_WARMUP"] < 0:
        raise ValueError("Batch sizes must be positive; warmup must be nonnegative")
    if values["SAO_BATCH"] % values["SAO_MICRO_BATCH"]:
        raise ValueError("SAO_BATCH must be divisible by SAO_MICRO_BATCH (the example uses data parallel size one)")
    if not 0 < values["SAO_CONTEXT"] <= 40960:
        raise ValueError("SAO_CONTEXT must be between 1 and 40960; this example does not enable YaRN")
    return values


if __name__ == "__main__":
    for name, value in training_config().items():
        print(f"export {name}={value}")
