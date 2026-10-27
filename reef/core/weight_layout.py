"""The layouts of saved weight files: what a training component's releases hold and what serving engines load."""

from typing import Literal

#: ``"full"`` is a Hugging Face checkpoint of every model weight; ``"adapter"`` is a PEFT adapter directory.
WeightLayout = Literal["full", "adapter"]
WEIGHT_LAYOUTS: tuple[WeightLayout, ...] = ("full", "adapter")
