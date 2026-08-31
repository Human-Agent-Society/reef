from typing import TypedDict, TypeVar, Tuple, Dict
import torch

input_t = TypeVar("input_t", bound=Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict])
output_t = TypeVar("output_t", bound=Tuple[torch.Tensor, Dict])


class TestSpec(TypedDict):
    seqlen: int
    bs: int
    dim: int
    hiddendim: int
    seed: int
    nomask: bool