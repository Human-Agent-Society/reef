from utils import make_match_reference, DisableCuDNNTF32
from task import input_t, output_t

import torch
from torch import nn, einsum
import math

# Reference code in PyTorch
class TriMul(nn.Module):
    # Based on https://github.com/lucidrains/triangle-multiplicative-module/blob/main/triangle_multiplicative_module/triangle_multiplicative_module.py
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        device="cuda",
        dtype=""
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)

        self.to_out_norm = nn.LayerNorm(hidden_dim, device=device)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, device=device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [bs, seq_len, seq_len, dim]
        mask: [bs, seq_len, seq_len]

        Returns:
            output: [bs, seq_len, seq_len, dim]
        """
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left, right)
        # This einsum is the same as the following:
        # out = torch.zeros(batch_size, seq_len, seq_len, dim, device=x.device)
        
        # # Compute using nested loops
        # for b in range(batch_size):
        #     for i in range(seq_len):
        #         for j in range(seq_len):
        #             # Compute each output element
        #             for k in range(seq_len):
        #                 out[b, i, j] += left[b, i, k, :] * right[b, j, k, :]

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of TriMul using PyTorch.
    
    Args:
        data: Tuple of (input: torch.Tensor, mask: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, seq_len, dim]
            - mask: Mask tensor of shape [batch_size, seq_len, seq_len]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters
    """

    # Use deterministic kernels and disable TF32 for accuracy
    with DisableCuDNNTF32():
        input_tensor, mask, weights, config = data
        trimul = TriMul(dim=config["dim"], hidden_dim=config["hidden_dim"], device=input_tensor.device)

        # Fill in the given weights of the model
        trimul.norm.weight = nn.Parameter(weights['norm.weight'])
        trimul.norm.bias = nn.Parameter(weights['norm.bias'])
        trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'])
        trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'])
        trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'])
        trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'])
        trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'])
        trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'])
        trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'])
        trimul.to_out.weight = nn.Parameter(weights['to_out.weight'])

        output = trimul(input_tensor, mask)

        return output


# Input generation for the reference code
def generate_input(
    seqlen: int,
    bs: int,
    dim: int,
    hiddendim: int,
    seed: int,
    nomask: bool,
    distribution: str,
) -> input_t:

    # Really dumb but for now _ isn't parsing correctly.
    batch_size = bs
    seq_len = seqlen
    hidden_dim = hiddendim
    no_mask = nomask

    config = {
        "hidden_dim": hidden_dim,
        "dim": dim,
    }

    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)

    weights = {}

    # Generate input tensor based on distribution
    if distribution == "cauchy":
        # FAST: sample Cauchy(0, 2) directly on GPU via inverse CDF
        u = torch.empty(
            (batch_size, seq_len, seq_len, dim),
            device="cuda",
            dtype=torch.float32,
        )
        u.uniform_(0.0, 1.0, generator=gen)
        input_tensor = 2.0 * torch.tan(math.pi * (u - 0.5))
    else:  # normal distribution
        input_tensor = torch.randn(
            (batch_size, seq_len, seq_len, dim),
            device='cuda',
            dtype=torch.float32,
            generator=gen
        ).contiguous()

    if no_mask:
        mask = torch.ones(batch_size, seq_len, seq_len, device=input_tensor.device)
    else:
        mask = torch.randint(0, 2, (batch_size, seq_len, seq_len), device=input_tensor.device, generator=gen)

    # Initialize model weights based on distribution
    weights["norm.weight"] = torch.randn(dim, device="cuda", dtype=torch.float32)
    weights["norm.bias"] = torch.randn(dim, device="cuda", dtype=torch.float32)
    weights["left_proj.weight"] = torch.randn(hidden_dim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hidden_dim)
    weights["right_proj.weight"] = torch.randn(hidden_dim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hidden_dim)
    weights["left_gate.weight"] = torch.randn(hidden_dim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hidden_dim)
    weights["right_gate.weight"] = torch.randn(hidden_dim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hidden_dim)
    weights["out_gate.weight"] = torch.randn(hidden_dim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hidden_dim)
    weights["to_out_norm.weight"] = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)
    weights["to_out.weight"] = torch.randn(dim, hidden_dim, device="cuda", dtype=torch.float32) / math.sqrt(dim)
    weights["to_out_norm.bias"] = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)

    return (input_tensor, mask, weights, config)


check_implementation = make_match_reference(ref_kernel, rtol=2e-2, atol=2e-2)