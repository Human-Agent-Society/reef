# ruff: noqa: E501

# Vendored verbatim from TTT-Discover examples/gpu_mode/prompt.py (MIT).
# See tasks/assets/trimul_evaluator/SOURCE.md for source and license details.
TRIMUL_PROMPT = r'''You are an expert Triton engineer tasked with translating PyTorch code into highly optimized Triton kernel code.

You will be implementing a Triangle Multiplicative Update (TriMul) module that is a core operation
for AlphaFold3, Chai, Protenix, and other protein structure prediction models in BioML.

The TriMul operator operates over a 4D tensor of shape [B, N, N, C]. 

Your task:
- Implement the "outgoing" version of the TriMul operator from the AlphaFold3 paper.
- You will not have to compute or store gradients for this version. You will only need to implement the forward pass.

Your function should be defined as 'custom_kernel' with the following signature:
Input:
- `data`: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
    - input: Input tensor of shape [bs, seq_len, seq_len, dim]
    - mask: Mask tensor of shape [bs, seq_len, seq_len]
    - weights: Dictionary containing model weights
    - config: Dictionary containing model configuration parameters

Output:
- output: Processed tensor [bs, seq_len, seq_len, dim]

**Problem Constraints:**
- B ∈ {1,2}, N ∈ {128,256,512,1024}, c ∈ {128}, c_z ∈ {128,384,768}
- The input distribution will be sampled from a standard Normal distribution, or a heavy-tailed Cauchy distribution (gamma = 2).
- There will either be no mask, or a randomly sampled mask over the inputs.

**Remarks.** So why is this problem so annoying? Because you have to choose whether to load / deal with either the channel dimensions c,c_z that the LayerNorms require (otherwise you have to do a synchronize to compute the statistics like mean / variance) or the sequence dimension N. 
The sequence dimension is particularly annoying because it's quite large, but also because we compute pair-wise operations at the last operation that sum over another sequence dimension (this is N^3!). 
However, I really like this kernel because it only consists of “simple” operations, and is really easy to understand. It is a true test of “fusions” that torch.compile() doesn't do that well.

Here is a pytorch implementation of the TriMul module. You will want to implement a kernel for the operations in the forward call:

```python
import torch
from torch import nn, einsum
import math

# Reference code in PyTorch
class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

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
```

Here is some example skeleton code of the entrypoint function you will create:
```python
def custom_kernel(data)
    input_tensor, mask, weights, config = data
    dim, hidden_dim = config["dim"], config["hidden_dim"]

    # Access the given weights of the model
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    left_proj_weight = weights["left_proj.weight"]
    right_proj_weight = weights["right_proj.weight"]
    left_gate_weight = weights["left_gate.weight"]
    right_gate_weight = weights["right_gate.weight"]
    out_gate_weight = weights["out_gate.weight"]
    to_out_norm_weight = weights["to_out_norm.weight"]
    to_out_norm_bias = weights["to_out_norm.bias"]
    to_out_weight = weights["to_out.weight"]

    # Perform TriMul

    return out
```

To help you understand which triton version we are using, here is some example triton code for an unrelated task:
```python
import triton
import triton.language as tl

@triton.jit
def matmul_persistent_ws_kernel(
   a_ptr, b_ptr, c_ptr, M, N, K,
   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
   pid = tl.program_id(axis=0) # async_task 0, 1, 2
   num_pid_m = tl.cdiv(M, BLOCK_M) # async_task 0, 1, 2
   num_pid_n = tl.cdiv(N, BLOCK_N) # async_task 0, 1, 2
   pid_m = pid // num_pid_m # async_task 0, 1, 2
   pid_n = pid % num_pid_n # async_task 0, 1, 2
   offs_m_1 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M // 2) # async_task 0, 1, 2
   offs_m_2 = pid_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M) # async_task 0, 1, 2
   offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_N) # async_task 0, 1, 2
   offs_k = tl.arange(0, BLOCK_K) # async_task 0
   a_ptrs_1 = a_ptr + (offs_m_1[:, None] * stride_am + offs_k[None, :] * stride_ak) # async_task 0
   a_ptrs_2 = a_ptr + (offs_m_2[:, None] * stride_am + offs_k[None, :] * stride_ak) # async_task 0
   b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn) # async_task 0
   acc_1 = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.float32) # async_task 1
   acc_1 = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.float32) # async_task 2
   for k in range(0, tl.cdiv(K, BLOCK_K)): # async_task 0, 1, 2
       a_1 = tl.load(a_ptrs_1)   # async_task 0
       a_2 = tl.load(a_ptrs_2)   # async_task 0
       b = tl.load(b_ptrs)   # async_task 0
       acc_1 += tl.dot(a_1, b)   # async_task 1
       acc_2 += tl.dot(a_2, b)   # async_task 2
       a_ptrs_1 += BLOCK_K * stride_ak # async_task 0
       a_ptrs_2 += BLOCK_K * stride_ak # async_task 0
       b_ptrs += BLOCK_K * stride_bk # async_task 0
   c_1 = acc_1.to(tl.float16) # async_task 1
   c_2 = acc_2.to(tl.float16) # async_task 2
   c_ptrs_1 = c_ptr_1 + stride_cm * offs_m_1[:, None] + stride_cn * offs_n[None, :] # async_task 1
   c_ptrs_2 = c_ptr_2 + stride_cm * offs_m_2[:, None] + stride_cn * offs_n[None, :] # async_task 2
   tl.store(c_ptrs_1, c_1) # async_task 1
   tl.store(c_ptrs_2, c_2) # async_task 2
```

A few general triton tips:
- tl.arange only takes in constexpr arguments (static or tl.constexpr)
- You cannot use continue in your kernel code
- tl.dot can only take in two input tensors
- There is no tl.mean

Here are the different configs that your kernel will be tested on ("nomask" sets whether there will be no mask, or a randomly sampled mask over the inputs):

Test Cases for correctness and runtime (optimize runtime for these):
  - {"seqlen": 256, "bs": 2, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "normal"}
  - {"seqlen": 768, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "cauchy"}
  - {"seqlen": 256, "bs": 2, "dim": 384, "hidden_dim": 128, "nomask": False, "distribution": "normal"}
  - {"seqlen": 512, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "normal"}
  - {"seqlen": 1024, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "cauchy"}
  - {"seqlen": 768, "bs": 1, "dim": 384, "hidden_dim": 128, "nomask": False, "distribution": "normal"}
  - {"seqlen": 1024, "bs": 1, "dim": 384, "hidden_dim": 128, "nomask": True, "distribution": "normal"}
'''


MLA_DECODE_PROMPT = r'''You are an expert Triton engineer tasked with translating PyTorch code into highly optimized Triton kernel code.

Below is a pytorch implementation of the multi-head latent attention (MLA) module. You will want to implement a Triton kernel for the operations in the forward call:

```python
import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        theta = 10000 ** (-torch.arange(0, d_model//2,dtype=torch.bfloat16) / (d_model//2))
        self.register_buffer("theta", theta)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.size(-2)
        d_model = x.size(-1)
        assert d_model == self.d_model
        seq_idx = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        idx_theta = torch.einsum('s,d->sd', seq_idx, self.theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)
        cos = idx_theta2.cos().to(torch.bfloat16)
        sin = idx_theta2.sin().to(torch.bfloat16)
        return x * cos + self.rotate_half(x) * sin

class KVCache(nn.Module):
    def __init__(self, kv_cache_shape: tuple) -> None:
        super().__init__()
        self.register_buffer('data', torch.zeros(kv_cache_shape, dtype=torch.bfloat16, device='cuda'))
        self.seq_len = 0
        self.zero()

    def zero(self) -> None:
        self.data.zero_()
    
    def get_data(self) -> torch.Tensor:
        return self.data

    def forward(self, c_kv: torch.Tensor) -> torch.Tensor:
        assert self.seq_len + c_kv.size(1) <= self.data.size(1), "KV Cache Exceeded"

        self.data = self.data.to(c_kv.dtype)
        self.data[
            :, self.seq_len : self.seq_len + c_kv.size(1), :
        ] = c_kv
        self.seq_len += c_kv.size(1)

        return self.data[:, :self.seq_len], self.seq_len
    
@dataclass
class Config:
    batch_size: int
    dim: int
    n_heads: int
    q_lora_rank: int 
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    seq_len: int
    max_seq_len: int
    kv_cache_shape: tuple
    Q_proj_down_weight: torch.Tensor
    Q_proj_up_weight: torch.Tensor
    KV_proj_down_weight: torch.Tensor
    KV_proj_up_weight: torch.Tensor
    wo_weight: torch.Tensor

class MLA(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.nope_head_dim = config.qk_nope_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        # Down-projection matrices
        self.Q_proj_down = nn.Linear(self.dim, self.q_lora_rank, bias=False, dtype=torch.bfloat16)
        self.KV_proj_down = nn.Linear(self.dim, self.kv_lora_rank + self.rope_head_dim, bias=False, dtype=torch.bfloat16)

        # Up-projection and rope projection matrices
        self.Q_proj_up = nn.Linear(self.q_lora_rank, (self.nope_head_dim + self.rope_head_dim) * self.n_heads, bias=False, dtype=torch.bfloat16)
        self.KV_proj_up = nn.Linear(self.kv_lora_rank, (self.nope_head_dim + self.v_head_dim) * self.n_heads, bias=False, dtype=torch.bfloat16)

        # RoPE on half embeddings
        self.q_rope = RoPE(self.rope_head_dim)
        self.k_rope = RoPE(self.rope_head_dim)

        # Output projection
        self.wo = nn.Linear(self.v_head_dim * self.n_heads, self.dim, dtype=torch.bfloat16, bias=False)
        self.eps = 1e-6
   
    def forward(self, x: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        # seq_len = 1 always here
        batch_size, seq_len, model_dim = x.size()

        ## Step 1: Handle down-projection + KV cache ##
        
        q_lora = self.Q_proj_down(x)
        kv_lora = self.KV_proj_down(x)
        kv_lora, kv_len = kv_cache(kv_lora)
        query_pos = kv_len - 1

        ## Step 2: Up-project and prepare NoPE + RoPE ##
        
        # Handle queries Q first
        q_nope_and_rope = self.Q_proj_up(q_lora).view(
            batch_size, seq_len, self.n_heads, self.nope_head_dim + self.rope_head_dim)
        q_nope, q_rope = torch.split(q_nope_and_rope, [self.nope_head_dim, self.rope_head_dim], dim=-1)

        # Handle keys and values K/V. V does not need RoPE
        kv_nope, k_rope = torch.split(kv_lora, [self.kv_lora_rank, self.rope_head_dim], dim=-1)
        kv_nope = self.KV_proj_up(kv_nope).view(
            batch_size, kv_len, self.n_heads, self.nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_nope, [self.nope_head_dim, self.v_head_dim], dim=-1)

        ## Step 3: Handle RoPE Stream ##
        
        # Compute RoPE for queries and combine with no-RoPE part
        q_rope = q_rope.permute(0, 2, 1, 3) # bs x n_heads x seq_len x rope_head_dim
        q_rope = self.q_rope(q_rope, start_pos=query_pos)

        q_nope = q_nope.permute(0, 2, 1, 3) # bs x n_heads x seq_len x rope_head_dim
        q = torch.concat([q_nope, q_rope], dim=-1)

        # Compute RoPE for keys and combine with no-RoPE part
        k_rope = k_rope[:, None, :, :]
        k_rope = self.k_rope(k_rope).expand(-1,self.n_heads,-1,-1)
        k_nope = k_nope.permute(0, 2, 1, 3) # bs x kv_len x n_heads x rope_head_dim
        k = torch.concat([k_nope, k_rope], dim=-1)
                
        ## Step 4: Compute Multi-head Attention ##
        
        v = v.permute(0, 2, 1, 3) # bs x n_heads x kv_len x v_head_dim
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.rope_head_dim + self.nope_head_dim)
        attn = F.softmax(scores, dim=-1).to(torch.bfloat16)
        y = torch.matmul(attn, v).view(batch_size, 1, -1)
        y = self.wo(y)

        return y, kv_cache.get_data()
```

Your function should be defined as 'custom_kernel' (skeleton provided below)

```python
### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
from reference import KVCache, Config  # Definition of KVCache and Config classes are shown above. Must import this way. Do not rewrite yourself.
### END OF IMPORT STATEMENTS BLOCK ###

### Import other packages here if needed

def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, KVCache]:
    config, x, kv_cache = data
    
    bs = config.batch_size
    sl = config.seq_len
    pl = kv_cache.seq_len
    msl = config.max_seq_len
    nh = config.n_heads
    d =  config.dim
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    dnope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim

    wDQ  = config.Q_proj_down_weight
    wDKV = config.KV_proj_down_weight
    wUQ  = config.Q_proj_up_weight
    wUKV = config.KV_proj_up_weight
    wO   = config.wo_weight                      
    
    # Perform MLA operations to process data into output and updated kv_cache

    return output, kv_cache.data
```

with the following signature:

Input:
- `data`: Tuple of (config: Config, x: torch.Tensor, kv_cache: KVCache)
    - config: An instance of class `Config` containing model configurations and weights
    - x: Input tensor of shape [batch_size, seq_len, dim]
    - kv_cache: An instance of KVCache class for caching the keys and values

Output:
- output: Output tensor [batch_size, seq_len, dim]
- kv_cache.data: The data field of the updated `KVCache` instance with the new keys and values added

Below are the different configs that your kernel will be tested on:

Common configs:
  - {"batch_size": 128, "seq_len": 1, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "v_head_dim": 128, "n_heads": 128, "dim": 7168, "q_lora_rank": 1536, "max_seq_len": 8192}

For correctness check:
  - {"prefill": 128}
  - {"prefill": 512}
  - {"prefill": 1024}
  - {"prefill": 2048}

For performance benchmark (optimize runtime for these):
  - {"prefill": 6144}

'''

MLA_DECODE_PROMPT_END = r'''

Rules:
- The tensors arguments passed in will be already on your cuda device.
- The weights for all parameters in the MLA will be given as input.
- All weights and data will be in `torch.bfloat16` format.
- Define all of your code in one final ```python ``` block.
- The entrypoint to your code must be named 'custom_kernel'.
- You will be using triton 3.4.0 and your kernels will be run on an Nvidia H200 GPU.
- Consider optimizing multiple operations with triton, not just limited to softmax. E.g., rope, attention, etc.
- You are allowed to use torch.compile().

Important rules in triton 3.4.0:
- `tl.load` does not have an argument called `dtype`. Never use it like `tl.load(..., dtype=...)`.
- Triton dtypes are not callable, so never use them like `tl.float16(1.0)`, `tl.float32(0.0)`.
- `tl.arange(start, end)`:
    - range length (end - start) must be power-of-2
    - start, end must be of type `tl.constexpr`
- `tl.range(start, end, step, num_stages)`:
    - keep loop index type stable, don't reassign it
    - start, end, step do not have to be `tl.constexpr` but must stay scalar integer types
    - num_stages must be `tl.constexpr`
- Do not something like x[0] or offs[0] inside a Triton kernel. Triton tensors are SIMD vectors; scalar indexing like [0] is not generally supported.

Here's an simple example correctly following these rules:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_right(
    x_ptr, y_ptr, out_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,          # ✅ constexpr; also power-of-2 for tl.arange
    ROW_STEP: tl.constexpr,
    NUM_STAGES: tl.constexpr,     # ✅ constexpr; used by tl.range(num_stages=...)
):
    pid = tl.program_id(axis=0)

    # ------------------------------------------------------------
    # arange: ✅ constexpr args + ✅ power-of-2 range
    # ------------------------------------------------------------
    offs = pid * BLOCK + tl.arange(0, BLOCK)   # (0, BLOCK) are constexpr
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)

    # ------------------------------------------------------------
    # Dtypes not callable: ✅ typed constants and casting
    # ------------------------------------------------------------
    one_f32 = tl.full([], 1.0, tl.float32)               # typed scalar
    acc = tl.zeros((BLOCK,), dtype=tl.float32)           # typed vector
    acc = tl.cast(x, tl.float32) + tl.cast(y, tl.float32) + one_f32

    # ------------------------------------------------------------
    # Avoid x[0]: ✅ scalar address load + broadcast
    # ------------------------------------------------------------
    base = tl.full([], pid * BLOCK, tl.int32)
    x0 = tl.load(x_ptr + base, mask=(base < n_elements), other=0.0)
    x0_vec = tl.full((BLOCK,), x0, tl.float32)

    out_vec = acc + x0_vec

    # ------------------------------------------------------------
    # tl.range: ✅ keep loop index type stable, don't reassign it
    #
    # WRONG (causes "Loop-carried variable ... type stays consistent" assertion):
    #   for row in tl.range(row, n_rows, row_step):
    #       row = tl.load(...)  # ❌ row (int32) reassigned to tensor/bf16/...
    #
    # RIGHT:
    #   - use a fresh name for loop index (e.g., r)
    #   - compute offsets/tensors into *different* vars
    #   - keep r as an integer index (int32) throughout
    # ------------------------------------------------------------
    # We'll do a tiny staged reduction over "rows" just as a demo.
    n_rows = tl.full([], 4, tl.int32)  # small fixed count for demo (scalar int32)

    extra = tl.zeros((BLOCK,), dtype=tl.float32)
    for r in tl.range(0, n_rows, ROW_STEP, num_stages=NUM_STAGES):
        # r is an int32 loop index. Keep it that way.

        # Use r to build an integer shift; keep shifts as ints too.
        shift = r * tl.full([], 1, tl.int32)

        # Compute new offsets (int) without mutating r:
        offs_r = offs + shift

        # Load something; store into a separate var (tensor), not r:
        xr = tl.load(x_ptr + offs_r, mask=(offs_r < n_elements), other=0.0)
        extra += tl.cast(xr, tl.float32)

    out_vec = out_vec + extra

    tl.store(out_ptr + offs, tl.cast(out_vec, tl.float16), mask=mask)
```

'''


MLA_DECODE_INITIAL_STATE = r'''import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config

@triton.jit
def rope_swap_halves_kernel(
    x_ptr,                      # [B, T, D] bf16/fp16/fp32
    cos_ptr, sin_ptr,            # [T, D] or [D] depending on stride_cos_t/stride_sin_t
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,             # must be even
    stride_xb, stride_xt, stride_xd,
    stride_cos_t, stride_cos_d,
    stride_sin_t, stride_sin_d,
    BLOCK_HALF: tl.constexpr,    # processes D/2 in blocks
):
    pid = tl.program_id(0)
    bt = pid
    b = bt // T
    t = bt - b * T

    half = D // 2

    off = tl.arange(0, BLOCK_HALF)
    mask = off < half

    # pointers for x halves
    x_base = x_ptr + b * stride_xb + t * stride_xt
    x0_ptr = x_base + off * stride_xd                  # first half
    x1_ptr = x_base + (half + off) * stride_xd         # second half

    # pointers for cos/sin halves
    # If stride_cos_t == 0 => broadcast over t (query case)
    cos_base = cos_ptr + t * stride_cos_t
    sin_base = sin_ptr + t * stride_sin_t

    c_ptr = cos_base + off * stride_cos_d
    s_ptr = sin_base + off * stride_sin_d

    # load
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(c_ptr,  mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(s_ptr,  mask=mask, other=0.0).to(tl.float32)

    # RoPE with rotate_half swap-halves:
    # out0 = x0*c - x1*s
    # out1 = x1*c + x0*s
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    # store back in-place (or change to out_ptr if you want out-of-place)
    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def rope_inplace_query(q_rope: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor):
    # q_rope: (bs, nh, d_rope) bf16
    # cos_q/sin_q: (d_rope,) bf16 (or fp16/fp32)
    assert q_rope.is_cuda
    assert q_rope.shape[-1] % 2 == 0
    bs, nh, d_rope = q_rope.shape

    # Use half-dim block; pick a power-of-2 up to 256
    half = d_rope // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()
    # BLOCK_HALF = min(BLOCK_HALF, 256)

    grid = (bs * nh,)

    rope_swap_halves_kernel[grid](
        q_rope,
        cos_q, sin_q,
        B=bs, T=nh, D=d_rope,
        stride_xb=q_rope.stride(0),
        stride_xt=q_rope.stride(1),
        stride_xd=q_rope.stride(2),
        # broadcast cos/sin across t by setting stride_*_t = 0
        stride_cos_t=0, stride_cos_d=cos_q.stride(0),
        stride_sin_t=0, stride_sin_d=sin_q.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )

# ----------------------------------------------------------------------
# 0️⃣  RoPE cache (cos / sin tables) – built once per config
# ----------------------------------------------------------------------
_rope_cache = {}

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return cached (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
            torch.bfloat16
        )  # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (max_seq_len, 1)
        idx = pos * theta[None, :]          # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)  # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
# 1️⃣  Triton row‑wise softmax (bf16)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in = row * stride_in
    row_off_out = row * stride_out

    # ---------- max ----------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---------- normalize ----------
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)

def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row‑wise softmax for a 2‑D bf16 tensor using Triton."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    n_rows, n_cols = x.shape

    # pick a power‑of‑2 block size (capped at 1024)
    if n_cols <= 32:
        BLOCK_SIZE = 32
    elif n_cols <= 64:
        BLOCK_SIZE = 64
    elif n_cols <= 128:
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 1 << (n_cols - 1).bit_length()
        BLOCK_SIZE = min(BLOCK_SIZE, 1024)

    out = torch.empty_like(x)
    grid = (n_rows,)
    _softmax_kernel[grid](
        out,
        x,
        out.stride(0),
        x.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=2,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
# 2️⃣  Custom kernel – MLA forward (optimised)
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward step of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor    # shape (batch, seq_len, dim), bf16
    kv_cache_tensor : torch.Tensor   # updated KV‑cache tensor
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Unpack configuration (readability)
    # ------------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len               # always 1 in the provided configs
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # ------------------------------------------------------------------
    # Extract weight tensors (already on device & bf16)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight           # (dq, dim)
    wDKV  = config.KV_proj_down_weight          # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight             # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight            # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                    # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 1️⃣  Down‑project
    # ------------------------------------------------------------------
    q_lora = F.linear(x, wDQ)                     # (bs, sl, dq)
    kv_lora_input = F.linear(x, wDKV)             # (bs, sl, dkv + d_rope)

    # ------------------------------------------------------------------
    # 2️⃣  Update KV‑cache (in‑place)
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input)    # kv_lora: (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                         # absolute position for query RoPE

    # ------------------------------------------------------------------
    # 3️⃣  Up‑project queries
    # ------------------------------------------------------------------
    # sl == 1 ⇒ squeeze before linear
    q_up = F.linear(q_lora.squeeze(1), wUQ)        # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)     # (bs, nh, d_total)
    q_nope = q_up[..., :d_nope]                   # (bs, nh, d_nope)
    q_rope = q_up[..., d_nope:]                   # (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣  Split KV into latent (no‑PE) and RoPE parts
    # ------------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]            # (bs, kv_len, dkv)
    k_rope_input = kv_lora[..., dkv:]            # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 5️⃣  RoPE – use cached cosine / sine tables
    # ------------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # query side (single position)
    cos_q = cos_table[query_pos].view(d_rope).contiguous()  # (d_rope,)
    sin_q = sin_table[query_pos].view(d_rope).contiguous()  # (d_rope,)
    rope_inplace_query(q_rope, cos_q, sin_q)

    # key side (all cached positions)
    cos_k = cos_table[:kv_len]                        # (kv_len, d_rope)
    sin_k = sin_table[:kv_len]                        # (kv_len, d_rope)
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k   # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 6️⃣  Latent projection for the “no‑PE” query part
    # ------------------------------------------------------------------
    # wUKV shape: ((d_nope+dv)*nh, dkv) → view as (nh, d_nope+dv, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)          # (nh, d_nope+dv, dkv)
    wK = wUKV_view[:, :d_nope, :]                        # (nh, d_nope, dkv)
    # q_nope: (bs, nh, d_nope)  wK: (nh, d_nope, dkv) → (bs, nh, dkv)
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)   # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 7️⃣  Compute attention scores (latent + RoPE)
    # ------------------------------------------------------------------
    # latent part: q_nope_latent @ kv_nope_input^T
    kv_nope_T = kv_nope_input.transpose(1, 2)            # (bs, dkv, kv_len)
    scores_nope = torch.matmul(q_nope_latent, kv_nope_T) # (bs, nh, kv_len)

    # RoPE part: q_rope @ k_rope^T
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))  # (bs, nh, kv_len)

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale        # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 8️⃣  Softmax (Triton) → attention weights
    # ------------------------------------------------------------------
    scores_flat = scores.reshape(bs * nh, kv_len)       # (B*H, kv_len)
    attn_flat = _triton_softmax(scores_flat)            # (B*H, kv_len) bf16
    attn = attn_flat.view(bs, nh, kv_len)               # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 9️⃣  Weighted sum of latent keys (M)
    # ------------------------------------------------------------------
    M = torch.matmul(attn, kv_nope_input)               # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 🔟  Project aggregated latent keys to per‑head values
    # ------------------------------------------------------------------
    wV = wUKV_view[:, d_nope:, :]                       # (nh, dv, dkv)
    wV_T = wV.permute(0, 2, 1)                          # (nh, dkv, dv)
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)      # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 1️⃣1️⃣ Merge heads & final linear projection
    # ------------------------------------------------------------------
    y = y_head.reshape(bs, nh * dv)                     # (bs, nh*dv)
    y = y.unsqueeze(1)                                   # (bs, 1, nh*dv)
    output = F.linear(y, wO)                            # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return the output and the updated KV‑cache tensor
    # ------------------------------------------------------------------
    return output, kv_cache.data
'''

MLA_DECODE_INITIAL_VALUE = -3846.045  # H200
