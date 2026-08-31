import random
from typing import Tuple

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(use_cuda: bool = True) -> torch.device:
    """Get the appropriate device (GPU or CPU)."""
    if use_cuda:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("No compatible GPU found. Falling back to CPU.")
    return torch.device("cpu")


# Adapted from https://github.com/linkedin/Liger-Kernel/blob/main/test/utils.py
@torch.no_grad()
def verbose_allclose(
        received: torch.Tensor,
        expected: torch.Tensor,
        rtol=1e-05,
        atol=1e-08,
        max_print=5
) -> Tuple[bool, list[str]]:
    """
    Assert that two tensors are element-wise equal within a tolerance, providing detailed information about mismatches.

    Parameters:
    received (torch.Tensor): Tensor we actually got.
    expected (torch.Tensor): Tensor we expected to receive.
    rtol (float): Relative tolerance; relative to expected
    atol (float): Absolute tolerance.
    max_print (int): Maximum number of mismatched elements to print.
    """
    # Check if the shapes of the tensors match
    if received.shape != expected.shape:
        return False, ["SIZE MISMATCH"]

    # Calculate the difference between the tensors
    diff = torch.abs(received.to(torch.float32) - expected.to(torch.float32))

    # Determine the tolerance
    tolerance = atol + rtol * torch.abs(expected)

    # Find tolerance mismatched elements
    tol_mismatched = diff > tolerance

    # Find nan mismatched elements
    nan_mismatched = torch.logical_xor(torch.isnan(received), torch.isnan(expected))

    # Find +inf mismatched elements
    posinf_mismatched = torch.logical_xor(torch.isposinf(received), torch.isposinf(expected))
    # Find -inf mismatched elements
    neginf_mismatched = torch.logical_xor(torch.isneginf(received), torch.isneginf(expected))

    # Find all mismatched elements
    mismatched = torch.logical_or(
        torch.logical_or(tol_mismatched, nan_mismatched),
        torch.logical_or(posinf_mismatched, neginf_mismatched),
    )

    mismatched_indices = torch.nonzero(mismatched)

    # Count the number of mismatched elements
    num_mismatched = mismatched.count_nonzero().item()

    # Generate detailed information if there are mismatches
    if num_mismatched >= 1:
        mismatch_details = [f"Number of mismatched elements: {num_mismatched}"]

        for index in mismatched_indices[:max_print]:
            i = tuple(index.tolist())
            mismatch_details.append(f"ERROR at {i}: {received[i]} {expected[i]}")
        if num_mismatched > max_print:
            mismatch_details.append(f"... and {num_mismatched - max_print} more mismatched elements.")
        return False, mismatch_details

    return True, [f"Maximum error: {torch.max(diff)}"]


@torch.no_grad()
def verbose_allequal(received: torch.Tensor, expected: torch.Tensor, max_print: int = 5) -> Tuple[bool, list[str]]:
    """
    Assert that two tensors are element-wise perfectly equal, providing detailed information about mismatches.

    Parameters:
    received (torch.Tensor): Tensor we actually got.
    expected (torch.Tensor): Tensor we expected to receive.
    max_print (int): Maximum number of mismatched elements to print.

    Returns:
         Empty string if tensors are equal, otherwise detailed error information
    """
    mismatched = torch.not_equal(received, expected)
    mismatched_indices = torch.nonzero(mismatched)

    # Count the number of mismatched elements
    num_mismatched = mismatched.count_nonzero().item()

    # Generate detailed information if there are mismatches
    if num_mismatched >= 1:
        mismatch_details = [f"Number of mismatched elements: {num_mismatched}"]

        for index in mismatched_indices[:max_print]:
            i = tuple(index.tolist())
            mismatch_details.append(f"ERROR at {i}: {received[i]} {expected[i]}")
        if num_mismatched > max_print:
            mismatch_details.append(f"... and {num_mismatched - max_print} more mismatched elements.")
        return False, mismatch_details

    return True, []


def match_reference(data, output, reference: callable, rtol=1e-05, atol=1e-08):
    """
    Convenient "default" implementation for tasks' `check_implementation` function.
    """
    expected = reference(data)
    good, reasons = verbose_allclose(output, expected, rtol=rtol, atol=atol)

    if len(reasons) > 0:
        return good, "\\n".join(reasons)

    return good, ''


def make_match_reference(reference: callable, **kwargs):
    def wrapped(data, output):
        return match_reference(data, output, reference=reference, **kwargs)
    return wrapped


class DisableCuDNNTF32:
    def __init__(self):
        self.allow_tf32 = torch.backends.cudnn.allow_tf32
        self.deterministic = torch.backends.cudnn.deterministic
        pass

    def __enter__(self):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.backends.cudnn.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.deterministic = self.deterministic

def clear_l2_cache():
    # import cupy as cp
    # cp.cuda.runtime.deviceSetLimit(cp.cuda.runtime.cudaLimitPersistingL2CacheSize, 0)
    # create a large dummy tensor
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda")
    # write stuff to
    dummy.fill_(42)
    del dummy
