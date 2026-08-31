"""Task verifier adapters used by Guidance-TTT."""

from .polyomino import extract_cpp_solution_code, verify_polyomino_solution_text
from .trimul import extract_trimul_solution_code, verify_trimul_solution_text
from .vliw_kernel import extract_vliw_solution_code, verify_vliw_solution_text

__all__ = [
    "extract_cpp_solution_code",
    "extract_trimul_solution_code",
    "extract_vliw_solution_code",
    "verify_polyomino_solution_text",
    "verify_trimul_solution_text",
    "verify_vliw_solution_text",
]
