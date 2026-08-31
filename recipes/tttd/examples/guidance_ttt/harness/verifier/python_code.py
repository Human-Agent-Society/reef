from __future__ import annotations

import re


def extract_python_solution_code(text: str) -> str | None:
    """Extract the final fenced Python file from an execution response."""
    solution_match = re.search(r"<solution>\s*([\s\S]*?)\s*</solution>", text)
    if solution_match:
        return extract_python_solution_code(solution_match.group(1))
    matches = list(re.finditer(r"```python\s*([\s\S]*?)\s*```", text))
    if not matches:
        return None
    return matches[-1].group(1).strip() or None
