"""Answer extraction and strict equivalence, the rule the Harbor verifier applies.

Copied from harbor/imo-*/tests/grade.py rather than imported, because the
harness is installed on its own into reef-eval's environment.
"""

from __future__ import annotations

import re


def _latex_to_float(expression: str) -> float | None:
    """Numerically evaluate simple competition-answer LaTeX (frac/sqrt/pi)."""
    text = expression.strip().strip("$").replace(" ", "").replace("\\left", "").replace("\\right", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*").replace("\\!", "").replace("\\,", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\pi", "P")
    for _ in range(20):
        new = re.sub(r"\\sqrt\{([^{}]*)\}", r"s(\1)", text)
        new = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", new)
        if new == text:
            break
        text = new
    if "\\" in text or "{" in text or "}" in text:
        return None
    for _ in range(4):
        text = re.sub(r"([\dP)])\s*([Ps(])", r"\1*\2", text)
    try:
        value = eval(text, {"__builtins__": {}}, {"P": 3.141592653589793, "s": lambda v: v**0.5})
        return float(value)
    except Exception:
        return None


def answers_equal(gold: str, predicted: str | None) -> bool:
    if predicted is None:
        return False
    gold_text = str(gold).strip().strip("$").rstrip(".").replace(" ", "")
    predicted_text = predicted.strip().strip("$").rstrip(".").replace(" ", "")
    if predicted_text == gold_text:
        return True
    gold_value = _latex_to_float(gold_text)
    predicted_value = _latex_to_float(predicted_text)
    if gold_value is None or predicted_value is None:
        return False
    return abs(predicted_value - gold_value) <= 1e-6 * max(1.0, abs(gold_value))


def extract_answer(text: str) -> str | None:
    """Prefer the last \\boxed{...}; fall back to the last standalone integer."""
    starts = [m.end() for m in re.finditer(r"\\boxed\{", text)]
    for start in reversed(starts):
        depth, i = 1, start
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            candidate = text[start : i - 1].strip()
            if candidate:
                return candidate
    tail_ints = re.findall(r"(?<![\d.])(\d+)(?![\d.])", text[-400:])
    return tail_ints[-1] if tail_ints else None
