"""Harbor verifier for IMO problem_idx 8 (gold: $-\\frac{2023}{2024^2}$).

Extracts \\boxed{} from /workspace/answer.txt and checks it against the gold
answer with the same strict equivalence checker the math-eval harness uses
(recipes/sao/examples/sao/math_eval/eval.py). Binary reward: 1.0 correct, 0.0 wrong.
"""

import re
from pathlib import Path

GOLD_ANSWER = r"$-\frac{2023}{2024^2}$"


def _latex_to_float(expression):
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


def answers_equal(gold, predicted):
    if predicted is None:
        return False
    gold_text = str(gold).strip()
    predicted_text = predicted.strip().strip("$").rstrip(".").replace(" ", "")
    if predicted_text == gold_text.replace(" ", ""):
        return True
    gold_value = _latex_to_float(gold_text)
    predicted_value = _latex_to_float(predicted_text)
    if gold_value is None or predicted_value is None:
        return False
    return abs(predicted_value - gold_value) <= 1e-6 * max(1.0, abs(gold_value))


def extract_answer(text):
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


def main():
    reward_path = Path("/logs/verifier/reward.txt")
    reward_path.parent.mkdir(parents=True, exist_ok=True)

    answer_file = Path("/workspace/answer.txt")
    if not answer_file.exists():
        reward_path.write_text("0.0\n")
        return

    text = answer_file.read_text(encoding="utf-8")
    predicted = extract_answer(text)
    if predicted is not None and answers_equal(GOLD_ANSWER, predicted):
        reward_path.write_text("1.0\n")
    else:
        reward_path.write_text("0.0\n")


if __name__ == "__main__":
    main()
