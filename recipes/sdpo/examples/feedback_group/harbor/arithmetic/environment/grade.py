"""Private training grader for the arithmetic feedback-group smoke task."""

import base64
import json
import re
import sys
from pathlib import Path

ANSWER = 391


def grade(text: str) -> dict[str, float | str]:
    match = re.findall(r"<answer>\s*(-?\d+)\s*</answer>", text)
    if not match:
        return {"score": 0.0, "feedback": "Missing a final <answer>integer</answer> tag."}
    value = int(match[-1])
    if value == ANSWER:
        Path("/workspace/solved.txt").write_text("391", encoding="utf-8")
        return {"score": 1.0, "feedback": "The answer passed the arithmetic check."}
    return {"score": 0.0, "feedback": "The answer is too low." if value < ANSWER else "The answer is too high."}


if __name__ == "__main__":
    print(json.dumps(grade(base64.b64decode(sys.argv[1]).decode("utf-8"))))
