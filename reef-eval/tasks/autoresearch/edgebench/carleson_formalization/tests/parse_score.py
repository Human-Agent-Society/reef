"""Extract the judge's numeric score from its output (stdin).

Mirrors EdgeBench's extraction: a structured JSON object with a "score"
field wins; otherwise the last `TOTAL_SCORE <n>` line; otherwise reward 0
with the reason recorded. Writes Harbor's numbers-only reward.json.
"""

import json
import re
import sys
from pathlib import Path

VERIFIER_DIR = Path("/logs/verifier")


def extract(output: str):
    start = output.find(">>>>> Start Structured Result")
    end = output.find(">>>>> End Structured Result")
    if start != -1 and end != -1:
        block = output[start + len(">>>>> Start Structured Result") : end].strip()
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and "score" in obj:
                return float(obj["score"]), "structured result block"
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    matches = re.findall(
        r"TOTAL_SCORE\s+(inf|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", output
    )
    if matches:
        return float(matches[-1]), "TOTAL_SCORE line"
    return None, "no structured score or TOTAL_SCORE found in judge output"


def main() -> None:
    output = sys.stdin.read()
    score, reason = extract(output)
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    rewards = {"reward": score if score is not None else 0.0}
    (VERIFIER_DIR / "reward.json").write_text(json.dumps(rewards))
    (VERIFIER_DIR / "reason.txt").write_text(reason)
    print(json.dumps(rewards), reason)


if __name__ == "__main__":
    main()
