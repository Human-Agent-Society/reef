"""Run two short rollouts and one TTTD update against tutorials/tinker/serve.yaml.

Rewards are synthetic so the update has a nonzero signal. This verifies the
mechanism; it is not a quality evaluation. Running it consumes Tinker credits.
"""

import argparse
import os
import time
import uuid

from reef_client import ReefClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8900")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    client = ReefClient(args.url, token=os.environ["REEF_TOKEN"], timeout_s=args.timeout)
    scenario = f"tinker-smoke-{uuid.uuid4().hex[:8]}"
    for rollout in range(2):
        response, record = client.inference_with_record(
            scenario,
            "/v1/chat/completions",
            {
                "model": "Qwen/Qwen3-8B",
                "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                "max_tokens": 32,
                "temperature": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        print(f"rollout {rollout}: {response['choices'][0]['message']['content']}")
        client.report(
            scenario,
            {
                "score": float(rollout),
                "metadata": {
                    "algorithm": "tttd",
                    "step": 0,
                    "group": 0,
                    "rollout": rollout,
                    "groups_per_step": 1,
                    "rollouts_per_group": 2,
                    "comparison_set": "tttd-step-0-group-0",
                },
            },
            references=[record],
        )
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        status = client.get("/reef/status")
        block = status.get("scenarios", {}).get(scenario) or {}
        if block.get("scenario_step", 0) >= 1 and block.get("current_runtime_load_id"):
            print(f"Committed one Tinker update for {scenario}")
            return
        time.sleep(2)
    raise TimeoutError("Reef did not commit the Tinker update; inspect service training status and logs")


if __name__ == "__main__":
    main()
