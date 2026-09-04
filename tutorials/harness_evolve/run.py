"""The harness evolution loop, in the open.

One pass:

    record - each task goes once through reef inference, so reef serves the
             reply and records the exchange against a receipt
    report - the reply is graded with the same grader the evolve gate uses
             and the score is reported against the receipt; every failure
             (score 0.0, inside the max_score window) batches and triggers
             one gated evolve step - the model proposes a mutation over its
             own failures, real episodes score it, a win publishes
    pull   - GET /reef/harness returns the winning composition; the evolved
             skill, tool, hook, and graph files are printed

Start this through ./run.sh: it writes work/tasks.json and starts the Reef
these constants point at, on pi (serve.yaml) or on reef's native harness
(./run.sh native, serve-native.yaml). This loop is the same for both.
"""

import json
import time
from pathlib import Path

from reef_client import ReefClient, ReefClientError

from harness import evolution

SERVICE_URL = "http://127.0.0.1:8900"  # the Reef run.sh started
SCENARIO = "harness-evolve-demo"  # this workload's isolated lane
TOKEN = "reef-local"  # matches serve.yaml
MODEL = "qwen3-8b"  # matches serve.yaml's upstream_model
PULL_TIMEOUT_S = 900.0

# run.sh copies serve.yaml's evolution.tasks here; the recorded traffic and
# the evolve episodes run the same three tasks.
TASKS_FILE = Path(__file__).resolve().parent / "work" / "tasks.json"


def main():
    tasks = json.loads(TASKS_FILE.read_text())
    client = ReefClient(SERVICE_URL, token=TOKEN, timeout_s=300.0)

    # record + report: the traffic the evolve steps learn from.
    failures = 0
    for index, task in enumerate(tasks, start=1):
        body, receipt = client.inference_with_record(
            SCENARIO,
            "/v1/chat/completions",
            # A cap on the reply: a local single slot server stalls behind one unbounded generation.
            {"model": MODEL, "messages": [{"role": "user", "content": task}], "max_tokens": 2048},
        )
        prefix = task.split(maxsplit=1)[0]  # keys evolution.py's ANSWERS table
        score = evolution.grade_text(task, body["choices"][0]["message"]["content"])
        client.report(
            SCENARIO,
            {"agent_record_id": f"harness-evolve-{index}", "score": score, "feedback": prefix},
            references=[receipt],
        )
        if score == 0.0:
            failures += 1
        print(f"task {index} {prefix}: score {score} (receipt {receipt})")

    if failures == 0:
        print("every task passed: nothing batched, no evolve step runs")
        return
    print(f"{failures} failing report(s) batched; each triggers one gated evolve step (episodes take minutes)")

    # pull: GET /reef/harness 404s until a winning step publishes its tree.
    manifest = None
    deadline = time.monotonic() + PULL_TIMEOUT_S
    while manifest is None and time.monotonic() < deadline:
        try:
            manifest = client.get("/reef/harness", extra_headers={"x-reef-scenario": SCENARIO})
        except ReefClientError as exc:  # noqa: PERF203 - publish poll
            if exc.status != 404:  # 404 only means nothing has published yet
                raise
            if error := client.get("/reef/status").get("error"):
                raise SystemExit(f"evolve step failed: {error}; check work/reef.log") from exc
            time.sleep(2.0)
        except (TimeoutError, OSError):
            # The evolve step runs inside the service, so a long gate (a graph that loops on verify,
            # a slow local model) leaves a poll unanswered; keep polling until the deadline.
            time.sleep(2.0)
    if manifest is None:
        print(f"no skill mutation won a gate within {PULL_TIMEOUT_S:.0f}s; rerun ./run.sh for another attempt")
        return

    print(f"published: artifact {manifest['release_id']} (parent {manifest['parent_release_id']})")
    print("gate metrics (the evolve step that published this artifact):")
    print(json.dumps(manifest["gate"], indent=2, sort_keys=True))
    print("evolved node files:")
    for path, text in sorted(manifest["files"].items()):
        if any(segment in path for segment in ("/skills/", "/tools/", "/hooks/", "/graphs/")):
            print(f"--- {path} ---")
            print(text)


if __name__ == "__main__":
    main()
