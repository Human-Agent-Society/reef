"""The harness evolution loop, in the open.

One pass on pi (``./run.sh``):

    record - each task goes once through reef inference, so reef serves the
             reply and records the exchange against a receipt
    report - the reply is graded with the same grader the evolve gate uses
             and the score is reported against the receipt; every failure
             (score 0.0, inside the max_score window) batches and triggers
             one gated evolve step - the model proposes a mutation over its
             own failures, real episodes score it, a win publishes
    pull   - GET /reef/harness returns the winning composition; the evolved
             skill, tool, hook, and graph files are printed

On reef's native harness (``./run.sh native``) the same pass runs through a
resident ``reef-native serve`` process, so the publish lands in a process
that is already running:

    pull   - ``python3 run.py pull`` writes the served seed tree under
             work/tree with the model binding at this Reef; run.sh then
             starts ``reef-native serve`` on it, following the head
    turns  - ``python3 run.py native`` sends each task to the process as one
             turn, grades the reply, and reports the score through the
             wrapper's ``report`` command, which claims the turn's receipts
    mount  - the failing report batches and triggers the evolve step; the
             winning tree publishes and the process mounts it while it runs,
             which its log shows as ``harness/mount``
    again  - the first task goes to the same process once more, and the
             new session shows the stages of the mounted graph

Start this through ./run.sh: it writes work/tasks.json and starts the Reef
these constants point at, on pi (serve.yaml) or on reef's native harness
(./run.sh native, serve-native.yaml).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from reef_client import ReefClient, ReefClientError

from harness import evolution

SERVICE_URL = "http://127.0.0.1:8900"  # the Reef run.sh started
SCENARIO = "harness-evolve-demo"  # this workload's isolated lane
TOKEN = "reef-local"  # matches serve.yaml
MODEL = "qwen3-8b"  # matches serve.yaml's upstream_model
PULL_TIMEOUT_S = 900.0
# run.sh polls the head every 5 s, but a poll that started during the evolve step waits behind the step's
# lock until the client's 30 s timeout; the mount lands on the poll after that one.
MOUNT_TIMEOUT_S = 120.0

WORK = Path(__file__).resolve().parent / "work"
# run.sh copies serve.yaml's evolution.tasks here; the recorded traffic and
# the evolve episodes run the same three tasks.
TASKS_FILE = WORK / "tasks.json"
# The native variant: the pulled tree the serve process runs, and the workspace its tools work in.
TREE_DIR = WORK / "tree"
SCRATCH_DIR = WORK / "scratch"
SIDECAR = ".reef-harness-release"


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


# -- the native variant: the serve form ----------------------------------------------------------------------


def _manifest(client):
    # A scenario exists once traffic has named it: the manifest route answers 404 for a name it has never
    # seen, so one minimal inference creates the scenario, whose base release is the seed tree.
    try:
        return client.get("/reef/harness", extra_headers={"x-reef-scenario": SCENARIO})
    except ReefClientError as exc:
        if exc.status != 404:
            raise
    client.inference_with_record(
        SCENARIO,
        "/v1/chat/completions",
        {"model": MODEL, "messages": [{"role": "user", "content": "Reply with one word."}], "max_tokens": 1},
    )
    return client.get("/reef/harness", extra_headers={"x-reef-scenario": SCENARIO})


def pull():
    """Write the served tree under work/tree, the sidecar naming its release, and the model binding at this Reef."""
    client = ReefClient(SERVICE_URL, token=TOKEN, timeout_s=300.0)
    manifest = _manifest(client)
    shutil.rmtree(TREE_DIR, ignore_errors=True)
    for relative, text in manifest["files"].items():
        target = TREE_DIR / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    sidecar = {
        "release_id": manifest["release_id"],
        "content_id": manifest["content_id"],
        "files": sorted(manifest["files"]),
    }
    (TREE_DIR / SIDECAR).write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    binding = {"api": "openai", "base_url": SERVICE_URL, "api_key": TOKEN, "model": MODEL}
    (TREE_DIR / "native" / "models.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    print(f"pulled release {manifest['release_id']} into {TREE_DIR}")


def _wrapper_env():
    """The five settings the install script bakes into reef-<adapter>, so the wrapper's report claims the spool."""
    return {
        **os.environ,
        "REEF_HARNESS_BINARY": shutil.which("reef-native") or "reef-native",
        "REEF_HARNESS_COMPOSE": str(TREE_DIR / "native"),
        "REEF_HARNESS_SCENARIO": SCENARIO,
        "REEF_HARNESS_ADAPTER": "native",
        "REEF_HARNESS_ENV_VAR": "REEF_NATIVE_DIR",
        "REEF_TOKEN": TOKEN,
    }


def turn(prompt, session=None):
    """One turn on the serve process; the events as written and the result event."""
    command = ["reef-native", "turn", "--tree", str(TREE_DIR), "-p", prompt, "--workdir", str(SCRATCH_DIR)]
    if session:
        command += ["--session", session]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0 and not done.stdout.strip():
        raise SystemExit(f"reef-native turn failed: {done.stderr.strip()}; check work/serve.log")
    events = [json.loads(line) for line in done.stdout.splitlines() if line.strip()]
    return events[:-1], events[-1]["data"]


def report(score, feedback):
    """The wrapper's report: it claims the oldest turn's receipts and posts the score against them."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "reef.harness.harness_wrapper",
            "report",
            "--score",
            str(score),
            "--feedback",
            feedback,
        ],
        env=_wrapper_env(),
        check=True,
    )


def _mount_events(release_id):
    """Every harness/mount of ``release_id`` the serve process logged, in serve.jsonl or in a session."""
    found = []
    for path in sorted((TREE_DIR / "native" / "sessions").rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event["type"] == "harness/mount" and event["data"].get("release_id") == release_id:
                found.append((path.relative_to(TREE_DIR).as_posix(), event["data"]))
    return found


def native_main():
    tasks = json.loads(TASKS_FILE.read_text())
    client = ReefClient(SERVICE_URL, token=TOKEN, timeout_s=300.0)
    seed = json.loads((TREE_DIR / SIDECAR).read_text())["release_id"]

    # turns + report: each task is one turn on the resident process; the score goes against the turn's receipts.
    failures = 0
    for index, task in enumerate(tasks, start=1):
        _, result = turn(task)
        prefix = task.split(maxsplit=1)[0]
        score = evolution.grade_text(task, result["text"])
        report(score, prefix)
        if score == 0.0:
            failures += 1
        print(f"task {index} {prefix}: score {score} (session {result['session']}, exit {result['exit']})")

    if failures == 0:
        print("every task passed: nothing batched, no evolve step runs")
        return
    print(f"{failures} failing report(s) batched; each triggers one gated evolve step (episodes take minutes)")

    # The head moves when a winning step publishes; until then the seed release is served.
    manifest = None
    deadline = time.monotonic() + PULL_TIMEOUT_S
    while manifest is None and time.monotonic() < deadline:
        try:
            current = _manifest(client)
        except (ReefClientError, TimeoutError, OSError) as exc:
            if isinstance(exc, ReefClientError) and exc.status != 404:
                raise
            time.sleep(2.0)
            continue
        if current["release_id"] != seed:
            manifest = current
            continue
        if error := client.get("/reef/status").get("error"):
            raise SystemExit(f"evolve step failed: {error}; check work/reef.log")
        time.sleep(2.0)
    if manifest is None:
        print(f"no mutation won a gate within {PULL_TIMEOUT_S:.0f}s; rerun ./run.sh native for another attempt")
        return
    release = manifest["release_id"]
    print(f"published: artifact {release} (parent {manifest['parent_release_id']})")
    print("gate metrics (the evolve step that published this artifact):")
    print(json.dumps(manifest["gate"], indent=2, sort_keys=True))

    # The process follows the head: no reinstall, no restart, one harness/mount line in its log.
    deadline = time.monotonic() + MOUNT_TIMEOUT_S
    while not _mount_events(release) and time.monotonic() < deadline:
        time.sleep(1.0)
    mounts = _mount_events(release)
    if not mounts:
        print(f"the serve process did not mount {release} within {MOUNT_TIMEOUT_S:.0f}s; check work/serve.log")
        return
    for path, data in mounts:
        print(f"{path}: harness/mount {json.dumps(data, sort_keys=True)}")

    # The same process, the first task again: the new session runs the mounted tree.
    task = tasks[0]
    events, result = turn(task)
    stages = [event["data"]["stage"] for event in events if event["type"] == "stage/enter"]
    print(f"second pass {task.split(maxsplit=1)[0]}: stages {' -> '.join(stages)}")
    print(f"second pass answer: {result['text'].strip().splitlines()[-1] if result['text'].strip() else ''}")
    print(f"second pass score: {evolution.grade_text(task, result['text'])} (session {result['session']})")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "pull":
        pull()
    elif mode == "native":
        native_main()
    else:
        main()
