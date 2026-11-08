"""Run one or two OPD feasibility batches through the routed Reef frontend and verify the published student."""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import partial
from pathlib import Path
from urllib.parse import quote

from bridge import MARKER, NATIVE_URL, SCENARIO, sampling_digest
from reef_client import ReefClient, ReefClientError

from recipes.opd.examples.math.run import Question, load_questions, report_id, write_json
from reef.storage.records import RecordStore
from reef.storage.samples import consumed_sample_id
from reef.storage.sqlite import SQLiteScenarioStorage

SCENARIO_PATH = f"/reef/scenarios/{quote(SCENARIO, safe='')}"


def sample(
    outer: ReefClient,
    native: ReefClient,
    records: RecordStore,
    prompt: str,
    seed: int,
    tokens: int,
    role: str,
    artifact_id: str,
    *,
    selector: str = "auto",
) -> dict:
    """Run one completion and match the routed record to the native receipt for the expected student weights."""
    body = {
        "model": selector,
        "messages": [{"role": "user", "content": prompt}],
        "seed": seed,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": tokens,
    }
    started = time.monotonic()
    reply, headers = outer.post("/v1/chat/completions", SCENARIO, body)
    elapsed = time.monotonic() - started
    receipt = headers["x-reef-agent-record-id"]
    record = records.get(SCENARIO, receipt)
    if record is None or record.artifact_id is not None:
        raise RuntimeError(f"Missing or unexpected weight-bound routed record {receipt}")
    routing = record.payload["metadata"]["routing"]
    expected = {
        "target_id": role,
        "provider_model": role,
        "original_selector": selector,
        "attempt_index": 1,
        "prior_attempt_ids": [],
        "execution_outcome": "completed",
        "client_visible": True,
    }
    reason = "explicit_pin" if selector != "auto" else ("rule" if role == "teacher" else "default")
    if (
        any(routing[key] != value for key, value in expected.items())
        or routing["reason"] != reason
        or bool(routing["router_config_hash"]) != (selector == "auto")
        or (role == "teacher" and routing["rule_name"] != "teacher_control")
    ):
        raise RuntimeError(f"Unexpected route for {receipt}: {routing}")
    link = reply["experiment_bridge"]
    if (
        record.payload["response"]["experiment_bridge"] != link
        or link["request_sha256"] != sampling_digest(body)
        or link["role"] != role
    ):
        raise RuntimeError(f"Backend link does not match the routed receipt {receipt}")
    if role == "student":
        native_receipt = link["native_receipt"]
        if not native_receipt or native_receipt == receipt:
            raise RuntimeError("Expected a distinct native student receipt")
        stored = native.get(f"{SCENARIO_PATH}/records/{quote(native_receipt, safe='')}")
        payload = stored["payload"]
        if (
            stored["agent_record_id"] != native_receipt
            or stored["artifact_id"] != artifact_id
            or link["artifact_id"] != artifact_id
            or sampling_digest(payload) != sampling_digest(body)
            or payload["response"]["choices"] != reply["choices"]
            or payload["response"]["usage"] != reply["usage"]
        ):
            raise RuntimeError(f"Student receipt {native_receipt} does not match its response and expected weights")
    elif link["native_receipt"] is not None or link["artifact_id"] is not None:
        raise RuntimeError("Teacher serving control unexpectedly produced a native training receipt")
    return {
        "outer_receipt": receipt,
        **link,
        "selector": selector,
        "reason": reason,
        "rule_name": routing["rule_name"],
        "router_config_hash": routing["router_config_hash"],
        "snapshot_version": routing["snapshot_version"],
        "seed": seed,
        "completion_tokens": reply["usage"]["completion_tokens"],
        "wall_seconds": elapsed,
    }


def weights(native: ReefClient) -> list[dict]:
    """Return the native weight history, newest first, with exactly the newest artifact active."""
    rows = [row for row in native.get(f"{SCENARIO_PATH}/artifacts")["artifacts"] if row["component"] == "weights"]
    if not rows or [row["active"] for row in rows] != [True] + [False] * (len(rows) - 1):
        raise RuntimeError("Expected the newest native weight artifact to be active")
    return rows


def await_update(
    native: ReefClient, before: list[dict], consumed: set[str], timeout: float
) -> tuple[list[dict], dict]:
    """Wait for one publication and verify the submitted records and checkpoint belong to the same training job."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = weights(native)
        if len(history) > len(before):
            break
        error = native.get("/reef/status").get("error")
        if error:
            raise RuntimeError(f"Native training failed: {error}")
        time.sleep(2)
    else:
        raise TimeoutError("Timed out waiting for the native OPD weight publication")
    previous = before[0]["artifact_id"]
    published = history[0]
    if (
        len(history) != len(before) + 1
        or history[1]["artifact_id"] != previous
        or published["parent_artifact_id"] != previous
        or published["operation"] != "training"
    ):
        raise RuntimeError("Expected exactly one new weight artifact descended from the preceding one")
    commits = native.get(f"{SCENARIO_PATH}/commits?after_step=0&limit=100")["commits"]
    submissions = [
        c
        for c in commits
        if set(c["consumed_ids"]) == consumed and c["metrics"].get("training_job", {}).get("event") == "submitted"
    ]
    if len(submissions) != 1:
        raise RuntimeError("No unique training job consumed exactly these student receipts, reports and samples")
    submission = submissions[0]
    job_id = submission["metrics"]["training_job"]["job_id"]
    checkpoints = [
        c
        for c in commits
        if c["metrics"].get("training_job", {}).get("job_id") == job_id
        and c["metrics"]["training_job"]["event"] == "checkpoint"
        and c["activated"]
        and c["components"]["weights"] == published["artifact_id"]
    ]
    if (
        len(checkpoints) != 1
        or not submission["operation_verified"]
        or submission["activated"]
        or not checkpoints[0]["operation_verified"]
        or checkpoints[0]["consumed_ids"]
        or submission["components"]["weights"] != previous
        or submission["step"] >= checkpoints[0]["step"]
    ):
        raise RuntimeError(f"No matching published checkpoint for submitted training job {job_id}")
    return history, {"submission": submission, "checkpoint": checkpoints[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True, help="DeepMath JSONL with id and prompt fields")
    parser.add_argument("--steps", type=int, choices=(1, 2), default=2)
    parser.add_argument("--train-tokens", type=int, default=256)
    parser.add_argument("--update-timeout", type=float, default=1800, help="Seconds to wait for each new artifact")
    args = parser.parse_args()
    if not 1 <= args.train_tokens <= 256 or args.update_timeout <= 0:
        parser.error("choose 1 to 256 train tokens and a positive update timeout")
    for variable in ("ROUTE_EXP_DIR", "ROUTER_RECORD_DIR", "REEF_TOKEN", "ROUTER_TOKEN"):
        if not os.environ.get(variable):
            parser.error(f"set {variable} before starting the driver")
    questions = load_questions(args.train_data)[: 2 * args.steps]
    if (
        len({q.id for q in questions}) != 2 * args.steps
        or len({q.prompt for q in questions}) != 2 * args.steps
        or any(not q.prompt.strip() or MARKER.casefold() in q.prompt.casefold() for q in questions)
    ):
        parser.error("each update needs two distinct prompts and IDs without the teacher keyword")
    output = Path(os.environ["ROUTE_EXP_DIR"]) / "driver"
    output.mkdir(parents=True, exist_ok=False)
    with (
        closing(SQLiteScenarioStorage(Path(os.environ["ROUTER_RECORD_DIR"]))) as storage,
        closing(storage.open(SCENARIO)) as scenario,
    ):
        run(args, questions, output, scenario.records)


def run(args: argparse.Namespace, questions: list[Question], output: Path, records: RecordStore) -> None:
    """Submit one four-record OPD batch per step and check the weights served after the last update."""
    outer = ReefClient(
        os.environ.get("OPD_ROUTER_URL", "http://127.0.0.1:18920"),
        token=os.environ["ROUTER_TOKEN"],
        timeout_s=args.update_timeout,
    )
    native = ReefClient(NATIVE_URL, token=os.environ["REEF_TOKEN"], timeout_s=args.update_timeout)
    try:
        native.get(f"{SCENARIO_PATH}/artifacts")
    except ReefClientError as exc:
        if exc.status != 404:
            raise
    else:
        raise RuntimeError(f"Use a fresh native Scenario; {SCENARIO} already exists")
    native.post("/reef/scenarios", SCENARIO, {"name": SCENARIO})
    history = weights(native)
    if len(history) != 1 or history[0]["operation"] != "creation":
        raise RuntimeError("Expected a fresh weight history")
    infer = partial(sample, outer, native, records)
    for step in range(1, args.steps + 1):
        current = history[0]["artifact_id"]
        started = time.monotonic()
        prompt = f"{MARKER} Briefly give the sum of {step} and 2."
        teacher = infer(prompt, 10000 + step, 32, "teacher", current)
        rollout_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            tasks = [
                pool.submit(infer, q.prompt, step * 2 + repeat, args.train_tokens, "student", current)
                for q in questions[(step - 1) * 2 : step * 2]
                for repeat in range(2)
            ]
            students = [task.result() for task in tasks]
        report_start = time.monotonic()
        receipts = {row["native_receipt"] for row in students}
        if len(receipts) != 4:
            raise RuntimeError("Expected four unique student generations")
        consumed = set()
        for row in students:
            receipt = row["native_receipt"]
            report = report_id(SCENARIO, receipt)
            result = native.report(
                SCENARIO, {"agent_record_id": report, "metadata": {"teacher_context": ""}}, references=[receipt]
            )
            if result["agent_record_id"] != report:
                raise RuntimeError(f"Unexpected report acknowledgement for {receipt}")
            consumed.update((receipt, report, consumed_sample_id(f"{report}:0")))
        wait_start = time.monotonic()
        history, job = await_update(native, history, consumed, args.update_timeout)
        timing = {
            "teacher_control_seconds": rollout_start - started,
            "student_collection_seconds": report_start - rollout_start,
            "report_seconds": wait_start - report_start,
            "wait_and_verify_publication_seconds": time.monotonic() - wait_start,
        }
        write_json(
            output / f"step-{step}.json",
            {
                "step": step,
                "teacher_control": teacher,
                "students": students,
                "published_artifact_id": history[0]["artifact_id"],
                **job,
                "timing": timing,
            },
        )
        print(f"Step {step}: published {history[0]['artifact_id']} after four student rollouts", flush=True)
    latest = history[0]["artifact_id"]
    final = infer("Briefly give the sum of 6 and 7.", 20001, 32, "student", latest, selector="student")
    write_json(
        output / "completed.json",
        {
            "steps": args.steps,
            "training_samples": 4 * args.steps,
            "teacher_serving_controls": args.steps,
            "final_unreported_student": final,
            "final_artifact_id": latest,
        },
    )


if __name__ == "__main__":
    main()
