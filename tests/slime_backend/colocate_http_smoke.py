"""GPU acceptance of an HTTP receipt, teacher feedback, training, and publication."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import requests
import yaml
from reef_client import ReefClient


def main() -> None:
    run = Path(os.environ["SMOKE_RUN"])
    run.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        teacher_port = listener.getsockname()[1]
    raw = yaml.safe_load(Path("recipes/opd/examples/math/serve.yaml").read_text())
    raw["reef"].update(port=port, token="${REEF_TOKEN}")
    raw["training"].update(colocate=True, **{"ray-address": os.environ["RAY_ADDRESS"], "timeout-s": 600})
    raw["training"]["config"].update(global_batch_size=2)
    raw["training"]["config"]["checkpoint_retention"].update(max_storage="20GB", min_free_space="2GB")
    options = raw["training"]["options"]
    options.update(
        {
            "actor-num-gpus-per-node": 1,
            "num-gpus-per-node": 2,
            "tensor-model-parallel-size": 1,
            "seq-length": 2048,
            "rollout-max-response-len": 32,
            "rollout-max-context-len": 2048,
            "max-tokens-per-gpu": 2048,
            "megatron-lora-rank": 8,
            "megatron-lora-alpha": 8,
        }
    )
    options.pop("sequence-parallel")
    raw["inference"].update({"num-gpus": 1, "tensor-parallel-size": 1, "timeout-s": 600})
    raw["inference"]["options"].update({"context-length": 2048, "mem-fraction-static": 0.3})
    raw["inference"]["options"].pop("router-port")
    raw["inference"]["handler-config"]["sampling_defaults"]["max_new_tokens"] = 32
    raw["teacher"].update({"num-gpus": 1, "port": teacher_port})
    raw["teacher"]["options"].update({"context-length": 2048, "mem-fraction-static": 0.3})
    raw["pipeline"]["record2sample"]["config"]["max_teacher_tokens"] = 2048
    config = run / "config.yaml"
    config.write_text(yaml.safe_dump(raw))
    token = uuid4().hex
    environment = {
        **os.environ,
        "OPD_RUN_DIR": str(run),
        "OPD_MODEL_PATH": os.environ["MODEL_DIR"],
        "OPD_TEACHER_PATH": os.environ["MODEL_DIR"],
        "REEF_TOKEN": token,
    }
    process = subprocess.Popen([sys.executable, "-m", "reef", "serve", "-c", str(config)], env=environment)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 900
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"HTTP deployment exited: {process.returncode}")
            try:
                if requests.get(url + "/healthz", timeout=2).status_code == 200:
                    break
            except requests.ConnectionError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("HTTP deployment did not become ready")
            time.sleep(1)
        client = ReefClient(url, token=token, timeout_s=600)
        scenario = "teacher-smoke"
        versions = []
        for step in range(2):
            receipts = []
            for index in range(2):
                response, headers = client.post(
                    "/v1/chat/completions",
                    scenario,
                    {
                        "model": os.environ["MODEL_DIR"],
                        "messages": [{"role": "user", "content": f"What is {index+2} plus 3?"}],
                        "temperature": 0.7,
                        "max_tokens": 32,
                    },
                )
                assert response["choices"]
                receipt = headers["x-reef-agent-record-id"]
                record = client.get(f"/reef/scenarios/{scenario}/records/{receipt}")
                assert record["agent_record_id"] == receipt
                release = record["artifact_ref"]["release_id"]
                if step:
                    assert release == versions[-1]
                receipts.append(receipt)
            for receipt in receipts:
                client.report(
                    scenario,
                    {"agent_record_id": uuid4().hex, "metadata": {"teacher_context": ""}},
                    references=[receipt],
                )
            deadline = time.monotonic() + 600
            while True:
                rows = client.get(f"/reef/scenarios/{scenario}/artifacts")["artifacts"]
                trained = [row for row in rows if row.get("operation") == "training"]
                status = client.get("/reef/status")
                assert not status.get("error"), status
                progress = status["scenarios"][scenario]
                if (
                    len(trained) == step + 1
                    and progress["finished_jobs"] == step + 1
                    and progress["training_job"] is None
                ):
                    active = [row for row in trained if row["active"]]
                    assert len(active) == 1
                    versions.append(active[0]["release_id"])
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(f"no publication for HTTP batch {step}: {status}")
                time.sleep(1)
            with (run / "results.jsonl").open("a") as output:
                output.write(json.dumps({"step": step, "receipts": receipts, "published": versions[-1]}) + "\n")
            print(f"COLOCATE_HTTP_STEP_PASS step={step}", flush=True)
        response, headers = client.post(
            "/v1/chat/completions",
            scenario,
            {
                "model": os.environ["MODEL_DIR"],
                "messages": [{"role": "user", "content": "What is 2 plus 3?"}],
                "max_tokens": 16,
            },
        )
        assert response["choices"]
        record = client.get(f"/reef/scenarios/{scenario}/records/{headers['x-reef-agent-record-id']}")
        assert record["artifact_ref"]["release_id"] == versions[-1]
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise RuntimeError("HTTP deployment did not shut down") from None
    assert process.returncode == 0, process.returncode
    print("COLOCATE_HTTP_GPU_PASS", flush=True)


if __name__ == "__main__":
    main()
