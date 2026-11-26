"""Run one LangGraph task through Reef and Harbor, keeping records without a training report."""

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from reef_client import ReefClient
from smoke_provider import SmokeHandler

from reef.harness.adapters import get_adapter
from reef.harness.harbor.agent import tree_agent, trial_directory
from reef.harness.harbor.rollouts import TaskRolloutRunner
from reef.harness.tree import HarnessTree, validate_tree

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="use scripted replies; no model or API key")
    parser.add_argument("--upstream-url", help="OpenAI Chat Completions provider URL, without /v1")
    parser.add_argument("--model", help="provider model name; must support tool calling")
    parser.add_argument("--agent-host", default="host.docker.internal", help="host address reachable from Docker")
    parser.add_argument("--work-dir", type=Path, default=HERE / "work")
    args = parser.parse_args()
    if args.smoke and (args.upstream_url or args.model):
        parser.error("--smoke cannot be combined with --upstream-url or --model")
    if not args.smoke and (not args.upstream_url or not args.model):
        parser.error("provide --upstream-url and --model, or use --smoke")
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
    descriptor = get_adapter("python")
    tree = HarnessTree.read(HERE / "tree")
    validation = validate_tree(tree, descriptor)
    if not validation.passed:
        raise SystemExit(validation.message())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=args.work_dir)).resolve()
    print(f"Run files: {work}", flush=True)
    with ExitStack() as cleanup:
        environment = dict(os.environ)
        if args.smoke:
            provider = ThreadingHTTPServer(("127.0.0.1", 0), SmokeHandler)
            cleanup.callback(provider.server_close)
            thread = threading.Thread(target=provider.serve_forever, daemon=True)
            thread.start()
            cleanup.callback(provider.shutdown)
            upstream_url = f"http://127.0.0.1:{provider.server_port}"
            model = "langgraph-smoke"
            environment["REEF_UPSTREAM_API_KEY"] = ""
        else:
            upstream_url, model = args.upstream_url.rstrip("/"), args.model
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        token = secrets.token_urlsafe(24)
        environment.update(
            CUSTOM_AGENT_PORT=str(port),
            CUSTOM_AGENT_TOKEN=token,
            CUSTOM_AGENT_UPSTREAM_URL=upstream_url,
            CUSTOM_AGENT_MODEL=model,
            CUSTOM_AGENT_WORK=str(work),
        )
        log = cleanup.enter_context((work / "reef.log").open("w", encoding="utf-8"))
        process = subprocess.Popen(
            [sys.executable, "-m", "reef", "serve", "-c", str(HERE / "serve.yaml")],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            reef_url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"Reef did not become ready; see {work / 'reef.log'}")
                try:
                    with urlopen(f"{reef_url}/healthz", timeout=2):
                        break
                except (URLError, TimeoutError):
                    time.sleep(0.2)
            client = ReefClient(reef_url, token=token)
            scenario = "custom-agent"
            client.post("/reef/scenarios", scenario, {"name": scenario})
            deadline = time.monotonic() + 120
            while True:
                status = client.get(f"/reef/scenarios/{scenario}/inference/status")
                if status["state"] == "ready":
                    break
                if status["state"] == "failed" or time.monotonic() > deadline:
                    raise RuntimeError(f"Inference did not become ready; see {work / 'reef.log'}")
                time.sleep(0.2)
            tree.write(work / "tree")
            runner = TaskRolloutRunner(
                reef_url=reef_url,
                scenario=scenario,
                model=model,
                token=token,
                work_dir=work / "rollout",
                agent=tree_agent(
                    descriptor,
                    api="openai",
                    tree_dir=work / "tree",
                    files_dir=work / "rendered",
                    timeout=240,
                ),
                agent_host=args.agent_host,
                is_reporting=False,
            )
            result = runner.run(HERE / "harbor")
            summary = {
                **result.as_line(),
                "smoke": args.smoke,
                "model": model,
                "receipt_ids": list(result.receipts),
                "trial_uri": result.trial_uri,
            }
            (work / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(summary, indent=2))
            trial = trial_directory(result.trial_uri)
            command_file = None if trial is None else trial / "agent" / "command.json"
            if command_file is None or not command_file.is_file():
                raise RuntimeError(f"The agent did not run; inspect {work}")
            command = json.loads(command_file.read_text(encoding="utf-8"))
            if command["exit_code"] != 0:
                raise RuntimeError(f"The agent exited with {command['exit_code']}; inspect {command_file}")
            if result.error or result.reward != 1.0 or len(result.receipts) < 2:
                raise RuntimeError(f"The task did not pass with two recorded model calls; inspect {work}")
            print("Passed: model -> multiply tool -> model -> answer.txt -> verifier reward 1.0")
            return 0
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
