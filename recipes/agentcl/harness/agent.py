"""Harbor adapter for the recorded student code agent; never submits feedback."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shlex
import sys
import tempfile
import uuid
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from reef_client import ReefClient

from .episode import EpisodeFault, EpisodeRunner, EpisodeSandbox, EpisodeSettings, JsonObject, Phase, ReefRecordedModel

SAFE_COMMAND_PREFIX = (
    "env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/home/student "
    "MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0 "
)


class HarborSandbox(EpisodeSandbox):
    """All execution and answer writes occur inside the supplied Harbor environment."""

    def __init__(self, environment: BaseEnvironment) -> None:
        self.environment = environment
        self.kernel_pid: int | None = None
        self.launch_receipt_path: str | None = None

    async def start(self) -> None:
        nonce = uuid.uuid4().hex
        self.launch_receipt_path = f"/tmp/agentcl-launch-{nonce}.json"
        script = (
            "import json,pathlib,subprocess,sys,time\n"
            "socket_path = pathlib.Path('/tmp/agentcl-kernel.sock')\n"
            "socket_path.unlink(missing_ok=True)\n"
            "pathlib.Path('/workspace/answer.py').unlink(missing_ok=True)\n"
            "with open('/tmp/agentcl-kernel.log', 'wb') as output:\n"
            "    process = subprocess.Popen([sys.executable, '/opt/agentcl/kernel.py', '--serve'], "
            "cwd='/workspace', stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, "
            "start_new_session=True)\n"
            "ready = False\n"
            "try:\n"
            "    deadline = time.monotonic() + 2.0\n"
            "    while process.poll() is None:\n"
            "        if socket_path.is_socket():\n"
            f"            pathlib.Path({self.launch_receipt_path!r}).write_text(json.dumps("
            f"{{'pid': process.pid, 'socket': str(socket_path), 'nonce': {nonce!r}}}))\n"
            "            print(process.pid, flush=True)\n"
            "            ready = True\n"
            "            break\n"
            "        if time.monotonic() >= deadline:\n"
            "            raise RuntimeError('isolated Python tool did not become ready')\n"
            "        time.sleep(0.02)\n"
            "    if not ready:\n"
            "        raise RuntimeError(f'isolated Python tool exited before readiness: {process.returncode}')\n"
            "finally:\n"
            "    if not ready:\n"
            "        if process.poll() is None:\n"
            "            process.kill()\n"
            "        process.wait()\n"
            "        socket_path.unlink(missing_ok=True)\n"
        )
        command = SAFE_COMMAND_PREFIX + "python3 -I -S -c " + shlex.quote(script)
        result = await self.environment.exec(command, user="student", timeout_sec=10)
        if result.return_code != 0:
            raise EpisodeFault(
                f"failed to start the isolated Python tool: exit={result.return_code}; "
                f"stdout={(result.stdout or '')[-2000:]}; stderr={(result.stderr or '')[-2000:]}; "
                f"check /tmp/agentcl-kernel.log and {self.launch_receipt_path}"
            )
        try:
            kernel_pid = int((result.stdout or "").strip())
        except ValueError as error:
            raise EpisodeFault("isolated Python tool launch returned an invalid process ID") from error
        if kernel_pid <= 1:
            raise EpisodeFault("isolated Python tool launch returned an invalid process ID")
        self.kernel_pid = kernel_pid

    async def execute(self, code: str, timeout_seconds: int, output_limit: int) -> JsonObject:
        request = base64.b64encode(
            json.dumps({"code": code, "timeout_seconds": timeout_seconds, "output_limit": output_limit}).encode()
        ).decode()
        command = SAFE_COMMAND_PREFIX + "python3 /opt/agentcl/kernel.py --execute " + shlex.quote(request)
        result = await self.environment.exec(command, user="student", timeout_sec=timeout_seconds + 5)
        if result.return_code != 0:
            raise EpisodeFault("isolated Python tool transport failed")
        observation = json.loads(result.stdout)
        if (
            not isinstance(observation, dict)
            or observation.get("status") not in ("ok", "error", "timeout")
            or not isinstance(observation.get("output"), str)
        ):
            raise EpisodeFault("isolated Python tool returned an invalid observation")
        return observation

    async def submit(self, code: str) -> None:
        encoded = base64.b64encode(code.encode()).decode()
        script = (
            "import base64,pathlib; pathlib.Path('/workspace/answer.py').write_bytes(base64.b64decode("
            + repr(encoded)
            + "))"
        )
        result = await self.environment.exec(
            SAFE_COMMAND_PREFIX + "python3 -c " + shlex.quote(script), user="student", timeout_sec=10
        )
        if result.return_code != 0:
            failure = EpisodeFault(
                f"failed to write the isolated answer artifact: exit={result.return_code}; "
                f"stdout={(result.stdout or '')[-2000:]}; stderr={(result.stderr or '')[-2000:]}"
            )
            output = (result.stdout or "") + (result.stderr or "")
            if not (
                result.return_code == 255
                and "Error: open /var/lib/containers/storage/vfs-containers/" in output
                and "/exit/" in output
                and "no such file or directory" in output
            ):
                raise failure
            with tempfile.TemporaryDirectory(prefix="agentcl-answer-") as directory:
                answer_file = Path(directory) / "answer.py"
                try:
                    await self.environment.download_file("/workspace/answer.py", answer_file)
                    answer_sha256 = hashlib.sha256(answer_file.read_bytes()).digest()
                except (OSError, RuntimeError) as error:
                    raise failure from error
                if answer_sha256 != hashlib.sha256(code.encode()).digest():
                    raise failure

    async def close(self) -> None:
        # Harbor removes the container; retain the primary fault if owned-kernel cleanup also fails.
        if self.kernel_pid is None:
            return
        active_error = sys.exception()
        receipt_id = uuid.uuid4().hex
        receipt_path = f"/tmp/agentcl-cleanup-{receipt_id}.json"
        expected = {"receipt_id": receipt_id, "kernel_pid": self.kernel_pid, "killed": True, "socket_removed": True}
        script = (
            "import json,os,pathlib,signal\n"
            "try:\n"
            f"    os.kill({self.kernel_pid}, signal.SIGKILL)\n"
            "except ProcessLookupError:\n"
            "    pass\n"
            "pathlib.Path('/tmp/agentcl-kernel.sock').unlink(missing_ok=True)\n"
            f"pathlib.Path({receipt_path!r}).write_text(json.dumps({expected!r}))\n"
        )

        async def verify_receipt(failure: Exception) -> None:
            with tempfile.TemporaryDirectory(prefix="agentcl-cleanup-") as directory:
                receipt_file = Path(directory) / "receipt.json"
                try:
                    await self.environment.download_file(receipt_path, receipt_file)
                    receipt = json.loads(receipt_file.read_text())
                except (OSError, RuntimeError, ValueError) as error:
                    raise EpisodeFault(f"isolated Python kernel cleanup could not be confirmed: {failure}") from error
                if receipt != expected:
                    raise EpisodeFault("isolated Python kernel cleanup returned an invalid receipt")

        try:
            try:
                result = await self.environment.exec(
                    SAFE_COMMAND_PREFIX + "python3 -I -S -c " + shlex.quote(script), user="student", timeout_sec=5
                )
            except RuntimeError as error:
                if str(error) != "Command timed out after 5 seconds":
                    raise
                await verify_receipt(error)
            else:
                if result.return_code != 0:
                    output = (result.stdout or "") + (result.stderr or "")
                    failure = EpisodeFault(
                        f"isolated Python kernel cleanup failed: exit={result.return_code}; output={output[-2000:]}"
                    )
                    if not (
                        result.return_code == 255
                        and "Error: open /var/lib/containers/storage/vfs-containers/" in output
                        and "/exit/" in output
                        and "no such file or directory" in output
                    ):
                        raise failure
                    # Independently confirm effects when the transport loses the exit result.
                    await verify_receipt(failure)
        except (OSError, RuntimeError):
            if active_error is None:
                raise
            active_error.add_note("isolated Python kernel cleanup also failed; Harbor owns container removal")
        else:
            self.kernel_pid = None


class HarborAgent(BaseAgent):
    """One student episode with ordered exact receipts and host-side artifacts."""

    def __init__(
        self,
        *args,
        episode_id: str,
        expected_release: str,
        phase: Phase = "train",
        service_url: str | None = None,
        scenario: str | None = None,
        expected_runtime_load_id: str | None = None,
        episode_output: str | None = None,
        max_turns: int = 8,
        max_response_tokens: int = 2048,
        max_episode_tokens: int = 8192,
        tool_timeout_seconds: int = 30,
        max_tool_output_chars: int = 8000,
        temperature: float = 0.7,
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        service_url = service_url or os.environ.get("REEF_SERVICE_URL", "")
        scenario = scenario or os.environ.get("REEF_SCENARIO", "")
        if not service_url or not scenario:
            raise ValueError("REEF_SERVICE_URL and REEF_SCENARIO are required")
        self.client = ReefClient(
            service_url, token=os.environ.get("REEF_TOKEN"), timeout_s=float(os.environ.get("REEF_TIMEOUT_S", "300"))
        )
        self.scenario = scenario
        self.settings = EpisodeSettings(
            episode_id=episode_id,
            model_name=self.model_name or "reef",
            expected_release=expected_release,
            phase=phase,
            expected_runtime_load_id=expected_runtime_load_id,
            max_turns=max_turns,
            max_response_tokens=max_response_tokens,
            max_episode_tokens=max_episode_tokens,
            tool_timeout_seconds=tool_timeout_seconds,
            max_tool_output_chars=max_tool_output_chars,
            temperature=temperature,
            seed=seed,
        )
        self.episode_output = Path(episode_output) if episode_output is not None else None

    @staticmethod
    def name() -> str:
        return "reef-agentcl"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        """The task image supplies the isolated Python kernel and dependencies."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        runner = EpisodeRunner(
            ReefRecordedModel(self.client, self.scenario),
            HarborSandbox(environment),
            self.settings,
            self.logs_dir,
            self.episode_output,
        )
        try:
            await runner.run(instruction)
        finally:
            episode = runner.snapshot()
            context.metadata = {**(context.metadata or {}), "agentcl": episode}
            context.n_input_tokens = episode["prompt_tokens"]
            context.n_output_tokens = episode["completion_tokens"]
