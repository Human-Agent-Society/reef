"""GPU qualification of Reef's native engines, traffic splits, logprobs, engine recovery, public deployment lifecycle
and Semantic Router routing.

``run_qualification`` serves two synthetic checkpoints with disjoint words: A on the native engine behind Reef's
remote inference component, and B on the other engine as a raw OpenAI provider. The ``features`` profile serves A
as a published artifact through the production full-weight runtime, whose component keeps each request on its
weights, and checks the inference app's splits, pins, rollouts and records. Its engine shares prefixes: the last
check counts the cached prompt tokens of A's requests, loads B into that engine during an A request, and counts
them again for B's requests. ``routing`` puts both engines behind a real Semantic Router. A router target takes
OpenAI chat requests, which a component that serves published weights refuses; there the component's engine
answers with its base model, A, as a plain endpoint without a weight-serving runtime. ``recovery`` starts the
native engine only. It removes the component's engines before the runtime's first activation, and again while they
serve B as a published artifact: each time the runtime's health watcher must restore them, the second time with B
under its runtime load ID. ``lifecycle`` changes what a scenario serves through Reef's public API only, with a real
dispatcher and its serving worker, in two scenarios. The first serves the weights A and B at the same time, both on
the native engine: two inference components, each with its own engines, behind the runtime that the remote serving
factory builds for ``inference.remote.slot-urls``. It creates the scenario with A, imports B, and serves B beside
A: each answers by its named route from its own component, under its own runtime load ID, also while the other
answers. Its updates then split the traffic 50/50, give all of it to B, and give it back to A; none of them loads
anything. An update without A ends A's route, and the serving worker releases A's component after the grace time,
while B serves. An update with A again loads it into that component, under a new runtime load ID. An update that
names a third weight artifact is refused, as two components serve two. Last, the run removes the engines of A's
component, its one change that is no public request: the watcher of A's runtime must restore A under its runtime
load ID, while B serves unchanged. Then both components stop, and a harness scenario on B's raw provider serves two
harness artifacts at the same time: it imports the second, splits the traffic 50/50, gives one artifact all the
traffic, and stops serving the other. A run writes its results into a new output directory, keeps credentials in
memory, and stops only the processes that it started.
``python -m tests.smoke.native_inference`` runs it from a shell; ``--help`` lists the options.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any
from urllib.parse import quote

from aiohttp import ClientError, ClientSession, ClientTimeout, web
from tests.smoke.inference_features import (
    TOP_ALTERNATIVES,
    Checkpoint,
    Exchange,
    ExpectedTraffic,
    LogprobOracle,
    QualificationReport,
    ReefClient,
    answer_values,
    artifact_of_bucket,
    check_logprobs,
    check_native_alternatives,
    check_private_fields_absent,
    check_secrets_absent,
    check_stored_response,
    check_traffic,
    check_words,
    file_sha256,
    private_fields_in,
    replay_records,
    write_records,
)
from tests.smoke.inference_lifecycle import ControlClient, NoTraining, SeededWeights
from tests.smoke.inference_recovery import (
    WATCHER_THREAD,
    ObservedEngines,
    ServingLog,
    ended_checks,
    first_check,
    serving_log,
)

from reef.artifact import Artifact, InMemoryRepositoryBackend
from reef.core.records_types import ROLLOUT_HEADER, AgentRecord
from reef.core.served_models import ACTIVE, CACHE_SECONDS, ScenarioModel, ScheduledStop, ServedArtifact
from reef.core.values import JsonValue
from reef.core.weight_layout import ArtifactFormat
from reef.inference.checkpoint_slots import CheckpointSlotsRuntime
from reef.inference.deployment import BUILTIN_ENGINES
from reef.inference.http import InferenceProxyRuntime, ProviderRequestHeaders
from reef.inference.interfaces import (
    InferenceHandler,
    InferenceRuntime,
    RuntimeLoadId,
    full_weight_name,
    new_runtime_load_id_incarnation,
    parse_full_weight_name,
)
from reef.inference.registry import RuntimeRegistry
from reef.inference.remote.config import (
    DEFAULT_CONTROL_TOKEN_ENV,
    DEFAULT_INFERENCE_TOKEN_ENV,
    RemoteInferenceSettings,
)
from reef.inference.remote.runtime import ComponentEngines, serving_engine_config
from reef.inference.remote.service import HEALTH_TIMEOUT_S, RemoteInferenceBackend, RemoteInferenceService
from reef.inference.serving import HEALTH_INTERVAL_S, FullWeightServingRuntime
from reef.inference.sglang.plugin import REEF_SGLANG_PLUGIN_ENV
from reef.recipe.base import WEIGHTS_COMPONENT, Recipe
from reef.scenario.live_objects import LiveObjects
from reef.service.engine_clients import DeploymentEngineClients
from reef.service.inference_service import (
    ARTIFACT_HEADER,
    RECORD_HEADER,
    SESSION_KEY_HEADER,
    build_inference_app,
    conversation_bucket,
)
from reef.service.routing import (
    DEFAULT_RULE,
    RoutingAdapter,
    RoutingSnapshot,
    is_router_config_hash,
    parse_scenario,
    parse_target,
    scenario_entrypoint,
)
from reef.service.routing_preview import PreviewSettings
from reef.service.wire import SCENARIO_HEADER
from reef.storage.coordination import Coordination
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage

PROFILES = ("features", "routing", "recovery", "lifecycle")
ENGINES = ("sglang", "vllm")
#: When set, the SHA-256 that the router binary must have.
ROUTER_SHA256_ENV = "REEF_QUALIFY_ROUTER_SHA256"
#: Versions and hashes that a launcher may set; the report copies each one that is set, as the launcher reported it.
LAUNCHER_ENV = (
    "REEF_QUALIFY_COMMIT",
    "REEF_QUALIFY_DIFF_SHA256",
    "REEF_QUALIFY_UNTRACKED_SHA256",
    "REEF_QUALIFY_IMAGE_ID",
    "REEF_QUALIFY_LAUNCHER_SHA256",
)
#: The ONNX Runtime library that the router loads from its library directory.
ONNX_RUNTIME_LIBRARY = "libonnxruntime.so"
REPOSITORY = Path(__file__).resolve().parents[2]
HARNESS_FILES = (
    "tests/smoke/native_inference.py",
    "tests/smoke/inference_features.py",
    "tests/smoke/inference_recovery.py",
    "tests/smoke/inference_lifecycle.py",
)
#: The owned processes of the lifecycle profile's two inference components, the first and the second, by name.
SLOT_COMPONENTS = ("component", "component-b")
#: The request timeout of the lifecycle profile's runtime (``inference.timeout-s``; a deployment's default is 300).
#: The serving worker may stop a checkpoint that left the routes this long plus ``CACHE_SECONDS`` after the update.
LIFECYCLE_INFERENCE_TIMEOUT_S = 10.0
#: How long the lifecycle profile waits for the serving worker to stop a checkpoint that left the routes.
RETIREMENT_TIMEOUT_S = 120.0
#: The file that marks the lifecycle profile's capacity fixture, a copy of checkpoint B.
CAPACITY_MARKER = "capacity_fixture.json"

SCENARIO = "qualification"
SEED = 1145
#: Tokens of each scored request; the in-flight request generates ``LONG_TOKENS``, within the 256-token context.
SCORED_TOKENS = 8
LONG_TOKENS = 192
INFERENCE_TIMEOUT_S = 120.0
REQUEST_TIMEOUT_S = 180.0
ENGINE_START_TIMEOUT_S = 600.0
COMPONENT_LISTEN_TIMEOUT_S = 120.0
ROUTER_READY_TIMEOUT_S = 120.0
#: The name that the raw provider serves checkpoint B under.
PROVIDER_MODEL = "qualification-provider"
#: The routing target variable that holds the component's inference token; the value stays in memory.
COMPONENT_TOKEN_ENV = "REEF_QUALIFY_COMPONENT_TOKEN"
AUTOMATIC_MODEL = "auto"
ROUTING_RULE = "escalate"
ROUTING_KEYWORD = "reef_provider_required"
PLAIN_PROMPT = "List three words."
KEYWORD_PROMPT = f"{ROUTING_KEYWORD}: list three words."
#: The prompt of the prefix-cache check. No other prompt of the run starts with its first byte, so only its own
#: requests share a prefix with it. It fits the context with ``LONG_TOKENS`` generated tokens, and it is longer
#: than one 16-token block, the unit that vLLM caches.
CACHE_PROMPT = "Prefix cache check: list three words again."

NATIVE_WORDS = ("ALPHA", "AMBER", "ASPEN")
PROVIDER_WORDS = ("BRAVO", "BASIL", "BIRCH")
#: The logits of a checkpoint's three words, in word order, and of every other token: exact in float16 as row sums.
WORD_LOGITS = (0.25, 0.0, -0.25)
OTHER_LOGIT = -24.0
VOCAB_SIZE = 320
HIDDEN_SIZE = 128
CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
END_OF_TEXT = "<|endoftext|>"
#: Native engine options sized for two small engines on one GPU; Reef adds what its serving needs.
NATIVE_OPTIONS: dict[str, dict[str, Any]] = {
    "sglang": {
        "dtype": "float16",
        "mem_fraction_static": 0.2,
        "context_length": 256,
        "max_total_tokens": 1024,
        "max_running_requests": 8,
        "disable_cuda_graph": True,
        "attention_backend": "triton",
        "disable_custom_all_reduce": True,
    },
    "vllm": {
        "dtype": "float16",
        "gpu_memory_utilization": 0.2,
        "max_model_len": 256,
        "max_num_seqs": 8,
        "enforce_eager": True,
    },
}
ENGINE_URL_LINE = re.compile(r"started \w+ engines on \d+ GPUs; serving (http://\S+)")
RUNNING_METRIC = "vllm:num_requests_running"
#: vLLM's prefix-cache counters: the prompt tokens that it looked up in the cache, and those that it found there.
CACHE_QUERIES_METRIC = "vllm:prefix_cache_queries_total"
CACHE_HITS_METRIC = "vllm:prefix_cache_hits_total"
#: How long the counters of a finished request may take to reach the engine's metrics.
METRIC_DELAY_S = 10.0
#: The health interval of the recovery profile's runtime, so that one run sees many checks; a deployment's runtime
#: checks every ``HEALTH_INTERVAL_S`` seconds.
WATCHER_INTERVAL_S = 2.0
#: The passed health checks in a row that show a watcher at rest, and how long they may take.
QUIET_CHECKS = 3
QUIET_TIMEOUT_S = 90.0
#: How long the watcher may take to restore removed engines: one engine start and one load.
RECOVERY_TIMEOUT_S = ENGINE_START_TIMEOUT_S + 120.0
#: How long the recovery profile waits after a refused request, as the refusal's ``Retry-After`` tells a client.
REFUSED_RETRY_S = 1.0


@dataclass(frozen=True)
class QualificationInputs:
    """What one run qualifies, and where it writes.

    Values: ``profile`` is one of ``PROFILES``; ``native_engine`` is one of ``ENGINES``, and the other engine is the
    raw provider, which the recovery profile does not start. ``output`` is an absolute path outside this repository,
    to a directory that does not exist yet or is empty. ``router_binary`` is an executable Semantic Router binary,
    required by and only allowed for the routing profile; ``router_library_dir`` is None or a directory with
    ``ONNX_RUNTIME_LIBRARY``, which only the router process loads its libraries from. Construction checks these
    conditions and raises ``ValueError``.
    Purpose: ``main`` makes it from the command line; ``run_qualification`` runs it.
    """

    profile: str
    native_engine: str
    output: Path
    router_binary: Path | None = None
    router_library_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError(f"the profile must be one of {', '.join(PROFILES)}, not {self.profile!r}")
        if self.native_engine not in ENGINES:
            raise ValueError(f"the native engine must be one of {', '.join(ENGINES)}, not {self.native_engine!r}")
        if not self.output.is_absolute():
            raise ValueError(f"the output directory {self.output} must be an absolute path")
        resolved = self.output.resolve()
        if resolved == REPOSITORY or REPOSITORY in resolved.parents:
            raise ValueError(f"the output directory {self.output} must be outside the repository")
        if self.output.exists() and (not self.output.is_dir() or any(self.output.iterdir())):
            raise ValueError(f"the output directory {self.output} must not exist yet or be empty")
        if self.profile != "routing":
            if self.router_binary is not None or self.router_library_dir is not None:
                raise ValueError("the router binary and library directory apply only to the routing profile")
            return
        if self.router_binary is None:
            raise ValueError("the routing profile needs the Semantic Router binary")
        if not self.router_binary.is_file() or not os.access(self.router_binary, os.X_OK):
            raise ValueError(f"the router binary {self.router_binary} is not an executable file")
        if self.router_library_dir is not None and not (self.router_library_dir / ONNX_RUNTIME_LIBRARY).is_file():
            raise ValueError(f"the router library directory {self.router_library_dir} has no {ONNX_RUNTIME_LIBRARY}")

    @property
    def provider_engine(self) -> str:
        return "vllm" if self.native_engine == "sglang" else "sglang"


@dataclass(frozen=True)
class Credentials:
    """The bearer tokens of one run: random, held only in memory, and never written to a file or a log.

    Values: ``control`` and ``inference`` are the component's tokens, ``service`` is the Reef app's, and
    ``management`` and ``selection`` are the router's. All are distinct URL-safe strings.
    """

    control: str = field(repr=False)
    inference: str = field(repr=False)
    service: str = field(repr=False)
    management: str = field(repr=False)
    selection: str = field(repr=False)

    @classmethod
    def generate(cls) -> Credentials:
        return cls(*(secrets.token_urlsafe(32) for _ in range(5)))

    def values(self) -> tuple[str, ...]:
        return (self.control, self.inference, self.service, self.management, self.selection)


class RunLayout:
    """The files of one run under ``root``, the run's output directory.

    Values: ``models``, ``logs``, ``records``, ``dispatcher`` and ``router`` are directories, and ``wire``,
    ``control``, ``records_file``, ``report`` and ``error`` are files, each a fixed name under ``root``.
    ``component_work`` gives each inference component its own work directory under ``root``.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.models = root / "models"
        self.logs = root / "logs"
        self.records = root / "records"
        self.dispatcher = root / "dispatcher"
        self.router = root / "router"
        self.wire = root / "wire.jsonl"
        self.control = root / "control.jsonl"
        self.records_file = root / "records.jsonl"
        self.report = root / "report.json"
        self.error = root / "error.txt"

    def create(self) -> None:
        for directory in (self.root, self.models, self.logs):
            directory.mkdir(parents=True, exist_ok=True)

    def component_work(self, name: str) -> Path:
        """The work directory of the inference component that runs as the owned process ``name``."""
        return self.root / f"{name}-work"


# -- Processes ---------------------------------------------------------------------------------------------------


def group_alive(group: int) -> bool:
    """Whether a process of the process group ``group`` still exists; a group of someone else counts as gone."""
    try:
        os.killpg(group, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def descendants_of(pid: int) -> list[Any]:
    """The descendants of ``pid`` as ``psutil`` processes, which remember their start time; empty without psutil."""
    try:
        import psutil
    except ImportError:
        return []
    try:
        return list(psutil.Process(pid).children(recursive=True))
    except psutil.Error:
        return []


class OwnedProcess:
    """One external server that this run started, as the leader of its own session, with its output in one file.

    Values: ``name`` labels it in errors and the report; ``command`` is its argument list, which holds no
    credential; ``log`` is the file of its output; ``grace_s`` is how long ``stop`` waits after asking it to exit;
    ``process`` is the child.
    Purpose: ``OwnedProcesses.start`` makes it. ``stop`` signals the child, then whatever remains of its process
    group, then the descendants that it had when ``stop`` began, if they still run; nothing else.
    """

    def __init__(
        self,
        name: str,
        command: Sequence[str],
        *,
        environment: Mapping[str, str],
        log: Path,
        grace_s: float,
        cwd: Path | None = None,
    ) -> None:
        self.name = name
        self.command = list(command)
        self.log = log
        self.grace_s = grace_s
        with log.open("ab") as stream:
            self.process = subprocess.Popen(
                self.command,
                env=dict(environment),
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    def check_running(self) -> None:
        status = self.process.poll()
        if status is not None:
            raise RuntimeError(f"{self.name} exited with status {status}; see {self.log}")

    def stop(self) -> dict[str, Any]:
        group = self.process.pid
        descendants = descendants_of(group)
        if self.process.poll() is None:
            # The leader first, so that its own shutdown stops what it started.
            self.process.terminate()
            with suppress(subprocess.TimeoutExpired):
                self.process.wait(self.grace_s)
        signals = []
        for signal_number in (signal.SIGTERM, signal.SIGKILL):
            if not group_alive(group):
                break
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(group, signal_number)
            signals.append(signal_number.name)
            deadline = time.monotonic() + 10
            while group_alive(group) and time.monotonic() < deadline:
                time.sleep(0.1)
        if self.process.poll() is None:
            self.process.kill()
        with suppress(subprocess.TimeoutExpired):
            self.process.wait(10)
        killed = []
        for process in descendants:
            # psutil compares start times, so a reused process ID is not this descendant.
            with suppress(Exception):
                if process.is_running() and process.status() != "zombie":
                    process.kill()
                    killed.append(process)
        remaining = 0
        if killed:
            import psutil

            remaining = len(psutil.wait_procs(killed, timeout=10)[1])
        return {
            "name": self.name,
            "exit_status": self.process.returncode,
            "group_signals": signals,
            "group_remaining": group_alive(group),
            "descendants_killed": len(killed),
            "descendants_remaining": remaining,
        }

    @staticmethod
    def problem(outcome: Mapping[str, Any]) -> str | None:
        """What ``stop`` left wrong according to its ``outcome``: a failure, or a process that still runs."""
        if "error" in outcome:
            return f"{outcome['name']} did not stop: {outcome['error']}"
        if outcome["exit_status"] is None or outcome["group_remaining"] or outcome["descendants_remaining"]:
            return f"{outcome['name']} left owned processes running"
        return None


class OwnedProcesses:
    """The external servers of one run, in start order; ``stop_all`` stops them in reverse order."""

    def __init__(self, logs: Path) -> None:
        self.logs = logs
        self.started: list[OwnedProcess] = []

    def start(
        self,
        name: str,
        command: Sequence[str],
        *,
        environment: Mapping[str, str],
        grace_s: float,
        cwd: Path | None = None,
    ) -> OwnedProcess:
        process = OwnedProcess(
            name, command, environment=environment, log=self.logs / f"{name}.log", grace_s=grace_s, cwd=cwd
        )
        self.started.append(process)
        return process

    def stop_all(self) -> list[dict[str, Any]]:
        outcomes = []
        for process in reversed(self.started):
            try:
                outcomes.append(process.stop())
            except Exception as exc:  # Stop the others as well; the report keeps the failure.
                outcomes.append({"name": process.name, "error": f"{type(exc).__name__}: {exc}"})
        self.started.clear()
        return outcomes


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


async def wait_for_port(port: int, process: OwnedProcess, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        process.check_running()
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{process.name} did not listen on port {port} within {timeout_s:.0f} s") from None
            await asyncio.sleep(0.5)
            continue
        writer.close()
        await writer.wait_closed()
        return


async def wait_for_models(
    session: ClientSession, url: str, model: str, process: OwnedProcess, timeout_s: float
) -> list[str]:
    """Wait until the OpenAI server at ``url`` lists ``model``; return the models it lists."""
    deadline = time.monotonic() + timeout_s
    while True:
        process.check_running()
        try:
            async with session.get(url + "/v1/models", timeout=ClientTimeout(total=10)) as response:
                payload = await response.json() if response.status == 200 else None
        except (ClientError, TimeoutError, ValueError):
            payload = None
        if isinstance(payload, dict):
            listed = sorted(str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict))
            if model not in listed:
                raise RuntimeError(f"{process.name} serves {listed}, not {model!r}")
            return listed
        if time.monotonic() > deadline:
            raise TimeoutError(f"{process.name} did not serve {model!r} within {timeout_s:.0f} s; see {process.log}")
        await asyncio.sleep(1)


def engine_urls(component_log: Path) -> list[str]:
    """The private engine address of each engine stack that the component logged the start of, in start order."""
    return [str(url) for url in ENGINE_URL_LINE.findall(component_log.read_text(errors="replace"))]


def engine_url(component_log: Path) -> str:
    """The private engine address that the component logged when it last started engines; observation only."""
    urls = engine_urls(component_log)
    if not urls:
        raise RuntimeError("the component log names no engine address")
    return urls[-1]


def metric_total(metrics: str, name: str) -> float | None:
    """The sum of the samples named ``name`` in the Prometheus text ``metrics``; None when it has no such sample."""
    values = [
        float(line.rsplit(" ", 1)[1]) for line in metrics.splitlines() if line.startswith((name + "{", name + " "))
    ]
    return sum(values) if values else None


async def cancelled(task: asyncio.Task[Exchange]) -> None:
    """Cancel ``task`` and wait until it ended, whatever its outcome."""
    task.cancel()
    with suppress(BaseException):
        await task


# -- Checkpoints ---------------------------------------------------------------------------------------------------


def byte_symbols() -> list[str]:
    """The printable symbol of each byte in byte order: GPT-2's byte-level alphabet."""
    # The bytes from "!" to "~", from U+00A1 to U+00AC and from U+00AE to U+00FF are their own symbols.
    printable = {*range(0x21, 0x7F), *range(0xA1, 0xAD), *range(0xAE, 0x100)}
    symbols = []
    unprintable = 0
    for byte in range(256):
        if byte in printable:
            symbols.append(chr(byte))
        else:
            symbols.append(chr(256 + unprintable))
            unprintable += 1
    return symbols


def write_checkpoints(root: Path) -> tuple[Checkpoint, Checkpoint]:
    """Write checkpoint A (``native``) and B (``provider``) under ``root``, and check each from its saved files.

    Both are two-layer Qwen2 models with zero attention and MLP weights, so every hidden state is the all-ones
    embedding after normalization and the language-model head alone sets the next-token distribution, the same at
    every position: ``WORD_LOGITS`` for the checkpoint's three words and ``OTHER_LOGIT`` for every other token, end
    of text included. Each head row holds its logit divided by ``HIDDEN_SIZE``, so every logit is exact in float16.
    Sampling at temperature 1 then emits the checkpoint's words only, with three distinct log probabilities.
    Transformers' own ``Qwen2Tokenizer`` writes the shared tokenizer: every byte is a token, so any prompt encodes
    without merges, and each word is one more token.
    """
    import torch
    from tokenizers import pre_tokenizers
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer

    symbols = byte_symbols()
    if set(symbols) != set(pre_tokenizers.ByteLevel.alphabet()):
        raise RuntimeError("the byte alphabet differs from the tokenizers ByteLevel alphabet")
    space = symbols[ord(" ")]
    vocab = {symbol: index for index, symbol in enumerate(symbols)}
    for word in (*NATIVE_WORDS, *PROVIDER_WORDS):
        vocab[space + word] = len(vocab)
    vocab[END_OF_TEXT] = len(vocab)
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=[], chat_template=CHAT_TEMPLATE)
    end_id = vocab[END_OF_TEXT]
    config = Qwen2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=False,
        bos_token_id=end_id,
        eos_token_id=end_id,
        pad_token_id=end_id,
    )
    checkpoints = []
    for name, words in (("native", NATIVE_WORDS), ("provider", PROVIDER_WORDS)):
        directory = root / name
        directory.mkdir(parents=True)
        model = Qwen2ForCausalLM(config).to(dtype=torch.float16)
        logits = torch.full((VOCAB_SIZE,), OTHER_LOGIT)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.model.embed_tokens.weight.fill_(1)
            for parameter_name, parameter in model.named_parameters():
                if "norm.weight" in parameter_name:
                    parameter.fill_(1)
            model.lm_head.weight.fill_(OTHER_LOGIT / HIDDEN_SIZE)
            for word, logit in zip(words, WORD_LOGITS, strict=True):
                model.lm_head.weight[vocab[space + word]].fill_(logit / HIDDEN_SIZE)
                logits[vocab[space + word]] = logit
        model.save_pretrained(directory, safe_serialization=True)
        tokenizer.save_pretrained(directory)
        checkpoint = Checkpoint(name, directory, words)
        check_checkpoint(checkpoint, [vocab[space + word] for word in words], logits)
        checkpoints.append(checkpoint)
    return checkpoints[0], checkpoints[1]


def check_checkpoint(checkpoint: Checkpoint, word_ids: Sequence[int], logits: Any) -> None:
    """Check one saved checkpoint as the engines load it: the tokenizer loads unchanged, each word is one token,
    the float16 model computes exactly the designed logits, and in float32 the words are the three most likely
    tokens, in word order, with every other token below a log probability of -20."""
    import torch
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    directory = checkpoint.directory
    tokenizer = AutoTokenizer.from_pretrained(directory)
    saved = json.loads((directory / "tokenizer.json").read_text())
    loaded = json.loads(tokenizer.backend_tokenizer.to_str())
    if type(tokenizer).__name__ != "Qwen2Tokenizer" or any(
        saved[part] != loaded[part] for part in ("normalizer", "pre_tokenizer", "decoder", "model")
    ):
        raise RuntimeError(f"{checkpoint.name}: transformers does not load the saved tokenizer unchanged")
    for token, token_id in zip(checkpoint.tokens(), word_ids, strict=True):
        if tokenizer.decode([token_id]) != token:
            raise RuntimeError(f"{checkpoint.name}: token {token_id} decodes to {tokenizer.decode([token_id])!r}")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": PLAIN_PROMPT}], tokenize=True, add_generation_prompt=True
    )
    prompt = list(rendered["input_ids"] if isinstance(rendered, Mapping) else rendered)
    with torch.no_grad():
        half = Qwen2ForCausalLM.from_pretrained(directory, dtype=torch.float16)
        computed = half(torch.tensor([prompt])).logits[0, -1].float()
        if not torch.equal(computed, logits):
            raise RuntimeError(f"{checkpoint.name}: float16 logits differ from the design")
        full = Qwen2ForCausalLM.from_pretrained(directory, dtype=torch.float32)
        scores = torch.log_softmax(full(torch.tensor([prompt])).logits[0, -1].float(), dim=-1)
    ranked = torch.argsort(scores, descending=True).tolist()
    if ranked[:3] != list(word_ids) or float(scores[ranked[3]]) > -20:
        raise RuntimeError(f"{checkpoint.name}: float32 scores do not rank the words first: {ranked[:4]}")


def directory_digests(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): "sha256:" + file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def revision_of(checkpoint: Checkpoint) -> str:
    return "sha256:" + file_sha256(checkpoint.directory / "model.safetensors")


def write_capacity_fixture(source: Checkpoint, directory: Path) -> None:
    """Copy the files of the checkpoint ``source`` to the new ``directory``, and add ``CAPACITY_MARKER``, a file
    that says what the copy is.

    The copy is the third weight artifact of the lifecycle profile's capacity check. It is no third model: its
    weights are those of ``source``, and no engine loads it.
    """
    shutil.copytree(source.directory, directory)
    marker = {"purpose": "a third weight artifact for the capacity check, not a third model", "copy_of": source.name}
    (directory / CAPACITY_MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


# -- Engines -------------------------------------------------------------------------------------------------------


def provider_command(engine: str, model: Path, port: int) -> list[str]:
    """The raw OpenAI server of ``engine`` for ``model`` on loopback, with the options of the native engine."""
    if engine == "vllm":
        command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model)]
    else:
        command = [sys.executable, "-m", "sglang.launch_server", "--model-path", str(model)]
    command += ["--host", "127.0.0.1", "--port", str(port), "--served-model-name", PROVIDER_MODEL]
    for name, value in NATIVE_OPTIONS[engine].items():
        # Each option is a value or a switch that is on.
        flag = "--" + name.replace("_", "-")
        command += [flag] if value is True else [flag, str(value)]
    if engine == "vllm":
        # vLLM's own sampling defaults, not the checkpoint's generation_config.json.
        command += ["--generation-config", "vllm"]
    return command


@dataclass
class Run:
    """The state of one run: its inputs, files, report, credentials, processes, checkpoints and native runtime.

    ``runtime`` is set once the native runtime exists, so that cleanup shuts it down before its components stop.
    The lifecycle profile sets it to None again once the runtime has shut down and its components have stopped.
    """

    inputs: QualificationInputs
    layout: RunLayout
    report: QualificationReport
    credentials: Credentials
    processes: OwnedProcesses
    oracle: LogprobOracle | None = None
    native: Checkpoint | None = None
    provider: Checkpoint | None = None
    runtime: InferenceRuntime | None = None

    def checkpoints(self) -> tuple[Checkpoint, Checkpoint]:
        if self.native is None or self.provider is None:
            raise RuntimeError("the checkpoints are not written yet")
        return self.native, self.provider

    def scores(self) -> LogprobOracle:
        if self.oracle is None:
            raise RuntimeError("the logprob oracle is not loaded yet")
        return self.oracle


@dataclass(frozen=True)
class Engines:
    """The started engines: the component at ``component_url`` with the native engine of A, and the raw provider of
    B at ``provider_url``. ``runtime`` serves A as a published artifact; it is None when the engine answers with its
    base model only (the ``routing`` profile)."""

    runtime: FullWeightServingRuntime | None
    component_url: str
    provider_url: str


def native_service(run: Run, component_url: str, *, artifact_serving: bool) -> RemoteInferenceService:
    """The component's control client for the native engine, with the engine input of the remote serving factory.

    With ``artifact_serving``, the client makes the component keep each request on its weights, as the factory
    always does for published weights, and the engine shares prefixes (``inference.prefix-sharing: true``).
    Without it, the component forwards every request to the engine, as for a trainer's rollout engines, serves no
    published artifact, and the engine keeps no prefix cache.
    """
    engine = run.inputs.native_engine
    native, _ = run.checkpoints()
    config = serving_engine_config(
        engine,
        model_path=str(native.directory),
        num_gpus=1,
        tensor_parallel_size=1,
        options=NATIVE_OPTIONS[engine],
        artifact_format=ArtifactFormat(layout="full"),
        adapter_capacity=1,
        prefix_sharing=artifact_serving,
        executor="auto",
        executor_options={},
        environ=os.environ,
    )
    config["startup_timeout"] = ENGINE_START_TIMEOUT_S
    config["request_timeout"] = INFERENCE_TIMEOUT_S
    return RemoteInferenceService(
        RemoteInferenceSettings(url=component_url, startup_timeout_s=ENGINE_START_TIMEOUT_S + 60),
        engine=engine,
        engine_config=config,
        num_gpus=1,
        control_token=run.credentials.control,
        request_pinning=artifact_serving,
        drain_timeout_s=INFERENCE_TIMEOUT_S,
    )


def native_runtime(
    run: Run, engines: ComponentEngines, *, health_interval_s: float = 3600
) -> FullWeightServingRuntime:
    """The production full-weight runtime of the native engine on the component of ``engines``, as the remote
    serving factory builds it: the engine's own handler with the inference token, and the component's control
    client. Its watcher checks the engines every ``health_interval_s`` seconds: by default, never within a run."""
    native, _ = run.checkpoints()
    model_path = str(native.directory)
    service = engines.service
    handler_class = InferenceHandler.resolve_class(
        BUILTIN_ENGINES[service.engine].handler, setting="inference handler"
    )
    handler = handler_class.from_config(
        service.settings.url,
        model_path=model_path,
        timeout_s=INFERENCE_TIMEOUT_S,
        request_headers=ProviderRequestHeaders(run.credentials.inference),
    )
    return FullWeightServingRuntime(
        engines,
        handler,
        model_path=model_path,
        inference_timeout_s=INFERENCE_TIMEOUT_S,
        health_interval_s=health_interval_s,
    )


async def start_component_process(run: Run, name: str) -> tuple[str, OwnedProcess]:
    """Start one remote inference component as the owned process ``name``; return its base URL and the process.

    The component has its own port and work directory, and its output goes to ``<name>.log``. It serves the run's
    native engine and gives its engines one GPU. No engine runs yet.
    """
    port = free_port()
    process = run.processes.start(
        name,
        [
            sys.executable,
            "-m",
            "reef.inference.remote",
            "--engine",
            run.inputs.native_engine,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--work-dir",
            str(run.layout.component_work(name)),
            "--ray-num-gpus",
            "1",
        ],
        environment={
            **os.environ,
            DEFAULT_CONTROL_TOKEN_ENV: run.credentials.control,
            DEFAULT_INFERENCE_TOKEN_ENV: run.credentials.inference,
        },
        grace_s=120,
    )
    await wait_for_port(port, process, COMPONENT_LISTEN_TIMEOUT_S)
    return f"http://127.0.0.1:{port}", process


async def start_component(run: Run, *, artifact_serving: bool) -> RemoteInferenceService:
    """Start Reef's remote inference component, and return the control client for its native engine.

    No engine runs yet. The engine facts are recorded here, so that a failed engine start still reports what it
    ran; ``start_order`` lists the engines that started, which the caller appends to.
    """
    url, component = await start_component_process(run, SLOT_COMPONENTS[0])
    service = native_service(run, url, artifact_serving=artifact_serving)
    run.report.facts["engines"] = {
        "start_order": [],
        "native": {
            "engine": run.inputs.native_engine,
            "request_pinning": artifact_serving,
            "component_command": component.command,
            "engine_config": service.engine_config,
        },
    }
    return service


async def start_provider(run: Run) -> str:
    """Start the raw provider of B and wait until it lists its model; return its base URL.

    The engine facts must exist. They receive the provider's command and models, and its entry in ``start_order``.
    """
    provider_engine = run.inputs.provider_engine
    _, provider = run.checkpoints()
    engine_facts = run.report.facts["engines"]
    provider_facts: dict[str, Any] = engine_facts.setdefault("provider", {"engine": provider_engine})
    # The port is taken last, so that no engine started before takes it.
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    command = provider_command(provider_engine, provider.directory, port)
    provider_facts["command"] = command
    # The raw provider loads no Reef scheduler hook.
    environment = {key: value for key, value in os.environ.items() if key != REEF_SGLANG_PLUGIN_ENV}
    process = run.processes.start("provider", command, environment=environment, grace_s=60)
    async with ClientSession() as session:
        provider_facts["models"] = await wait_for_models(session, url, PROVIDER_MODEL, process, ENGINE_START_TIMEOUT_S)
    engine_facts["start_order"].append(f"raw {provider_engine} provider")
    return url


async def start_engines(run: Run, *, artifact_serving: bool) -> Engines:
    """Start the component and its native engine, and start the raw provider.

    With ``artifact_serving``, the production full-weight runtime opens the engine, which stays paused until the
    caller activates an artifact. Without it, the engine answers with its base model and no runtime exists.
    SGLang always starts first, because it sizes its memory from what is free when it starts.
    """
    native_engine = run.inputs.native_engine
    service = await start_component(run, artifact_serving=artifact_serving)
    runtime = native_runtime(run, ComponentEngines(service)) if artifact_serving else None
    run.runtime = runtime
    engine_facts = run.report.facts["engines"]
    order: list[str] = engine_facts["start_order"]
    engine_facts["provider"] = {"engine": run.inputs.provider_engine}

    def open_engine() -> None:
        if runtime is None:
            backend = service.backend(service.start())
            backend.initialize_version(str(RuntimeLoadId(new_runtime_load_id_incarnation(), 0)))
            backend.resume()
        else:
            runtime.set_owner_sequence(1)
            runtime.open()

    async def open_native() -> None:
        await asyncio.to_thread(open_engine)
        order.append(f"native {native_engine} on the component")

    if native_engine == "sglang":
        await open_native()
        provider_url = await start_provider(run)
    else:
        provider_url = await start_provider(run)
        await open_native()
    return Engines(runtime, service.settings.url, provider_url)


@asynccontextmanager
async def serving(app: web.Application) -> AsyncIterator[str]:
    """Serve ``app`` on a free loopback port for the block; yield its base URL."""
    runner = web.AppRunner(app)
    try:
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        yield f"http://127.0.0.1:{runner.addresses[0][1]}"
    finally:
        await runner.cleanup()


def chat_body(
    text: str,
    *,
    stream: bool,
    max_tokens: int = SCORED_TOKENS,
    history: Sequence[Mapping[str, Any]] = (),
    model: object = None,
) -> dict[str, Any]:
    """A scored chat request: fixed seed, temperature 1 and top-p 1, and logprobs with ``TOP_ALTERNATIVES``."""
    body: dict[str, Any] = {
        "messages": [*history, {"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": SEED,
        "logprobs": True,
        "top_logprobs": TOP_ALTERNATIVES,
        "stream": stream,
    }
    if model is not None:
        body["model"] = model
    return body


def mode(stream: bool) -> str:
    return "streamed" if stream else "buffered"


# -- Requests to served artifacts ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One served artifact that a profile sends requests to.

    Values: ``name`` labels it in the checks, and is unique within a run; ``artifact_id`` is the artifact that
    answers; ``checkpoint`` is its weights; ``native_facade`` is whether Reef's native chat facade answers, rather
    than a raw provider.
    """

    name: str
    artifact_id: str
    checkpoint: Checkpoint
    native_facade: bool


@dataclass(frozen=True)
class RecordedRequest:
    """A request that must have exactly one record: its exchange, the arm that answered it and its split."""

    exchange: Exchange
    arm: Arm
    traffic: ExpectedTraffic | None


class RequestChecks:
    """Requests to the served artifacts of one inference app, with the checks of their answers and records.

    Values: ``recorded`` lists requests that must have records, in completion order; ``sampled`` holds the token
    IDs that each arm generated, by the arm's name.
    Purpose: the checks of the features, recovery and lifecycle profiles extend it.
    """

    def __init__(self, run: Run, client: ReefClient, coordination: Coordination) -> None:
        self.run = run
        self.client = client
        self.coordination = coordination
        self.recorded: list[RecordedRequest] = []
        self.sampled: dict[str, set[int]] = {}

    # -- Publication, as control would write it

    def lease(self) -> int:
        lease = self.coordination.control_lease.acquire("native-qualification")
        if lease is None:
            raise RuntimeError("the run could not take its isolated control lease")
        return lease.number

    def publish(self, rows: Mapping[str, ServedArtifact]) -> None:
        number = self.lease()
        for name, served in rows.items():
            self.coordination.served_models.write(ScenarioModel(SCENARIO, name), served, control_lease_number=number)

    def set_weights(self, weights: Mapping[str, int]) -> None:
        self.coordination.served_models.write_traffic_weights(
            SCENARIO, dict(weights), control_lease_number=self.lease()
        )

    # -- Requests

    def route(self, arm: Arm) -> str:
        return f"/reef/scenarios/{SCENARIO}/artifacts/{quote(arm.artifact_id, safe='')}/v1/chat/completions"

    def conversation(self, arm: Arm, weights: Mapping[str, int]) -> tuple[str, int]:
        """A first message whose conversation the split ``weights`` give to ``arm``, and its bucket."""
        for index in range(1000):
            text = f"Conversation {index}: list three words."
            bucket = conversation_bucket(SCENARIO, chat_body(text, stream=False))
            if bucket is not None and artifact_of_bucket(bucket, weights) == arm.artifact_id:
                return text, bucket
        raise RuntimeError(f"no conversation of the {arm.name} arm among the candidates")

    async def chat(
        self,
        label: str,
        arm: Arm,
        body: Mapping[str, Any],
        *,
        named: bool = False,
        headers: Mapping[str, str] | None = None,
        traffic: ExpectedTraffic | None = None,
        rollout: bool = False,
        first_content: asyncio.Future[float] | None = None,
    ) -> Exchange:
        """Send one request, check that ``arm`` answered it correctly, and expect its record unless ``rollout``."""
        sent_headers = {SCENARIO_HEADER: SCENARIO, **(headers or {})}
        if rollout:
            sent_headers[ROLLOUT_HEADER] = "1"
        path = self.route(arm) if named else "/v1/chat/completions"
        exchange = await self.client.post(label, path, body, headers=sent_headers, first_content=first_content)
        self.check_answer(arm, exchange, recorded=not rollout)
        if not rollout:
            self.recorded.append(RecordedRequest(exchange, arm, traffic))
        return exchange

    async def explicit_request(self, label: str, arm: Arm, *, stream: bool, named: bool = False) -> Exchange:
        """Send ``PLAIN_PROMPT`` to ``arm``'s artifact: by its named route when ``named``, else with the artifact ID
        as the body's model. No split applies, so its record must store no traffic metadata."""
        model = None if named else arm.artifact_id
        return await self.chat(label, arm, chat_body(PLAIN_PROMPT, stream=stream, model=model), named=named)

    def check_answer(self, arm: Arm, exchange: Exchange, *, recorded: bool) -> None:
        label = exchange.label
        exchange.require_success()
        answered = exchange.response_headers.get(ARTIFACT_HEADER)
        assert answered == arm.artifact_id, f"{label}: answered by {answered!r}, expected the {arm.name} arm"
        if recorded:
            exchange.receipt()
        else:
            assert RECORD_HEADER not in exchange.response_headers, f"{label}: an unrecorded call has a receipt"
            assert b"agent_record_id" not in exchange.content, f"{label}: an unrecorded call has a receipt"
        check_words(arm.checkpoint, exchange.text(), exchange.body["max_tokens"], label)
        oracle = self.run.scores()
        token_ids = check_logprobs(oracle, arm.checkpoint, exchange, native_facade=arm.native_facade)
        self.sampled.setdefault(arm.name, set()).update(token_ids)
        if arm.native_facade:
            check_private_fields_absent(answer_values(exchange), label)
            meta = exchange.meta_info()
            if self.run.inputs.native_engine == "sglang":
                check_native_alternatives(oracle, arm.checkpoint, exchange, token_ids)
            else:
                listed = sorted(key for key in meta if "logprob" in key)
                assert not listed, f"{label}: vLLM's native answer passes on alternatives in {listed}"

    async def refusal(self, label: str, arm: Arm, text: str = CACHE_PROMPT) -> Exchange | None:
        """Send ``text`` to ``arm``'s named route. Return the exchange when Reef refused it as it refuses a route
        whose weights the engine does not serve now: 503 with ``Retry-After`` and no receipt. Otherwise ``arm``
        must have answered it; the request then has a record, and the result is None."""
        exchange = await self.client.post(
            label, self.route(arm), chat_body(text, stream=False), headers={SCENARIO_HEADER: SCENARIO}
        )
        if exchange.status == 200:
            self.check_answer(arm, exchange, recorded=True)
            self.recorded.append(RecordedRequest(exchange, arm, None))
            return None
        answer = exchange.content.decode(errors="replace")
        refused = exchange.status == 503 and exchange.response_headers.get("retry-after") == "1"
        assert refused, f"{label}: HTTP {exchange.status}: {answer!r}"
        receipt = RECORD_HEADER in exchange.response_headers or "agent_record_id" in answer
        assert not receipt, f"{label}: a refused request has a receipt"
        return exchange

    # -- The native engine's prefix cache

    async def engine_metrics(self, component_log: Path | None = None) -> str:
        """The Prometheus metrics of a native engine, read from its private address; observation only.

        The engine is the one that the component with the log ``component_log`` started last; None selects the
        run's first component.
        """
        url = engine_url(component_log or self.run.layout.logs / f"{SLOT_COMPONENTS[0]}.log") + "/metrics"
        async with self.client.session.get(url) as response:
            assert response.status == 200, f"the engine's metrics answered HTTP {response.status}"
            return await response.text()

    async def cache_counters(self, component_log: Path | None = None) -> tuple[float, float]:
        """The prefix-cache counters of a native vLLM engine (``engine_metrics``): the prompt tokens that it looked
        up, and those it found."""
        metrics = await self.engine_metrics(component_log)
        queries, hits = metric_total(metrics, CACHE_QUERIES_METRIC), metric_total(metrics, CACHE_HITS_METRIC)
        if queries is None or hits is None:
            published = sorted(
                {re.split(r"[{ ]", line)[0] for line in metrics.splitlines() if "cache" in line and line[:1] != "#"}
            )
            raise AssertionError(
                f"the engine publishes no {CACHE_QUERIES_METRIC} with {CACHE_HITS_METRIC}; it has {published}"
            )
        return queries, hits

    async def cached_request(self, label: str, arm: Arm, component_log: Path | None = None) -> dict[str, Any]:
        """Send ``CACHE_PROMPT`` to ``arm``'s named route while no other request runs on its engine; return the
        prompt's token count and ``cached_tokens``, the prompt tokens that the engine reused from its prefix cache.

        SGLang reports the reused tokens in the answer's ``meta_info``. vLLM's counters give them: around the
        request, the looked-up tokens must grow by exactly this prompt, which neither an engine without a cache
        nor a second request gives, and the found tokens grow by the reused ones. vLLM's answer may also carry the
        count in its usage (``answer_usage_cached_tokens``, else None); it must then be the same. The counters are
        those of the engine that serves ``arm``: ``component_log`` is the log of its component, and None selects
        the run's first component (``engine_metrics``).
        """
        body = chat_body(CACHE_PROMPT, stream=False)
        prompt_tokens = len(self.run.scores().prompt_ids(body["messages"]))
        measured: dict[str, Any] = {"prompt_tokens": prompt_tokens}
        if self.run.inputs.native_engine == "sglang":
            exchange = await self.chat(label, arm, body, named=True)
            cached = exchange.meta_info().get("cached_tokens")
            assert type(cached) is int, f"{label}: SGLang reported cached_tokens {cached!r}"
        else:
            sought_before, found_before = await self.cache_counters(component_log)
            exchange = await self.chat(label, arm, body, named=True)
            deadline = time.monotonic() + METRIC_DELAY_S
            while True:
                sought, found = await self.cache_counters(component_log)
                if sought - sought_before >= prompt_tokens or time.monotonic() > deadline:
                    break
                await asyncio.sleep(0.05)
            looked_up, cached = sought - sought_before, int(found - found_before)
            assert looked_up == prompt_tokens, f"{label}: vLLM looked up {looked_up} tokens, not the {prompt_tokens}"
            usage = exchange.meta_info().get("usage") or {}
            reported = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            assert reported is None or reported == cached, f"{label}: usage names {reported} cached tokens"
            measured["answer_usage_cached_tokens"] = reported
        assert 0 <= cached <= prompt_tokens, f"{label}: {cached} cached tokens of a {prompt_tokens}-token prompt"
        measured["cached_tokens"] = cached
        return measured

    # -- Records

    def check_records(self, records: Sequence[Any]) -> dict[str, Any]:
        """Check that every recorded request has exactly one record, which keeps its artifact, split and answer,
        and that no other record exists. Return the records by their ID."""
        by_id = {record.agent_record_id: record for record in records}
        expected = [request.exchange.receipt() for request in self.recorded]
        assert len(by_id) == len(records) and len(set(expected)) == len(expected), "a record ID repeats"
        assert set(by_id) == set(expected), f"{len(records)} records for {len(expected)} recorded requests"
        for request in self.recorded:
            exchange, arm = request.exchange, request.arm
            label = exchange.label
            record = by_id[exchange.receipt()]
            payload = record.payload
            assert record.scenario == SCENARIO, f"{label}: scenario {record.scenario!r}"
            assert record.artifact_id == arm.artifact_id == payload.get("artifact_id"), f"{label}: stored artifact"
            assert payload.get("messages") == exchange.body["messages"], f"{label}: stored messages"
            assert payload.get("stream") is exchange.body["stream"], f"{label}: stored stream flag"
            check_traffic((payload.get("metadata") or {}).get("traffic"), request.traffic, arm.artifact_id, label)
            check_stored_response(payload["response"], exchange)
            if arm.native_facade:
                check_private_fields_absent(payload, f"{label} record")
        return by_id


# -- Features profile ----------------------------------------------------------------------------------------------


class FeatureChecks(RequestChecks):
    """The requests and checks of the features profile, against one inference app.

    Values: ``native`` and ``provider`` are the two arms of its splits, and ``sampled`` has an entry for each;
    ``session_records`` holds the record IDs of the session-key requests.
    """

    def __init__(
        self,
        run: Run,
        client: ReefClient,
        coordination: Coordination,
        native: Arm,
        provider: Arm,
    ) -> None:
        super().__init__(run, client, coordination)
        self.native = native
        self.provider = provider
        self.sampled.update({native.name: set(), provider.name: set()})
        self.session_records: list[str] = []

    # -- Checks

    async def logprobs(self) -> None:
        """Both arms, buffered and streamed, outside any split."""
        for stream in (False, True):
            await self.chat(f"native {mode(stream)} logprobs", self.native, chat_body(PLAIN_PROMPT, stream=stream))
            await self.explicit_request(f"provider {mode(stream)} logprobs", self.provider, stream=stream, named=True)
        self.run.report.passed("logprobs of both arms, buffered and streamed")

    async def split(self) -> dict[str, tuple[str, int]]:
        """A 50/50 split: both arms, conversation stickiness, session-key stickiness and a rollout."""
        weights = {self.native.artifact_id: 50, self.provider.artifact_id: 50}
        self.set_weights(weights)
        conversations = {arm.name: self.conversation(arm, weights) for arm in (self.native, self.provider)}
        for arm, stream in ((self.native, False), (self.provider, True)):
            text, bucket = conversations[arm.name]
            first = await self.chat(
                f"50/50 {arm.name} conversation, turn 1",
                arm,
                chat_body(text, stream=stream),
                traffic=ExpectedTraffic(weights, bucket),
            )
            history = [{"role": "user", "content": text}, {"role": "assistant", "content": first.text()}]
            await self.chat(
                f"50/50 {arm.name} conversation, turn 2",
                arm,
                chat_body("Continue.", stream=not stream, history=history),
                traffic=ExpectedTraffic(weights, bucket),
            )
        session = {SESSION_KEY_HEADER: "qualification-session"}
        answered = []
        for index, stream in enumerate((False, True)):
            body = chat_body(f"Session request {index}: list three words.", stream=stream)
            exchange = await self.client.post(
                f"50/50 session-key request {index}",
                "/v1/chat/completions",
                body,
                headers={SCENARIO_HEADER: SCENARIO, **session},
            )
            answered_by = exchange.response_headers.get(ARTIFACT_HEADER)
            arm = self.provider if answered_by == self.provider.artifact_id else self.native
            self.check_answer(arm, exchange, recorded=True)
            self.recorded.append(RecordedRequest(exchange, arm, ExpectedTraffic(weights)))
            self.session_records.append(exchange.receipt())
            answered.append(arm.name)
        assert answered[0] == answered[1], f"one session key reached two arms: {answered}"
        text, _ = conversations[self.provider.name]
        await self.chat(
            "50/50 rollout of a provider-bucket conversation",
            self.native,
            chat_body(text, stream=False),
            rollout=True,
        )
        self.run.report.passed("50/50 split reaches both arms and keeps conversations")
        return conversations

    async def weights_and_pins(self, conversations: Mapping[str, tuple[str, int]]) -> None:
        """100/0, a traffic change during an in-flight native request, then 0/100; explicit pins to the arm without
        weight in each."""
        native_only = {self.native.artifact_id: 100, self.provider.artifact_id: 0}
        provider_only = {self.native.artifact_id: 0, self.provider.artifact_id: 100}
        self.set_weights(native_only)
        text, bucket = conversations[self.provider.name]
        await self.chat(
            "100/0 provider-bucket conversation",
            self.native,
            chat_body(text, stream=False),
            traffic=ExpectedTraffic(native_only, bucket),
        )
        await self.explicit_request("100/0 named route to the provider", self.provider, stream=True, named=True)
        await self.explicit_request("100/0 body pin to the provider", self.provider, stream=False)
        await self.in_flight_change(native_only, provider_only)
        text, bucket = conversations[self.native.name]
        await self.chat(
            "0/100 native-bucket conversation",
            self.provider,
            chat_body(text, stream=True),
            traffic=ExpectedTraffic(provider_only, bucket),
        )
        await self.explicit_request("0/100 named route to the native arm", self.native, stream=False, named=True)
        await self.explicit_request("0/100 body pin to the native arm", self.native, stream=True)
        self.run.report.passed("100/0 and 0/100 follow the weights; named and body pins bypass them")

    async def in_flight_change(self, before: Mapping[str, int], after: Mapping[str, int]) -> None:
        """Change the weights while a long native stream is generating: it keeps its decision and its stored split,
        and a new request of the same conversation follows the new weights."""
        text = "In-flight conversation: list many words."
        bucket = conversation_bucket(SCENARIO, chat_body(text, stream=True))
        assert bucket is not None
        long_task, signal_name, observed = await self.long_stream(
            "in-flight native stream", text, traffic=ExpectedTraffic(before, bucket)
        )
        try:
            self.set_weights(after)
            switched = time.monotonic()
            follow = await self.chat(
                "request of the same conversation after the traffic change",
                self.provider,
                chat_body(text, stream=False),
                traffic=ExpectedTraffic(after, bucket),
            )
            long = await long_task
        except BaseException:
            await cancelled(long_task)
            raise
        assert long.ended > switched, "the in-flight stream ended before the traffic change: no overlap"
        self.run.report.passed(
            "an in-flight request keeps its split decision across a traffic change",
            in_flight_signal=signal_name,
            seconds_from_stream_start={
                "signal": round(observed - long.sent, 4),
                "weights_changed": round(switched - long.sent, 4),
                "new_request_sent": round(follow.sent - long.sent, 4),
                "new_request_ended": round(follow.ended - long.sent, 4),
                "stream_ended": round(long.ended - long.sent, 4),
            },
            new_request_ended_before_stream=follow.ended < long.ended,
            stream_tokens=LONG_TOKENS,
        )

    async def long_stream(
        self, label: str, text: str, *, named: bool = False, traffic: ExpectedTraffic | None = None
    ) -> tuple[asyncio.Task[Exchange], str, float]:
        """Start a native stream of ``LONG_TOKENS`` tokens for ``text``, and wait until the engine generates it.

        Return the stream's task, which the caller awaits or cancels, the signal that showed the generation, and
        the moment it showed. The stream has not ended at that moment.
        """
        emitting: asyncio.Future[float] = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(
            self.chat(
                label,
                self.native,
                chat_body(text, stream=True, max_tokens=LONG_TOKENS),
                named=named,
                traffic=traffic,
                first_content=emitting,
            )
        )
        try:
            if self.run.inputs.native_engine == "sglang":
                await asyncio.wait({emitting, task}, timeout=REQUEST_TIMEOUT_S, return_when=asyncio.FIRST_COMPLETED)
                if not emitting.done():
                    if task.done():
                        task.result()
                    raise AssertionError(f"{label}: the stream generated no text in time")
                signal_name = "the first generated text reached the client"
            else:
                await self.wait_until_running(task)
                signal_name = f"engine metric {RUNNING_METRIC} >= 1"
            observed = time.monotonic()
            assert not task.done(), f"{label}: the stream ended as soon as it generated: nothing can overlap it"
        except BaseException:
            await cancelled(task)
            raise
        return task, signal_name, observed

    async def wait_until_running(self, task: asyncio.Task[Exchange]) -> None:
        """Wait until the native vLLM engine reports a running request; its stream arrives as one burst."""
        deadline = time.monotonic() + REQUEST_TIMEOUT_S
        while True:
            running = metric_total(await self.engine_metrics(), RUNNING_METRIC)
            if running is not None and running >= 1:
                return
            if task.done():
                task.result()
                raise AssertionError("the in-flight request ended before the engine reported it running")
            if time.monotonic() > deadline:
                raise TimeoutError("the engine never reported the in-flight request as running")
            await asyncio.sleep(0.005)

    async def concurrency(self) -> None:
        """Requests of both arms at once, pinned and split, buffered and streamed, under 0/100."""
        provider_only = {self.native.artifact_id: 0, self.provider.artifact_id: 100}
        text = "Concurrent conversation: list three words."
        requests = [
            self.explicit_request("concurrent named native, buffered", self.native, stream=False, named=True),
            self.explicit_request("concurrent body pin native, streamed", self.native, stream=True),
            self.explicit_request("concurrent named provider, streamed", self.provider, stream=True, named=True),
            self.explicit_request("concurrent body pin provider, buffered", self.provider, stream=False),
        ]
        for stream in (True, False):
            body = chat_body(text, stream=stream)
            label = f"concurrent split provider, {mode(stream)}"
            requests.append(self.chat(label, self.provider, body, traffic=ExpectedTraffic(provider_only)))
        exchanges = await asyncio.gather(*requests)
        native, provider = exchanges[:2], exchanges[2:]
        overlapping = sum(a.sent < b.ended and b.sent < a.ended for a in native for b in provider)
        assert overlapping, "no native request overlapped a provider request"
        self.run.report.passed(
            "concurrent requests of both arms stay on their artifacts", overlapping_native_provider_pairs=overlapping
        )

    async def prefix_cache(self, runtime: FullWeightServingRuntime, successor: Checkpoint) -> None:
        """The native engine shares a prompt's prefix among the requests of one artifact, never across a load.

        Two requests on A read the cache: the first reuses nothing, the second reuses the prompt. Then the runtime
        activates ``successor``, B, as a new artifact on the same engine while a long A stream of that prompt
        generates. The component ends the stream on A before the load, and Reef refuses a new A request meanwhile
        and afterwards, without a record. The first B request of the same prompt must reuse nothing, and the
        second must reuse the prompt again. No checkpoint reads its context, so no output shows a stale cache
        entry: the cached-token counts do.
        """
        pause_mode = self.run.report.facts["engines"]["native"]["engine_config"]["pause_mode"]
        assert pause_mode == "retract", f"the native engine pauses {pause_mode}, so it shares no prefixes"
        cold = await self.cached_request("prefix cache: first request on A", self.native)
        warm = await self.cached_request("prefix cache: second request on A", self.native)
        assert cold["cached_tokens"] == 0 and warm["cached_tokens"] > 0, f"A reused {cold}, then {warm}"
        artifact = Artifact.local(successor.directory)

        async def activate() -> tuple[str, float]:
            load_id = await asyncio.to_thread(runtime.activate_checkpoint, artifact, scenario=SCENARIO)
            return load_id, time.monotonic()

        long_task, signal_name, observed = await self.long_stream(
            "A stream during the load of B", CACHE_PROMPT, named=True
        )
        called = time.monotonic()
        activation = asyncio.create_task(activate())
        during: Exchange | None = None
        admitted = 0
        try:
            # The activation first closes A's admission and waits for the stream. A request that arrives before
            # that is answered by A; the next one is refused.
            while during is None and not long_task.done():
                during = await self.refusal(f"new A request during the load of B, attempt {admitted}", self.native)
                if during is None:
                    admitted += 1
            long = await long_task
            load_id, loaded = await activation
        except BaseException:
            await cancelled(long_task)
            with suppress(BaseException):
                await activation
            raise
        assert (
            during is not None and during.ended < long.ended
        ), "no A request was refused while the A stream generated: the load did not overlap the stream"
        after = await self.refusal("A request after the load of B", self.native)
        assert after is not None, "A's route answered after the engine loaded B"
        served = runtime.served_artifact(artifact.artifact_id, SCENARIO)
        if served is None or parse_full_weight_name(served.model_name) != (runtime.model_path, load_id):
            raise RuntimeError(f"the activation of B established no route that names its load: {served!r}")
        loaded_arm = Arm("native after the load", served.artifact_id, successor, native_facade=True)
        # A's row stays as it is: the engine refuses its model name from now on.
        self.publish({ACTIVE: served, served.artifact_id: served})
        first = await self.cached_request("prefix cache: first request on B", loaded_arm)
        second = await self.cached_request("prefix cache: second request on B", loaded_arm)
        assert first["cached_tokens"] == 0 and second["cached_tokens"] > 0, f"B reused {first}, then {second}"
        self.run.report.passed(
            "the prefix cache serves one artifact: a load waits for the requests in flight, then clears it",
            pause_mode=pause_mode,
            cached_tokens_from=(
                "meta_info.cached_tokens of each answer"
                if self.run.inputs.native_engine == "sglang"
                else f"the engine's {CACHE_HITS_METRIC} around each request"
            ),
            requests={"A first": cold, "A second": warm, "B first": first, "B second": second},
            in_flight_signal=signal_name,
            seconds_from_stream_start={
                "signal": round(observed - long.sent, 4),
                "activation_called": round(called - long.sent, 4),
                "new_request_refused": round(during.ended - long.sent, 4),
                "stream_ended": round(long.ended - long.sent, 4),
                "activation_returned": round(loaded - long.sent, 4),
            },
            requests_answered_before_admission_closed=admitted,
            stream_tokens=LONG_TOKENS,
            loaded_route=served.to_json(),
        )

    def verify_records(self, records: Sequence[Any]) -> None:
        """Every recorded request has exactly one record, which keeps its artifact, split and answer. The requests
        of one session key stored one bucket, and each arm sampled more than one token."""
        by_id = self.check_records(records)
        buckets = {by_id[record_id].payload["metadata"]["traffic"]["bucket"] for record_id in self.session_records}
        assert len(buckets) == 1, f"one session key stored the buckets {sorted(buckets)}"
        for arm in (self.native, self.provider):
            assert len(self.sampled[arm.name]) >= 2, f"the {arm.name} arm sampled only {self.sampled[arm.name]}"
        self.run.report.passed("records match their requests exactly", records=len(records))


async def qualify_features(run: Run) -> None:
    native, provider = run.checkpoints()
    engines = await start_engines(run, artifact_serving=True)
    runtime = engines.runtime
    if runtime is None:
        raise RuntimeError("the features profile needs the full-weight runtime")
    artifact = Artifact.local(native.directory)
    load_id = await asyncio.to_thread(runtime.activate_checkpoint, artifact, scenario=SCENARIO)
    native_served = runtime.served_artifact(artifact.artifact_id, SCENARIO)
    if native_served is None or native_served.url != engines.component_url:
        raise RuntimeError(f"the native activation established no component route: {native_served!r}")
    if parse_full_weight_name(native_served.model_name) != (str(native.directory), load_id):
        raise RuntimeError(f"the native route does not name its load: {native_served.model_name!r}")
    provider_id = f"artifact:raw-provider-{revision_of(provider).removeprefix('sha256:')[:16]}"
    provider_served = ServedArtifact(provider_id, engines.provider_url, PROVIDER_MODEL)
    native_arm = Arm("native", native_served.artifact_id, native, native_facade=True)
    provider_arm = Arm("provider", provider_id, provider, native_facade=False)
    run.report.facts["routes"] = {
        "native": native_served.to_json(),
        "native_runtime_load_id": load_id,
        "provider": provider_served.to_json(),
    }
    coordination = memory_coordination()
    storage = SQLiteScenarioStorage(run.layout.records)
    try:
        app = build_inference_app(
            coordination.served_models,
            storage,
            DeploymentEngineClients(runtime, provider_key=None, timeout_seconds=INFERENCE_TIMEOUT_S),
            tokens=run.credentials.service,
            timeout_seconds=INFERENCE_TIMEOUT_S,
        )
        async with (
            serving(app) as url,
            ClientSession(timeout=ClientTimeout(total=REQUEST_TIMEOUT_S)) as session,
        ):
            client = ReefClient(session, url, run.credentials.service, run.layout.wire)
            checks = FeatureChecks(run, client, coordination, native_arm, provider_arm)
            checks.publish(
                {ACTIVE: native_served, native_arm.artifact_id: native_served, provider_id: provider_served}
            )
            await checks.logprobs()
            conversations = await checks.split()
            await checks.weights_and_pins(conversations)
            await checks.concurrency()
            await checks.prefix_cache(runtime, provider)
    finally:
        storage.close()
        coordination.close()
    records = replay_records(run.layout.records, SCENARIO)
    write_records(run.layout.records_file, records)
    checks.verify_records(records)


# -- Recovery profile ----------------------------------------------------------------------------------------------


class RecoveryChecks(RequestChecks):
    """The requests and checks of the recovery profile, against one inference app and the runtime behind it.

    Values: ``runtime`` is the open full-weight runtime of the native engine, whose watcher checks ``engines``;
    ``log`` lists the warnings and errors that the runtime logs during the run; ``startup`` is the load that new
    engines hold, the base model A, under a route that only this run publishes; ``component_log`` is the log in
    which the component names each engine stack that it starts.
    Purpose: ``qualify_recovery`` makes it, and runs ``before_activation`` and then ``after_activation``.
    """

    def __init__(
        self,
        run: Run,
        client: ReefClient,
        coordination: Coordination,
        runtime: FullWeightServingRuntime,
        engines: ObservedEngines,
        log: ServingLog,
    ) -> None:
        super().__init__(run, client, coordination)
        self.runtime = runtime
        self.engines = engines
        self.log = log
        startup, _ = run.checkpoints()
        startup_id = f"artifact:startup-load-{revision_of(startup).removeprefix('sha256:')[:16]}"
        self.startup = Arm("startup load", startup_id, startup, native_facade=True)
        self.component_log = run.layout.logs / "component.log"

    async def load_ids(self) -> list[str]:
        """The runtime load IDs that the component reports for its engines, one for each engine."""
        service = self.engines.service
        backend = RemoteInferenceBackend(service.control, service)
        return list(await asyncio.to_thread(backend.runtime_load_ids))

    async def at_rest(self, label: str) -> None:
        """Wait for ``QUIET_CHECKS`` more health checks. The watcher must make each one, each one must pass, and
        the runtime must log no warning or error meanwhile."""
        first, logged = len(self.engines.checks), len(self.log.lines)
        checks = await ended_checks(self.engines, first, QUIET_CHECKS, QUIET_TIMEOUT_S)
        unexpected = [check for check in checks if check.thread != WATCHER_THREAD or check.error is not None]
        assert not unexpected, f"{label}: the watcher is not at rest: {unexpected}"
        lines = self.log.lines[logged:]
        assert not lines, f"{label}: the runtime logged {lines} while its engines passed their health checks"

    async def unbound(self, label: str) -> None:
        """The runtime has no verified binding, and its engines hold the startup load and generate nothing: Reef
        refuses a request for that load's route, without a receipt."""
        runtime = self.runtime
        bound = runtime.served_artifact_id is not None or not runtime.awaits_first_activation
        assert not bound, f"{label}: the runtime has a binding: it serves {runtime.served_artifact_id!r}"
        assert runtime.recovery_pending, f"{label}: the runtime does not wait for an activation to resume generation"
        held = await self.load_ids()
        assert held == [runtime.base_version], f"{label}: the engines hold the runtime loads {held}"
        refused = await self.refusal(f"{label}: request for the startup load", self.startup, PLAIN_PROMPT)
        assert refused is not None, f"{label}: the paused engines answered a request"

    async def removal(self, label: str, served: Arm | None) -> dict[str, Any]:
        """Remove the component's engines, and wait until the runtime's watcher has restored them.

        The removal is the component's control operation ``DELETE engines``, sent as the engines' owner with the
        runtime's own control client. The component stops its engine stack, so the GPU engines are lost. Nothing
        here starts engines or calls the runtime. One health check of the watcher must fail, and its recovery must
        succeed at the first attempt: the next check passes, the component has started one more engine stack, and
        the runtime has logged one warning and no error.

        With ``served``, an artifact that the runtime served before the removal, Reef must refuse the requests for
        its route until the watcher resumes generation, and its first answer ends the wait. A new stack holds the
        base model: Reef must refuse each request for the startup load's route too. Return what was measured.
        """
        first, logged = len(self.engines.checks), len(self.log.lines)
        stacks = engine_urls(self.component_log)
        control = self.engines.service.control
        called = time.monotonic()
        # The service's own ``close`` gives the component this long to stop its engines.
        answer = await asyncio.to_thread(control.call, "DELETE", "engines", timeout_s=HEALTH_TIMEOUT_S)
        removed = time.monotonic()
        assert answer == {"ok": True}, f"{label}: the component answered {answer!r}"
        refused = 0
        answered: float | None = None
        if served is not None:
            while True:
                stale = await self.refusal(f"{label}: startup route, request {refused}", self.startup, PLAIN_PROMPT)
                assert stale is not None, f"{label}: the startup load's route answered"
                if await self.refusal(f"{label}: served route, request {refused}", served, PLAIN_PROMPT) is None:
                    break
                refused += 1
                waited = time.monotonic() - called
                assert waited < RECOVERY_TIMEOUT_S, f"{label}: {served.name} is still refused after {waited:.0f} s"
                await asyncio.sleep(REFUSED_RETRY_S)
            answered = time.monotonic()
            assert refused, f"{label}: {served.name} answered at once: the removal did not interrupt its route"
        failed = await first_check(self.engines, first, passed=False, timeout_seconds=QUIET_TIMEOUT_S)
        restored = await first_check(self.engines, failed + 1, passed=True, timeout_seconds=RECOVERY_TIMEOUT_S)
        checks, lines = self.engines.checks[first : restored + 1], self.log.lines[logged:]
        started = engine_urls(self.component_log)
        assert restored == failed + 1, f"{label}: {restored - failed} health checks failed: {checks}"
        strangers = sorted({check.thread for check in checks} - {WATCHER_THREAD})
        assert not strangers, f"{label}: the threads {strangers} checked the engines, not only the watcher"
        warned = len(lines) == 1 and not lines[0].error and lines[0].thread == WATCHER_THREAD
        assert warned, f"{label}: the runtime logged {lines}, not one warning of its watcher"
        assert len(started) == len(stacks) + 1, f"{label}: the engine stacks {stacks} became {started}"
        failure, success = self.engines.checks[failed], self.engines.checks[restored]
        return {
            "control_operation": "DELETE engines",
            "failed_health_check": failure.error,
            "seconds_from_call": {
                "engines_removed": round(removed - called, 3),
                "health_check_failed": round(failure.ended - called, 3),
                "served_route_answered": None if answered is None else round(answered - called, 3),
                "health_check_passed": round(success.ended - called, 3),
            },
            "refused_requests_of_served_route": refused,
            "engine_stacks_started": started,
        }

    async def before_activation(self) -> None:
        """Before its first activation, the runtime's watcher only replaces lost engines.

        The runtime knows no artifact yet. Its watcher checks the engines and logs nothing while they are healthy.
        After their removal it starts new engines, which stay paused under the startup load, and is at rest again.
        The run publishes a route for the startup load, which Reef never does, to show that no request generates.
        """
        runtime = self.runtime
        route = ServedArtifact(self.startup.artifact_id, runtime.base_url, runtime.served_model_name(SCENARIO))
        self.publish({self.startup.artifact_id: route})
        await self.at_rest("before the first activation")
        await self.unbound("before the first activation")
        removal = await self.removal("removal before the first activation", None)
        await self.unbound("after the replacement")
        await self.at_rest("after the replacement")
        self.run.report.passed(
            "before the first activation the watcher replaces removed engines, which stay paused and unbound",
            startup_route=route.to_json(),
            removal=removal,
        )

    async def serves(self, label: str, arm: Arm, load_id: str) -> dict[str, Any]:
        """The engines hold ``load_id`` only, and ``arm`` answers with its words, logprobs and receipts: by the
        default route and by its named route, buffered and streamed. Reef still refuses the startup load's route.
        The first request of ``CACHE_PROMPT`` must reuse nothing, because these engines did not see it under this
        load, and the second must reuse the prompt. Return the two cache measurements."""
        held = await self.load_ids()
        assert held == [load_id], f"{label}: the engines hold the runtime loads {held}, not {load_id}"
        await self.chat(f"{label}: default route, buffered", arm, chat_body(PLAIN_PROMPT, stream=False))
        await self.explicit_request(f"{label}: named route, streamed", arm, stream=True, named=True)
        stale = await self.refusal(f"{label}: request for the startup load", self.startup, PLAIN_PROMPT)
        assert stale is not None, f"{label}: the startup load's route answered"
        first = await self.cached_request(f"{label}: first request of the cache prompt", arm)
        second = await self.cached_request(f"{label}: second request of the cache prompt", arm)
        assert first["cached_tokens"] == 0 and second["cached_tokens"] > 0, f"{label}: reused {first}, then {second}"
        return {"first": first, "second": second}

    async def after_activation(self, successor: Checkpoint) -> None:
        """The watcher restores removed engines to the activated artifact, under its runtime load ID.

        The runtime activates ``successor``, B, into the engines that started with A, and B answers. Then the run
        removes the engines and calls nothing else: the watcher alone must start new engines, load B again under
        the same runtime load ID, and resume. The runtime's route for B is the same as before, B answers again,
        and the new engines' prefix cache is empty at first. No checkpoint reads its context, so a wrong load
        shows by its words, and a stale cache entry by the cached-token counts.
        """
        runtime = self.runtime
        artifact = Artifact.local(successor.directory)
        load_id = await asyncio.to_thread(runtime.activate_checkpoint, artifact, scenario=SCENARIO)
        served = runtime.served_artifact(artifact.artifact_id, SCENARIO)
        if served is None or parse_full_weight_name(served.model_name) != (runtime.model_path, load_id):
            raise RuntimeError(f"the activation of B established no route that names its load: {served!r}")
        assert load_id != runtime.base_version, f"the activation of B kept the startup load ID {load_id}"
        arm = Arm("activated", served.artifact_id, successor, native_facade=True)
        self.publish({ACTIVE: served, served.artifact_id: served})
        activated = await self.serves("B after the activation", arm, load_id)
        removal = await self.removal("removal while B serves", arm)
        route = runtime.served_artifact(artifact.artifact_id, SCENARIO)
        assert route == served, f"the runtime's route for B is {route!r} after the restoration, not {served!r}"
        restored = await self.serves("B after the restoration", arm, load_id)
        await self.at_rest("after the restoration")
        self.run.report.passed(
            "the watcher restores removed engines to the activated artifact under its runtime load ID",
            route=served.to_json(),
            runtime_load_id=load_id,
            removal=removal,
            cache_prompt_requests={"after the activation": activated, "after the restoration": restored},
        )


async def qualify_recovery(run: Run) -> None:
    _, successor = run.checkpoints()
    service = await start_component(run, artifact_serving=True)
    engines = ObservedEngines(service)
    runtime = native_runtime(run, engines, health_interval_s=WATCHER_INTERVAL_S)
    run.runtime = runtime
    runtime.set_owner_sequence(1)
    coordination = memory_coordination()
    storage = SQLiteScenarioStorage(run.layout.records)
    with serving_log() as log:
        opened = time.monotonic()
        try:
            await asyncio.to_thread(runtime.open)
            run.report.facts["engines"]["start_order"].append(f"native {service.engine} on the component")
            app = build_inference_app(
                coordination.served_models,
                storage,
                DeploymentEngineClients(runtime, provider_key=None, timeout_seconds=INFERENCE_TIMEOUT_S),
                tokens=run.credentials.service,
                timeout_seconds=INFERENCE_TIMEOUT_S,
            )
            async with (
                serving(app) as url,
                ClientSession(timeout=ClientTimeout(total=REQUEST_TIMEOUT_S)) as session,
            ):
                client = ReefClient(session, url, run.credentials.service, run.layout.wire)
                checks = RecoveryChecks(run, client, coordination, runtime, engines, log)
                await checks.before_activation()
                await checks.after_activation(successor)
            failed = sum(check.error is not None for check in engines.checks)
            assert failed == 2 and len(log.lines) == 2, (
                f"{failed} health checks failed and the runtime logged {len(log.lines)} lines for two removals: "
                f"{log.lines}"
            )
            run.report.passed(
                "the engines failed a health check only at each removal, and the runtime logged one warning for each",
                health_checks=len(engines.checks),
            )
        finally:
            run.report.facts["watcher"] = {
                "interval_seconds": WATCHER_INTERVAL_S,
                "deployment_interval_seconds": HEALTH_INTERVAL_S,
                "health_checks": len(engines.checks),
                "checking_threads": sorted({check.thread for check in engines.checks}),
                "failed_health_checks": [
                    {"seconds_after_open_call": round(check.ended - opened, 3), "error": check.error}
                    for check in engines.checks
                    if check.error is not None
                ],
                "logged": [asdict(line) for line in log.lines],
            }
            storage.close()
            coordination.close()
    records = replay_records(run.layout.records, SCENARIO)
    write_records(run.layout.records_file, records)
    checks.check_records(records)
    run.report.passed("records match their requests exactly", records=len(records))


# -- Lifecycle profile ---------------------------------------------------------------------------------------------


def active_artifact(served: Mapping[str, int]) -> str:
    """The active artifact of ``served``, artifact weights of one component: the one with the largest weight, the
    first in sorted order among equals, as ``Dispatcher.set_served`` selects it."""
    return min(served, key=lambda artifact_id: (-served[artifact_id], artifact_id))


@dataclass
class NativeSlot:
    """One inference component of the lifecycle profile's runtime, with what the run expects it to hold.

    Values: ``log`` is the file in which the component names each engine stack that it starts. ``runtime`` is the
    full-weight runtime that the factory built for the component: one slot of the profile's runtime. ``load_id`` is
    the runtime load ID that the component's engines must hold: the slot's startup load until its first checkpoint,
    and a released checkpoint's load until the next one. ``arm`` is the checkpoint that the slot must serve, or
    None while it must serve none. ``stacks`` are the engine stacks that the component must have started.
    Purpose: ``LifecycleChecks.native_weights`` changes ``load_id``, ``arm`` and ``stacks`` where it expects a
    change, and ``slots_hold`` compares them with the published routes, the runtime and the component.
    """

    log: Path
    runtime: FullWeightServingRuntime
    load_id: str
    arm: Arm | None = None
    stacks: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        """The base URL of the slot's component, as the slot's runtime names it."""
        return self.runtime.base_url

    @property
    def service(self) -> RemoteInferenceService:
        """The control client of the slot's component. The factory built it, and its calls carry the sequence of
        the serving worker that owns the engines."""
        engines = self.runtime._engines
        if not isinstance(engines, ComponentEngines):
            raise TypeError(f"the slot at {self.url} runs no component engines, but {type(engines).__name__}")
        return engines.service

    async def load_ids(self) -> list[str]:
        """The runtime load IDs that the component reports for its engines, one for each engine; a read only."""
        service = self.service
        backend = RemoteInferenceBackend(service.control, service)
        return list(await asyncio.to_thread(backend.runtime_load_ids))

    def bind(self, arm: Arm, route: ServedArtifact) -> str:
        """Expect ``arm`` in this slot from now on, under the runtime load ID that its published ``route`` names;
        return that ID.

        The route must be on this component, and must name a load of the runtime's model that the engines did not
        hold before: a slot loads each checkpoint that it binds.
        """
        loaded = parse_full_weight_name(route.model_name)
        assert (
            loaded is not None and loaded[0] == self.runtime.model_path
        ), f"{arm.name}'s route names no load: {route}"
        assert route.url == self.url, f"{arm.name} serves at {route.url}, not on the component at {self.url}"
        assert loaded[1] != self.load_id, f"{arm.name} serves under the load ID that the slot had, {self.load_id}"
        self.arm, self.load_id = arm, loaded[1]
        return self.load_id


async def start_checkpoint_slots(run: Run) -> tuple[CheckpointSlotsRuntime, list[NativeSlot]]:
    """Start the lifecycle profile's two inference components, and build the runtime that serves one checkpoint on
    each; return the runtime and its slots, the first component's first.

    The remote serving factory builds the runtime from an ordinary runtime section, which names the first component
    as ``url`` and the second in ``slot-urls``. Building it starts no engine, and nothing here opens it: the serving
    worker of the dispatcher that gets the runtime does. Both components run the native engine on the run's one GPU,
    each in its own local Ray session. The engine facts are recorded here, so that a failed engine start still
    reports what it ran.
    """
    engine = run.inputs.native_engine
    native, _ = run.checkpoints()
    started = [await start_component_process(run, name) for name in SLOT_COMPONENTS]
    urls = [url for url, _ in started]
    config = {
        "type": "remote_serving",
        "engine": engine,
        "remote": {"url": urls[0], "slot-urls": urls[1:], "startup-timeout-s": ENGINE_START_TIMEOUT_S + 60},
        "artifact_format": ArtifactFormat(layout="full").to_dict(),
        "num_gpus": 1,
        "tensor_parallel_size": 1,
        "prefix_sharing": True,
        "options": NATIVE_OPTIONS[engine],
        "inference_timeout_s": LIFECYCLE_INFERENCE_TIMEOUT_S,
    }
    # The tokens reach the factory as a deployment's would: in the environment, under the default variable names.
    environ = {
        **os.environ,
        DEFAULT_CONTROL_TOKEN_ENV: run.credentials.control,
        DEFAULT_INFERENCE_TOKEN_ENV: run.credentials.inference,
    }
    runtime = RuntimeRegistry().build(config, model_path=str(native.directory), environ=environ)
    if not isinstance(runtime, CheckpointSlotsRuntime):
        raise TypeError(f"the factory built a {type(runtime).__name__} for two components, not checkpoint slots")
    run.runtime = runtime
    slots = [
        NativeSlot(process.log, child, child.base_version)
        for child, (_, process) in zip(runtime.slots, started, strict=True)
    ]
    run.report.facts["engines"] = {
        "start_order": [],
        "native": {
            "engine": engine,
            "request_pinning": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "components": {
                name: {"url": url, "command": process.command}
                for name, (url, process) in zip(SLOT_COMPONENTS, started, strict=True)
            },
            "runtime_config": config,
            "engine_config": slots[0].service.engine_config,
        },
    }
    capacity = runtime.served_artifact_capacity
    assert capacity == len(SLOT_COMPONENTS), f"the runtime serves {capacity} artifacts at once, not one a component"
    assert [slot.url for slot in slots] == urls, f"the runtime's slots are at {[slot.url for slot in slots]}"
    engines_started = {slot.url: engine_urls(slot.log) for slot in slots}
    idle = not runtime.opened and not any(engines_started.values())
    assert idle, f"building the runtime opened it or started engines: {engines_started}"
    return runtime, slots


def endpoint_routes(provider_url: str, served: Mapping[str, int]) -> dict[str, ServedArtifact]:
    """The routes of ``served``, harness artifacts with their weights, when the provider's one model at
    ``provider_url`` answers for each: a route under each artifact's own ID, and the active artifact's also as
    ``ACTIVE``."""
    routes = {artifact_id: ServedArtifact(artifact_id, provider_url, PROVIDER_MODEL) for artifact_id in served}
    return {ACTIVE: routes[active_artifact(served)], **routes}


@dataclass(frozen=True)
class ScenarioState:
    """What the public API and the coordination store say about the scenario at one moment.

    Values: ``settings``, ``artifacts`` and ``operations`` are the answers of the public routes of those names;
    ``routes`` are the routes that the serving worker published, by name; ``traffic_weights`` are the weights of
    the traffic split that the store holds for the scenario, by artifact, and empty without a split;
    ``serving_request`` is the ID of the scenario's current serving request, or None when it has none.
    Purpose: ``LifecycleChecks`` checks one state, and compares the states before and after a step that must
    change none of this: a refused update, the same update again, and the restoration of removed engines.
    """

    settings: dict[str, JsonValue]
    artifacts: dict[str, JsonValue]
    operations: dict[str, JsonValue]
    routes: dict[str, ServedArtifact]
    traffic_weights: dict[str, int]
    serving_request: str | None


class LifecycleChecks(RequestChecks):
    """The requests and checks of the lifecycle profile, against one Reef app with its dispatcher.

    Values: ``control`` sends the public control requests; ``component`` is the scenario component whose artifacts
    the scenario serves.
    Purpose: ``public_app`` makes one for each scenario of the profile. It changes what the scenario serves only
    through ``control``. It reads the coordination store to wait for the serving worker and to compare the routes,
    the traffic weights and the scheduled stops, and never writes to it. Its one other change is a fault:
    ``restored_after_removal`` removes the engines of one inference component.
    """

    def __init__(
        self, run: Run, client: ReefClient, coordination: Coordination, control: ControlClient, component: str
    ) -> None:
        super().__init__(run, client, coordination)
        self.control = control
        self.component = component

    # -- The scenario's state

    async def wait_for_routes(self, label: str, served: Mapping[str, int]) -> dict[str, ServedArtifact]:
        """Wait until the serving worker finished the scenario's serving request with the routes of ``served``,
        artifact weights of the component; return the routes by name.

        The routes are one for each artifact of ``served``, and ``ACTIVE`` for the active one. The wait only reads
        the coordination store. A failed serving request fails it.
        """
        active_id = active_artifact(served)
        deadline = time.monotonic() + ENGINE_START_TIMEOUT_S
        while True:
            request = self.coordination.served_models.current_request(SCENARIO)
            if request is not None and request.state == "failed":
                raise AssertionError(f"{label}: the serving request {request.id} failed: {request.error}")
            routes = dict(self.coordination.served_models.read_scenario(SCENARIO))
            finished = request is not None and request.state == "finished"
            if finished and set(routes) == {ACTIVE, *served} and routes[ACTIVE].artifact_id == active_id:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"{label}: after {ENGINE_START_TIMEOUT_S:.0f} s the serving request is {request} and the routes "
                    f"are {routes}, not those of {dict(served)}"
                )
            await asyncio.sleep(0.1)
        for artifact_id in served:
            named = routes[artifact_id].artifact_id
            assert named == artifact_id, f"{label}: the route of {artifact_id} names the artifact {named}"
        return routes

    async def serve(self, label: str, served: Mapping[str, int]) -> dict[str, ServedArtifact]:
        """Make ``served`` the artifacts that the scenario serves of the component, with their weights, by a public
        update; return the routes that the serving worker then published."""
        answer = await self.control.request(
            label, "POST", f"/reef/scenarios/{SCENARIO}/update", {"served": dict(served), "component": self.component}
        )
        assert answer.get("served") == dict(served), f"{label}: the update answered {answer}"
        return await self.wait_for_routes(label, served)

    async def scenario_state(self, label: str) -> ScenarioState:
        path = f"/reef/scenarios/{SCENARIO}"
        settings = await self.control.request(f"{label}: settings", "GET", f"{path}/settings")
        artifacts = await self.control.request(f"{label}: artifacts", "GET", f"{path}/artifacts")
        operations = await self.control.request(f"{label}: operations", "GET", f"{path}/operations")
        store = self.coordination.served_models
        request = store.current_request(SCENARIO)
        return ScenarioState(
            settings,
            artifacts,
            operations,
            dict(store.read_scenario(SCENARIO)),
            dict(store.read_traffic_weights(SCENARIO)),
            None if request is None else request.id,
        )

    async def selection(self, label: str, served: Mapping[str, int], artifact_ids: Sequence[str]) -> ScenarioState:
        """Check that the public API lists ``artifact_ids`` as the artifacts of the scenario, all of the component,
        and ``served`` as the weights of those that the scenario serves; return the state that it read."""
        state = await self.scenario_state(label)
        mapping = state.settings["served"]
        assert mapping == {self.component: dict(served)}, f"{label}: the settings name the served artifacts {mapping}"
        listed = state.artifacts["artifacts"]
        assert {row["artifact_id"] for row in listed} == set(artifact_ids), f"{label}: the artifact list is {listed}"
        active_id = active_artifact(served)
        for row in listed:
            artifact_id = row["artifact_id"]
            # A row's ``served`` means that the scenario serves the artifact beside the active one.
            available = bool(row["active"] or row["served"])
            agrees = bool(row["active"]) == (artifact_id == active_id) and available == (artifact_id in served)
            assert row["component"] == self.component and agrees, f"{label}: the artifact list has {row}"
        return state

    def pending_stops(self) -> set[ScheduledStop]:
        """The set of the stops that the store holds, without their order: each route that left the published
        routes and that the serving worker has not stopped yet, with the time from which the worker may stop it. A
        read only: it asks for the stops that are due at the latest time that a float holds."""
        return set(self.coordination.served_models.due_stops(sys.float_info.max))

    # -- Requests

    async def answers(self, label: str, arm: Arm) -> None:
        """``arm`` answers a buffered request that names no artifact, and a streamed request to its named route."""
        await self.chat(f"{label}: no artifact named, buffered", arm, chat_body(PLAIN_PROMPT, stream=False))
        await self.explicit_request(f"{label}: named route, streamed", arm, stream=True, named=True)

    async def split_conversations(
        self, label: str, arms: Sequence[Arm], weights: Mapping[str, int]
    ) -> dict[str, tuple[str, int]]:
        """Hold a conversation of two turns with each of ``arms``, the two artifacts that share the traffic by the
        split ``weights``; return each one's first message and its bucket, by the arm's name.

        No request names an artifact: the bucket of the first message selects the arm, and the second turn must
        stay with it. The first arm's turns are buffered and then streamed, the second arm's streamed and then
        buffered. Each record must store the split and the bucket.
        """
        conversations = {arm.name: self.conversation(arm, weights) for arm in arms}
        for arm, stream in zip(arms, (False, True), strict=True):
            text, bucket = conversations[arm.name]
            traffic = ExpectedTraffic(weights, bucket)
            first = await self.chat(
                f"{label} conversation of {arm.name}, turn 1", arm, chat_body(text, stream=stream), traffic=traffic
            )
            history = [{"role": "user", "content": text}, {"role": "assistant", "content": first.text()}]
            await self.chat(
                f"{label} conversation of {arm.name}, turn 2",
                arm,
                chat_body("Continue.", stream=not stream, history=history),
                traffic=traffic,
            )
        return conversations

    async def cache_after_load(self, label: str, arm: Arm, component_log: Path) -> dict[str, JsonValue]:
        """Send the cache prompt to ``arm`` twice after its load: the first request must reuse no prompt token, and
        the second some. ``component_log`` is the log of the component that serves ``arm`` (``cached_request``).
        Return what each request measured."""
        first = await self.cached_request(f"{label}: first cache-prompt request", arm, component_log)
        second = await self.cached_request(f"{label}: second cache-prompt request", arm, component_log)
        assert first["cached_tokens"] == 0 and second["cached_tokens"] > 0, f"{label}: reused {first}, then {second}"
        return {"first": first, "second": second}

    async def still_cached(self, label: str, slots: Sequence[NativeSlot]) -> dict[str, JsonValue]:
        """Send the cache prompt once to the checkpoint of each of ``slots`` that serves one. Each request must
        reuse prompt tokens: the component's engines still hold the prefix cache of the checkpoint's earlier
        requests. Return what each request measured, by the checkpoint's name."""
        measured: dict[str, JsonValue] = {}
        for slot in slots:
            if slot.arm is None:
                continue
            reused = await self.cached_request(f"{label}: cache prompt of {slot.arm.name}", slot.arm, slot.log)
            assert reused["cached_tokens"] > 0, f"{label}: {slot.arm.name} reused no prompt token: {reused}"
            measured[slot.arm.name] = reused
        return measured

    async def component_holds(self, label: str, slot: NativeSlot) -> None:
        """The component of ``slot`` reports the slot's runtime load ID for its one engine, and has started the
        slot's engine stacks and no other. The check only reads."""
        held, started = await slot.load_ids(), engine_urls(slot.log)
        assert held == [slot.load_id], f"{label}: the engines at {slot.url} hold {held}, not {slot.load_id}"
        assert started == slot.stacks, f"{label}: the component at {slot.url} started {started}, not {slot.stacks}"

    async def slots_hold(
        self, label: str, runtime: CheckpointSlotsRuntime, slots: Sequence[NativeSlot], served: Mapping[str, int]
    ) -> dict[str, ServedArtifact]:
        """Check that the published routes, ``runtime`` and the components agree with what ``slots`` expect while
        the scenario serves ``served``, weight artifacts with their weights; return the published routes.

        Each slot with a checkpoint gives one route: the checkpoint on the slot's component, under the slot's
        runtime load ID. Those must be the bindings of ``runtime``, which must give the same route for each one,
        and the store must hold the routes of ``served`` among them, the active one also as ``ACTIVE``. Each
        component must hold what its slot expects (``component_holds``). The check only reads.
        """
        expected = {
            slot.arm.artifact_id: ServedArtifact(
                slot.arm.artifact_id, slot.url, full_weight_name(slot.runtime.model_path, slot.load_id)
            )
            for slot in slots
            if slot.arm is not None
        }
        bindings = dict(runtime.bindings)
        assert bindings == expected, f"{label}: the runtime binds {bindings}, not {expected}"
        for artifact_id, route in expected.items():
            given = runtime.served_artifact(artifact_id, SCENARIO)
            assert given == route, f"{label}: the runtime serves {artifact_id} as {given}, not as {route}"
        routes = dict(self.coordination.served_models.read_scenario(SCENARIO))
        published = {ACTIVE: expected[active_artifact(served)], **{name: expected[name] for name in served}}
        assert routes == published, f"{label}: the published routes are {routes}, not {published}"
        for slot in slots:
            await self.component_holds(label, slot)
        return routes

    async def not_served(self, label: str, arm: Arm) -> None:
        """The named route of ``arm``, an artifact that the scenario does not serve, answers HTTP 404 without a
        receipt."""
        exchange = await self.client.post(
            label, self.route(arm), chat_body(PLAIN_PROMPT, stream=False), headers={SCENARIO_HEADER: SCENARIO}
        )
        answer = exchange.content.decode(errors="replace")
        assert exchange.status == 404, f"{label}: HTTP {exchange.status}: {answer!r}"
        receipt = RECORD_HEADER in exchange.response_headers or "agent_record_id" in answer
        assert not receipt, f"{label}: a refused request has a receipt"

    async def serves_unchanged(
        self, label: str, runtime: CheckpointSlotsRuntime, slot: NativeSlot, route: ServedArtifact
    ) -> tuple[Exchange, dict[str, JsonValue]]:
        """The checkpoint of ``slot``, which has all the traffic, serves as it did when ``route`` was its route;
        return the exchange of its first request here, and what its cache-prompt request measured.

        The checkpoint answers a request that names no artifact. Then it answers the cache prompt by its named
        route, and its component's engines must still hold that prompt in their prefix cache. The store must
        publish the checkpoint and ``runtime`` must bind it as ``route``, and its component must hold what the
        slot expects (``component_holds``).
        """
        arm = slot.arm
        assert arm is not None, f"{label}: the slot at {slot.url} must hold a checkpoint"
        exchange = await self.chat(f"{label}: no artifact named", arm, chat_body(PLAIN_PROMPT, stream=False))
        cache = await self.still_cached(label, [slot])
        published = dict(self.coordination.served_models.read_scenario(SCENARIO)).get(arm.artifact_id)
        binding = runtime.bindings.get(arm.artifact_id)
        same = published == route and binding == route
        assert same, f"{label}: {arm.name} is published as {published} and bound as {binding}, not as {route}"
        await self.component_holds(label, slot)
        return exchange, cache

    # -- Checks

    async def all_traffic(
        self,
        label: str,
        runtime: CheckpointSlotsRuntime,
        slots: Sequence[NativeSlot],
        served: Mapping[str, int],
        artifact_ids: Sequence[str],
        text: str,
    ) -> dict[str, JsonValue]:
        """Give one of the two checkpoints of ``slots`` all the traffic by the public update ``served``, which
        keeps the other at weight 0; return the published routes and what the cache-prompt requests measured.

        Nothing may load: each component must keep its checkpoint under its runtime load ID, in the engines that
        it had, with the cache prompt in its prefix cache. ``text`` starts a conversation that a 50/50 split gave
        to the checkpoint at weight 0. The selected checkpoint must answer it now, as it answers each request that
        names no artifact, and the records store no traffic metadata: one weight above 0 is no split. The other
        checkpoint still answers by its named route.
        """
        await self.serve(label, served)
        await self.slots_hold(label, runtime, slots, served)
        await self.selection(label, served, artifact_ids)
        arms = {slot.arm.artifact_id: slot.arm for slot in slots if slot.arm is not None}
        selected = arms.pop(active_artifact(served))
        (other,) = arms.values()
        await self.chat(
            f"{label}: conversation that the split gave to {other.name}", selected, chat_body(text, stream=True)
        )
        await self.answers(f"{label}, {selected.name}", selected)
        for stream in (False, True):
            await self.explicit_request(
                f"{label}: {mode(stream)} named route of {other.name}", other, stream=stream, named=True
            )
        cache = await self.still_cached(label, slots)
        routes = await self.slots_hold(f"{label}, after the requests", runtime, slots, served)
        return {"routes": {name: route.to_json() for name, route in routes.items()}, "cache_prompt_requests": cache}

    async def released_after_grace(
        self,
        runtime: CheckpointSlotsRuntime,
        leaving: NativeSlot,
        staying: NativeSlot,
        asked: float,
        stop: ScheduledStop,
    ) -> dict[str, JsonValue]:
        """Wait until the serving worker has stopped the checkpoint of the slot ``leaving``, which left the
        published routes; return when the wait saw what, in seconds after ``asked``.

        ``asked`` is the time of the update that removed the checkpoint. ``stop`` is the stop that the update
        scheduled: the checkpoint's route, and the time from which the worker may stop it. Both times are those of
        ``time.time``. The wait reads the store and the runtime and sends requests: it never stops the checkpoint,
        changes the stop or drives the worker. Until the stop is due, the runtime must keep the checkpoint bound.
        The worker then releases the slot and deletes the stop, in this order. Meanwhile the checkpoint's named
        route answers 404, and the checkpoint of ``staying`` answers with its binding, runtime load ID, engines and
        prefix cache unchanged (``serves_unchanged``).
        """
        retired, kept = leaving.arm, staying.arm
        assert retired is not None and kept is not None, "each slot must hold a checkpoint when the wait starts"
        kept_route = runtime.bindings[kept.artifact_id]
        deadline = time.monotonic() + RETIREMENT_TIMEOUT_S
        bound_until: float | None = None
        checks = 0
        while True:
            # The stop is read first: the worker releases the slot before it deletes the stop.
            stops = self.pending_stops()
            bound = retired.artifact_id in runtime.bindings
            now = time.time()
            assert stops <= {stop}, f"the store holds the stops {stops}, not only {stop}"
            early = stop.stop_after - now
            assert bound or early <= 0, f"the runtime released {retired.name} {early:.1f} s before its stop was due"
            assert stops or not bound, f"the stop of {retired.name} is gone, and the runtime still binds it"
            if not stops:
                break
            if bound:
                bound_until = now
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"after a wait of {RETIREMENT_TIMEOUT_S:.0f} s the stop of {retired.name} is still pending, "
                    f"{-early:.1f} s after it was due; the runtime binds {dict(runtime.bindings)}"
                )
            checks += 1
            label = f"while the stop of {retired.name} is pending, check {checks}"
            await self.not_served(f"{label}: the named route of {retired.name}", retired)
            await self.serves_unchanged(label, runtime, staying, kept_route)
            await asyncio.sleep(0.5)
        released = time.time()
        serving = leaving.runtime.served_artifact_id
        assert serving is None, f"the stop of {retired.name} is gone, and its slot still serves {serving}"
        assert bound_until is not None, f"the wait never saw {retired.name} bound while its stop was pending"
        return {
            "stop_due_s": round(stop.stop_after - asked, 3),
            "bound_until_s": round(bound_until - asked, 3),
            "released_seen_s": round(released - asked, 3),
            "checks_while_pending": checks,
        }

    async def restored_after_removal(
        self, runtime: CheckpointSlotsRuntime, losing: NativeSlot, staying: NativeSlot
    ) -> dict[str, JsonValue]:
        """Remove the engines of the slot ``losing``, and wait until the health watcher of its runtime has restored
        its checkpoint; return what the wait measured, with times in seconds after the removal's call.

        The removal is the control operation ``DELETE engines`` of that slot's component, sent as the engines'
        owner with the slot's own control client, as the recovery profile sends it. The component stops its engine
        stack, and stays. Nothing else here changes anything: no call starts engines, loads a checkpoint, reaches
        the runtime or writes to the store. The watcher alone must find the loss at a health check, start engines,
        load the checkpoint again and resume. The runtimes must log one warning for it, on a watcher's thread, and
        no error.

        Until then Reef must refuse each request for the lost checkpoint's named route, without a receipt, and
        its first answer ends the wait. After each refusal the checkpoint of ``staying`` must serve unchanged
        (``serves_unchanged``), with two requests. Such a round lies within the outage that the run saw when the
        next request for the lost checkpoint was refused too; at least one round must. The requests for the lost
        checkpoint carry the plain prompt, so the new engines' prefix cache has not seen the cache prompt.
        """
        lost, kept = losing.arm, staying.arm
        assert lost is not None and kept is not None, "each slot must hold a checkpoint when the engines go"
        kept_route = runtime.bindings[kept.artifact_id]
        refusals: list[Exchange] = []
        rounds: list[tuple[Exchange, dict[str, JsonValue]]] = []
        warning_seen: float | None = None
        with serving_log() as log:
            called = time.monotonic()
            # The service's own ``close`` gives the component this long to stop its engines.
            answer = await asyncio.to_thread(
                losing.service.control.call, "DELETE", "engines", timeout_s=HEALTH_TIMEOUT_S
            )
            removed = time.monotonic()
            assert answer == {"ok": True}, f"the component at {losing.url} answered {answer!r}"
            while True:
                label = f"after the removal of {lost.name}'s engines, round {len(rounds)}"
                refused = await self.refusal(f"{label}: the named route of {lost.name}", lost, PLAIN_PROMPT)
                if refused is None:
                    break
                refusals.append(refused)
                waited = time.monotonic() - called
                assert waited < RECOVERY_TIMEOUT_S, f"{label}: {lost.name} is still refused after {waited:.0f} s"
                rounds.append(await self.serves_unchanged(label, runtime, staying, kept_route))
                if warning_seen is None and log.lines:
                    warning_seen = time.monotonic()
                await asyncio.sleep(REFUSED_RETRY_S)
            answered = time.monotonic()
        lines = list(log.lines)
        warned = len(lines) == 1 and not lines[0].error and lines[0].thread == WATCHER_THREAD
        assert warned, f"the runtimes logged {lines} for one removal, not one warning of a health watcher"

        def after_call(moment: float) -> float:
            return round(moment - called, 3)

        # The last round ended before the lost checkpoint answered, not before another refusal: it does not count.
        within: list[JsonValue] = []
        for earlier, (exchange, cache), later in zip(refusals[:-1], rounds[:-1], refusals[1:], strict=True):
            between = earlier.ended <= exchange.sent and exchange.ended <= later.sent
            assert between, f"{exchange.label}: not between two refusals of {lost.name}"
            within.append(
                {
                    "refused_before": after_call(earlier.ended),
                    "request_sent": after_call(exchange.sent),
                    "request_answered": after_call(exchange.ended),
                    "receipt": exchange.receipt(),
                    "cache_prompt_request": cache[kept.name],
                    "refused_after": after_call(later.ended),
                }
            )
        assert within, (
            f"{lost.name} answered after {len(refusals)} refusals: the run saw no request of {kept.name} between "
            f"two of them"
        )
        return {
            "control_operation": "DELETE engines",
            "component": losing.url,
            "watcher_interval_seconds": losing.runtime._health_interval_s,
            "seconds_from_call": {
                "engines_removed": after_call(removed),
                "first_refusal": after_call(refusals[0].ended),
                # The end of the first round that found the watcher's warning for the failed health check in the
                # log: the watcher logged it during that round or the wait before it. None when no round found it.
                "watcher_warning_seen": None if warning_seen is None else after_call(warning_seen),
                "last_refusal": after_call(refusals[-1].ended),
                "served_route_answered": after_call(answered),
            },
            "refused_requests_of_served_route": len(refusals),
            "staying_checkpoint": kept.name,
            "requests_of_staying_checkpoint_between_two_refusals": 2 * len(within),
            "rounds_of_staying_checkpoint_between_two_refusals": within,
            "runtime_log": [asdict(line) for line in lines],
        }

    async def native_weights(
        self, runtime: CheckpointSlotsRuntime, slots: Sequence[NativeSlot], native: Checkpoint, successor: Checkpoint
    ) -> list[str]:
        """The public lifecycle of two weight artifacts that serve at the same time on the native engine. Return
        the IDs of the scenario's weight artifacts: A, B and the capacity fixture.

        ``slots`` are the two components of ``runtime``. Each step but the last is a public request. The creation
        serves A: the serving worker opens both components and loads A into the first. After its import, B must
        load into the second under its own runtime load ID, while A's route, load and engines stay as they were.
        Then each answers by its named route with its own words while the other answers, and each component's
        prefix cache serves its own checkpoint.

        Three updates then move the traffic: a 50/50 split, all of it to B, and all of it back to A. None may load
        anything: both checkpoints keep their routes, runtime load IDs, engines and prefix caches. An update
        without A ends its route at once. The serving worker must release A's component only after the grace time,
        while B serves unchanged; the component keeps its engines. An update that names A again must load it into
        that component under a new runtime load ID, with an empty prefix cache, and leave B as it was. Then the
        scenario imports the capacity fixture, a copy of B, as a third artifact: two components serve no three
        artifacts, so an update that names all three must be refused and change nothing.

        The last step is a fault, and no public request: the run removes the engines of A's component
        (``restored_after_removal``). The health watcher of A's runtime must restore A under the runtime load ID
        that it had, in one new engine stack with an empty prefix cache. B must serve during A's outage with its
        route, runtime load ID, engines and prefix cache unchanged, and the scenario's public state, routes and
        traffic weights must be the same before and after.

        After the worker opened the components, engines start only once more: A's, for that restoration.
        """
        path = f"/reef/scenarios/{SCENARIO}"
        first, second = slots
        created = await self.control.request(
            "create the scenario", "POST", "/reef/scenarios", {"name": SCENARIO}, status=201
        )
        root_id = str(created["artifact_id"])
        root = Arm("A", root_id, native, native_facade=True)
        alone = {root_id: 100}
        routes = await self.wait_for_routes("create the scenario", alone)
        root_route = routes[root_id]
        root_load = first.bind(root, root_route)
        # The serving worker opened both components before it loaded A, so each one has started its engines by now.
        # They must keep these.
        order: list[str] = self.run.report.facts["engines"]["start_order"]
        for name, slot in zip(SLOT_COMPONENTS, slots, strict=True):
            slot.stacks = engine_urls(slot.log)
            assert len(slot.stacks) == 1, f"the component at {slot.url} started the engine stacks {slot.stacks}"
            order.append(f"native {self.run.inputs.native_engine} on {name}")
        stacks = {slot.url: slot.stacks for slot in slots}
        await self.slots_hold("after the creation", runtime, slots, alone)
        await self.selection("after the creation", alone, [root_id])
        await self.answers("A after the creation", root)
        self.run.report.passed(
            "a public scenario creation serves the root weights A on the first component, and the second holds its "
            "startup load",
            route=root_route.to_json(),
            runtime_load_ids={slot.url: slot.load_id for slot in slots},
            engine_stacks_started=stacks,
        )

        imported = await self.control.request(
            "import B", "POST", f"{path}/artifacts", {"source": successor.directory.as_uri()}, status=201
        )
        successor_id = str(imported["artifact_id"])
        assert successor_id != root_id and imported.get("component") == self.component, f"the import: {imported}"
        beside = Arm("B", successor_id, successor, native_facade=True)
        artifact_ids = [root_id, successor_id]
        state = await self.selection("after the import of B", alone, artifact_ids)
        assert state.routes == routes, f"the import changed the routes from {routes} to {state.routes}"
        await self.slots_hold("after the import of B", runtime, slots, alone)
        await self.not_served("the named route of B before it serves", beside)
        await self.answers("A after the import of B", root)
        self.run.report.passed(
            "a public import adds the weights B, and A stays the only served artifact under its runtime load ID",
            artifact_id=successor_id,
        )

        together = {root_id: 100, successor_id: 0}
        routes = await self.serve("serve B beside A", together)
        successor_load = second.bind(beside, routes[successor_id])
        assert routes[root_id] == root_route, f"A's route changed from {root_route} to {routes[root_id]}"
        distinct = successor_load != root_load and second.url != first.url
        assert distinct, f"B serves at {second.url} under {successor_load}, and A at {first.url} under {root_load}"
        await self.slots_hold("A and B together", runtime, slots, together)
        await self.selection("A and B together", together, artifact_ids)
        # Both named routes answer at the same time, buffered and streamed.
        arms = (root, beside)
        exchanges = await asyncio.gather(
            *(
                self.explicit_request(
                    f"A and B together: {mode(stream)} named route of {arm.name}", arm, stream=stream, named=True
                )
                for arm in arms
                for stream in (False, True)
            )
        )
        overlapping = sum(a.sent < b.ended and b.sent < a.ended for a in exchanges[:2] for b in exchanges[2:])
        assert overlapping, "no request of A overlapped a request of B"
        # Neither component has seen the cache prompt: each one's first request of it must reuse nothing.
        cache = {
            arm.name: await self.cache_after_load(f"{arm.name} while both serve", arm, slot.log)
            for arm, slot in zip(arms, slots, strict=True)
        }
        await self.answers("A beside B", root)
        routes = await self.slots_hold("A and B after their requests", runtime, slots, together)
        self.run.report.passed(
            "a public update serves B beside A: each on its own component under its own runtime load ID, with named "
            "routes that answer at the same time and its own prefix cache",
            routes={name: route.to_json() for name, route in routes.items()},
            runtime_load_ids={root.name: root_load, beside.name: successor_load},
            overlapping_request_pairs_of_a_and_b=overlapping,
            cache_prompt_requests=cache,
            engine_stacks_started=stacks,
        )

        # The next three updates move the traffic between two bound checkpoints. The slots expect no change, so
        # ``slots_hold`` checks each time that both kept their routes, runtime load IDs and engine stacks.
        halves = {root_id: 50, successor_id: 50}
        await self.serve("serve A and B 50/50", halves)
        await self.slots_hold("at 50/50", runtime, slots, halves)
        await self.selection("at 50/50", halves, artifact_ids)
        conversations = await self.split_conversations("50/50", arms, halves)
        routes = await self.slots_hold("after the 50/50 conversations", runtime, slots, halves)
        self.run.report.passed(
            "a public update splits the traffic 50/50 between A and B: each keeps its conversations, its component "
            "and its runtime load ID",
            routes={name: route.to_json() for name, route in routes.items()},
            conversation_buckets={name: bucket for name, (_, bucket) in conversations.items()},
        )

        promoted = {root_id: 0, successor_id: 100}
        promotion = await self.all_traffic(
            "the promotion of B", runtime, slots, promoted, artifact_ids, conversations[root.name][0]
        )
        self.run.report.passed(
            "a public update promotes B to all the traffic and loads nothing: A still answers by its named route, "
            "and both prefix caches stay",
            **promotion,
        )
        rollback = await self.all_traffic(
            "the rollback to A", runtime, slots, together, artifact_ids, conversations[beside.name][0]
        )
        self.run.report.passed(
            "a public update rolls the traffic back to A and loads nothing: B still answers by its named route, "
            "and both prefix caches stay",
            **rollback,
        )

        # A leaves the routes. The serving worker must keep A's component bound for the grace time, in which a
        # request that A's route admitted may still run, and release it then. The run only watches that.
        only = {successor_id: 100}
        grace = CACHE_SECONDS + runtime.inference_timeout_s
        stops = self.pending_stops()
        assert not stops, f"the store holds stops before A leaves the routes: {stops}"
        asked = time.time()
        await self.serve("stop serving A", only)
        published = time.time()
        stops = self.pending_stops()
        assert [stop.served for stop in stops] == [root_route], f"the stops after the update without A are {stops}"
        (stop,) = stops
        due = stop.stop_after
        assert asked + grace <= due <= published + grace, (
            f"A's stop is due {due - asked:.3f} s after the update's request, which took {published - asked:.3f} s: "
            f"not the grace time of {grace:g} s after the publication"
        )
        await self.selection("after the update without A", only, artifact_ids)
        # The first slot expects A until the wait has seen its release: A is off the routes, and still bound.
        await self.slots_hold("after the update without A", runtime, slots, only)
        release = await self.released_after_grace(runtime, first, second, asked, stop)
        first.arm = None
        # The released component keeps its engines, which still hold A's load: the slot's load ID is unchanged.
        routes = await self.slots_hold("after the release of A", runtime, slots, only)
        unbound = runtime.served_artifact(root_id, SCENARIO)
        assert unbound is None, f"the runtime still serves A after its release, as {unbound}"
        await self.not_served("the named route of A after its release", root)
        await self.answers("B alone", beside)
        cache = await self.still_cached("B alone", slots)
        self.run.report.passed(
            "a public update without A ends its route at once, and the serving worker releases A's component after "
            "the grace time, while B serves unchanged; the component keeps its engines",
            grace_s=grace,
            **release,
            routes={name: route.to_json() for name, route in routes.items()},
            cache_prompt_requests=cache,
            engine_stacks_started=stacks,
        )

        # A serves again. The first component is the runtime's only free slot: it must load A under a new runtime
        # load ID into the engines that it kept, with an empty prefix cache. The second slot expects no change.
        again = {root_id: 0, successor_id: 100}
        routes = await self.serve("serve A again beside B", again)
        second_load = first.bind(root, routes[root_id])
        assert second_load != successor_load, f"A's second load has B's runtime load ID, {second_load}"
        stops = self.pending_stops()
        assert not stops, f"the store holds stops after A serves again: {stops}"
        await self.slots_hold("A again beside B", runtime, slots, again)
        await self.selection("A again beside B", again, artifact_ids)
        reloaded = await self.cache_after_load("A after its second load", root, first.log)
        warm = await self.still_cached("B beside A's second load", [second])
        await self.answers("B beside A's second load", beside)
        for stream in (False, True):
            await self.explicit_request(
                f"A after its second load: {mode(stream)} named route", root, stream=stream, named=True
            )
        routes = await self.slots_hold("A again beside B, after their requests", runtime, slots, again)
        self.run.report.passed(
            "a public update serves A again beside B: the released component loads A under a new runtime load ID "
            "into the engines that it kept, with an empty prefix cache, and B serves unchanged",
            routes={name: route.to_json() for name, route in routes.items()},
            runtime_load_ids={"A": second_load, "A before its release": root_load, "B": successor_load},
            cache_prompt_requests={root.name: reloaded, **warm},
            engine_stacks_started=stacks,
        )

        # A third weight artifact is more than the two components serve. The fixture only has to be a third
        # artifact: it is a copy of B with a marker file, and nothing loads it.
        fixture = self.run.layout.models / "capacity"
        await asyncio.to_thread(write_capacity_fixture, successor, fixture)
        imported = await self.control.request(
            "import the capacity fixture", "POST", f"{path}/artifacts", {"source": fixture.as_uri()}, status=201
        )
        fixture_id = str(imported["artifact_id"])
        added = fixture_id not in artifact_ids and imported.get("component") == self.component
        assert added, f"the import of the capacity fixture: {imported}"
        fixture_arm = Arm("the capacity fixture", fixture_id, successor, native_facade=True)
        artifact_ids = [*artifact_ids, fixture_id]
        state = await self.selection("after the import of the capacity fixture", again, artifact_ids)
        assert state.routes == routes, f"the import changed the routes from {routes} to {state.routes}"
        await self.slots_hold("after the import of the capacity fixture", runtime, slots, again)
        capacity = runtime.served_artifact_capacity
        over = {root_id: 0, successor_id: 100, fixture_id: 0}
        before = await self.scenario_state("before the refused update")
        refused = await self.control.request(
            "serve three weight artifacts",
            "POST",
            f"{path}/update",
            {"served": over, "component": self.component},
            status=400,
        )
        after = await self.scenario_state("after the refused update")
        assert after == before, f"the refused update changed the scenario from {before} to {after}"
        refusal = str(refused["error"])
        explained = f"at most {capacity} artifacts" in refusal and all(artifact_id in refusal for artifact_id in over)
        assert explained, f"the refusal does not name the capacity of {capacity} and the three artifacts: {refusal!r}"
        stops = self.pending_stops()
        assert not stops, f"the store holds stops after the refused update: {stops}"
        await self.slots_hold("after the refused update", runtime, slots, again)
        await self.not_served("the named route of the capacity fixture", fixture_arm)
        await self.answers("B after the refused update", beside)
        for stream in (False, True):
            await self.explicit_request(
                f"A after the refused update: {mode(stream)} named route", root, stream=stream, named=True
            )
        cache = await self.still_cached("after the refused update", slots)
        await self.slots_hold("after the requests that follow the refused update", runtime, slots, again)
        self.run.report.passed(
            "a public update that names three weight artifacts is refused, as the runtime has two components: the "
            "scenario, its routes and both loads stay as they were",
            capacity=capacity,
            refusal=refusal,
            fixture={"artifact_id": fixture_id, "copy_of": beside.name, "marker": CAPACITY_MARKER},
            cache_prompt_requests=cache,
        )

        # The last step is a fault, and the profile's only change that is no public request: the run removes A's
        # engines. The watcher of A's runtime must restore A, and nothing may change for B or for the scenario.
        before = await self.selection("before the removal of A's engines", again, artifact_ids)
        routes = await self.slots_hold("before the removal of A's engines", runtime, slots, again)
        lost_stacks = first.stacks
        removal = await self.restored_after_removal(runtime, first, second)
        started = engine_urls(first.log)
        one_more = len(started) == len(lost_stacks) + 1 and started[:-1] == lost_stacks
        assert one_more, f"A's component had started the engine stacks {lost_stacks}, and now {started}: not one more"
        first.stacks = started
        # Both slots expect the runtime load IDs that they had: A's new engines must hold A's second load again.
        restored = await self.slots_hold("after the restoration of A", runtime, slots, again)
        assert restored == routes, f"the restoration of A changed the routes from {routes} to {restored}"
        after = await self.selection("after the restoration of A", again, artifact_ids)
        assert after == before, f"the restoration of A changed the scenario from {before} to {after}"
        stops = self.pending_stops()
        assert not stops, f"the store holds stops after the restoration of A: {stops}"
        # A's requests since the removal carried the plain prompt: its new engines must reuse nothing at first.
        reloaded = await self.cache_after_load("A after its restoration", root, first.log)
        warm = await self.still_cached("B after the restoration of A", [second])
        for arm in arms:
            for stream in (False, True):
                await self.explicit_request(
                    f"after the restoration of A: {mode(stream)} named route of {arm.name}",
                    arm,
                    stream=stream,
                    named=True,
                )
        await self.slots_hold("after the requests that follow the restoration of A", runtime, slots, again)
        self.run.report.passed(
            "the watcher of A's runtime restores A's removed engines: A serves again under its runtime load ID from "
            "one new engine stack with an empty prefix cache, and B served unchanged during the outage that the "
            "run saw",
            removal=removal,
            routes={name: route.to_json() for name, route in restored.items()},
            runtime_load_ids={root.name: first.load_id, beside.name: second.load_id},
            engine_stacks_started={"A before the removal": lost_stacks, "A": started, "B": second.stacks},
            cache_prompt_requests={root.name: reloaded, **warm},
        )
        return artifact_ids

    async def harness_artifacts(self, provider_url: str, provider: Checkpoint, successor_tree: Path) -> list[str]:
        """The public lifecycle of harness artifacts that serve at the same time: create the scenario with the root
        harness A, import B from the directory ``successor_tree``, serve both in a 50/50 split, give B all the
        traffic, and stop serving A. Return the IDs of A and B.

        A harness artifact loads no weights: the model of ``provider`` at ``provider_url`` answers for both, and
        the artifact ID of each answer and record tells them apart.
        """
        path = f"/reef/scenarios/{SCENARIO}"
        created = await self.control.request(
            "create the harness scenario", "POST", "/reef/scenarios", {"name": SCENARIO}, status=201
        )
        root_id = str(created["artifact_id"])
        root = Arm("harness A", root_id, provider, native_facade=False)
        alone = {root_id: 100}
        routes = await self.wait_for_routes("create the harness scenario", alone)
        assert routes == endpoint_routes(provider_url, alone), f"the routes of harness A alone are {routes}"
        await self.selection("after the creation of harness A", alone, [root_id])
        await self.answers("harness A after the creation", root)
        self.run.report.passed(
            "a public scenario creation serves the root harness A on the provider's model",
            route=routes[ACTIVE].to_json(),
        )

        imported = await self.control.request(
            "import harness B", "POST", f"{path}/artifacts", {"source": successor_tree.as_uri()}, status=201
        )
        successor_id = str(imported["artifact_id"])
        assert successor_id != root_id and imported.get("component") == self.component, f"the import: {imported}"
        successor = Arm("harness B", successor_id, provider, native_facade=False)
        artifact_ids = [root_id, successor_id]
        state = await self.selection("after the import of harness B", alone, artifact_ids)
        assert state.routes == routes, f"the import changed the routes from {routes} to {state.routes}"
        await self.not_served("the named route of harness B before it serves", successor)
        await self.answers("harness A after the import of harness B", root)
        self.run.report.passed(
            "a public import adds the harness B, and A stays the only served artifact", artifact_id=successor_id
        )

        halves = {root_id: 50, successor_id: 50}
        routes = await self.serve("serve harness A and B 50/50", halves)
        assert routes == endpoint_routes(provider_url, halves), f"the routes of the 50/50 split are {routes}"
        await self.selection("at 50/50", halves, artifact_ids)
        for stream in (False, True):
            # Both named routes answer at the same time.
            label = f"50/50 {mode(stream)} named route"
            await asyncio.gather(
                self.explicit_request(f"{label} of harness A", root, stream=stream, named=True),
                self.explicit_request(f"{label} of harness B", successor, stream=stream, named=True),
            )
        conversations = await self.split_conversations("50/50", (root, successor), halves)
        self.run.report.passed(
            "a public update serves harness A and B at the same time: named routes, and a 50/50 split that keeps "
            "conversations",
            routes={name: route.to_json() for name, route in routes.items()},
            conversation_buckets={name: bucket for name, (_, bucket) in conversations.items()},
        )

        shifted = {root_id: 0, successor_id: 100}
        routes = await self.serve("give harness B all the traffic", shifted)
        assert routes == endpoint_routes(provider_url, shifted), f"the routes at 0/100 are {routes}"
        await self.selection("at 0/100", shifted, artifact_ids)
        # One weight above 0 is no split: B answers the conversation that was A's, and its record stores no traffic.
        text, _ = conversations[root.name]
        await self.chat(
            "0/100 conversation that the split gave to harness A", successor, chat_body(text, stream=False)
        )
        await self.explicit_request("0/100 named route of harness A", root, stream=True, named=True)
        self.run.report.passed(
            "a public update to 0/100 gives harness B each request that names no artifact, and A answers by name",
            routes={name: route.to_json() for name, route in routes.items()},
        )

        only = {successor_id: 100}
        routes = await self.serve("stop serving harness A", only)
        assert routes == endpoint_routes(provider_url, only), f"the routes of harness B alone are {routes}"
        await self.selection("after harness A stopped", only, artifact_ids)
        await self.not_served("the named route of harness A after it stopped", root)
        await self.answers("harness B alone", successor)
        before = await self.scenario_state("before the same update again")
        await self.serve("the same update again", only)
        after = await self.scenario_state("after the same update again")
        assert after == before, f"the same update again changed the scenario from {before} to {after}"
        self.run.report.passed(
            "a public update without harness A stops its route and keeps it stored, and the same update again changes "
            "nothing",
            route=routes[ACTIVE].to_json(),
        )
        return artifact_ids

    # -- Records

    def stored_records(self, phase: Path) -> list[AgentRecord]:
        """Replay the scenario's records from the closed store under ``phase``, check them against the requests of
        these checks, and write them to ``records.jsonl`` under ``phase``; return them."""
        records = replay_records(phase / "records", SCENARIO)
        self.check_records(records)
        write_records(phase / "records.jsonl", records)
        return records


@asynccontextmanager
async def public_app(
    run: Run, phase: Path, recipe: Recipe, live: LiveObjects, component: str
) -> AsyncIterator[LifecycleChecks]:
    """Serve Reef's public app for ``recipe`` and its ``live`` objects during the block; yield the checks against it.

    A real dispatcher with its serving worker backs the app. Its coordination store is in memory; its records, its
    repository and its staged artifacts are under ``phase``, an existing directory. ``component`` is the component
    of ``recipe`` whose artifacts the checks serve. The end of the block closes the dispatcher, which shuts down
    the runtime of ``live`` and closes the record store.
    """
    # The dispatcher and the public app load only for the lifecycle profile.
    from reef.dispatcher import Dispatcher
    from reef.service.app import create_app

    coordination = memory_coordination()
    dispatcher = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(directory=phase / "repository"),
        live=live,
        scenario_storage=SQLiteScenarioStorage(phase / "records"),
        coordination=coordination,
        local_artifact_dir=phase / "staged",
    )
    try:
        app = create_app(dispatcher, tokens=run.credentials.service)
        async with (
            serving(app) as url,
            ClientSession(timeout=ClientTimeout(total=REQUEST_TIMEOUT_S)) as session,
        ):
            yield LifecycleChecks(
                run,
                ReefClient(session, url, run.credentials.service, run.layout.wire),
                coordination,
                ControlClient(session, url, run.credentials.service, run.layout.control),
                component,
            )
    finally:
        await asyncio.to_thread(dispatcher.close)


async def qualify_lifecycle(run: Run) -> None:
    # The recipes of the two scenarios load only for this profile.
    from tests.support.harness_jobs import with_harness_component
    from tests.support.policy_recipe import TestPolicyRecipe
    from tests.support.runtime_stubs import StubFacts

    from reef.evaluation.harness import EvaluationSettings
    from reef.harness.sync import HARNESS_COMPONENT
    from reef.harness.tree import HarnessTree
    from reef.recipe.harness import HarnessEvolutionRecipe
    from reef.train.harness_backend.settings import EvolutionSettings

    native, provider = run.checkpoints()
    assert revision_of(native) != revision_of(provider), "the checkpoints A and B have the same weights"
    runtime, slots = await start_checkpoint_slots(run)

    # Weight artifacts on the native engine, A and B at the same time. The dispatcher's serving worker alone opens
    # the runtime and loads a checkpoint into a component; after the engine removal, the watcher of A's runtime
    # loads A again.
    weight_phase = run.layout.root / "lifecycle-weights"
    weight_phase.mkdir()
    trainer = SeededWeights(native.directory)
    # The app runs every service, so its process holds the training component beside the facts of its backend.
    weight_live = LiveObjects(runtime=runtime, training_component=trainer, training_facts=StubFacts(trainer))
    async with public_app(run, weight_phase, TestPolicyRecipe(), weight_live, WEIGHTS_COMPONENT) as weight_checks:
        weight_ids = await weight_checks.native_weights(runtime, slots, native, provider)
    assert not trainer.submitted, f"the weight scenario submitted {len(trainer.submitted)} training jobs"
    run.report.passed("the weight scenario submitted no training job")
    weight_records = weight_checks.stored_records(weight_phase)

    # The first dispatcher shut the runtime down when it closed, which leaves the components with their engines.
    # Both components stop here, so that no native engine holds the GPU while the raw provider serves.
    stopped = await asyncio.to_thread(run.processes.stop_all)
    problems = [problem for outcome in stopped if (problem := OwnedProcess.problem(outcome))]
    run.report.facts["native_cleanup"] = {"processes": stopped, "problems": problems}
    names = sorted(str(outcome["name"]) for outcome in stopped)
    assert not problems and names == sorted(SLOT_COMPONENTS), f"the components did not stop: {names}, {problems}"
    run.runtime = None
    run.report.passed("both inference components stopped before the raw provider started", processes=names)

    # Harness artifacts on the raw provider, which starts only now. The proxy runtime of this scenario forwards to
    # the provider, and neither starts nor stops it.
    provider_url = await start_provider(run)
    harness_phase = run.layout.root / "lifecycle-harness"
    successor_tree = harness_phase / "successor"
    successor_tree.mkdir(parents=True)
    (successor_tree / "rules.md").write_text("Harness B.\n", encoding="utf-8")
    harness_recipe = HarnessEvolutionRecipe(
        evolution=EvolutionSettings(state_dir=str(harness_phase / "harness"), binary="/bin/false"),
        evaluation=EvaluationSettings(policy="none"),
        seed=HarnessTree({"rules.md": "Harness A.\n"}),
    )
    # The evaluation policy is none, so the harness training component takes no scorer.
    harness_live = with_harness_component(
        harness_recipe,
        NoTraining(),
        score=None,
        live=LiveObjects(runtime=InferenceProxyRuntime(base_url=provider_url, model_path=PROVIDER_MODEL)),
    )
    async with public_app(run, harness_phase, harness_recipe, harness_live, HARNESS_COMPONENT) as harness_checks:
        harness_ids = await harness_checks.harness_artifacts(provider_url, provider, successor_tree)
    harness_records = harness_checks.stored_records(harness_phase)

    weight_receipts = {record.agent_record_id for record in weight_records}
    shared = sorted(weight_receipts.intersection(record.agent_record_id for record in harness_records))
    assert not shared, f"the two scenarios stored records under the same IDs: {shared}"
    write_records(run.layout.records_file, [*weight_records, *harness_records])
    run.report.passed(
        "records match their requests exactly",
        weight_scenario={"artifacts": weight_ids, "records": len(weight_records)},
        harness_scenario={"artifacts": harness_ids, "records": len(harness_records)},
    )


# -- Routing profile -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutedExpectation:
    """The decision that one routed request must get: its target, reason, rule and selector."""

    target_id: str
    reason: str
    rule_name: str | None
    selector: str


class RoutingChecks:
    """The requests and checks of the routing profile, against one routed app.

    Values: ``snapshot`` is the routing policy; ``runtime_hash`` the router runtime that selects for it;
    ``checkpoints`` maps each target ID to the weights behind it; ``recorded`` lists every routed request with its
    expected decision; ``engine_fields`` lists, per request, the ``PRIVATE_FIELDS`` names that a target's own answer
    contained, which the run reports and does not judge.
    """

    def __init__(
        self,
        run: Run,
        client: ReefClient,
        snapshot: RoutingSnapshot,
        runtime_hash: str,
        checkpoints: Mapping[str, Checkpoint],
    ) -> None:
        self.run = run
        self.client = client
        self.snapshot = snapshot
        self.runtime_hash = runtime_hash
        self.checkpoints = dict(checkpoints)
        self.recorded: list[tuple[Exchange, RoutedExpectation]] = []
        self.engine_fields: dict[str, list[str]] = {}

    async def routed(self, label: str, expectation: RoutedExpectation, body: Mapping[str, Any]) -> Exchange:
        exchange = await self.client.post(label, "/v1/chat/completions", body, headers={SCENARIO_HEADER: SCENARIO})
        exchange.require_success()
        assert ARTIFACT_HEADER not in exchange.response_headers, f"{label}: a routed answer names a Reef artifact"
        exchange.receipt()
        checkpoint = self.checkpoints[expectation.target_id]
        check_words(checkpoint, exchange.text(), body["max_tokens"], label)
        check_logprobs(self.run.scores(), checkpoint, exchange, native_facade=False)
        found = private_fields_in(answer_values(exchange))
        if found:
            self.engine_fields[label] = found
        self.recorded.append((exchange, expectation))
        return exchange

    async def refused(
        self, label: str, body: Mapping[str, Any], status: int, code: str, *, headers: Mapping[str, str] | None = None
    ) -> None:
        exchange = await self.client.post(
            label, "/v1/chat/completions", body, headers={SCENARIO_HEADER: SCENARIO, **(headers or {})}
        )
        text = exchange.content.decode(errors="replace")
        assert exchange.status == status and text.startswith(f"{code}:"), f"{label}: HTTP {exchange.status}: {text!r}"
        assert RECORD_HEADER not in exchange.response_headers, f"{label}: a refused request has a receipt"

    async def run_all(self, native_target: str, provider_target: str) -> None:
        default = RoutedExpectation(native_target, "default", DEFAULT_RULE, AUTOMATIC_MODEL)
        keyword = RoutedExpectation(provider_target, "rule", ROUTING_RULE, AUTOMATIC_MODEL)
        native_pin = RoutedExpectation(native_target, "explicit_pin", None, native_target)
        provider_pin = RoutedExpectation(provider_target, "explicit_pin", None, provider_target)
        # Each routed case sends its expectation's selector as the request's model.
        routed_cases = (
            ("automatic default", default, PLAIN_PROMPT),
            ("automatic keyword rule", keyword, KEYWORD_PROMPT),
            ("explicit native pin over the keyword", native_pin, KEYWORD_PROMPT),
            ("explicit provider pin", provider_pin, PLAIN_PROMPT),
        )
        for stream in (False, True):
            for name, expectation, prompt in routed_cases:
                body = chat_body(prompt, stream=stream, model=expectation.selector)
                await self.routed(f"{name}, {mode(stream)}", expectation, body)
        self.run.report.passed("Semantic Router default, keyword rule and explicit pins, buffered and streamed")
        # Label, the request's model (None: no model), extra headers, then the refusal's status and code.
        refusals = (
            ("unknown target", "no-such-target", {}, 400, "unknown_target"),
            ("private router entrypoint", scenario_entrypoint(SCENARIO), {}, 400, "unknown_target"),
            ("malformed target", 7, {}, 400, "missing_model"),
            ("missing target", None, {}, 400, "missing_model"),
            ("Reef artifact pin", AUTOMATIC_MODEL, {ARTIFACT_HEADER: "artifact:any"}, 400, "release_pin"),
            ("unknown Scenario", AUTOMATIC_MODEL, {SCENARIO_HEADER: "not-routed"}, 403, "unknown_scenario"),
        )
        for label, model, headers, status, code in refusals:
            body = chat_body(PLAIN_PROMPT, stream=False, model=model)
            await self.refused(label, body, status, code, headers=headers)
        self.run.report.passed("unknown, private, malformed and missing targets and Reef pins are refused")

    def verify_records(self, records: Sequence[Any]) -> None:
        """Each routed request has one record with its decision, its provider request and its answer, and no Reef
        artifact; refused requests have none."""
        by_id = {record.agent_record_id: record for record in records}
        assert len(by_id) == len(records) == len(self.recorded), f"{len(records)} records for {len(self.recorded)}"
        for exchange, expectation in self.recorded:
            label = exchange.label
            record = by_id[exchange.receipt()]
            target = self.snapshot.targets[expectation.target_id]
            payload = record.payload
            assert record.artifact_id is None and "artifact_id" not in payload, f"{label}: a Reef artifact is stored"
            assert payload.get("model") == target.provider_model, f"{label}: provider model {payload.get('model')!r}"
            assert payload.get("messages") == exchange.body["messages"], f"{label}: stored messages"
            routing = payload["metadata"]["routing"]
            pinned = expectation.reason == "explicit_pin"
            wanted = {
                "target_id": target.target_id,
                "role": target.role,
                "provider_model": target.provider_model,
                "model_revision": target.revision,
                "identity_source": "deployment_manifest",
                "policy_id": "static",
                "policy_version": self.snapshot.version,
                "snapshot_version": self.snapshot.version,
                "router_config_hash": None if pinned else self.runtime_hash,
                "original_selector": expectation.selector,
                "reason": expectation.reason,
                "rule_name": expectation.rule_name,
                "attempt_index": 1,
                "prior_attempt_ids": [],
                "purpose": "serve",
                "execution_outcome": "completed",
                "delivery_outcome": "unconfirmed",
                "client_visible": True,
            }
            stored = {key: routing.get(key) for key in wanted}
            assert stored == wanted, f"{label}: stored routing {stored!r}"
            assert isinstance(routing.get("request_id"), str) and routing["request_id"], f"{label}: request ID"
            check_stored_response(payload["response"], exchange)
        self.run.report.passed("routed records keep their decisions and answers", records=len(records))
        self.run.report.facts["engine_answer_fields"] = {
            "note": "PRIVATE_FIELDS names in a target's own answer, which Reef relays unchanged; reported, not judged",
            "found": self.engine_fields,
        }


def routing_policy(run: Run, engines: Engines) -> tuple[dict[str, Any], str, str]:
    """The routing manifest's policy without its deployment section, and the native and provider target IDs."""
    native, provider = run.checkpoints()
    native_target = f"native-{run.inputs.native_engine}-component"
    provider_target = f"raw-{run.inputs.provider_engine}-provider"
    policy = {
        "schema_version": 2,
        "automatic_model": AUTOMATIC_MODEL,
        "targets": {
            native_target: {
                # The component's engine serves its model under the model path.
                "provider_model": str(native.directory),
                "revision": revision_of(native),
                "url": engines.component_url,
                "role": "reef-component",
                "api_key_env": COMPONENT_TOKEN_ENV,
            },
            provider_target: {
                "provider_model": PROVIDER_MODEL,
                "revision": revision_of(provider),
                "url": engines.provider_url,
                "role": "raw-provider",
            },
        },
        "scenarios": {
            SCENARIO: {
                "default_target": native_target,
                "targets": [native_target, provider_target],
                "rules": [{"name": ROUTING_RULE, "keywords": [ROUTING_KEYWORD], "target": provider_target}],
            }
        },
    }
    return policy, native_target, provider_target


async def wait_for_router(process: OwnedProcess, management: Any) -> Any:
    """Wait until the router is ready and serves the runtime it generated, as a replica checks it; return that
    ``RouterRuntime``. ``management`` is the router's ``RouterManagementClient``."""
    from reef.service.deploy.routing.policy import RouterPolicyError

    deadline = time.monotonic() + ROUTER_READY_TIMEOUT_S
    runtime = None
    while True:
        process.check_running()
        # The management listener refuses connections until the router has started.
        with suppress(RouterPolicyError):
            if await asyncio.to_thread(management.ready):
                runtime = await asyncio.to_thread(management.runtime_hashes)
                if (
                    runtime.activation_status == "active"
                    and runtime.active_runtime_hash == runtime.generated_runtime_hash
                    and is_router_config_hash(runtime.active_runtime_hash)
                ):
                    return runtime
        if time.monotonic() > deadline:
            raise TimeoutError(f"the router served no verified runtime in time ({runtime!r}); see {process.log}")
        await asyncio.sleep(0.5)


async def qualify_routing(run: Run) -> None:
    # The routed one-process app, its dispatcher and the router's deployment code load only for this profile.
    from reef.dispatcher import build_default_dispatcher
    from reef.service.app import create_app
    from reef.service.deploy.routing.policy import RouterManagementClient
    from reef.service.deploy.routing.renderer import (
        MANAGEMENT_TOKEN_ENV,
        SELECTION_TOKEN_ENV,
        router_configuration,
        write_private_document,
    )
    from reef.service.deploy.routing.settings import RoutingDeploymentSettings

    inputs = run.inputs
    binary = inputs.router_binary
    if binary is None:
        raise RuntimeError("the routing profile needs the router binary")
    binary_sha256 = file_sha256(binary)
    expected_sha256 = os.environ.get(ROUTER_SHA256_ENV, "")
    if expected_sha256 and binary_sha256 != expected_sha256:
        raise RuntimeError(f"the router binary has SHA-256 {binary_sha256}, not {expected_sha256}")
    native, provider = run.checkpoints()
    # A router target is a plain endpoint: Reef knows no artifact behind it, so nothing serves published weights.
    engines = await start_engines(run, artifact_serving=False)
    policy, native_target, provider_target = routing_policy(run, engines)
    canonical = json.dumps(policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    version = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    run.layout.router.mkdir()
    (run.layout.router / "policy.json").write_text(canonical + "\n", encoding="utf-8")
    # The manifest parsers check each target and Scenario; RoutingSnapshot.from_mapping would also require
    # deployment.semantic_router_image pinned by a repository digest, which a locally built router image lacks.
    snapshot = RoutingSnapshot(
        policy["automatic_model"],
        {target_id: parse_target(target_id, value) for target_id, value in policy["targets"].items()},
        {name: parse_scenario(name, value) for name, value in policy["scenarios"].items()},
        version,
    )
    router_port = free_port()
    settings = RoutingDeploymentSettings(lifecycle="external", sr_management_port=router_port)
    config_path = run.layout.router / "config.yaml"
    write_private_document(config_path, router_configuration(snapshot, settings))
    environment = {
        **os.environ,
        MANAGEMENT_TOKEN_ENV: run.credentials.management,
        SELECTION_TOKEN_ENV: run.credentials.selection,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    library = inputs.router_library_dir
    if library is not None:
        # Only the router process loads the router's libraries.
        environment["LD_LIBRARY_PATH"] = str(library)
        environment["ORT_DYLIB_PATH"] = str(library / ONNX_RUNTIME_LIBRARY)
    router = run.processes.start(
        "router",
        [str(binary), f"-config={config_path}", "-enable-api=true"],
        environment=environment,
        grace_s=15,
        cwd=run.layout.router,
    )
    management = RouterManagementClient(router_port, run.credentials.management)
    runtime = await wait_for_router(router, management)
    runtime_hash = str(runtime.active_runtime_hash)
    run.report.facts["router"] = {
        "binary": str(binary),
        "binary_sha256": binary_sha256,
        "expected_binary_sha256": expected_sha256 or None,
        "onnx_runtime_sha256": None if library is None else file_sha256(library / ONNX_RUNTIME_LIBRARY),
        "command": router.command,
        "policy_version": version,
        "rendered_config_sha256": file_sha256(config_path),
        "runtime": {
            "activation_status": runtime.activation_status,
            "generated_runtime_hash": runtime.generated_runtime_hash,
            "active_runtime_hash": runtime.active_runtime_hash,
        },
    }
    adapter = RoutingAdapter.from_snapshot(
        snapshot,
        router_config_hash=runtime_hash,
        credential_environment={COMPONENT_TOKEN_ENV: run.credentials.inference},
        timeout_s=INFERENCE_TIMEOUT_S,
    )
    dispatcher = build_default_dispatcher(
        local_artifact_dir=run.layout.dispatcher,
        scenario_storage=SQLiteScenarioStorage(run.layout.records),
        coordination=memory_coordination(),
    )
    try:
        app = create_app(
            dispatcher,
            tokens=run.credentials.service,
            routing=adapter,
            preview=PreviewSettings(
                f"http://127.0.0.1:{router_port}", run.credentials.selection, settings.selection_timeout_seconds
            ),
        )
        async with (
            serving(app) as url,
            ClientSession(timeout=ClientTimeout(total=REQUEST_TIMEOUT_S)) as session,
        ):
            # The app's startup checked that each target lists its model and that one preview per Scenario passes.
            run.report.passed("routing readiness")
            checks = RoutingChecks(
                run,
                ReefClient(session, url, run.credentials.service, run.layout.wire),
                snapshot,
                runtime_hash,
                {native_target: native, provider_target: provider},
            )
            await checks.run_all(native_target, provider_target)
        after = await asyncio.to_thread(management.runtime_hashes)
        assert after.active_runtime_hash == runtime_hash, f"the router runtime changed during the run: {after!r}"
    finally:
        await asyncio.to_thread(dispatcher.close)
    records = replay_records(run.layout.records, SCENARIO)
    write_records(run.layout.records_file, records)
    checks.verify_records(records)


# -- Run -----------------------------------------------------------------------------------------------------------


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("sglang", "vllm", "torch", "transformers", "tokenizers", "ray", "aiohttp", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def sglang_scheduler_sha256() -> str | None:
    """The SHA-256 of the SGLang scheduler that this interpreter's engines load, found without importing SGLang."""
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(next(iter(spec.submodule_search_locations))) / "srt" / "managers" / "scheduler.py"
    return file_sha256(path) if path.is_file() else None


def base_facts(inputs: QualificationInputs) -> dict[str, Any]:
    """The facts that a report starts with: the inputs, the versions, and what the profile does not show."""
    limitations = ["tiny synthetic checkpoints on one GPU; no performance or capacity claim"]
    if inputs.profile == "features":
        provider_engine: str | None = inputs.provider_engine
        limitations += [
            "the run writes served_models rows and traffic weights itself; control-plane publication is not exercised",
            "two prepublished routes are not a public multi-artifact deployment",
            "the checkpoints ignore their context, so a stale prefix-cache entry would change no output: the features "
            "profile shows the cache's reset by cached-token counts only",
        ]
    elif inputs.profile == "recovery":
        provider_engine = None
        limitations += [
            f"the runtime's watcher checks the engines every {WATCHER_INTERVAL_S:g} s; a deployment's runtime checks "
            f"them every {HEALTH_INTERVAL_S:g} s",
            "the run removes its engines with the component's control operation DELETE engines, as their owner: the "
            "watcher restores lost engines, which is not an engine actor or a server process that crashes, and not a "
            "component that restarts",
            "the run writes served_models rows itself, also one for the startup load, which Reef never publishes; "
            "control-plane publication and the public deployment lifecycle are not exercised",
            "no raw provider runs and no Reef service restarts: no split, and no claim about stored serving state",
            "the checkpoints ignore their context, so a stale prefix-cache entry would change no output: the recovery "
            "profile shows the new engines' empty cache by cached-token counts only",
        ]
    elif inputs.profile == "lifecycle":
        provider_engine = inputs.provider_engine
        limitations += [
            "the lifecycle profile changes what a scenario serves through the public API only, in two scenarios: "
            "weight artifacts on the native engine, then harness artifacts on the raw provider",
            "the weight scenario serves A and B at the same time on two inference components of the native engine. "
            "The run starts both as processes that share the one GPU on purpose, each with its own local Ray "
            "session and the engine's 0.2 memory fraction: no claim about GPU isolation, placement, capacity or "
            "performance",
            f"the serving worker may release A's component {CACHE_SECONDS + LIFECYCLE_INFERENCE_TIMEOUT_S:g} s after "
            f"the update without A: {CACHE_SECONDS:g} s plus the runtime's request timeout, which the run sets to "
            f"{LIFECYCLE_INFERENCE_TIMEOUT_S:g} s where a deployment's default is 300 s. No request of A is in "
            "flight when its route ends. The released component keeps its engines and their GPU memory: the run "
            "shows no freed memory",
            "the third weight artifact of the capacity check is a copy of B's checkpoint with a marker file: a third "
            "artifact, not a third model, and no engine loads it. The run shows that an update which names three "
            "weight artifacts is refused, and nothing about a runtime with more components",
            "the weight scenario has one engine fault, which the run controls: it removes A's engines once with the "
            "control operation DELETE engines of A's component, as their owner, while no request runs. The watcher "
            "of A's runtime restores them. That is not an engine actor or a server process that crashes, not a "
            "component that restarts, and not a fault of B's component or of both",
            "A's weights are also the model that new engines start with, so A's words do not tell A's restored load "
            "from the startup load of the new engines: the runtime load ID that A's component reports does",
            "both components run on the one GPU of the run. B's requests between A's refused requests show what B "
            "did during A's outage in this run: no general claim that a fault of one component leaves another "
            "alone. One restoration measures neither how long a recovery takes nor how reliably it succeeds: it is "
            "no durability benchmark",
            "the native components stop before the raw provider starts: the two scenarios never serve at the same "
            "time",
            "both harness artifacts answer from the raw provider's one model, B: the run shows the artifact IDs, "
            "routes, traffic split and records of two artifacts that serve at the same time, not two models that "
            "serve at the same time, and nothing executes the harness trees",
            "each scenario has its own dispatcher and serving worker in the test process, with an in-memory "
            "coordination store: no Reef service restarts, and no claim about stored serving state",
            "the run starts no training: the weight scenario must submit no training job, and nothing asks the "
            "harness scenario for an evolution step",
            "the checkpoints ignore their context, so a prefix-cache entry of other weights would change no output: "
            "the lifecycle profile shows each component's own cache by cached-token counts only",
        ]
    else:
        provider_engine = inputs.provider_engine
        limitations += [
            "the routing profile runs the router as its own subprocess: no managed lifecycle, container, policy "
            "update or SIGHUP, and no deployment-image validation",
            "the routing profile's native target is the component's engine with its base model and without request "
            "pinning: a component that serves published weights forwards no OpenAI chat request",
        ]
    return {
        "profile": inputs.profile,
        "native_engine": inputs.native_engine,
        "provider_engine": provider_engine,
        "output": str(inputs.output),
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "sglang_scheduler_sha256": sglang_scheduler_sha256(),
        "harness_sha256": {name: file_sha256(REPOSITORY / name) for name in HARNESS_FILES},
        "launcher": {name: os.environ[name] for name in LAUNCHER_ENV if os.environ.get(name)},
        "sampling": {"seed": SEED, "temperature": 1.0, "top_p": 1.0, "top_logprobs": TOP_ALTERNATIVES},
        "limitations": limitations,
    }


async def qualify(run: Run) -> None:
    """Run the profile, then shut down the runtime and stop every owned process, also after a failure.

    A cleanup that fails or leaves an owned process running fails the run; an earlier failure stays its error.
    """
    failure: BaseException | None = None
    try:
        native, provider = await asyncio.to_thread(write_checkpoints, run.layout.models)
        run.native, run.provider = native, provider
        run.report.facts["checkpoints"] = {
            checkpoint.name: {
                "words": list(checkpoint.words),
                "word_logits": list(WORD_LOGITS),
                "other_logit": OTHER_LOGIT,
                "files": directory_digests(checkpoint.directory),
            }
            for checkpoint in (native, provider)
        }
        run.oracle = await asyncio.to_thread(LogprobOracle, (native, provider))
        if run.inputs.profile == "features":
            await qualify_features(run)
        elif run.inputs.profile == "recovery":
            await qualify_recovery(run)
        elif run.inputs.profile == "lifecycle":
            await qualify_lifecycle(run)
        else:
            await qualify_routing(run)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        cleanup: dict[str, Any] = {}
        problems = []
        if run.runtime is not None:
            try:
                await asyncio.to_thread(run.runtime.shutdown)
            except Exception as exc:
                cleanup["runtime_shutdown_error"] = f"{type(exc).__name__}: {exc}"
                problems.append(f"the runtime did not shut down: {cleanup['runtime_shutdown_error']}")
        cleanup["processes"] = await asyncio.to_thread(run.processes.stop_all)
        problems.extend(problem for outcome in cleanup["processes"] if (problem := OwnedProcess.problem(outcome)))
        cleanup["problems"] = problems
        run.report.facts["cleanup"] = cleanup
        if problems and failure is None:
            raise RuntimeError("cleanup failed: " + "; ".join(problems))


def write_report(layout: RunLayout, report: QualificationReport) -> None:
    layout.report.write_text(json.dumps(report.document(), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def interrupt(signal_number: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt(f"signal {signal_number}")


def run_qualification(inputs: QualificationInputs) -> dict[str, Any]:
    """Run ``inputs`` and return its report document, which is also in ``report.json`` under the output directory.

    A failed check fails the run, and its traceback goes to ``error.txt``. The processes stop before the report is
    written, also after a failure or a termination signal; then every output file is checked for credentials.
    """
    layout = RunLayout(inputs.output)
    layout.create()
    report = QualificationReport()
    credentials = Credentials.generate()
    run = Run(inputs, layout, report, credentials, OwnedProcesses(layout.logs))
    report.facts.update(base_facts(inputs))
    started = time.monotonic()
    interrupted: BaseException | None = None
    # A terminated run still stops its processes: the signal becomes an exception that unwinds the run.
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        asyncio.run(qualify(run))
    except BaseException as exc:
        report.error = f"{type(exc).__name__}: {exc}"
        layout.error.write_text(traceback.format_exc(), encoding="utf-8")
        if not isinstance(exc, Exception):
            interrupted = exc
    finally:
        signal.signal(signal.SIGTERM, previous)
    report.facts["seconds"] = round(time.monotonic() - started, 1)
    write_report(layout, report)
    try:
        report.facts["files_checked_for_credentials"] = check_secrets_absent(layout.root, credentials.values())
    except AssertionError as exc:
        report.error = report.error or f"AssertionError: {exc}"
    write_report(layout, report)
    print(json.dumps({"passed": report.error is None, "error": report.error, "report": str(layout.report)}))
    if interrupted is not None:
        raise interrupted
    return report.document()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.smoke.native_inference",
        description="Qualify native engines, traffic splits, logprobs, engine recovery, the public deployment "
        "lifecycle or Semantic Router routing on one GPU.",
    )
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--native-engine", required=True, choices=ENGINES)
    parser.add_argument("--output", required=True, type=Path, help="a new or empty directory outside the repository")
    parser.add_argument("--router-binary", type=Path, help="the Semantic Router binary (routing profile)")
    parser.add_argument(
        "--router-library-dir", type=Path, help=f"the router's library directory with {ONNX_RUNTIME_LIBRARY}"
    )
    arguments = parser.parse_args(argv)
    try:
        inputs = QualificationInputs(
            arguments.profile,
            arguments.native_engine,
            arguments.output,
            arguments.router_binary,
            arguments.router_library_dir,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return 0 if run_qualification(inputs)["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
