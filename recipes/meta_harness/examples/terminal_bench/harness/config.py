"""Pinned inputs shared by the Meta-Harness runner and deterministic tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

REEF_COMMIT = "6b0d9a1345191a0df1a7e324fa09875e8581a83f"
META_HARNESS_COMMIT = "95175f70c758dd1145b395edfe8b67e6f9d80fbd"
TERMINAL_BENCH_COMMIT = "1a6ffa9674b571da0ed040c470cb40c4d85f9b9b"
TERMINAL_BENCH_DATASET_URL = "https://github.com/laude-institute/terminal-bench-2.git"
TERMINAL_BENCH_DATASET_COMMIT = "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"
HARBOR_VERSION = "0.3.0"
TERMINAL_BENCH_VERSION = "0.2.18"
PI_VERSION = "0.84.2"
TARGET_MODEL = "gpt-5.4-mini-2026-03-17"
TARGET_MODEL_PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5.4-mini"
TARGET_MODEL_PRICING_OBSERVED_AT = "2026-08-30"
PROPOSER_MODEL = "gpt-5.5"
PROPOSER_REASONING_EFFORT = "high"
SEARCH_ROUNDS = 2
TRIALS_PER_TASK = 2

# Exact 30-task development subset shipped at the pinned Meta-Harness commit.
HARD_TASKS = (
    "bn-fit-modify",
    "cancel-async-tasks",
    "circuit-fibsqrt",
    "configure-git-webserver",
    "dna-assembly",
    "extract-moves-from-video",
    "feal-differential-cryptanalysis",
    "feal-linear-cryptanalysis",
    "fix-code-vulnerability",
    "fix-ocaml-gc",
    "gpt2-codegolf",
    "install-windows-3.11",
    "llm-inference-batching-scheduler",
    "make-doom-for-mips",
    "make-mips-interpreter",
    "mcmc-sampling-stan",
    "model-extraction-relu-logits",
    "password-recovery",
    "path-tracing",
    "path-tracing-reverse",
    "polyglot-rust-c",
    "protein-assembly",
    "regex-chess",
    "sam-cell-seg",
    "sparql-university",
    "torch-pipeline-parallelism",
    "torch-tensor-parallelism",
    "train-fasttext",
    "video-processing",
    "write-compressor",
)

# A fixed 10/10/10 partition of the pinned hard subset. The proposer receives
# train trajectories and aggregate dev scores; test ids are withheld until the
# search loop returns its fixed selection.
TRAIN_TASKS = HARD_TASKS[:10]
DEV_TASKS = HARD_TASKS[10:20]
TEST_TASKS = HARD_TASKS[20:]

# The live smoke keeps one task from each fixed split while preserving the
# reproduction's two rounds and two trials. This exercises later-round history,
# held-out sealing, and publication without silently changing search semantics.
SMOKE_TRAIN_TASKS = TRAIN_TASKS[:1]
SMOKE_DEV_TASKS = DEV_TASKS[:1]
SMOKE_TEST_TASKS = TEST_TASKS[:1]


@dataclass(frozen=True)
class TokenPricing:
    """Reproducible text-token pricing snapshot for one exact model."""

    model: str
    input_usd_per_million: float
    cached_input_usd_per_million: float
    output_usd_per_million: float
    source_url: str
    observed_at: str

    def __post_init__(self) -> None:
        if not self.model or not self.source_url or not self.observed_at:
            raise ValueError("token pricing requires model, source URL, and observation date")
        rates = (
            self.input_usd_per_million,
            self.cached_input_usd_per_million,
            self.output_usd_per_million,
        )
        if any(rate < 0 for rate in rates):
            raise ValueError("token prices must be non-negative")

    def estimate_usd(self, usage: Mapping[str, int]) -> float:
        counts = {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "cached_input_tokens": int(usage.get("cached_input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
        }
        if any(count < 0 for count in counts.values()):
            raise ValueError("token usage must be non-negative")
        return (
            counts["input_tokens"] * self.input_usd_per_million
            + counts["cached_input_tokens"] * self.cached_input_usd_per_million
            + counts["output_tokens"] * self.output_usd_per_million
        ) / 1_000_000

    def to_jsonable(self) -> dict[str, str | float]:
        return asdict(self)


TARGET_MODEL_PRICING = TokenPricing(
    model=TARGET_MODEL,
    input_usd_per_million=0.75,
    cached_input_usd_per_million=0.075,
    output_usd_per_million=4.50,
    source_url=TARGET_MODEL_PRICING_SOURCE,
    observed_at=TARGET_MODEL_PRICING_OBSERVED_AT,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """One auditable Meta-Harness reproduction configuration."""

    target_model: str = TARGET_MODEL
    proposer_model: str = PROPOSER_MODEL
    proposer_reasoning_effort: str = PROPOSER_REASONING_EFFORT
    rounds: int = SEARCH_ROUNDS
    trials_per_task: int = TRIALS_PER_TASK
    target_api_key_env: str = "OPENAI_API_KEY"
    target_base_url: str = "https://api.openai.com"

    def __post_init__(self) -> None:
        if not self.target_model or not self.proposer_model:
            raise ValueError("target and proposer model names must be non-empty")
        if self.proposer_reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            raise ValueError("unsupported proposer reasoning effort")
        if self.rounds < 2:
            raise ValueError("the reproduction requires at least two rounds")
        if self.trials_per_task < 2:
            raise ValueError("the reproduction requires at least two trials per task")
        if not self.target_base_url.startswith(("http://", "https://")):
            raise ValueError("target_base_url must be an HTTP(S) URL")
        if not self.target_api_key_env:
            raise ValueError("target_api_key_env must be non-empty")
        if self.target_model != TARGET_MODEL_PRICING.model:
            raise ValueError("target_model has no pinned token-pricing snapshot")
