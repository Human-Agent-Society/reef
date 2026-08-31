"""Paths, models, and campaign constants, ported from the sealed campaign.

The sealed source is benchmarks/skill_claw/config.py at commit 0519eefb
(branch bo/skillclaw-paper). Adaptations: paths root under this example's
./work directory, and the upstream endpoint and executor model come from the
REEF_UPSTREAM_URL / REEF_MODEL / REEF_UPSTREAM_API_KEY environment. Only the
driver (run.py, the deployment) consumes the endpoint and key: it hands them
to the recipe as its runtime, and the method reaches the model through the
binding reef passes to ``propose``.
"""

from __future__ import annotations

import os
from pathlib import Path

# The example directory (this file lives in its harness/ package): work/
# and the campaign state root under it. The frozen task list ships in this
# package (harness/tasks.json); harbor/ is the Harbor-format smoke task.
HERE = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent
WORKDIR = Path(os.environ.get("REEF_SC_WORKDIR", str(HERE / "work" / "campaign")))

# Where a campaign writes its rounds, and which of the two runs this is:
# "skillclaw" evolves the pool each night, "frozen" is the control.
RUN = os.environ.get("REEF_SC_RUN", "skillclaw")

# The benchmark's search tasks call the Brave API from inside the task
# container, and its gateway refuses to start without a key. Supplying the
# key restores those tasks; the placeholder keeps the other tasks alive.
BRAVE_KEYFILE = Path(os.environ.get("REEF_SC_BRAVE_KEYFILE", os.path.expanduser("~/.config/reef/brave.env")))
BRAVE_PLACEHOLDER = "placeholder-no-search"

# The executor and evolver model, and the OpenAI-compatible endpoint the driver
# proxies to (no /v1 suffix). run.sh repeats these two defaults because
# skillclaw.yaml interpolates them and a YAML cannot carry one; the pair here
# is what lets this module import without run.sh. serving/ overrides both.
AGENT_MODEL = os.environ.get("REEF_MODEL", "qwen/qwen3-max")
UPSTREAM_BASE = os.environ.get("REEF_UPSTREAM_URL", "https://openrouter.ai/api").rstrip("/")
# What the agent container presents to the embedded reef service. The service
# runs without authentication, so the value only has to be non-empty; it must
# never be the upstream key, which would land in the container's config.
GATEWAY_KEY = "reef-embedded"
# The benchmark's own in-container judge. The judge is measurement
# instrumentation, so its endpoint stays pinned to OpenRouter no matter
# where the model under test is served; only the explicit judge settings
# move it.
JUDGE_MODEL = os.environ.get("REEF_SC_JUDGE_MODEL", "openai/gpt-5.4")
OPENROUTER_BASE = os.environ.get("REEF_SC_JUDGE_BASE", "https://openrouter.ai/api/v1")

USERS = int(os.environ.get("REEF_SC_USERS", "8"))
# The setting keeps its published REEF_SC_ROUNDS name.
NIGHTS = int(os.environ.get("REEF_SC_ROUNDS", "6"))
SUCCESS_THRESHOLD = 0.5

# The run's own Reef service. Task containers reach it through Docker's host
# gateway. The night budget bounds waiting for background evolution.
REEF_SC_PORT = int(os.environ.get("REEF_SC_PORT", "8901"))
NIGHT_TIMEOUT_S = float(os.environ.get("REEF_SC_NIGHT_TIMEOUT_S", "10800"))
# A grader that returns no score reports this sentinel so the report still
# pairs and batches (0.0 would be a real failing grade); the night's judge
# backfills the real score.
UNSCORED_SENTINEL = -1.0

# Where the agent reads skill bodies inside a task container, and therefore
# the <location> root of every catalog entry; day.run_task copies the served
# pool there, the path their executor uses.
PUBLIC_SKILL_ROOT = "/root/skills"

# The paper's Day 1 column, used only by the advisory regime check.
PAPER_DAY_ONE = {
    "03_Social_Interaction": 54.01,
    "04_Search_Retrieval": 22.73,
    "05_Creative_Synthesis": 11.57,
    "06_Safety_Alignment": 24.00,
}
REGIME_TOLERANCE = 20.0


def _first_value(keyfile: Path) -> str | None:
    if not keyfile.exists():
        return None
    for line in keyfile.read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if "=" in line and not line.startswith("#"):
            return line.split("=", 1)[1].strip()
    return None


def read_key() -> str:
    """The upstream provider key, for the driver's own proxy backend only."""
    key = os.environ.get("REEF_UPSTREAM_API_KEY", "")
    if not key:
        raise RuntimeError("REEF_UPSTREAM_API_KEY is required (the upstream provider key)")
    return key


def read_judge_key() -> str:
    """The judge's OpenRouter key: REEF_SC_JUDGE_KEY, else the upstream key.

    The fallback matches the sealed campaign, which billed the judge and the
    model under test against the same OpenRouter key. A deployment whose
    upstream is not OpenRouter must set REEF_SC_JUDGE_KEY, or the in
    container judge cannot reach its pinned endpoint.
    """
    return os.environ.get("REEF_SC_JUDGE_KEY", "") or read_key()


def brave_key() -> str:
    return os.environ.get("BRAVE_API_KEY") or _first_value(BRAVE_KEYFILE) or BRAVE_PLACEHOLDER
