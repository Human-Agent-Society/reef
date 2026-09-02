"""The pins and experiment constants the runner and the tests share.

Everything that decides what a live run measures lives here, so a report can
name its inputs exactly and the deterministic tests can assert them. The
provider credential is deliberately absent: the runner reads it from the
environment at execution time and never writes it anywhere.
"""

from __future__ import annotations

# The upstream GEPA release whose AIME quickstart this reproduces (the v0.1.2
# tag, commit 92dadfff, which the retained record names) and the Pi release
# the Reef arms run. The runner refuses to start on any other version.
GEPA_VERSION = "0.1.2"
PI_VERSION = "0.84.2"

# Dated snapshots of the quickstart's task and reflection models, so the
# comparison cannot drift under an alias; sampling stays at provider defaults
# as upstream leaves it. The credential is named, not held.
TASK_MODEL = "gpt-4.1-mini-2025-04-14"
REFLECTION_MODEL = "gpt-5-2025-08-07"
OPENAI_BASE_URL = "https://api.openai.com"
API_KEY_ENV = "OPENAI_API_KEY"

# The quickstart's search budget and its default seed, plus two repetitions
# for robustness. One search cell is 150 metric calls; a full run adds two
# passes over the 150-example test split per cell.
SEARCH_BUDGET = 150
SEEDS = (0, 1, 2)

# Pi's pinned default output limit, made explicit in its models.json so the
# baseline gate can give the direct arm the same cap. The exact reference
# search leaves the cap unset, as upstream does.
PI_TASK_MAX_TOKENS = 16_384

# Evaluation concurrency only; none of these change sampling or search. The
# direct arm matches upstream's LiteLLM worker default, and --workers
# overrides all three for paired high-throughput runs.
REFERENCE_WORKERS = 10
REEF_WORKERS = 32
HELDOUT_WORKERS = 16

# The pre-search gate: each arm scores the seed prompt on ten repetitions of
# the 45 validation problems, and search starts only if the one-sided 95%
# lower bound (t at 44 degrees of freedom, one cluster per problem) shows Reef
# within ten points of the direct path. Powered to reject a persistent
# 13/45-versus-20/45 gap without treating one noisy vector as a failure.
BASELINE_REPETITIONS = 10
BASELINE_MARGIN = 0.10
BASELINE_CRITICAL_VALUE = 1.6802299765721167

# The upstream loader does not pin Hugging Face revisions; these do, and the
# split hash catches same-size content drift before any paid call.
AIME_TRAIN_REVISION = "13f9e12f613e720c2a2b2f345dd04b998a29494d"
AIME_TEST_REVISION = "c94da77eb22bbd6439e62a323bec18493a421302"
AIME_SPLIT_SIZES = {"train": 45, "validation": 45, "test": 150}
AIME_DATASET_SHA256 = "74e81306a9a1debadd64c49a4ab3588615f7bb698b695a59c17c65dd3b895185"
