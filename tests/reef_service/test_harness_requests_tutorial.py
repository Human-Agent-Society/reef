"""Guarantees of the tutorials/harness-requests demos, hermetic: the deployment builds in manual mode with the harness
requests defaults and the gate's selection set to always, the driver parses, the demo requests and the measurement's
fixed list pass admission's screens on POST /reef/train, the bugfix fixture fails its one test, and the README keeps
the shape the rows land in."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from reef_service.config_helpers import load_harness_deployment as load_config

from reef.dispatcher import training_request_refusal
from reef.harness.tree.nodes import directive_shaped, secret_shaped
from reef.recipe.cordis import CordisRecipe
from reef.service.deploy.settings import service_settings_from_config
from reef.train.evaluation.evaluators import BackendAlwaysSelectPlugin

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = REPO_ROOT / "tutorials" / "harness-requests"
METHOD_ROOT = REPO_ROOT / "tutorials" / "evolve-your-harness"
RUNS_HEADER = (
    "| Demo | Model | Run | Date | Code | Proposal | Verdict | W / L / T | Pending | Promoted | Requires "
    "| Show session | Proposer (s) | Ask to install (s) |"
)
RUNS_COLUMNS = 14
MEASURE_HEADER = (
    "| Run | Model | Date | Code | Parser | Requests | Answered | Admitted | Won | Published | Skipped | Median (s) |"
)
MEASURE_COLUMNS = 12
#: Additional retired concepts in the tutorial; check-doc-contracts.mjs checks simplified terminology repository-wide.
DROPPED_WORDS = (
    r"\blineage\b",
    r"\battribution\b",
    r"\bexecution runtime\b",
    r"\bTrack [AB]\b",
    r"\bscored records\b",
    r"\bnever stops\b",
    r"\bnever pauses\b",
)


@pytest.fixture
def run_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """tutorials/harness-requests/run.py as a module, loaded from its path: it is a script, not a package."""
    monkeypatch.setenv("REEF_UPSTREAM_MODEL", "provider/model-a")
    spec = importlib.util.spec_from_file_location("harness_requests_run", TUTORIAL / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("run.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clear_method_package() -> None:
    # Every tutorial's method package is named ``harness``: a sibling's cached import must not win.
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        sys.modules.pop(name)


def test_deployment_yaml_builds_the_recipe_with_the_requests_defaults_and_selection_always(monkeypatch) -> None:
    """The demos' deployment is the other tutorial's with its state moved, training_mode manual (one step per
    accepted instruction and no failure driven step between them) and the gate's selection set to always, so a
    workflow change that ties every task still publishes and a code_extension still waits."""
    from reef.recipe.registry import build_named_recipe
    from reef.service.assembly import _upstream_runtime

    monkeypatch.setenv("REEF_UPSTREAM_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("REEF_UPSTREAM_MODEL", "provider/model-a")
    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")
    monkeypatch.setenv("REEF_PYTHON", sys.executable)
    monkeypatch.setenv("PWD", str(REPO_ROOT))
    path = TUTORIAL / "configs" / "deployment.yaml"
    config = load_config(path)
    section = config["evolution"]
    assert section["requests"] is True and section["version_check"] is True
    assert section["review_kinds"] == ["code_extension"]
    assert section["selection"] == "always"
    assert config["data"]["training_mode"] == "manual"
    assert section["tasks"] == load_config(METHOD_ROOT / "configs" / "deployment.yaml")["evolution"]["tasks"]
    env = next(service for service in config["services"] if service["name"] == "reef")["env"]
    assert (REPO_ROOT / env["REEF_RECIPE_CONFIG_DIR"] / "deployment.yaml").resolve() == path.resolve()
    method_root = Path(env["PYTHONPATH"].split(":")[0])
    assert method_root.resolve() == METHOD_ROOT.resolve()
    assert (method_root / "harness" / "evolution.py").is_file()
    for key in ("agent_record_dir", "artifact_repository", "artifact_work_dir", "artifact_cache_dir"):
        assert config["reef"][key].startswith("tutorials/harness-requests/work/")
    assert config["run_dir"].startswith("tutorials/harness-requests/work/")
    assert section["step_record_dir"].startswith("tutorials/harness-requests/work/")
    # The proposal inbox defaults to .reef/proposals under the service's directory, outside the run's state.
    assert section["proposals_dir"].startswith("tutorials/harness-requests/work/")
    assert config["reef"]["port"] == 8901 and config["reef"]["token"] == "reef-local"

    _clear_method_package()
    monkeypatch.syspath_prepend(str(method_root))
    service = service_settings_from_config(config)
    built = build_named_recipe(
        "deployment",
        {**os.environ, "REEF_RECIPE_CONFIG_DIR": str(TUTORIAL / "configs")},
        default_runtime=_upstream_runtime(service),
    )
    assert isinstance(built, CordisRecipe) and built.adapter == "pi"
    assert built.review_kinds == ("code_extension",)
    # Manual mode needs a proposer that names requests, which the other tutorial's does; the build refuses otherwise.
    assert built.training_mode == "manual" and built.propose.reads_requests
    assert built.candidate_plugin is BackendAlwaysSelectPlugin
    assert built.model_binding().model == "provider/model-a"
    assert [entry["id"] for entry in built.seed] == [
        "answer-style",
        "reef-version-check",
        "reef-requests",
        "reef-pi-extension-api",
    ]
    _clear_method_package()


def test_run_py_help_names_the_modes() -> None:
    done = subprocess.run(
        [sys.executable, str(TUTORIAL / "run.py"), "--help"], capture_output=True, text=True, cwd=TUTORIAL
    )
    assert done.returncode == 0, done.stderr
    for mode in ("install", "bugfix", "research", "measure"):
        assert mode in done.stdout
    assert (
        "--n"
        in subprocess.run(
            [sys.executable, str(TUTORIAL / "run.py"), "measure", "--help"], capture_output=True, text=True
        ).stdout
    )


def test_run_sh_refuses_an_unknown_mode_before_touching_anything() -> None:
    for argv in ([], ["nope"]):
        done = subprocess.run(["bash", str(TUTORIAL / "run.sh"), *argv], capture_output=True, text=True)
        assert done.returncode == 2
        assert "usage: ./run.sh bugfix | research | measure" in done.stderr


def test_the_measurement_requests_pass_the_request_screens(run_module) -> None:
    """The fixed list becomes proposer input through POST /reef/train: none of it may trip admission's directive
    or credential screen, and the default --n has that many to file."""
    requests = run_module.MEASURE_REQUESTS
    assert len(requests) >= 10 and len(set(requests)) == len(requests)
    for text in requests:
        assert not directive_shaped(text), text
        assert not secret_shaped(text), text
        assert training_request_refusal(text) is None, text
        assert "\n" not in text and len(text) <= 4000


def test_the_demo_requests_are_the_fenced_text_and_pass_the_request_screens(run_module) -> None:
    for demo in ("bugfix", "research"):
        text = run_module.request_text(demo)
        assert text and "\n" not in text
        assert text in (TUTORIAL / "demos" / f"{demo}.md").read_text(encoding="utf-8")
        assert training_request_refusal(text) is None
        assert demo in run_module.SHOW_PROMPTS
    assert "reproduce it first with a failing test" in run_module.request_text("bugfix")
    assert "answer with citations" in run_module.request_text("research")


def test_the_driver_reads_verdicts_and_mutations_as_the_page_does(run_module) -> None:
    pending = {
        "pending": True,
        "metrics": {"selected": True, "mutation": {"op": "create", "id": "x", "options": {"name": "code_extension"}}},
    }
    rejected = {"metrics": {"selected": False, "wins": 0, "losses": 1, "ties": 2}}
    skipped = {"metrics": {"skipped": "no proposal"}}
    assert run_module.verdict_of(pending) == "pending"
    assert run_module.verdict_of(rejected) == "rejected"
    assert run_module.verdict_of(skipped) == "skipped: no proposal"
    assert run_module.verdict_of({"operation": "promote"}) == "promote"
    assert run_module.mutations_of(pending["metrics"]) == ["create x (code_extension)"]
    assert run_module.kinds_of({"mutations": [{"options": {"name": "rules"}}, {"options": {"name": "skill"}}]}) == [
        "rules",
        "skill",
    ]
    assert run_module.tally(rejected["metrics"]) == "0 / 1 / 2"
    assert run_module.tally(skipped["metrics"]) == "- / - / -"
    # selection: always records the per task scores of both sides and no counts; the counts come from those.
    always = {
        "selected": True,
        "selection": {
            "evaluation": {"metrics": {"candidate_scores": [1.0, 0.0, 1.0], "current_scores": [1.0, 1.0, 0.0]}}
        },
    }
    assert run_module.tally_parts(always) == (1, 1, 1)
    assert run_module.tally(always) == "1 / 1 / 1"
    # A failed episode scores None and ranks below every number, as reef's own tally ranks it.
    failed = {
        "selection": {
            "evaluation": {"metrics": {"candidate_scores": [1.0, None, 1.0], "current_scores": [1.0, 1.0, 0.0]}}
        }
    }
    assert run_module.tally_parts(failed) == (1, 1, 1)
    both = {"selection": {"evaluation": {"metrics": {"candidate_scores": [None], "current_scores": [None]}}}}
    assert run_module.tally_parts(both) == (0, 0, 1)
    assert run_module.tally_parts({"selection": {"evaluation": {"metrics": {"candidate_scores": [1.0]}}}}) is None
    turns = [
        {
            "response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "pytest -q"}'}}]
                        }
                    }
                ]
            }
        },
        {"response": {"choices": [{"message": {"content": "done", "tool_calls": []}}]}},
    ]
    assert run_module._tool_calls(turns) == ["bash(pytest -q)"]
    assert run_module._last_answer(turns) == "done"


def test_the_measurement_counts_won_and_published_apart(run_module) -> None:
    """Under selection: always a publish says nothing about the gate, so won is the recorded tally and
    published the verdict; a request that never got a step or a mutation counts as filed only."""
    results = [
        {"request": "a", "filed": False, "verdict": "the request was not filed"},
        {"request": "b", "filed": True, "verdict": "no step"},
        {"request": "c", "filed": True, "verdict": "skipped: no proposal"},
        {"request": "d", "filed": True, "kinds": "rules", "verdict": "selected", "wins": 1, "losses": 0, "ties": 2},
        {"request": "e", "filed": True, "kinds": "skill", "verdict": "selected", "wins": 0, "losses": 0, "ties": 3},
        {"request": "f", "filed": True, "kinds": "rules", "verdict": "rejected", "wins": 0, "losses": 1, "ties": 2},
        {
            "request": "g",
            "filed": True,
            "kinds": "code_extension",
            "verdict": "pending",
            "wins": 0,
            "losses": 0,
            "ties": 3,
        },
        # A refusal at admission or a skip after three failed steps carries no mutation: filed, not answered.
        {"request": "h", "filed": True, "kinds": "-", "verdict": "skipped: entry 'x' already exists"},
        {"request": "i", "filed": True, "kinds": "-", "verdict": "skipped: instruction failed"},
    ]
    assert run_module.totals_of(results) == {
        "filed": 8,
        "answered": 4,
        "admitted": 4,
        "won": 1,
        "published": 2,
        "pending": 1,
    }


def test_the_driver_reads_the_wrapper_as_it_prints_and_spools() -> None:
    """The acceptance line run.py parses and the spool name it reads are the wrapper's own."""
    wrapper = (REPO_ROOT / "reef" / "harness" / "client" / "wrapper.py").read_text(encoding="utf-8")
    driver = (TUTORIAL / "run.py").read_text(encoding="utf-8")
    assert "training request {record_id} accepted" in wrapper
    assert r"training request (\S+) accepted" in driver
    # The phrases of the request store era are gone from both sides: nothing files, nothing reports a batch.
    for phrase in ("the step runs with this batch", "filed (", "_rearm"):
        assert phrase not in wrapper and phrase not in driver, phrase
    assert '"training_request")' in driver and '"request")' not in driver
    assert "{time.time_ns():020d}" in wrapper and r"-(\d{20})-" in driver


def test_the_workspace_fixture_fails_exactly_one_test() -> None:
    workspace = TUTORIAL / "demos" / "workspace"
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", str(workspace)],
        cwd=workspace,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert re.search(r"\b1 failed\b", done.stdout), done.stdout
    assert "passed" not in done.stdout.splitlines()[-1]


def test_the_readme_keeps_the_runs_table_shape_and_the_docs_words() -> None:
    readme = (TUTORIAL / "README.md").read_text(encoding="utf-8")
    assert readme.isascii()
    assert RUNS_HEADER in readme
    assert "<!-- rows -->" in readme
    # Every row is a live run naming the pull request it ran on; the marker stays where the next row lands.
    separator = "|" + "---|" * RUNS_COLUMNS
    table = readme.split(f"{RUNS_HEADER}\n{separator}\n", 1)[1].split("<!-- rows -->", 1)[0]
    rows = [line for line in table.splitlines() if line]
    assert rows and all(line.startswith("| ") and line.count("|") == RUNS_COLUMNS + 1 for line in rows)
    assert all(re.search(r"\| #\d+ \|", line) for line in rows)
    measure_separator = "|" + "---|" * MEASURE_COLUMNS
    measured = readme.split(f"{MEASURE_HEADER}\n{measure_separator}\n", 1)[1].split("<!-- measure rows -->", 1)[0]
    measure_rows = [line for line in measured.splitlines() if line]
    assert measure_rows and all(line.count("|") == MEASURE_COLUMNS + 1 for line in measure_rows)
    assert all(re.search(r"\| #\d+ \|", line) for line in measure_rows)
    for pattern in DROPPED_WORDS:
        assert re.search(pattern, readme, re.IGNORECASE) is None, pattern
    for heading in (
        "## Directory layout",
        "## Quick start",
        "## What each demo does",
        "## The measurement",
        "## Environment",
        "## Runs",
        "## Reading",
        "## Reproduce",
        "## Notes",
        "## Known limitations",
    ):
        assert heading in readme
    for needle in ("./run.sh bugfix", "REEF_UPSTREAM_URL", "REEF_UPSTREAM_MODEL", "REEF_UPSTREAM_API_KEY"):
        assert needle in readme
    assert "selection: always" in readme and "RFC #308" in readme
    # The rows ran on the request store head; the README names that commit until rows from this path land.
    assert "training_mode: manual" in readme and "7e3982bb" in readme
    # One paragraph is one line: no hard wraps inside prose, fenced blocks aside.
    lines = readme.splitlines()
    fenced = False
    for index, line in enumerate(lines[:-1]):
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced or not line or line.startswith(("#", "|", "-", " ", "<!--")) or line[0].isdigit():
            continue
        following = lines[index + 1]
        assert following == "" or following.startswith(("#", "|", "-", "`", "<!--")), line[:60]


def test_the_tutorial_is_listed_beside_the_other() -> None:
    listing = (REPO_ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    assert "harness-requests/README.md" in listing
    for name in (
        "run.sh",
        "run.py",
        "pyproject.toml",
        "configs/deployment.yaml",
        "demos/bugfix.md",
        "demos/research.md",
    ):
        assert (TUTORIAL / name).is_file(), name
    assert os.access(TUTORIAL / "run.sh", os.X_OK)


def test_the_proposer_call_budget_follows_the_environment(monkeypatch) -> None:
    """run.sh exports REEF_PROPOSER_TIMEOUT_S because the method package's defaults (60 s for a failure step,
    120 s for a request) are short of what a local 26B model needs; unset, the defaults stand."""
    _clear_method_package()
    monkeypatch.syspath_prepend(str(METHOD_ROOT))
    from harness import evolution

    monkeypatch.delenv("REEF_PROPOSER_TIMEOUT_S", raising=False)
    assert evolution._timeout_s(120.0) == 120.0
    monkeypatch.setenv("REEF_PROPOSER_TIMEOUT_S", "900")
    assert evolution._timeout_s(120.0) == 900.0
    monkeypatch.setenv("REEF_PROPOSER_TIMEOUT_S", " ")
    assert evolution._timeout_s(60.0) == 60.0
    monkeypatch.delenv("REEF_PROPOSER_MAX_TOKENS", raising=False)
    assert evolution._max_tokens(4096) == 4096
    monkeypatch.setenv("REEF_PROPOSER_MAX_TOKENS", "16384")
    assert evolution._max_tokens(4096) == 16384
    monkeypatch.setenv("REEF_PROPOSER_MAX_TOKENS", "16k")
    assert evolution._max_tokens(4096) == 4096
    run_sh = (TUTORIAL / "run.sh").read_text(encoding="utf-8")
    assert 'REEF_PROPOSER_TIMEOUT_S="${REEF_PROPOSER_TIMEOUT_S:-900}"' in run_sh
    assert 'REEF_PROPOSER_MAX_TOKENS="${REEF_PROPOSER_MAX_TOKENS:-16384}"' in run_sh
    _clear_method_package()
