"""Guarantees of the recipes/beta/reefine/examples demos, hermetic: the deployment builds in manual mode, answering in
text, with the evaluation's floor over the one health task, the driver parses and installs a source tree locally, the
demo requests and the measurement's fixed list pass the request screens on POST /reef/train, the bugfix fixture fails
its one test, and the README examples have valid shell syntax and local links resolve."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from tests.support.config_helpers import load_harness_deployment as load_config

from recipes.beta import reefine
from recipes.beta.reefine import AnswerScorer, ReefineMethod, ReefineRecipe
from recipes.beta.reefine import answer as reefine_answer
from recipes.beta.reefine.recipe import SEED_DIRECTORY
from recipes.beta.reefine.scorer import grade_text
from reef.dispatcher.ingest import training_request_refusal
from reef.evaluation.activation import AlwaysActivate
from reef.harness.tree import HarnessTree
from reef.harness.tree.checks import directive_shaped, secret_shaped
from reef.service.deploy.service_config import service_config_from_mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
TUTORIAL = REPO_ROOT / "recipes" / "beta" / "reefine" / "examples"
#: The one evaluation task, the profile's: the floor checks that the tree still runs a shell command and answers.
HEALTH_TASK = (
    "[health] Run the shell command `echo reef-ok` with your shell tool and reply with its exact output as a "
    "plain word alone on the last line."
)
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
    """recipes/beta/reefine/examples/run.py as a module, loaded from its path: it is a script, not a package."""
    monkeypatch.setenv("REEF_UPSTREAM_MODEL", "provider/model-a")
    spec = importlib.util.spec_from_file_location("harness_requests_run", TUTORIAL / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("run.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deployment_yaml_builds_the_recipe_answering_in_text_with_the_floor(monkeypatch) -> None:
    """The demos use the built-in recipe with tutorial-local state, no agent, training_mode manual (one step per
    accepted instruction and no failure driven step between them) and the evaluation's floor over the one health task,
    so a change publishes when the tree still works and an extension still waits."""
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
    # A person activates a tree that changes an extension; every other passing tree serves at once.
    assert config["activation"] == {"module": "always", "config": {"review_kinds": ["extension"]}}
    assert section["agent"] is None and section["evaluation"]["policy"] == "floor"
    assert config["data"]["training_mode"] == "manual"
    # The health task alone, the profile's own: the floor checks that the tree still works, not the change.
    assert section["evaluation"]["tasks"] == [HEALTH_TASK]
    serve = load_config(TUTORIAL / "configs" / "serve.yaml")["evolution"]
    assert serve["evaluation"]["tasks"] == [HEALTH_TASK]
    # The serve config's agent works in a step's directory, and pi reads every AGENTS.md above it: not this checkout's.
    assert not Path(serve["state_dir"]).expanduser().resolve().is_relative_to(REPO_ROOT)
    assert grade_text(HEALTH_TASK, "$ echo reef-ok\nreef-ok") == 1.0
    assert [service["name"] for service in config["services"]] == ["reef"]
    module_name = section["evaluation"]["scorer"].partition(":")[0]
    spec = importlib.util.find_spec(module_name)
    assert spec is not None
    # The scorer belongs to the active Reef installation, including a wheel outside the checkout.
    assert Path(spec.origin).resolve() == Path(reefine.__file__).with_name("scorer.py").resolve()
    for key in ("agent_record_dir", "artifact_repository", "artifact_work_dir", "artifact_cache_dir"):
        assert config["reef"][key].startswith("recipes/beta/reefine/examples/work/")
    assert config["run_dir"].startswith("recipes/beta/reefine/examples/work/")
    # The step workspaces, the history and the proposal inbox stay with the run's state.
    assert section["state_dir"].startswith("recipes/beta/reefine/examples/work/")
    assert config["reef"]["port"] == 8901 and config["reef"]["token"] == "reef-local"

    service = service_config_from_mapping(config)
    built = build_named_recipe(
        "deployment",
        {**os.environ, "REEF_RECIPE_CONFIG_DIR": str(TUTORIAL / "configs")},
        default_runtime=_upstream_runtime(service),
    )
    assert isinstance(built, ReefineRecipe) and built.evolution.adapter == "pi"
    rule = built.activation_rule()
    assert isinstance(rule, AlwaysActivate) and rule.review_kinds == {"extension"}
    # No agent: the method answers each request in text.
    assert isinstance(built.method, ReefineMethod) and built.agent is None and built.training_mode == "manual"
    assert (built.evaluation.policy, built.evaluation.floor_score) == ("floor", 1.0)
    assert isinstance(built.scorer, AnswerScorer)
    assert built.model_binding().model == "provider/model-a"
    # No seed in the deployment: the first tree is the recipe's own, the starter skill.
    assert built.seed == HarnessTree.read(SEED_DIRECTORY)


def test_run_py_help_names_the_modes() -> None:
    # run.sh puts the repository on PYTHONPATH so run.py can import the recipe; do the same here.
    environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    done = subprocess.run(
        [sys.executable, str(TUTORIAL / "run.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=TUTORIAL,
        env=environment,
    )
    assert done.returncode == 0, done.stderr
    for mode in ("install", "bugfix", "research", "measure"):
        assert mode in done.stdout
    measure = subprocess.run(
        [sys.executable, str(TUTORIAL / "run.py"), "measure", "--help"],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert "--n" in measure.stdout, measure.stderr


def test_run_sh_refuses_an_unknown_mode_before_touching_anything() -> None:
    for argv in ([], ["nope"]):
        done = subprocess.run(["bash", str(TUTORIAL / "run.sh"), *argv], capture_output=True, text=True)
        assert done.returncode == 2
        assert "usage: ./run.sh bugfix | research | measure" in done.stderr


def test_the_measurement_requests_pass_the_request_screens(run_module) -> None:
    """The fixed list becomes the method's input through POST /reef/train: none of it may trip the directive or
    credential screen, and the default --n has that many to file."""
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


def test_the_driver_reads_results_and_patches_as_the_page_does(run_module) -> None:
    patch = {"rules.md": None, "extensions/speak.ts": "export default function (pi) {}", "notes.txt": "x"}
    parts = {"rules.md": "rules", "extensions/speak.ts": "extension"}
    base = {"release_id": "rel-1", "artifact_id": "rel-1", "active": False, "activated": True, "checkpoint": False}
    inactive = {
        **base,
        "checkpoint": True,
        "activated": False,
        "metrics": {"selected": True, "patch": patch, "patch_parts": parts},
    }
    rejected = {**base, "metrics": {"selected": False, "wins": 0, "losses": 1, "ties": 2}}
    skipped = {**base, "metrics": {"skipped": "the method left the head tree unchanged"}}
    assert run_module.result_of(inactive) == "inactive"
    assert run_module.result_of({**inactive, "active": True, "activated": True}) == "selected"
    assert run_module.result_of(rejected) == "rejected"
    assert run_module.result_of(skipped) == "skipped"
    assert run_module.result_of({**base, "checkpoint": True, "activated": True, "operation": "creation"}) == "creation"
    assert run_module.changes_of(inactive["metrics"]) == [
        "write extensions/speak.ts",
        "write notes.txt",
        "remove rules.md",
    ]
    # A path that no part takes has no part: the page shows it as "?".
    assert run_module.parts_of(inactive["metrics"]) == ["?", "extension", "rules"]
    assert run_module.parts_of(rejected["metrics"]) == []
    assert run_module.tally(rejected["metrics"]) == "0 / 1 / 2"
    assert run_module.tally(skipped["metrics"]) == "-"
    # The floor records how many tasks the candidate passed and failed, and runs no current side.
    met = {"selected": True, "passed": 1, "failed": 0, "floor_score": 1.0, "evaluation_sides": ["candidate"]}
    missed = {"selected": False, "passed": 0, "failed": 1, "floor_score": 1.0, "evaluation_sides": ["candidate"]}
    assert run_module.tally_parts(met) == (1, 0) and run_module.tally(met) == "1 / 0"
    assert run_module.tally_parts(missed) == (0, 1) and run_module.tally(missed) == "0 / 1"
    assert run_module.evaluation_counts(met) == {"passed": 1, "failed": 0}
    assert run_module.evaluation_counts(rejected["metrics"]) == {"wins": 0, "losses": 1, "ties": 2}
    assert run_module.evaluation_counts(skipped["metrics"]) == {}


def test_the_measurement_counts_won_and_published_apart(run_module) -> None:
    """Won is the evaluation's own count: met the floor on a row that carries passed and failed, more wins than losses
    on an older row, where under selection: always a publish said nothing about the evaluation; published is the
    result. A request that never got a step or a patch counts as filed only."""
    results = [
        {"request": "a", "filed": False, "result": "the request was not filed"},
        {"request": "b", "filed": True, "result": "no step"},
        {"request": "c", "filed": True, "result": "skipped: the method left the head tree unchanged"},
        {"request": "d", "filed": True, "parts": "rules", "result": "selected", "wins": 1, "losses": 0, "ties": 2},
        {"request": "e", "filed": True, "parts": "skill", "result": "selected", "wins": 0, "losses": 0, "ties": 3},
        {"request": "f", "filed": True, "parts": "rules", "result": "rejected", "wins": 0, "losses": 1, "ties": 2},
        {
            "request": "g",
            "filed": True,
            "parts": "extension",
            "result": "inactive",
            "wins": 0,
            "losses": 0,
            "ties": 3,
        },
        # A refused patch or a skip after three failed steps carries no patch: filed, not answered.
        {"request": "h", "filed": True, "parts": "-", "result": "skipped: the patch changes nothing"},
        {"request": "i", "filed": True, "parts": "-", "result": "skipped: instruction failed"},
        # Under the floor a row carries passed and failed: met the floor is won, missed is evaluated and lost.
        {"request": "j", "filed": True, "parts": "rules", "result": "selected", "passed": 1, "failed": 0},
        {"request": "k", "filed": True, "parts": "skill", "result": "rejected", "passed": 0, "failed": 1},
        {"request": "l", "filed": True, "parts": "extension", "result": "inactive", "passed": 1, "failed": 0},
    ]
    assert run_module.totals_of(results) == {
        "filed": 11,
        "answered": 7,
        "admitted": 7,
        "won": 3,
        "published": 3,
        "inactive": 2,
    }


def test_install_renders_the_source_tree_for_pi_with_a_binding_at_this_reef(run_module, tmp_path, monkeypatch) -> None:
    """The demo is the client: it pulls the source tree, checks what the release requires on this machine, and
    renders the tree locally with its own model binding; a release that requires what the machine lacks stops it."""
    manifest: dict = {"release_id": "r1", "files": {"rules.md": "Answer briefly.\n"}, "requires": []}
    paths: list[str] = []

    class Client:
        def get(self, path, extra_headers=None):
            paths.append(path)
            return manifest

    monkeypatch.setattr(run_module, "_client", Client)
    monkeypatch.setattr(run_module, "_ensure_scenario", lambda client: None)
    monkeypatch.setattr(run_module, "INSTALL_ROOT", tmp_path / "harness")
    assert run_module.install() == "r1" and paths == ["/reef/harness?form=source"]
    rendered = tmp_path / "harness" / "pi-agent"
    assert "Answer briefly." in (rendered / "AGENTS.md").read_text()
    models = json.loads((rendered / "models.json").read_text())
    assert run_module.SERVICE_URL in models["providers"]["reef"]["baseUrl"]
    assert models["providers"]["reef"]["headers"] == {"x-reef-scenario": run_module.SCENARIO}
    record = json.loads((tmp_path / "harness" / run_module.RELEASE_FILE).read_text())
    assert record["release_id"] == "r1" and "pi-agent/AGENTS.md" in record["files"]
    monkeypatch.delenv("REEF_DEMO_PHONE", raising=False)
    manifest["requires"] = [{"name": "REEF_DEMO_PHONE", "kind": "env", "prompt": "The phone number to text"}]
    assert run_module.install("r2") is None and paths[-1] == "/reef/harness?form=source&release_id=r2"


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


def test_readme_shell_examples_have_valid_syntax() -> None:
    readme = (TUTORIAL / "README.md").read_text(encoding="utf-8")
    commands = re.findall(r"```bash\n(.*?)```", readme, re.DOTALL)
    assert commands
    for command in commands:
        done = subprocess.run(["bash", "-n"], input=command, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr


def test_readme_local_links_and_terminology() -> None:
    readme = (TUTORIAL / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)]+)\)", readme)
    assert links
    for link in links:
        if link.startswith(("https://", "http://", "#")):
            continue
        assert (TUTORIAL / link.partition("#")[0]).exists(), link
    for pattern in DROPPED_WORDS:
        assert re.search(pattern, readme, re.IGNORECASE) is None, pattern


def test_the_example_is_listed_with_the_beta_recipes() -> None:
    listing = (REPO_ROOT / "recipes" / "README.md").read_text(encoding="utf-8")
    assert "beta/reefine/README.md" in listing
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


def test_the_model_call_budget_follows_the_environment(monkeypatch) -> None:
    """run.sh exports REEF_REEFINE_CALL_TIMEOUT_S because the method package's defaults (60 s for a failure step,
    120 s for a request) are short of what a local 26B model needs, and pins REEF_REEFINE_CALL_MAX_TOKENS at the
    package's request default of 16384, the budget a thinking model needs; unset, the defaults stand."""
    monkeypatch.delenv("REEF_REEFINE_CALL_TIMEOUT_S", raising=False)
    assert reefine_answer.timeout_s(120.0) == 120.0
    monkeypatch.setenv("REEF_REEFINE_CALL_TIMEOUT_S", "900")
    assert reefine_answer.timeout_s(120.0) == 900.0
    monkeypatch.setenv("REEF_REEFINE_CALL_TIMEOUT_S", " ")
    assert reefine_answer.timeout_s(60.0) == 60.0
    monkeypatch.delenv("REEF_REEFINE_CALL_MAX_TOKENS", raising=False)
    assert reefine_answer.max_tokens(16384) == 16384
    monkeypatch.setenv("REEF_REEFINE_CALL_MAX_TOKENS", "32768")
    assert reefine_answer.max_tokens(16384) == 32768
    monkeypatch.setenv("REEF_REEFINE_CALL_MAX_TOKENS", "16k")
    assert reefine_answer.max_tokens(16384) == 16384
    run_sh = (TUTORIAL / "run.sh").read_text(encoding="utf-8")
    assert 'REEF_REEFINE_CALL_TIMEOUT_S="${REEF_REEFINE_CALL_TIMEOUT_S:-900}"' in run_sh
    assert 'REEF_REEFINE_CALL_MAX_TOKENS="${REEF_REEFINE_CALL_MAX_TOKENS:-16384}"' in run_sh
