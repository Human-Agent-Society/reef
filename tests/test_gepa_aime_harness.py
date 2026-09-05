"""Deterministic coverage for the GEPA AIME example.

Nothing here calls a model or launches Pi: the scorer and the feedback hook run
on synthetic episode results, and the driver's record lookup and
report run against a real embedded Reef service with a stubbed inference
backend, exactly as the campaign driver's suite does.
"""

from __future__ import annotations

import importlib
import json
import sys
import urllib.request
from pathlib import Path

import pytest

from reef.harness import EpisodeResult
from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "recipes" / "gepa" / "examples" / "aime"


def _purge_example_modules() -> None:
    # Every example names its package ``harness`` and its driver ``run``:
    # drop any cached import so this file's syspath entry wins, and drop
    # ours again afterwards so a sibling suite never inherits it.
    for name in [name for name in sys.modules if name in {"harness", "run"} or name.startswith("harness.")]:
        del sys.modules[name]


@pytest.fixture
def load(monkeypatch):
    """Import an example module fresh, and leave no trace for the next suite."""
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    _purge_example_modules()
    yield importlib.import_module
    _purge_example_modules()


@pytest.fixture
def aime(load):
    module = load("harness.aime")
    module.ANSWERS.clear()
    module.CONTEXTS.clear()
    return module


def pi_episode(answer: str, *, exit_code: int = 0, residue: tuple[str, ...] = ()) -> EpisodeResult:
    trajectory = ({"message": {"role": "assistant", "content": [{"type": "text", "text": answer}]}},)
    return EpisodeResult(exit_code, "", "", trajectory, residue)


# The scorer and the feedback hook ----------------------------------------


@pytest.mark.parametrize(
    ("response", "score"),
    [("work\n### 17", 1.0), ("### 17\nmore text", 1.0), ("The answer is 17", 0.0), ("### 16", 0.0)],
)
def test_scorer_matches_the_upstream_expected_string(aime, response, score):
    aime.register([{"input": "problem", "answer": "### 17"}])

    assert aime.evaluate("problem", pi_episode(response)) == score


@pytest.mark.parametrize(("exit_code", "residue"), [(1, ()), (0, ("pi-agent/stray.json",))])
def test_a_dirty_episode_scores_zero_however_it_answered(aime, exit_code, residue):
    aime.register([{"input": "problem", "answer": "### 17"}])

    assert aime.evaluate("problem", pi_episode("### 17", exit_code=exit_code, residue=residue)) == 0.0


def test_an_unregistered_task_refuses_rather_than_scoring_zero(aime):
    with pytest.raises(RuntimeError, match="no AIME answer is registered"):
        aime.evaluate("never seen", pi_episode("### 17"))


@pytest.mark.parametrize("selector", ["role", "worker"])
def test_driver_preserves_executor_profiles(load, monkeypatch, tmp_path, selector):
    driver = load("run")
    config = driver.load_config(EXAMPLE_DIR / "gepa.yaml")
    config["evolution"]["gepa"]["archive"] = str(tmp_path / "archive")
    config["executors"] = {"cpu-pool": {"backend": "local"}}
    config["execution"] = {"evolution": "cpu-pool"}
    if selector == "role":
        config["evolution"].pop("worker_executor")
    else:
        config["evolution"]["worker_executor"] = "cpu-pool"
        config["execution"]["evolution"] = "uni"
    monkeypatch.setattr(driver, "load_config", lambda path: config)
    _, recipe = driver.load_recipe(["problem"], api_key="dummy")
    assert recipe.worker_executor.backend == "local"


def test_default_driver_pool_keeps_registered_answers_visible(load, monkeypatch, tmp_path):
    from reef.records import RecordStore

    driver = load("run")
    driver.aime.register([{"input": "problem", "answer": "### 17"}])
    config = driver.load_config(EXAMPLE_DIR / "gepa.yaml")
    assert config["evolution"]["worker_executor"] == "local"
    config["evolution"]["episode_workers"] = 2
    config["evolution"]["gepa"]["archive"] = str(tmp_path / "archive")
    monkeypatch.setattr(driver, "load_config", lambda path: config)
    monkeypatch.setattr("reef.train.cordis_backend.backend.run_episode", lambda *args, **kwargs: pi_episode("### 17"))
    _, recipe = driver.load_recipe(["problem"], api_key="dummy")
    with RecordStore() as records:
        trainer = recipe.build("test", records)
        try:
            backend = trainer.training_backend
            assert backend._worker_selection.settings.backend == "local"
            assert [row[0] for row in backend._evaluate_pairings([({}, "problem"), ({}, "problem")])] == [1.0, 1.0]
            driver.aime.register([{"input": "problem", "answer": "### 18"}])
            assert backend._evaluate_pairings([({}, "problem")])[0][0] == 0.0
        finally:
            trainer.close()


def test_feedback_reproduces_the_upstream_evaluator_wording(aime):
    aime.register(
        [
            {"input": "with context", "answer": "### 42", "additional_context": {"solution": "full solution"}},
            {"input": "no context", "answer": "### 42"},
        ]
    )

    assert aime.feedback("with context", "### 42", 1.0) == (
        "The generated response is correct. The response include the correct answer '### 42'"
    )
    assert aime.feedback("no context", "The answer is 41", 0.0) == (
        "The generated response is incorrect. The correct answer is '### 42'. "
        "Ensure that the correct answer is included in the response exactly as it is."
    )
    assert aime.feedback("with context", "The answer is 41", 0.0) == (
        "The generated response is incorrect. The correct answer is '### 42'. "
        "Ensure that the correct answer is included in the response exactly as it is."
        " Here is some additional context that might be helpful:\nsolution: full solution"
    )


# The pinned dataset -------------------------------------------------------


def test_dataset_loader_rejects_same_size_content_drift(aime, monkeypatch):
    datasets = pytest.importorskip("datasets")  # the pinned loader's only dependency; absent from a bare checkout

    def load_dataset(name, *args, **kwargs):
        if name == "AI-MO/aimo-validation-aime":
            return [{"problem": f"source-{index}", "solution": "solution", "answer": 1} for index in range(90)]
        return [{"problem": f"test-{index}", "answer": 1} for index in range(30)]

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)

    with pytest.raises(RuntimeError, match="content changed"):
        aime.load_aime_splits()


def test_the_pins_a_report_names_stay_exact(aime):
    assert aime.PI_VERSION == "0.84.2"
    assert aime.TASK_MODEL == "gpt-4.1-mini-2025-04-14"
    assert aime.REFLECTION_MODEL == "gpt-5-2025-08-07"
    assert aime.SEARCH_BUDGET == 150
    assert aime.PI_TASK_MAX_TOKENS == 16_384
    assert aime.AIME_SPLIT_SIZES == {"train": 45, "validation": 45, "test": 150}
    assert aime.AIME_DATASET_SHA256 == "74e81306a9a1debadd64c49a4ab3588615f7bb698b695a59c17c65dd3b895185"


# The example's own wiring -------------------------------------------------


def test_the_seed_composition_carries_the_quickstart_envelope():
    import yaml

    seed = yaml.safe_load((EXAMPLE_DIR / "gepa.yaml").read_text(encoding="utf-8"))["evolution"]["seed"]
    entries = {str(entry["id"]): entry for entry in seed}

    assert entries["rules"]["config"]["text"].startswith("You are a helpful assistant.")
    assert entries["quickstart-settings"]["config"] == {"target": "primary", "data": {"defaultTools": []}}
    extension = entries["reef-quickstart-system"]["config"]["code"]
    assert 'pi.on("before_agent_start"' in extension
    assert 'pi.on("before_provider_request"' in extension
    # The extension reads the rules node instead of embedding its text, so the
    # envelope stays fixed while the evolving node underneath it changes.
    assert "AGENTS.md" in extension and "You are a helpful assistant" not in extension


def test_episode_files_merge_the_transient_binding_into_the_served_tree(load, monkeypatch, tmp_path):
    monkeypatch.setenv("REEF_WORK", str(tmp_path))
    driver = load("run")
    served = {"pi-agent/AGENTS.md": "rules\n", "pi-agent/settings.json": json.dumps({"defaultTools": []})}

    merged = driver.episode_files(served, ModelBinding("http://service.test", "task-model", "k"), get_adapter("pi"))

    assert merged["pi-agent/AGENTS.md"] == "rules\n"
    settings = json.loads(merged["pi-agent/settings.json"])
    # The served tree's own config survives the merge; the endpoint is added.
    assert settings["defaultTools"] == []
    assert settings["defaultModel"] == "reef/task-model"
    assert json.loads(merged["pi-agent/models.json"])["providers"]["reef"]["baseUrl"] == "http://service.test/v1"


# The driver against the embedded service ----------------------------------


def _stub_backend():
    from reef.runtime.inference import InferenceBackend

    class StubModel(InferenceBackend):
        async def inference(self, artifact, path, payload):
            del artifact, path, payload
            return {"choices": [{"message": {"role": "assistant", "content": "### 42"}}]}

    return StubModel()


def test_driver_reports_each_problem_against_its_recorded_request(load, monkeypatch, tmp_path):
    """The record path end to end on the real service: an episode's model call
    is recorded against the served composition, the driver finds it by problem
    text, and the report references it. Two of the three reports leave the
    batch open, so no training step runs and no Pi binary is needed."""
    from reef_client import ReefClient

    monkeypatch.setenv("REEF_WORK", str(tmp_path))
    monkeypatch.setenv("REEF_MODEL", "task-model")
    driver = load("run")
    problems = ["Find n such that ...", "Let ABC be a triangle ..."]
    recipe_name, recipe = driver.load_recipe(problems, api_key="dummy")
    bootstrap = tmp_path / "bootstrap"
    for relative, text in driver.seed_tree(recipe).items():
        (bootstrap / relative).parent.mkdir(parents=True, exist_ok=True)
        (bootstrap / relative).write_text(text, encoding="utf-8")
    service = driver.RunService(
        scenario="gepa-test",
        recipe_name=recipe_name,
        recipe=recipe,
        bootstrap_tree=bootstrap,
        run_dir=tmp_path / "run",
        upstream_url="http://127.0.0.1:9",
        upstream_key="dummy",
        port=0,
        inference_backend=_stub_backend(),
    )
    service.start()
    try:
        client = ReefClient(service.base_url, timeout_s=60.0)
        manifest = driver.pull_tree(client, service, tmp_path / "served")
        assert manifest["files"]["pi-agent/AGENTS.md"].startswith("You are a helpful assistant.")
        assert "providers" not in json.loads(manifest["files"]["pi-agent/models.json"])

        floor = driver.max_sequence(service.records(), service.scenario)
        references = []
        for problem in problems:
            # What a Pi episode does: an OpenAI-compatible call with no reef
            # headers, which the service stamps with this run's scenario.
            request = urllib.request.Request(
                f"{service.base_url}/v1/chat/completions",
                data=json.dumps({"model": "task-model", "messages": [{"role": "user", "content": problem}]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                assert json.load(response)["choices"][0]["message"]["content"] == "### 42"
            record = driver.find_task_record(service.records(), service.scenario, problem, after_sequence=floor)
            assert record is not None
            references.append(record.agent_record_id)

        assert len(set(references)) == len(problems)  # each problem found its own request
        for index, reference in enumerate(references):
            driver.report(
                service,
                client,
                {"agent_record_id": f"gepa-0-{index}", "score": 1.0, "references": [reference]},
            )
        # A replayed round re-reports the same id with different content: the
        # 409 means already reported, and the driver moves on.
        driver.report(service, client, {"agent_record_id": "gepa-0-0", "score": 0.0, "references": [references[0]]})
        assert service.training_step() == 0  # two of three reports: the batch is still open
    finally:
        service.stop()
