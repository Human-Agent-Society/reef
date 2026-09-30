"""A fault decides its pair (#698): the evaluation types every episode without a score by whose failure it is,
reruns the pairs an infrastructure fault hit, and ``paired_confidence`` counts a pair still faulted as a candidate
loss, where the score comparison counts a failed current episode as a candidate win."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from reef_service.test_harness_recipe import MODEL, batch, evaluate, make_binary

from reef.harness.adapters import get_adapter
from reef.harness.episodes.executor import (
    EpisodeExecutor,
    EpisodeLaunchError,
    EpisodeTimeout,
    LocalExecutor,
    ProcessOutcome,
)
from reef.harness.episodes.run import EpisodeError, EpisodeResult, EpisodeTimeoutError
from reef.recipe import RecipeConfigError
from reef.recipe.cordis import CordisRecipe
from reef.train.cordis_backend import (
    CordisBackend,
    HarnessCandidate,
    Mutation,
    PairedConfidencePlugin,
    PairedConfidencePluginFactory,
    ScoreComparisonPlugin,
)
from reef.train.cordis_backend.backend import EpisodeEvaluationWorker, _ScoredEpisode
from reef.train.cordis_backend.manifest import FailureObservation
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer, verifier_reward
from reef.train.evaluation import EvaluationResult, PairedConfidenceSettings, UpdateCandidate

MARKER = Mutation("create", "r1", {"name": "rules", "config": {"text": "marker rules"}})


def paired_backend(tmp_path: Path) -> CordisBackend:
    return CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(lambda nodes, samples, models: MARKER),
        score_episode=resolve_episode_scorer(evaluate),
        tasks=("task one", "task two"),
        models=MODEL,
        binary=str(make_binary(tmp_path)),
        step_record_dir=tmp_path / "record",
    )


def scored(score: float) -> _ScoredEpisode:
    return _ScoredEpisode(score, None)


def faulted(task: str) -> _ScoredEpisode:
    failure = FailureObservation(task=task, stage="launch", cause="cannot reach the docker daemon")
    return _ScoredEpisode(None, failure, fault="infrastructure", label="execution_error")


class ScriptedPairings:
    """Stands in for the evaluation pool: each call answers with the next scripted runs and keeps what it ran."""

    def __init__(self, *answers: list[_ScoredEpisode]) -> None:
        self.answers = list(answers)
        self.calls: list[list[tuple[Mapping[str, str], str, Path | None]]] = []

    def __call__(self, pairings: Sequence[tuple[Mapping[str, str], str, Path | None]]) -> list[_ScoredEpisode]:
        self.calls.append(list(pairings))
        return self.answers.pop(0)


def prepared_candidate(backend: CordisBackend) -> HarnessCandidate:
    candidate = backend.prepare_step(batch(), backend.initial_state(), 0).candidate
    assert isinstance(candidate, HarnessCandidate) and candidate.record_dir is not None
    return candidate


# Pairing order: task one's candidate and current episodes, then task two's.
CURRENT_FAULTED_ON_TASK_ONE = [scored(1.0), faulted("task one"), scored(0.5), scored(0.5)]


def test_a_current_side_infrastructure_fault_is_a_candidate_win_only_under_the_score_comparison(
    tmp_path: Path, monkeypatch
) -> None:
    b = paired_backend(tmp_path)
    candidate = prepared_candidate(b)
    monkeypatch.setattr(b, "_evaluate_pairings", ScriptedPairings(CURRENT_FAULTED_ON_TASK_ONE))
    evaluation = b.evaluate(candidate)
    assert evaluation.metrics["current_scores"] == (None, 0.5)
    assert evaluation.metrics["candidate_labels"] == ("valid", "valid")
    assert evaluation.metrics["current_labels"] == ("execution_error", "valid")
    assert evaluation.metrics["candidate_faults"] == (None, None)
    assert evaluation.metrics["current_faults"] == ("infrastructure", None)
    # Without infra_reruns an evaluation row keeps its shape.
    assert "rerun_rounds" not in evaluation.metrics and "rerun_pairs" not in evaluation.metrics
    # Unchanged: the missing score ranks below every real one, so the candidate wins the pair.
    compared = ScoreComparisonPlugin(b).decide(candidate, evaluation)
    assert compared.selected and (compared.metrics["wins"], compared.metrics["losses"]) == (1, 0)
    paired = PairedConfidencePlugin(b, PairedConfidenceSettings()).decide(candidate, evaluation)
    assert not paired.selected
    assert (paired.metrics["valid_pairs"], paired.metrics["void_pairs"]) == (1, 1)
    assert (paired.metrics["wins"], paired.metrics["losses"], paired.metrics["ties"]) == (0, 1, 1)


def test_an_infrastructure_fault_reruns_both_sides_of_its_pair_in_directories_of_their_own(
    tmp_path: Path, monkeypatch
) -> None:
    b = paired_backend(tmp_path)
    candidate = prepared_candidate(b)
    script = ScriptedPairings(CURRENT_FAULTED_ON_TASK_ONE, [scored(1.0), scored(0.0)])
    monkeypatch.setattr(b, "_evaluate_pairings", script)
    plugin = PairedConfidencePlugin(b, PairedConfidenceSettings(), infra_reruns=1)
    evaluation = plugin.evaluate(candidate)
    assert evaluation.metrics["rerun_rounds"] == 1 and evaluation.metrics["rerun_pairs"] == 1
    assert evaluation.metrics["candidate_scores"] == (1.0, 0.5) and evaluation.metrics["current_scores"] == (0.0, 0.5)
    assert evaluation.metrics["current_faults"] == (None, None) and evaluation.metrics["episode_failures"] == 0
    first, rerun = script.calls
    episodes = candidate.record_dir / "episodes"
    assert [pairing[2] for pairing in first] == [
        episodes / name for name in ("candidate-0", "current-0", "candidate-1", "current-1")
    ]
    # Only the faulted pair runs again, both of its sides, under the same trees.
    assert [(pairing[1], pairing[2]) for pairing in rerun] == [
        ("task one", episodes / "candidate-0-rerun-1"),
        ("task one", episodes / "current-0-rerun-1"),
    ]
    assert (rerun[0][0], rerun[1][0]) == (first[0][0], first[1][0])
    decision = plugin.decide(candidate, evaluation)
    assert (decision.metrics["valid_pairs"], decision.metrics["void_pairs"]) == (2, 0)
    assert (decision.metrics["wins"], decision.metrics["ties"]) == (1, 1)


def test_a_pair_still_faulted_after_every_rerun_stays_void(tmp_path: Path, monkeypatch) -> None:
    b = paired_backend(tmp_path)
    candidate = prepared_candidate(b)
    script = ScriptedPairings(
        CURRENT_FAULTED_ON_TASK_ONE,
        [scored(1.0), faulted("task one")],
        [faulted("task one"), scored(0.0)],
    )
    monkeypatch.setattr(b, "_evaluate_pairings", script)
    evaluation = b.evaluate(candidate, infra_reruns=2)
    assert evaluation.metrics["rerun_rounds"] == 2 and evaluation.metrics["rerun_pairs"] == 2
    assert [pairing[2].name for pairing in script.calls[2]] == ["candidate-0-rerun-2", "current-0-rerun-2"]
    # The last runs are the result: the candidate side faulted this time.
    assert evaluation.metrics["candidate_faults"] == ("infrastructure", None)
    assert evaluation.metrics["current_faults"] == (None, None)
    decision = PairedConfidencePlugin(b, PairedConfidenceSettings()).decide(candidate, evaluation)
    assert decision.metrics["void_pairs"] == 1 and not decision.selected

    # A clean evaluation reruns nothing and still records the rounds it ran.
    clean = ScriptedPairings([scored(1.0), scored(0.0), scored(1.0), scored(0.0)])
    monkeypatch.setattr(b, "_evaluate_pairings", clean)
    evaluation = b.evaluate(candidate, infra_reruns=3)
    assert len(clean.calls) == 1
    assert (evaluation.metrics["rerun_rounds"], evaluation.metrics["rerun_pairs"]) == (0, 0)
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="infra_reruns must be an integer of at least 0"):
            b.evaluate(candidate, infra_reruns=bad)


def test_a_paired_step_settles_with_its_labels_in_the_selection_record(tmp_path: Path) -> None:
    """Real episodes of the fake harness: the step record keeps each episode's label and fault, and the commit
    row keeps the decision's counts while the vectors stay inside the selection record."""
    b = paired_backend(tmp_path)
    prepared = b.prepare_step(batch(), b.initial_state(), 0)
    assert prepared.candidate is not None
    plugin = PairedConfidencePluginFactory(infra_reruns=1).build(b)
    evaluation = plugin.evaluate(prepared.candidate)
    assert evaluation.metrics["candidate_scores"] == (1.0, 1.0) and evaluation.metrics["current_scores"] == (0.0, 0.0)
    assert evaluation.metrics["current_labels"] == ("valid", "valid")
    decision = plugin.decide(prepared.candidate, evaluation)
    # Two wins are too few for the sign test at 0.95.
    assert not decision.selected and decision.metrics["selection_result"] == "insufficient_confidence"
    result = b.settle_step(prepared, decision)
    assert not {"candidate_labels", "current_labels", "candidate_faults", "current_faults"} & set(result.metrics)
    assert result.metrics["rerun_rounds"] == 0 and result.metrics["void_pairs"] == 0
    assert result.metrics["selection"]["evaluation"]["metrics"]["candidate_faults"] == (None, None)
    json.loads(json.dumps(result.metrics, allow_nan=False))
    record = json.loads((prepared.candidate.record_dir / "episodes" / "current-1" / "episode.json").read_text())
    assert (record["label"], record["fault"], record["score"]) == ("valid", None, 0.0)


class RaisingExecutor(EpisodeExecutor):
    """An executor whose launch fails the way a host or a hung harness does."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def preflight(self) -> None:
        return None

    def launch(self, argv, *, root, workspace, env, timeout, writable_paths=(), readonly_paths=()) -> ProcessOutcome:
        raise self.error


@pytest.mark.parametrize(
    ("error", "fault"),
    [
        (EpisodeTimeout("episode timed out after 10s"), "harness"),
        (EpisodeLaunchError("harness binary 'pi' not found"), "infrastructure"),
    ],
)
def test_a_timeout_is_the_harnesss_fault_and_another_launch_failure_the_infrastructures(
    tmp_path: Path, error: Exception, fault: str
) -> None:
    assert issubclass(EpisodeTimeoutError, EpisodeError)
    worker = EpisodeEvaluationWorker(
        descriptor=get_adapter("pi"),
        scorer=resolve_episode_scorer(evaluate),
        binary=None,
        timeout=10,
        executor=RaisingExecutor(error),
        forbid_residue=False,
    )
    result = worker.run({}, "task one", tmp_path / "episode")
    assert result.score is None and result.fault == fault and result.label == "execution_error"
    # The stage stays launch, so the failure's fingerprint does not change.
    assert result.failure == FailureObservation(task="task one", stage="launch", cause=str(error))
    record = json.loads((tmp_path / "episode" / "episode.json").read_text())
    assert (record["label"], record["fault"]) == ("execution_error", fault)


def test_every_failure_stage_names_whose_fault_it_is() -> None:
    def worker(adapter: str, forbid_residue: bool = False) -> EpisodeEvaluationWorker:
        return EpisodeEvaluationWorker(
            descriptor=get_adapter(adapter),
            scorer=resolve_episode_scorer(verifier_reward if adapter == "terminus" else evaluate),
            binary=None,
            timeout=10,
            executor=LocalExecutor(),
            forbid_residue=forbid_residue,
        )

    def result(*events: dict, exit_code: int = 0, residue: tuple[str, ...] = ()) -> EpisodeResult:
        return EpisodeResult(exit_code=exit_code, stdout="", stderr="boom", trajectory=events, residue=residue)

    marker = {"type": "agent_end", "rules": "marker"}
    littered = worker("pi", forbid_residue=True)._score_result(result(marker, residue=("stray.txt",)), "task one")
    session = {"type": "session", "seq": 0, "time": 0, "data": {"agent": "root"}}
    error = {"kind": "error", "error": {"code": "LOAD_ERROR", "message": "tool 'x' cannot load"}}
    end = {"type": "turn/end", "seq": 1, "time": 0, "data": {"turn": 1, "reason": error}, "rules": "marker"}
    errored = worker("pi")._score_result(result(session, end, exit_code=1), "task one")
    row = {"type": "verifier", "task": "task one", "rewards": {}, "reward": None, "failed": True, "error": "no image"}
    never_ran = worker("terminus")._score_result(result(row, exit_code=1), "task one")
    exited = worker("pi")._score_result(result(marker, exit_code=1), "task one")
    outcomes = {
        episode.failure.stage: (episode.label, episode.fault)
        for episode in (littered, errored, never_ran, exited)
        if episode.failure is not None
    }
    assert outcomes == {
        "residue": ("execution_error", "harness"),
        "graph": ("execution_error", "harness"),
        "trial": ("execution_error", "infrastructure"),
        # A nonzero exit that still scored is a valid episode, as the tally reads it.
        "exit": ("valid", None),
    }
    assert exited.score == 1.0


def recipe_config(tmp_path: Path, **evolution: object) -> dict[str, object]:
    return {
        "evolution": {
            "propose": "demo_paired:propose",
            "evaluate": "demo_paired:evaluate",
            "tasks": ["t"],
            "binary": str(make_binary(tmp_path)),
            **evolution,
        }
    }


def test_recipe_builds_the_paired_confidence_selection_from_its_keys(tmp_path: Path, monkeypatch) -> None:
    module = tmp_path / "demo_paired.py"
    module.write_text(
        "def propose(nodes, samples, model):\n    return None\n\ndef evaluate(task, result):\n    return 0.0\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    def build(**evolution: object) -> CordisRecipe:
        return CordisRecipe.from_environment({}, config=recipe_config(tmp_path, **evolution))

    assert build(selection="paired_confidence").candidate_plugin == PairedConfidencePluginFactory()
    tuned = build(
        selection="paired_confidence",
        min_valid_pairs=20,
        min_effect=0.05,
        confidence_level=0.9,
        infra_reruns=2,
        recheck_every=3,
    )
    assert tuned.candidate_plugin == PairedConfidencePluginFactory(
        PairedConfidenceSettings(min_valid_pairs=20, min_effect=0.05, confidence_level=0.9), infra_reruns=2
    )
    assert tuned.recheck_every == 3
    for key, value in (("min_valid_pairs", 3), ("min_effect", 0.1), ("confidence_level", 0.9), ("infra_reruns", 1)):
        with pytest.raises(
            RecipeConfigError, match=f"evolution.{key} applies only to the paired_confidence selection"
        ):
            build(**{key: value})
        with pytest.raises(
            RecipeConfigError, match=f"evolution.{key} applies only to the paired_confidence selection"
        ):
            build(selection="floor", **{key: value})
    for key, value, message in (
        ("confidence_level", 1, "evolution.confidence_level must be a number above 0.5 and below 1"),
        ("min_valid_pairs", 0, "evolution.min_valid_pairs must be an integer of at least 1"),
        ("min_effect", -0.5, "evolution.min_effect must be a finite number of at least 0"),
        ("infra_reruns", -1, "evolution.infra_reruns must be an integer of at least 0"),
        ("min_win_margin", 1, "evolution.min_win_margin applies only to the score_comparison selection"),
        ("floor_score", 0.5, "evolution.floor_score applies only to the floor selection"),
    ):
        with pytest.raises(RecipeConfigError, match=message):
            build(selection="paired_confidence", **{key: value})


class DecideOnlyBackend:
    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        raise AssertionError("this case exercises build(), not evaluate()")


def test_the_factory_binds_a_cordis_backend_only(tmp_path: Path) -> None:
    plugin = PairedConfidencePluginFactory(infra_reruns=1).build(paired_backend(tmp_path))
    assert isinstance(plugin, PairedConfidencePlugin) and plugin.infra_reruns == 1
    assert plugin.settings == PairedConfidenceSettings()
    with pytest.raises(TypeError, match="evaluates through a CordisBackend, not DecideOnlyBackend"):
        PairedConfidencePluginFactory().build(DecideOnlyBackend())
    with pytest.raises(ValueError, match="infra_reruns must be an integer of at least 0"):
        PairedConfidencePluginFactory(infra_reruns=True)
