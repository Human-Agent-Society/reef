"""Reefine configuration: its defaults, and the agent each scenario's backend gets."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from reef_terminus.runner import trial_record
from tests.support.harness_steps import COMPONENT, TreeReleases

from recipes.beta.reefine import ReefineMethod, ReefineRecipe
from recipes.beta.reefine.recipe import SEED_DIRECTORY
from recipes.beta.reefine.scorer import HEALTH_TASK_DIRECTORY, AnswerScorer
from reef.core.config import config_arguments
from reef.evaluation.activation import AlwaysActivate, NeverActivate
from reef.execution.sandbox.process import LocalExecutor
from reef.harness.rollouts.run import RolloutResult
from reef.harness.tree import HarnessTree
from reef.inference.http import InferenceProxyRuntime
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import build_recipe, recipe_class_for
from reef.train.backend import ComponentReleases

HEALTH = "[health] Run `echo reef-ok` and reply with its output."


def evolution(**section: object) -> dict[str, object]:
    """A deployment's ``evolution`` section: the health task as its one evaluation task, then ``section``."""
    return {"evaluation": {"tasks": [HEALTH]}, **section}


def test_dotted_recipe_defaults_and_config_are_independent() -> None:
    config = {"evolution": evolution(agent=None)}
    original = copy.deepcopy(config)
    built = build_recipe("recipes.beta.reefine:ReefineRecipe", {}, config)
    assert recipe_class_for("recipes.beta.reefine:ReefineRecipe") is ReefineRecipe
    assert isinstance(built, ReefineRecipe)
    assert built.name == "reefine" and built.training_mode == "manual"
    assert isinstance(built.method, ReefineMethod) and built.method.agent is None and built.agent is None
    # By default a person activates a tree that changes an extension.
    rule = built.activation_rule()
    assert isinstance(rule, AlwaysActivate) and rule.review_kinds == {"extension"}
    # The floor: the candidate alone must pass every evaluation task; the current release is not run.
    assert (built.evaluation.policy, built.evaluation.floor_score) == ("floor", 1.0)
    assert isinstance(built.scorer, AnswerScorer)
    # Without evolution.seed the first tree is the packaged one: the starter skill.
    assert built.seed == HarnessTree.read(SEED_DIRECTORY)
    assert list(built.seed.files) == ["skills/answer-style/SKILL.md"]
    assert config == original
    assert "training_mode" in {argument.name for argument in config_arguments(ReefineRecipe)}


def test_evolution_overrides_and_training_fields_remain_available(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "rules.md").write_text("Answer in one line.\n")
    built = ReefineRecipe.from_environment(
        {},
        config={
            "data": {"training_mode": "hybrid"},
            "pipeline": {"sample2job": {"config": {"batch_size": 3}}},
            "evolution": {
                "evaluation": {"tasks": ["[fib] Compute fib(90)."], "policy": "compare"},
                "seed": str(seed),
                "agent": None,
            },
            "activation": {"module": "never"},
        },
    )
    assert isinstance(built, ReefineRecipe)
    assert built.training_mode == "hybrid" and built.stages().sample2job.config == {"batch_size": 3}
    assert built.seed == HarnessTree({"rules.md": "Answer in one line.\n"})
    assert isinstance(built.activation_rule(), NeverActivate)
    assert built.evaluation.policy == "compare" and isinstance(built.scorer, AnswerScorer)


@pytest.mark.parametrize(
    "section",
    [
        None,
        [],
        {"evaluation": {"tasks": []}},
        {"evaluation": "floor"},
        {"evaluation": {"tasks": ["task"]}, "seed": 3},
        {"evaluation": {"tasks": ["task"]}, "seed": "/no/such/seed"},
        {"evaluation": {"tasks": ["task"]}, "multimodal": "openrouter"},
        {"evaluation": {"tasks": ["task"]}, "agent": "bwrap"},
        # The key's old name is an unknown field.
        {"evaluation": {"tasks": ["task"]}, "proposer_agent": None},
    ],
)
def test_invalid_evolution_config_is_rejected(section: object) -> None:
    with pytest.raises(RecipeConfigError):
        ReefineRecipe.from_environment({}, config={"evolution": section})


def test_a_seed_that_fails_validation_is_refused_at_startup(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not a harness file")
    with pytest.raises(RecipeConfigError, match=r"evolution\.seed fails validation: notes\.txt"):
        ReefineRecipe.from_environment({}, config={"evolution": evolution(seed=str(tmp_path))})


def test_on_terminus_the_health_task_runs_as_the_shipped_task_directory(tmp_path: Path) -> None:
    """reef-terminus runs a Harbor task directory, not a prompt: the profile's health prompt becomes the same check
    as the directory beside the recipe, and the bundled scorer reads that directory's verifier reward."""
    section = {"adapter": "terminus", "evaluation": {"tasks": [HEALTH, "/tasks/own-task"]}}
    built = ReefineRecipe.from_environment({}, config={"evolution": section})
    assert isinstance(built, ReefineRecipe)
    assert built.evaluation.tasks == (HEALTH_TASK_DIRECTORY, "/tasks/own-task")
    health = Path(HEALTH_TASK_DIRECTORY)
    for name in ("instruction.md", "task.toml", "tests/test.sh", "environment/Dockerfile"):
        assert (health / name).is_file(), name
    assert "reef-ok" in (health / "tests/test.sh").read_text()
    # A prompt adapter keeps the prompt.
    pi = ReefineRecipe.from_environment({}, config={"evolution": {**section, "adapter": "pi", "agent": None}})
    assert isinstance(pi, ReefineRecipe) and pi.evaluation.tasks[0] == HEALTH

    def ran(rewards: dict[str, float], error: str = "") -> RolloutResult:
        row = trial_record(HEALTH_TASK_DIRECTORY, rewards, tmp_path, error)
        events = ({"type": "verifier", **{key: value for key, value in row.items() if key != "steps"}},)
        return RolloutResult(exit_code=0, stdout="", stderr="", trajectory=events, residue=())

    scorer = AnswerScorer()
    assert scorer.score(HEALTH_TASK_DIRECTORY, ran({"reward": 1.0})) == 1.0
    assert scorer.score(HEALTH_TASK_DIRECTORY, ran({"reward": 0.0})) == 0.0
    assert scorer.score(HEALTH_TASK_DIRECTORY, ran({}, "the container never built")) == 0.0
    # The reply's last line is not the score of a task directory: a Harbor rollout has no assistant reply.
    assert scorer.score("[health] x", ran({"reward": 1.0})) == 0.0


def test_an_unknown_adapter_is_a_config_error() -> None:
    with pytest.raises(RecipeConfigError, match="acme"):
        ReefineRecipe.from_environment({}, config={"evolution": evolution(adapter="acme")})


def test_off_pi_no_agent_is_built_or_warned_about(caplog) -> None:
    """The agent answers requests on pi alone: another adapter builds no agent, so an unjailed agent setting
    neither warns nor needs a sandbox there, while pi still warns."""
    with caplog.at_level("WARNING"):
        built = ReefineRecipe.from_environment(
            {"REEF_AGENT_SANDBOX": "none"}, config={"evolution": evolution(adapter="dsh")}
        )
    assert isinstance(built, ReefineRecipe) and built.agent is None
    assert "sandbox is none" not in caplog.text
    with caplog.at_level("WARNING"):
        pi = ReefineRecipe.from_environment({"REEF_AGENT_SANDBOX": "none"}, config={"evolution": evolution()})
    assert isinstance(pi, ReefineRecipe) and pi.agent is not None and isinstance(pi.agent.executor, LocalExecutor)
    assert "sandbox is none" in caplog.text


@pytest.mark.parametrize("adapter", ["pi", "claude", "codex", "opencode", "hermes", "dsh", "terminus"])
def test_the_packaged_seed_is_the_first_tree_on_every_adapter(adapter: str) -> None:
    """The recipe validates its seed for the adapter at startup: the starter skill renders everywhere."""
    built = ReefineRecipe.from_environment({}, config={"evolution": evolution(adapter=adapter, agent=None)})
    assert isinstance(built, ReefineRecipe) and built.evolution.adapter == adapter
    assert built.seed == HarnessTree.read(SEED_DIRECTORY)


def test_each_scenarios_backend_runs_the_agent_on_the_bound_binary_with_the_gateway(tmp_path: Path) -> None:
    """The recipe binds the agent to pi once per scenario, with the deployment's binary and its multimodal gateway,
    and leaves its own method without one."""
    binary = tmp_path / "pi"
    section = evolution(
        binary=str(binary),
        state_dir=str(tmp_path / "state"),
        agent={"timeout_s": 120, "trial_timeout_s": 30},
        multimodal={"preset": "openrouter"},
    )
    upstream = InferenceProxyRuntime(base_url="https://openrouter.ai/api", api_key="sk-upstream", model_path="m")
    built = ReefineRecipe.from_environment(
        {"REEF_AGENT_SANDBOX": "none"}, config={"evolution": section}, runtime=upstream
    )
    assert isinstance(built, ReefineRecipe) and isinstance(built.method, ReefineMethod)
    assert built.method.agent is None
    backend = built.build_backend("demo", ComponentReleases(TreeReleases(built.seed), COMPONENT), None)
    method = backend.method
    assert isinstance(method, ReefineMethod) and method.agent is not None
    execution = method.agent.execution
    assert isinstance(execution.executor, LocalExecutor) and execution.binary == str(binary)
    assert (execution.timeout_s, method.agent.trial_timeout_s) == (120.0, 30.0)
    assert method.agent.provider == built.multimodal_provider and method.agent.provider.api_key == "sk-upstream"
    backend.close()


@pytest.mark.unit
def test_harbor_loads_the_reefine_health_task_directory() -> None:
    pytest.importorskip("harbor", reason="harbor is not installed")
    from harbor.models.task.task import Task

    assert Task.is_valid_dir(HEALTH_TASK_DIRECTORY)
    task = Task(HEALTH_TASK_DIRECTORY)
    assert "echo reef-ok" in task.instruction
    assert task.paths.test_path.is_file() and (task.paths.environment_dir / "Dockerfile").is_file()
