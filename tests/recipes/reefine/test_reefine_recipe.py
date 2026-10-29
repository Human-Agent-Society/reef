"""Reefine configuration: its defaults, the agent's isolation, and the agent each scenario's backend gets."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from tests.support.harness_steps import COMPONENT, TreeReleases

from recipes.beta.reefine import ReefineMethod, ReefineRecipe
from recipes.beta.reefine.recipe import SEED_DIRECTORY, agent_settings
from recipes.beta.reefine.scorer import HEALTH_TASK_DIRECTORY, AnswerScorer
from reef.core.config import config_arguments
from reef.evaluation.activation import AlwaysActivate, NeverActivate
from reef.execution.sandbox.e2b import E2BExecutor
from reef.execution.sandbox.modal import ModalExecutor
from reef.execution.sandbox.process import LocalExecutor, SandboxExecutor, SandboxUnavailable
from reef.execution.sandbox.remote import deployment_owner
from reef.harness.rollouts.run import RolloutResult
from reef.harness.runners.terminus.runner import trial_record
from reef.harness.tree import HarnessTree, TreeKind
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
    assert isinstance(rule, AlwaysActivate) and rule.review_kinds == {TreeKind.CODE_EXTENSION}
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


def test_agent_settings_choose_the_isolation(monkeypatch, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    assert agent_settings(None, {}, state_dir=state_dir) is None
    settings = agent_settings({"sandbox": "none", "timeout_s": 60}, {}, state_dir=state_dir)
    assert settings is not None and isinstance(settings.executor, LocalExecutor)
    assert (settings.timeout_s, settings.trial_timeout_s) == (60.0, 300.0)
    unset = agent_settings({}, {"REEF_AGENT_SANDBOX": "none"}, state_dir=state_dir)
    assert unset is not None and isinstance(unset.executor, LocalExecutor)
    # The variable's old name would be ignored, and the agent run with another isolation: it is refused.
    with pytest.raises(RecipeConfigError, match="set REEF_AGENT_SANDBOX instead"):
        agent_settings({}, {"REEF_PROPOSER_SANDBOX": "none"}, state_dir=state_dir)
    with pytest.raises(RecipeConfigError, match="one of 'bwrap', 'e2b', 'modal', 'none'"):
        agent_settings({"sandbox": "docker"}, {}, state_dir=state_dir)
    with pytest.raises(RecipeConfigError, match="positive number"):
        agent_settings({"timeout_s": 0}, {}, state_dir=state_dir)
    # E2B: the key from the section, else the environment; none at all is refused at startup.
    with pytest.raises(RecipeConfigError, match=r"sandbox is e2b, but .*E2B_API_KEY"):
        agent_settings({"sandbox": "e2b"}, {}, state_dir=state_dir)
    monkeypatch.setattr(E2BExecutor, "preflight", lambda self: None)  # the e2b package is an extra
    remote = agent_settings({"sandbox": "e2b", "timeout_s": 900}, {"E2B_API_KEY": "e2b-env"}, state_dir=state_dir)
    assert remote is not None and isinstance(remote.executor, E2BExecutor)
    assert (remote.executor.api_key, remote.executor.timeout_s, remote.executor.template) == ("e2b-env", 900.0, "")
    assert "e2b-env" not in repr(remote)
    # The deployment's state directory names its sandboxes, so a later start can stop the ones a stopped run left.
    assert remote.executor.owner == deployment_owner(state_dir)
    configured = agent_settings(
        {"sandbox": "e2b", "e2b_api_key": "e2b-cfg", "e2b_template": "mine"},
        {"E2B_API_KEY": "e2b-env"},
        state_dir=state_dir,
    )
    assert configured is not None and isinstance(configured.executor, E2BExecutor)
    assert (configured.executor.api_key, configured.executor.template) == ("e2b-cfg", "mine")
    # Modal: the provider's options from the section, the deployment that names its sandboxes, and the timeout.
    monkeypatch.setattr(ModalExecutor, "preflight", lambda self: None)
    modal = agent_settings({"sandbox": "modal", "modal_app": "reef-agents", "timeout_s": 90}, {}, state_dir=state_dir)
    assert modal is not None and isinstance(modal.executor, ModalExecutor)
    assert (modal.executor.app_name, modal.executor.timeout_s) == ("reef-agents", 90.0)
    assert modal.executor.owner == deployment_owner(state_dir)
    monkeypatch.undo()

    def unavailable(self) -> None:
        raise SandboxUnavailable("no bwrap here")

    monkeypatch.setattr(SandboxExecutor, "preflight", unavailable)
    # Unset on a host that cannot isolate: the agent is off and the served model answers in text.
    assert agent_settings({}, {}, state_dir=state_dir) is None
    with pytest.raises(RecipeConfigError, match="no bwrap here"):
        agent_settings({"sandbox": "bwrap"}, {}, state_dir=state_dir)


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
    assert isinstance(method.agent.executor, LocalExecutor) and method.agent.binary == str(binary)
    assert (method.agent.timeout_s, method.agent.trial_timeout_s) == (120.0, 30.0)
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
