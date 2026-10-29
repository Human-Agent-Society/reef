"""Reefine's recipe: the harness evolution recipe with Reefine's defaults, its agent and its multimodal gateway.

``ReefineRecipe`` answers a person's training instructions, one step per
instruction (``training_mode: manual``), with ``ReefineMethod``. It merges
Reefine's defaults into the deployment's ``evolution`` section: the method,
the seed beside this module, and an evaluation that runs the candidate alone
on the deployment's tasks, scored by ``AnswerScorer``, and keeps it when every
task meets the floor. On the health task the floor checks that the tree still
works, not that the requested change does; the step's design and review and
the person judge that. By default, a person activates a tree that changes an
extension (the ``activation`` rule's ``review_kinds``).

Two keys of the section are Reefine's own. ``agent`` says where the
coding agent that answers a request on pi runs (``agent_settings``), and
``multimodal`` names the gateway for images, embeddings, speech and
decisions (``recipes.beta.reefine.multimodal``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from recipes.beta.reefine.agent import AgentExecution
from recipes.beta.reefine.answer import EXTENSION_ADAPTER
from recipes.beta.reefine.method import ReefineMethod
from recipes.beta.reefine.multimodal import MultimodalProvider, MultimodalSettings, ProviderRelay
from recipes.beta.reefine.scorer import HEALTH_TASK_DIRECTORY, AnswerScorer
from reef.execution.sandbox.config import REMOTE_EXECUTORS, remote_executor
from reef.execution.sandbox.process import LocalExecutor, ProgramExecutor, SandboxExecutor, SandboxUnavailable
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import DescriptorError
from reef.harness.rollouts.vendor_install import AdapterProgram
from reef.harness.tree import TreeKind
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.runtime.interfaces import MultimodalRelay
from reef.storage.tasks import TaskStorage
from reef.train.backend import ComponentReleases
from reef.train.harness_backend.backend import HarnessBackend

#: The first tree of a new Reefine scenario: the starter skill.
SEED_DIRECTORY = Path(__file__).with_name("seed")
#: Reefine's defaults for the ``evolution`` section and its ``evaluation`` section; the deployment's keys win.
EVOLUTION_DEFAULTS: Mapping[str, Any] = {
    "method": f"{ReefineMethod.__module__}:{ReefineMethod.__qualname__}",
    "seed": str(SEED_DIRECTORY),
}
#: Reefine's ``activation`` section when the deployment names none. An extension runs in the harness's process with
#: the user's privileges, so a person activates a tree that changes one.
ACTIVATION_DEFAULT: Mapping[str, Any] = {
    "module": "always",
    "config": {"review_kinds": [TreeKind.CODE_EXTENSION.value]},
}
EVALUATION_DEFAULTS: Mapping[str, Any] = {
    "policy": "floor",
    "scorer": f"{AnswerScorer.__module__}:{AnswerScorer.__qualname__}",
}


@dataclass(frozen=True)
class AgentSettings:
    """``evolution.agent``, read: where Reefine's agent runs and for how long.

    Values: ``executor`` is the agent's isolation, not yet bound to pi's
    program; ``timeout_s`` > 0 bounds the whole agent run and
    ``trial_timeout_s`` > 0 each trial.
    Purpose: ``agent_settings`` reads it from the deployment, and
    ``ReefineRecipe.build_backend`` binds it for each scenario.
    """

    executor: ProgramExecutor
    timeout_s: float
    trial_timeout_s: float

    def __post_init__(self) -> None:
        if self.timeout_s <= 0 or self.trial_timeout_s <= 0:
            raise ValueError("the agent's timeouts must be positive")

    def bind(self, binary: str | None, provider: MultimodalProvider | None) -> AgentExecution:
        """The agent's execution: the executor bound to pi's program (``binary`` names the binary in place of the
        pinned one) and preflighted, the time limits, and ``provider`` for the trials."""
        executor, bound_binary = self.executor.for_program(
            AdapterProgram(get_adapter(EXTENSION_ADAPTER)), binary=binary
        )
        executor.preflight()
        return AgentExecution(executor, bound_binary, self.timeout_s, self.trial_timeout_s, provider)


def agent_settings(section: Any, environ: Mapping[str, str], *, state_dir: Path) -> AgentSettings | None:
    """``evolution.agent`` as the agent's isolation and its two time limits; ``None`` runs no agent.

    ``sandbox: bwrap`` jails the agent and gives it the internet but no host
    port but the gateway's, and refuses to start where bwrap or pasta is
    missing. A cloud sandbox (``sandbox: e2b`` or ``sandbox: modal``, with
    that provider's options in the section; its image is the harness's
    pinned binary, built on first use, unless an option names one) runs it
    where it reaches the gateway through a tunnel and nothing else of the
    host, and carries the deployment that ``state_dir`` names, so a later
    start stops the sandboxes a stopped run left. ``sandbox: none`` runs it
    unisolated with the service's privileges, and must be chosen. Left empty
    (and ``REEF_AGENT_SANDBOX`` unset), the agent is jailed where the host
    can, and off where it cannot: then the served model answers requests in
    text. No section runs no agent. The variable's old name,
    ``REEF_PROPOSER_SANDBOX``, is refused, so a deployment that still sets
    it does not start without the isolation it names.
    """
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise RecipeConfigError("evolution.agent must be a mapping")
    if "REEF_PROPOSER_SANDBOX" in environ:
        raise RecipeConfigError("REEF_PROPOSER_SANDBOX is no longer read: set REEF_AGENT_SANDBOX instead")
    timeouts = []
    for key, default in (("timeout_s", 1800.0), ("trial_timeout_s", 300.0)):
        value = section.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise RecipeConfigError(f"evolution.agent.{key} must be a positive number")
        timeouts.append(float(value))
    timeout_s, trial_timeout_s = timeouts
    sandbox = str(section.get("sandbox") or environ.get("REEF_AGENT_SANDBOX") or "").strip()
    if sandbox == "none":
        logging.getLogger(__name__).warning(
            "evolution.agent.sandbox is none: the agent runs with the service's own privileges and full "
            "network access, fed text from clients; use it only where you trust every client"
        )
        return AgentSettings(LocalExecutor(), timeout_s, trial_timeout_s)
    if sandbox in REMOTE_EXECUTORS:
        try:
            remote = remote_executor(sandbox, section, environ, timeout_s=timeout_s, owner_anchor=state_dir)
            remote.preflight()
        except SandboxUnavailable as exc:
            raise RecipeConfigError(f"evolution.agent.sandbox is {sandbox}, but {exc}") from exc
        return AgentSettings(remote, timeout_s, trial_timeout_s)
    if sandbox not in ("", "bwrap"):
        choices = ", ".join(f"'{name}'" for name in ("bwrap", *sorted(REMOTE_EXECUTORS), "none"))
        raise RecipeConfigError(f"evolution.agent.sandbox must be one of {choices}")
    executor = SandboxExecutor(network="isolated")
    try:
        executor.preflight()
    except SandboxUnavailable as exc:
        if sandbox == "bwrap":
            raise RecipeConfigError(f"evolution.agent.sandbox is bwrap, but {exc}") from exc
        logging.getLogger(__name__).warning(
            "the agent is off: %s. The served model answers requests in text; set "
            "evolution.agent.sandbox: none to run the agent without isolation",
            exc,
        )
        return None
    return AgentSettings(executor, timeout_s, trial_timeout_s)


@dataclass(frozen=True, kw_only=True)
class ReefineRecipe(HarnessEvolutionRecipe):
    """Refine a harness once per instruction; a person activates a tree that changes an extension.

    Values: the fields of ``HarnessEvolutionRecipe``, read with Reefine's
    defaults merged in. ``multimodal`` is ``evolution.multimodal`` read,
    ``None`` for no gateway: Reef relays the harness's image, embedding,
    speech and decision calls to it with its key, unrecorded, and the
    agent's trials reach the same one. ``agent`` is
    ``evolution.agent`` read, ``None`` when no agent answers
    requests (off pi, or where the host cannot jail it).
    Purpose: a deployment names it as its recipe; it builds each scenario's
    backend with ``ReefineMethod``. Override ``evolution.evaluation.policy``
    to compare scores against the head tree, and ``training-mode`` to
    learn from reports too.
    """

    name: str = field(default="reefine", kw_only=True)
    training_mode: str = config_field("manual")
    multimodal: MultimodalSettings | None = None
    agent: AgentSettings | None = None

    @property
    def multimodal_provider(self) -> MultimodalProvider | None:
        """The configured provider, its key the chat upstream's when the upstream is the same gateway."""
        if self.multimodal is None:
            return None
        binding = self.model_binding() if self.runtime is not None else None
        return self.multimodal.provider(
            None if binding is None else binding.base_url, None if binding is None else binding.api_key
        )

    @property
    def multimodal_relay(self) -> MultimodalRelay | None:
        provider = self.multimodal_provider
        return None if provider is None else ProviderRelay(provider)

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        section = settings.get("evolution")
        if not isinstance(section, Mapping):
            raise RecipeConfigError("reefine requires an 'evolution' config mapping")
        evaluation = section.get("evaluation", {})
        if not isinstance(evaluation, Mapping):
            raise RecipeConfigError("evolution.evaluation must be a mapping")
        adapter = str(section.get("adapter", EXTENSION_ADAPTER))
        try:
            descriptor = get_adapter(adapter)
        except DescriptorError as exc:
            raise RecipeConfigError(str(exc)) from exc
        evaluation = {**EVALUATION_DEFAULTS, **evaluation}
        tasks = evaluation.get("tasks")
        if descriptor.is_prompt_task_directory and isinstance(tasks, Sequence) and not isinstance(tasks, str):
            # The adapter's prompt is a Harbor task directory, so the health prompt runs as the same check in
            # that form, scored by its verifier.
            evaluation["tasks"] = [
                HEALTH_TASK_DIRECTORY if str(task).startswith("[health]") else task for task in tasks
            ]
        own = {key: value for key, value in section.items() if key not in ("agent", "multimodal")}
        kwargs = super()._recipe_kwargs(
            {
                **settings,
                "evolution": {**EVOLUTION_DEFAULTS, **own, "evaluation": evaluation},
                "activation": settings.get("activation", ACTIVATION_DEFAULT),
            },
            values,
        )
        try:
            kwargs["multimodal"] = MultimodalSettings.from_config(section.get("multimodal"), values)
        except ValueError as exc:
            raise RecipeConfigError(f"evolution.{exc}") from exc
        # The agent answers requests on pi alone; on another adapter the served model does, so no agent is
        # built, jailed or warned about.
        kwargs["agent"] = (
            agent_settings(
                section.get("agent", {}), values, state_dir=Path(kwargs["evolution"].state_dir).expanduser()
            )
            if adapter == EXTENSION_ADAPTER
            else None
        )
        return kwargs

    def build_backend(
        self, scenario: str, releases: ComponentReleases, task_storage: TaskStorage | None
    ) -> HarnessBackend:
        """The scenario's backend. With an agent configured, its Reefine method runs the agent, bound here to pi
        once per scenario as the evaluator is: the binary is resolved, and installed when missing, on the Reef
        host, or named in a cloud sandbox's image."""
        if self.agent is None or not isinstance(self.method, ReefineMethod):
            return super().build_backend(scenario, releases, task_storage)
        method = ReefineMethod(self.agent.bind(self.evolution.binary, self.multimodal_provider))
        return HarnessEvolutionRecipe.build_backend(replace(self, method=method), scenario, releases, task_storage)


__all__ = ["SEED_DIRECTORY", "AgentSettings", "ReefineRecipe", "agent_settings"]
