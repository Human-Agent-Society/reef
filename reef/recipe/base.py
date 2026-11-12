"""Base recipe contracts shared by every concrete recipe.

``Recipe`` is both the default record-only recipe and the base class for every
specialized recipe; ``WeightTrainingRecipe`` narrows the contract for recipes whose
backend updates model weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Self

from reef.artifact.artifact import Artifact, ArtifactValidator
from reef.artifact.components import AcceptAnyArtifact, ComponentSpec, ScenarioComponents
from reef.artifact.repository import DefaultStart
from reef.core.activation import ActivationRule
from reef.core.components import RECORDS_COMPONENT
from reef.core.plugins import PluginConfigError, PluginReference
from reef.core.reports import ReportBase
from reef.evaluation.activation import build_activation_rule
from reef.harness.tasks.model_binding import ModelBinding
from reef.inference.interfaces import InferenceRuntime
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.storage.data import DataStore
from reef.storage.samples import SAMPLE_KINDS, SampleKind
from reef.train.component import ScenarioRoot, TrainingArtifactLayout, TrainingComponent
from reef.train.pipeline import PipelineSettings, PipelineSpec


@dataclass(frozen=True)
class EveryCheck(ArtifactValidator):
    """Admit an artifact only when every check admits it, in order."""

    checks: tuple[ArtifactValidator, ...]

    def validate(self, artifact: Artifact) -> None:
        for check in self.checks:
            check.validate(artifact)


def every_check(*checks: ArtifactValidator) -> ArtifactValidator:
    """One check that admits an artifact only when each of ``checks`` does; a check that admits anything is left out."""
    kept = tuple(check for check in checks if not isinstance(check, AcceptAnyArtifact))
    if not kept:
        return AcceptAnyArtifact()
    return kept[0] if len(kept) == 1 else EveryCheck(kept)


@dataclass(frozen=True)
class ServedEndpoint:
    """Where this Reef answers inference itself: what a recipe's own evaluation calls target.

    ``url`` is the service's base URL and ``token`` a bearer token it accepts.
    ``component`` names the snapshot component the recipe evolves when it is
    one of several: its evaluation calls then leave that component's served
    hooks out, since the rollout runs a candidate of it.
    """

    url: str
    token: str | None = None
    component: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError("served endpoint requires a url")
        object.__setattr__(self, "url", self.url.strip().rstrip("/"))


def check_job_components(recipe: Recipe, components: Mapping[str, TrainingComponent]) -> None:
    """Raise ``ValueError`` unless ``components`` names exactly the job components of ``recipe``."""
    names = sorted(component.name for component in recipe.job_components())
    if sorted(components) != names:
        raise ValueError(f"the job components of {type(recipe).__name__} are {names}, not {sorted(components)}")


@dataclass(frozen=True, kw_only=True)
class Recipe:
    """Default inference recipe and base contract for update recipes.

    Instantiating ``Recipe`` directly records traffic without producing job
    data. Specialized recipes are frozen dataclasses over this base: they
    add their own fields and, when needed, narrow the ``runtime`` field.
    The scenario factory builds the trainers from a recipe's values
    (``reef.scenario.factory.read_trainer_specs``).
    """

    config_sections: ClassVar[tuple[str, ...]] = ()
    """Opaque config sections validated by the recipe alongside its declared fields."""

    refused_sections: ClassVar[Mapping[str, str]] = {}
    """Config sections that this recipe refuses, each with the message of the ``RecipeConfigError`` that the loader
    raises when the section is not empty: for example a weight recipe whose objective options belong to the backend
    refuses ``optimization``."""

    section_defaults: ClassVar[Mapping[str, Any]] = {}
    """The class's own defaults for its config sections, by section name. The loader merges the deployment's config
    over them (``reef.core.values.deep_merge``): nested objects merge key by key, and every other value of the
    deployment wins. A class that only sets other defaults for a base class's sections declares them here."""

    name: str = "recipe"
    runtime: InferenceRuntime | None = None
    training_component: TrainingComponent | None = None
    #: The store of the job data directories that the trainers write for their jobs; the dispatcher binds it
    #: (``with_job_components``).
    data_store: DataStore | None = field(default=None, repr=False, compare=False)

    training_mode: str = config_field("auto")
    #: The deployment's ``pipeline`` section over :meth:`pipeline_spec`; None is the recipe's stages with no config.
    pipeline_stages: PipelineSettings | None = field(default=None, repr=False, compare=False)
    #: The deployment's ``activation`` section: the rule that activates new artifacts; None is ``always``.
    activation: PluginReference | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.training_mode not in ("auto", "manual"):
            raise ValueError("training_mode must be 'auto' or 'manual'")
        # The stages' classes fail the deployment here, not the first scenario.
        self.stages()

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        """The recipe's default stage classes. The default passes records through and makes no job data."""
        return PipelineSpec()

    def activation_rule(self, reference: PluginReference | None = None) -> ActivationRule:
        """The rule that decides whether an artifact that the recipe's trainers make serves: the one ``reference``
        names, such as a scenario's own rule, else the ``activation`` section's."""
        try:
            return build_activation_rule(self.activation if reference is None else reference)
        except PluginConfigError as exc:
            raise RecipeConfigError(str(exc)) from exc

    def stages(self) -> PipelineSettings:
        """The pipeline's stage classes and their configs."""
        if self.pipeline_stages is not None:
            return self.pipeline_stages
        try:
            return PipelineSettings.from_setting(None, spec=type(self).pipeline_spec(), environ={})
        except PluginConfigError as exc:
            raise RecipeConfigError(str(exc)) from exc

    def with_record2sample_config(self, config: Mapping[str, Any]) -> Self:
        """This recipe with each key of ``config`` in place of that key of its record-to-sample stage's config
        (``PipelineSettings.with_record2sample_config``). An empty ``config`` returns this recipe."""
        if not config:
            return self
        return replace(self, pipeline_stages=Recipe.stages(self).with_record2sample_config(config))

    def for_record2sample_configs(
        self, components: ScenarioComponents, configs: Mapping[str, Mapping[str, Any]]
    ) -> Recipe:
        """The recipe from which the trainer builder builds the trainer of each component that ``configs`` names
        with those keys over its record-to-sample config (``ScenarioSettings.record2sample_config``).

        ``components`` is what the scenario serves. The default applies the config of the one component that
        the recipe's only trainer binds. Empty ``configs`` return this recipe. A recipe that binds several trainers
        overrides this method; the default raises ``RecipeConfigError`` for such a recipe.
        """
        if not configs:
            return self
        if len(components.names) > 1:
            raise RecipeConfigError(
                f"{type(self).__name__} serves components {list(components.names)}: override "
                "for_record2sample_configs to give each trainer its own record2sample config"
            )
        component = components.names[0] if components.names else RECORDS_COMPONENT
        return self.with_record2sample_config(configs.get(component, {}))

    def scenario_state_dirs(self, scenario: str) -> tuple[Path, ...]:
        """Directories that belong to one scenario alone, archived when the scenario is deleted.

        The value is a function of the recipe's configuration fields alone. The default is what the
        record-to-sample stage's config names (``PipelineSettings.scenario_state_dirs``). The dispatcher holds a
        recipe of any class, so each recipe class answers for its own configuration.
        """
        return self.stages().scenario_state_dirs(scenario)

    def job_components(self) -> tuple[TrainingComponent, ...]:
        """The training components whose jobs this recipe's trainers submit, each under its own ``name``: the
        weight trainer's ``training_component``, and the ones that a recipe adds, such as a harness's."""
        return () if self.training_component is None else (self.training_component,)

    def with_job_components(self, components: Mapping[str, TrainingComponent], data_store: DataStore) -> Recipe:
        """This recipe whose trainers submit the jobs of each job component to ``components[name]``, such as the
        jobs queue's view of it, and write the data of each job to ``data_store``. Raise ``ValueError`` unless
        ``components`` names exactly the job components."""
        check_job_components(self, components)
        if self.training_component is None:
            return replace(self, data_store=data_store)
        return replace(self, training_component=components[self.training_component.name], data_store=data_store)

    def with_served_endpoint(self, endpoint: ServedEndpoint) -> Recipe:
        """This recipe told where the service answers inference; the default has no calls of its own to point there."""
        return self

    @property
    def report_type(self) -> type[ReportBase] | None:
        """The typed external-report contract this recipe accepts.

        Recipes that consume reports override this property with their
        concrete :class:`~reef.core.reports.ReportBase` subclass. ``None`` keeps
        ingress open for record-only and computed-feedback recipes. This is a
        runtime capability, not a dataclass field or recipe configuration.
        """
        return None

    def build_components(self, scenario: str) -> ScenarioComponents:
        """The components that every snapshot of the named scenario binds.

        Each ``ComponentSpec`` gives the component's kind, the admission check
        run before it is published or restored, its import layout and the
        files a client pulls. The default declares none: the scenario records
        traffic only. Most recipes ignore ``scenario``.
        """
        return ScenarioComponents()

    def check_root_start(self, root: ScenarioRoot) -> None:
        """Check that training can start from ``root`` before an item start is registered.

        The default calls ``training_component.root_start`` when a training component is bound. A recipe with
        several trainers checks each trainer's component and root view. Raise ``ValueError`` on refusal.
        """
        if self.training_component is not None:
            self.training_component.root_start(root)

    def default_start(self) -> DefaultStart:
        """The content of a new scenario's root snapshot when no artifact item starts the scenario; empty by
        default."""
        return DefaultStart()

    def harness_state_dir(self, scenario: str, component: str) -> Path | None:
        """The directory of ``component``'s harness evolution state in ``scenario``: its history, its step records
        and the running step's progress (``reef.train.harness_backend.component``); ``None`` for a recipe that
        evolves no harness.

        The value is a function of the recipe's configuration fields alone (``EvolutionSettings.component_state_dir``
        for a harness recipe). The harness routes hold a recipe of any class, so each recipe class answers for its
        own configuration."""
        return None

    def harness_served_model(self, scenario: str, component: str) -> ModelBinding | None:
        """The model that ``component``'s harness in ``scenario`` calls: a client's layer asks Reef for this model
        by name and dialect. ``None`` for a recipe that evolves no harness."""
        return None

    @property
    def harness_adapter(self) -> str | None:
        """The adapter this recipe's harness artifacts render for, as ``GET /reef/harness`` serves them.

        ``None`` for a recipe that serves no harness tree. Every scenario of
        the process shares it, so the scenario list names it before a
        scenario loads.

        The value is configuration: an override reads only the recipe's own configuration fields, such as
        ``evolution.adapter``, and no runtime, training component or store. A service that holds any recipe
        reads it here.
        """
        return None


#: The component of a weight-training recipe's weights.
WEIGHTS_COMPONENT = "weights"


@dataclass(frozen=True, kw_only=True)
class WeightTrainingRecipe(Recipe):
    """Shared recipe contract for backend algorithms that update weights.

    Requires a :class:`~reef.train.component.TrainingComponent` in
    ``training_component`` (the first positional argument of every training
    recipe), which runs the recipe's training jobs.

    A job's ``parameters`` belong to its training component, and Reef reads
    none of them. :meth:`default_parameters` gives the recipe's own default
    parameters by sample kind, normally the method as ``objective``. The class
    method keeps the dataclass fields as the recipe's instance configuration.
    :meth:`configured_parameters` adds the ``job_parameters`` config; the service
    passes it to the backend when it connects the training component, so a backend
    that starts with one method reads it there.

    ``pipeline_stages`` is framework-owned configuration derived from an
    optional top-level ``pipeline`` section: each stage's class, the recipe's
    :meth:`pipeline_spec` unless the section names another, and its config.

    Configuration is declared as fields: a config field (see
    :mod:`reef.recipe.config_fields`) is simultaneously the YAML key, the
    environment fallback, and the type-aware parser. The loader
    (``reef.recipe.loading.load_recipe``) and the training deployment
    (``reef.service.deploy.training.training_recipe_config``) derive everything from those fields, so a
    recipe never repeats a setting's name or type anywhere else.
    """

    training_component: TrainingComponent = field(kw_only=False)
    runtime: InferenceRuntime = field(kw_only=True)
    job_parameters: Mapping[str, Any] | None = config_field(
        None,
        help=(
            "Job parameters that the recipe sends with every job, over its default parameters; the training "
            "component refuses a parameter it does not take."
        ),
    )

    @classmethod
    def default_parameters(cls) -> Mapping[SampleKind, Mapping[str, Any]]:
        """The recipe's own parameters of a training job, by the kind of the job's samples; empty by default.

        Each value belongs to the training component, and Reef passes it unread. A recipe normally names its method
        here, as the ``objective`` that the backends register. A kind can have its own method, and the training
        component switches the objective from one job to the next.
        """
        return {}

    @classmethod
    def configured_parameters(cls, job_parameters: Mapping[str, Any] | None) -> Mapping[SampleKind, Mapping[str, Any]]:
        """The parameters of each training job, by the kind of the job's samples, for the ``job_parameters`` config.

        The default is :meth:`default_parameters` with ``job_parameters`` over it, key by key, for every kind. A kind
        that a result leaves out trains only for a request that gives its parameters. The service reads this hook
        before it constructs the recipe, and passes the result to the backend when it connects the training component.
        """
        defaults = cls.default_parameters()
        return {kind: {**defaults.get(kind, {}), **(job_parameters or {})} for kind in SAMPLE_KINDS}

    def parameters_by_kind(self) -> Mapping[SampleKind, Mapping[str, Any]]:
        """This recipe's ``configured_parameters``.

        The pipeline's sample-to-job stage gives a job the parameters of its samples' kind, with its request's
        parameters merged in (``Sample2Job.job_parameters``). The component's own defaults fill the rest, such as its
        learning rate. The result must be the same at every call: it is part of each job's identity.
        """
        return type(self).configured_parameters(self.job_parameters)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.training_component is None:
            raise RecipeConfigError(
                f"{type(self).__name__} requires a training component; pass one via the 'training_component' argument"
            )
        if not isinstance(self.training_component, TrainingComponent):
            raise TypeError(
                f"{type(self).__name__} requires a TrainingComponent, got {type(self.training_component).__name__}"
            )
        if not isinstance(self.runtime, InferenceRuntime):
            raise RecipeConfigError("weight recipes require an independent inference runtime")

    def build_components(self, scenario: str) -> ScenarioComponents:
        weights = ComponentSpec(
            kind=self.runtime.served_weight_kind, layout=TrainingArtifactLayout(self.training_component)
        )
        return ScenarioComponents({WEIGHTS_COMPONENT: weights})

    def default_start(self) -> DefaultStart:
        """The training component's untrained checkpoint (``TrainingComponent.default_start``)."""
        return self.training_component.default_start()


def close_runtimes(
    training_components: Sequence[TrainingComponent], runtime: InferenceRuntime | None
) -> list[BaseException]:
    """Close a recipe's training components and inference runtime, and return their close errors in that order.

    Admission closes first, so that no new request starts. Then the training
    components close, and then the runtime shuts down. A failed close does not
    stop the next one. ``None`` stands for an absent runtime.
    """
    if runtime is not None:
        runtime.close_admission()
    errors: list[BaseException] = []
    for training_component in training_components:
        try:
            training_component.close()
        except BaseException as exc:
            errors.append(exc)
    if runtime is not None:
        try:
            runtime.shutdown()
        except BaseException as exc:
            errors.append(exc)
    return errors
