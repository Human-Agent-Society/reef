"""Base recipe contracts shared by every concrete recipe.

``Recipe`` is both the default record-only recipe and the base class for every
specialized recipe; ``WeightTrainingRecipe`` narrows the contract for recipes whose
backend updates model weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Self

from reef.artifact.artifact import Artifact, ArtifactValidator
from reef.artifact.components import AcceptAnyArtifact, ComponentSpec, ScenarioComponents
from reef.artifact.repository import DefaultStart
from reef.core.activation import ActivationRule
from reef.core.plugins import PluginConfigError, PluginReference
from reef.core.reports import ReportBase
from reef.evaluation.activation import build_activation_rule
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.storage.samples import SAMPLE_KINDS, SampleKind
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


@dataclass(frozen=True, kw_only=True)
class Recipe:
    """Default inference recipe and base contract for update recipes.

    Instantiating ``Recipe`` directly records traffic without producing job
    data. Specialized recipes are frozen dataclasses over this base: they
    add their own configuration fields. A recipe holds no runtime, no
    training component and no store (RFC #1390): the process holds them in
    ``reef.scenario.live_objects.LiveObjects``. The scenario factory builds
    the trainers from a recipe's values and those objects
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

    @property
    def report_type(self) -> type[ReportBase] | None:
        """The typed external-report contract this recipe accepts.

        Recipes that consume reports override this property with their
        concrete :class:`~reef.core.reports.ReportBase` subclass. ``None`` keeps
        ingress open for record-only and computed-feedback recipes. This is a
        runtime capability, not a dataclass field or recipe configuration.

        A ``CompositeRecipe`` declares none of its own: the contract of a deployment is
        ``reef.scenario.factory.scenario_report_type``, which reads each component's.
        """
        return None

    @property
    def supports_images(self) -> bool:
        """Whether this recipe trains on image content. The default is ``False``; no built-in recipe returns ``True``.

        A recipe returns ``True`` only when its training path renders images: ``reef.core.chat_rendering.trained_turns``
        refuses every part that is not text. A ``CompositeRecipe`` keeps the default and says nothing about its
        children. This is a runtime capability, not a dataclass field or recipe configuration.
        """
        return False

    @property
    def refuses_images(self) -> bool:
        """Whether this recipe refuses image content. The default is ``False``: a recipe that trains no weights keeps
        image content in its records.

        A ``CompositeRecipe`` keeps the default: a deployment refuses images when any of its recipes refuses them
        (``reef.recipe.deployment.refuses_images``). The inference service, the record import and the data items ask
        that function, and they find an image with ``reef.core.trajectories.has_image_content``.
        """
        return False

    def build_components(self, scenario: str) -> ScenarioComponents:
        """The components that every snapshot of the named scenario binds, as the recipe's configuration declares
        them.

        Each ``ComponentSpec`` gives the component's kind, the admission check
        run before it is published or restored, its import layout and the
        files a client pulls. The default declares none: the scenario records
        traffic only. Most recipes ignore ``scenario``. The value is a function
        of the recipe's configuration fields alone; the deployment's components are
        ``reef.scenario.live_objects.scenario_components``, which adds the facts that live objects know.
        """
        return ScenarioComponents()

    def default_start(self) -> DefaultStart:
        """The content of a new scenario's root snapshot when no artifact item starts the scenario, as the recipe's
        configuration gives it; empty by default. The deployment's default start is
        ``reef.scenario.live_objects.default_start``."""
        return DefaultStart()


#: The component of a weight-training recipe's weights.
WEIGHTS_COMPONENT = "weights"


@dataclass(frozen=True, kw_only=True)
class WeightTrainingRecipe(Recipe):
    """Shared recipe contract for backend algorithms that update weights.

    The deployment's training component runs the recipe's training jobs, and
    its runtime serves the weights. The recipe holds neither: the process holds
    them in ``reef.scenario.live_objects.LiveObjects``, and
    ``check_recipe_objects`` refuses a deployment without them.

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

    job_parameters: Mapping[str, Any] | None = config_field(
        None,
        help=(
            "Job parameters that the recipe sends with every job, over its default parameters; the training "
            "component refuses a parameter it does not take."
        ),
    )

    trains_task_jobs: ClassVar[bool] = False
    """Whether every job of the recipe is a task job. The deployment's training component must then take the
    recipe's task parameters (``reef.scenario.live_objects.check_recipe_objects``)."""

    @property
    def refuses_images(self) -> bool:
        """Refuse image content unless the recipe supports images, so that it does not store an image that it will not
        train on, and does not train on the text that remains without the image."""
        return not self.supports_images

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

    def build_components(self, scenario: str) -> ScenarioComponents:
        """One component, ``weights``. Its kind and its layout are not configuration yet: the runtime and the
        training component give them (``reef.scenario.live_objects.scenario_components``)."""
        return ScenarioComponents({WEIGHTS_COMPONENT: ComponentSpec()})
