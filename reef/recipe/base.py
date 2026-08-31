"""Base recipe contracts shared by every concrete recipe.

``Recipe`` is both the default record-only recipe and the base class for every
specialized recipe; ``WeightTrainingRecipe`` narrows the contract for recipes whose
backend updates model weights.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.core.reports import ReportBase
from reef.observability import ExperimentLogger
from reef.recipe.config import config_positive_int
from reef.recipe.config_fields import config_field, parse_int, recipe_config_fields, resolve_config_field_values
from reef.recipe.errors import RecipeConfigError
from reef.records import RecordStore
from reef.runtime.base import InferenceRuntime, TrainingRuntime
from reef.runtime.inference import InferenceBackend
from reef.runtime.proxy import resolve_proxy_runtime
from reef.scenario.binding import AcceptAnyArtifact, ArtifactValidator
from reef.scenario.checkpoint_strategy import CheckpointStrategy, EveryNVersions
from reef.surface.base import Surface
from reef.surface.weights import create_weight_surface
from reef.train.algos.registry import resolve_preparer
from reef.train.evaluation import CandidateEvaluationConfig, CandidateEvaluationConfigError, build_candidate_evaluation
from reef.train.processors.base import DataProcessor
from reef.train.trainer import Trainer


@dataclass(frozen=True, kw_only=True)
class Recipe:
    """Default inference recipe and base contract for update recipes.

    Instantiating ``Recipe`` directly records traffic without producing update
    batches. Specialized recipes are frozen dataclasses over this base: they
    add their own fields, override ``build`` and, when needed, narrow the
    ``runtime`` field.
    """

    name: str = "recipe"
    runtime: InferenceRuntime | None = None
    checkpoint_strategy: CheckpointStrategy = field(default_factory=lambda: EveryNVersions(1))

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        runtime: InferenceRuntime | None = None,
    ) -> Recipe:
        values = os.environ if environ is None else environ
        settings = config or {}
        artifact = settings.get("artifact", {})
        try:
            return cls(
                **cls._recipe_kwargs(settings, values),
                checkpoint_strategy=EveryNVersions(config_positive_int(artifact, "checkpoint_every_n_versions", 1)),
                runtime=cls._resolve_runtime(values, runtime),
                **resolve_config_field_values(cls, settings.get("data", {}), values),
            )
        except ValueError as exc:
            raise RecipeConfigError(f"invalid {cls.__name__} configuration: {exc}") from exc

    @classmethod
    def _resolve_runtime(cls, values: Mapping[str, str], runtime: InferenceRuntime | None) -> InferenceRuntime | None:
        return resolve_proxy_runtime(values, runtime)

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        """Recipe-specific keyword arguments resolved from config sections."""
        return {}

    @property
    def report_type(self) -> type[ReportBase] | None:
        """The typed external-report contract this recipe accepts.

        Recipes that consume reports override this property with their
        concrete :class:`~reef.core.reports.ReportBase` subclass. ``None`` keeps
        ingress open for record-only and computed-feedback recipes. This is a
        runtime capability, not a dataclass field or recipe configuration.
        """
        return None

    def build(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, Any] | None = None,
        experiment_logger: ExperimentLogger | None = None,
    ) -> Trainer:
        return Trainer.build(
            scenario,
            records,
            processor_factory=DataProcessor,
            algorithm_state=algorithm_state,
            report_type=self.report_type,
            experiment_logger=experiment_logger,
        )

    @property
    def inference_backend(self) -> InferenceBackend | None:
        """The inference backend composed by this recipe's runtime, if any.

        Use ``runtime`` for the runtime itself.
        """
        runtime = self.runtime
        return runtime.inference_backend if runtime is not None else None

    def build_surface(self, scenario: str) -> Surface:
        """Build the serving surface for the named scenario.

        Most recipes ignore ``scenario``; recipes whose serving state is
        scenario-specific (an adapter on a shared engine) route by it.
        """
        return Surface()

    def serving_status(self) -> Mapping[str, Any] | None:
        """Runtime-wide serving state this recipe owns, for ``/reef/status``.

        ``None`` when the recipe has nothing beyond per-scenario training
        state to report.
        """
        return None

    def build_artifact_validator(self) -> ArtifactValidator:
        """Build the artifact admission policy for one scenario."""
        return AcceptAnyArtifact()


@dataclass(frozen=True)
class WeightTrainingSpec:
    """Static machinery a weight-training recipe binds to its configuration.

    Keeping the binding behind one explicit hook leaves the recipe dataclass
    to describe instance configuration only. The service and training driver
    can still inspect a recipe class without constructing it.
    """

    step_preparer: str
    loss_family: str
    processor: type[DataProcessor] | None = None


@dataclass(frozen=True, kw_only=True)
class WeightTrainingRecipe(Recipe):
    """Shared recipe contract for backend algorithms that update weights.

    Narrows the base ``runtime`` field to a required :class:`TrainingRuntime`
    (the first positional argument of every training recipe).

    :meth:`training_spec` binds the data processor, step preparer, and backend
    loss family. Keeping that static machinery in one structured return value
    means the dataclass fields remain the recipe's instance configuration.
    The training driver receives its loss implementation independently through
    ``REEF_TRAINING_LOSS``. Cookbook deployments should give both the recipe
    and loss family explicit dotted references; Reef never infers one from a
    globally registered method name.

    ``max_staleness`` is the shared stale-sample admission window. Zero keeps
    exact-version training; a positive value admits same-incarnation samples
    up to that lag before the existing training path runs.

    ``candidate_evaluation`` is framework-owned configuration derived from an
    optional top-level ``evaluation`` section. Its dotted factory builds one
    candidate evaluator per scenario, replacing the backend's default
    metrics-only evaluator and selection policy without changing the recipe's
    method-specific config fields.

    Configuration is declared as fields: a config field
    (``batch_size: int = config_field(1, env="REEF_SAO_BATCH_SIZE")``, see
    :mod:`reef.recipe.config_fields`) is simultaneously the YAML key, the
    environment fallback, and the type-aware parser. :meth:`service_config`
    and :meth:`from_environment` derive everything from those fields, so a
    recipe never repeats a setting's name or type anywhere else.
    """

    runtime: TrainingRuntime = field(kw_only=False)
    max_staleness: int = config_field(0, env="REEF_MAX_STALENESS")
    candidate_evaluation: CandidateEvaluationConfig | None = field(default=None, repr=False, compare=False)

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        """Return the processor, preparer, and loss binding for this recipe.

        Concrete weight recipes override this hook. A recipe with bespoke
        trainer wiring may omit ``processor`` and override :meth:`build`.
        """
        return WeightTrainingSpec(step_preparer="", loss_family="")

    def __post_init__(self) -> None:
        if not isinstance(self.max_staleness, int) or isinstance(self.max_staleness, bool) or self.max_staleness < 0:
            raise ValueError("max_staleness must be a non-negative integer")
        runtime_max_staleness = self.runtime.max_staleness
        if runtime_max_staleness != self.max_staleness:
            raise ValueError(
                "max_staleness must match the training runtime; "
                f"recipe={self.max_staleness}, runtime={runtime_max_staleness}"
            )

    def build_surface(self, scenario: str) -> Surface:
        if self.runtime.concurrent_training_scenarios:
            # Every scenario owns an adapter on the shared base: route by
            # scenario and the frozen artifact's publication.
            return create_weight_surface(scenario=scenario)
        return create_weight_surface(adapter_name=self.runtime.serving_adapter_name())

    def serving_status(self) -> Mapping[str, Any] | None:
        """The engine's adapter residency on a runtime that serves per-scenario adapters."""
        report = getattr(self.runtime, "adapter_residency_status", None)
        status = report() if callable(report) else None
        return None if status is None else {"adapters": status}

    @classmethod
    def resolve_training_runtime(cls, runtime: InferenceRuntime | None) -> TrainingRuntime:
        if runtime is None:
            raise RecipeConfigError(
                f"{cls.__name__} requires a training runtime; pass one via the "
                "'runtime' argument (e.g. a RayRuntime injected from your training backend)"
            )
        if not isinstance(runtime, TrainingRuntime):
            raise TypeError(f"{cls.__name__} requires a TrainingRuntime, got {type(runtime).__name__}")
        return runtime

    @classmethod
    def service_config(cls, settings: Mapping[str, Any], *, model_path: str) -> dict[str, Any]:
        """Assemble this recipe's ``from_environment`` config from flat service settings.

        ``settings`` is the recipe-owned slice of the deployment's flat
        ``reef`` section — the caller (``reef.service.assembly``) removes the
        service's own keys first. Only keys the operator actually set are
        forwarded, so every default lives with the recipe's own config fields,
        never in the service layer. Parsing is type-aware per field annotation
        (a float field stays a float). A key this recipe does not declare
        is a loud error: the operator set a value nothing would consume.
        """
        config_fields = recipe_config_fields(cls)
        data: dict[str, Any] = {}
        unknown: list[str] = []
        checkpoint_every: Any = None
        for key, value in settings.items():
            if key == "checkpoint_every_n_versions":
                checkpoint_every = value
            elif key in config_fields:
                # A YAML key left empty (None) is unset, not a value to parse.
                if value is not None:
                    data[key] = config_fields[key].parse(value, f"reef.{key}")
            else:
                unknown.append(key)
        if unknown:
            known = ", ".join(f"reef.{name}" for name in [*sorted(config_fields), "checkpoint_every_n_versions"])
            raise RecipeConfigError(
                f"{cls.__name__} does not consume {', '.join(f'reef.{key}' for key in sorted(unknown))}; "
                f"the recipe would silently run with its defaults instead. "
                f"Settings {cls.__name__} consumes: {known}. "
                f"A new setting is declared with config_field() on the recipe dataclass "
                f"(see reef.recipe.config_fields)."
            )
        config: dict[str, Any] = {"data": data, "model": {"path": model_path}}
        if checkpoint_every is not None:
            config["artifact"] = {
                "checkpoint_every_n_versions": parse_int(checkpoint_every, "reef.checkpoint_every_n_versions")
            }
        return config

    @classmethod
    def _resolve_runtime(cls, values: Mapping[str, str], runtime: InferenceRuntime | None) -> TrainingRuntime:
        return cls.resolve_training_runtime(runtime)

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        cls._validate_config(settings)
        evaluation = settings.get("evaluation")
        if evaluation is None:
            return {}
        if not isinstance(evaluation, Mapping):
            raise RecipeConfigError("evaluation must be an object")
        try:
            return {"candidate_evaluation": CandidateEvaluationConfig.from_dict(evaluation, environ=values)}
        except CandidateEvaluationConfigError as exc:
            raise RecipeConfigError(str(exc)) from exc

    @classmethod
    def _validate_config(cls, settings: Mapping[str, Any]) -> None:
        """Internal hook: reject config sections this recipe refuses to own.

        Config field values never come through here; they resolve from
        their field declarations. This exists only for structural rules such
        as SAO refusing an ``optimization`` section.
        """

    def processor_config(self) -> dict[str, Any]:
        """The config mapping the default :meth:`build` hands the processor.

        Defaults to this recipe's data config field values. The shared
        ``max_staleness`` field configures the runtime and bridge, not data
        retention, so it is not included in processor config. Override this
        method to rename keys or add processor-only entries.
        """
        return {name: getattr(self, name) for name in recipe_config_fields(type(self)) if name != "max_staleness"}

    def build(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, Any] | None = None,
        experiment_logger: ExperimentLogger | None = None,
    ) -> Trainer:
        """Build a trainer from this recipe's processor, preparer, and report contract.

        Override with ``Trainer.build`` for bespoke wiring such as a local
        training backend, and pass :attr:`report_type` through there too.
        """
        return self._build_trainer(
            scenario,
            records,
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
        )

    def _build_trainer(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, Any] | None,
        experiment_logger: ExperimentLogger | None = None,
    ) -> Trainer:
        """Build the shared weight trainer with this recipe's report contract."""
        from reef.train.slime_backend.backend import SlimeTrainingBackend

        spec = type(self).training_spec()
        processor_class = spec.processor
        if processor_class is None:
            raise TypeError(
                f"{type(self).__name__} declares no processor: return a DataProcessor subclass "
                f"as `processor` from training_spec() or override build() for bespoke trainer wiring"
            )
        if not spec.step_preparer:
            raise TypeError(
                f"{type(self).__name__} declares no step_preparer: return a registered preparer name "
                f"(see reef.train.algos) or a dotted 'module:callable' path from training_spec(), "
                f"or override build()"
            )
        # Resolve the preparer now, so a recipe naming an unknown preparer
        # fails at recipe build — before GPUs spin up — instead of at its
        # first training step. Only the resolvability check happens here: the
        # trainer keeps carrying the *name*, because the runtime boundary
        # ships the string to the backend process, which resolves it again in
        # its own registry (``TrainingRuntime.prepare_training_step``). A
        # recipe whose preparer lives outside ``reef.train.algos`` must import
        # that module before calling this build.
        resolve_preparer(spec.step_preparer)
        config = self.processor_config()
        candidate_evaluator = None
        if self.candidate_evaluation is not None:
            candidate_evaluator = build_candidate_evaluation(
                self.candidate_evaluation,
                runtime=self.runtime,
                scenario=scenario,
            )
        return Trainer.build(
            scenario,
            records,
            processor_factory=lambda context: processor_class(context.with_config(config)),
            training_backend=SlimeTrainingBackend(
                self.runtime,
                spec.step_preparer,
                loss_family=spec.loss_family,
                scenario=scenario,
            ),
            candidate_evaluator=candidate_evaluator,
            algorithm_state=algorithm_state,
            report_type=self.report_type,
            experiment_logger=experiment_logger,
        )
