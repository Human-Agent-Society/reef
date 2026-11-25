"""The typed settings of the HTTP process of a service, and the public settings that derive from them.

``ServiceSettings`` holds the HTTP, storage and shared runtime settings that app assembly reads.
``service_config_arguments`` derives the public settings of the ``reef`` section from its fields, and
``service_owned_keys`` names the keys of that section that the service reads. ``reef.deploy.service_config``
reads a deployment configuration into a ``ServiceSettings``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from reef.artifact.repository import ConfigurableRepositoryBackendFactory
from reef.artifact.sources import ArtifactSourceReader
from reef.control.operations.tasks import DEFAULT_MAX_TASK_ARCHIVE_BYTES
from reef.control.scenario.admission import ScenarioAdmission, ScenarioLimits
from reef.core.config import ConfigArgument, config_arguments, config_metadata, config_option
from reef.evaluation.rollout_worker import ON_STOP_CHOICES, check_finish_timeout
from reef.evaluation.task_templates import TaskTemplateBuilder
from reef.evaluation.trials import TrialObserver
from reef.routing.settings import RoutedServiceSettings, RoutingDeploymentSettings
from reef.storage.data.segments import SegmentStore
from reef.storage.data.store import JOB_DIRECTORY_TTL_SECONDS, DataStore
from reef.storage.data.tasks import TaskFileStore
from reef.storage.scenario.records import RecordRetention
from reef.storage.scenario.store import ScenarioStorage
from reef.storage.sql.setup import postgres_url, validate_postgres_schema
from reef.train.manager import TrainingManager


@dataclass(frozen=True)
class ServiceSettings:
    """The HTTP service's settings, translated from a deployment config.

    Recipe-specific config fields (batch sizes, group counts)
    are not fields here: they stay in ``recipe_settings`` — the raw ``reef``
    config section — and each recipe extracts its own via
    ``reef.api.training.training_recipe_config``, so their defaults live with the recipe.

    ``routing_manifest`` is None, which turns routing off, or a non-empty manifest path; the public reader
    takes a blank value as not supplied, and nothing checks the file when the value is made.
    ``routing_deployment`` holds the router's settings; loading a configuration parses and checks
    them even when routing is off, but only a routed deployment acts on them. Starting managed routing also
    needs a deployment ID, a record-only recipe and a loopback host; the controller and app assembly check
    those when they start, not here.
    """

    recipe: str = config_option(
        public_path=("recipe", "implementation"),
        help=(
            "Recipe of the deployment: recipe (records only), a package.module:ClassName, or the name"
            " of a YAML preset under REEF_RECIPE_CONFIG_DIR."
        ),
    )
    host: str = config_option("0.0.0.0", public_path=("reef", "host"), help="HTTP bind address.")
    port: int = config_option(8900, public_path=("reef", "port"), help="HTTP bind port.")
    tokens: tuple[str, ...] = config_option(
        (), public_path=("reef", "tokens"), help="Accepted bearer tokens as a JSON/YAML list."
    )
    allow_unauthenticated: bool = config_option(
        False,
        public_path=("reef", "allow_unauthenticated"),
        help=(
            "Serve without bearer tokens on an address that other machines can reach. When false, "
            "startup refuses a reef.host that is not loopback unless a token is set."
        ),
    )
    served_url: str | None = config_option(
        None,
        public_path=("reef", "served_url"),
        help=(
            "URL at which a composite recipe's own evaluation calls reach this service (default: the "
            "bind address, loopback for a wildcard). Set it when the tasks run on another host; other"
            " recipes ignore it."
        ),
    )
    harness_inference_url: str | None = config_option(
        None,
        public_path=("harness", "inference_url"),
        help=(
            "URL at which clients reach the inference service: a client's harness layer sends its"
            " model calls there (default: the address the client gave). Set it when the inference"
            " service has its own address."
        ),
    )
    console_origins: tuple[str, ...] = config_option(
        (),
        public_path=("reef", "console_origins"),
        help=(
            "Exact HTTP(S) origins of the browser consoles that may call the service, as a JSON/YAML "
            "list, with no path or wildcard. Empty allows no console."
        ),
    )
    routing_manifest: str | None = config_option(
        None,
        public_path=("reef", "routing_manifest"),
        help=(
            "Path of the manifest that routes requests among external targets, for a record-only "
            "recipe. Unset turns routing off; reef.routing-deployment holds the router's settings."
        ),
    )
    routing_deployment: RoutingDeploymentSettings = field(default_factory=RoutingDeploymentSettings)
    ray_address: str | None = config_option(
        None,
        public_path=("training", "ray_address"),
        help=(
            "Ray cluster address that Slime's training workers and the inference component join; an "
            "in-process backend such as tinker refuses it. Unset: the RAY_ADDRESS variable, else Reef starts"
            " a local cluster."
        ),
    )
    ray_namespace: str = config_option(
        "reef", public_path=("training", "ray_namespace"), help="Ray namespace of the training component."
    )
    inference_url: str | None = config_option(
        None,
        public_path=("inference", "url"),
        help="Reserved. Startup refuses a value, because Reef starts or names its engines itself.",
    )
    model_path: str | None = config_option(
        None, public_path=("inference", "model_path"), help="Local model directory or Hugging Face repository ID."
    )
    inference_backend: str | None = config_option(
        None,
        public_path=("inference", "backend"),
        help=("Managed inference engine: sglang (default), vllm, or none. none trains without an inference engine."),
    )
    tensor_parallel_size: int | None = config_option(
        None,
        public_path=("inference", "tensor_parallel_size"),
        help="Tensor parallel GPU count per inference engine (default: 1).",
    )
    inference_num_gpus: int | None = config_option(
        None,
        public_path=("inference", "num_gpus"),
        help=(
            "Total inference GPUs, or those of each component with inference.remote.slot-urls; "
            "defaults to inference.tensor-parallel-size. A positive multiple of that size; "
            "standalone and vLLM serving need them equal."
        ),
    )
    inference_options: Mapping[str, Any] = field(
        default_factory=dict,
        metadata=config_metadata(
            (
                "Native options of the inference engine, by flag name without the leading dashes. Reef "
                "refuses the options that it sets itself."
            ),
            public_path=("inference", "options"),
        ),
    )
    inference_remote: Mapping[str, Any] = field(
        default_factory=dict,
        metadata=config_metadata(
            (
                "Run the engines that inference.backend names on a remote inference component, in "
                "place of a local one."
            ),
            public_path=("inference", "remote"),
        ),
    )
    inference_serving: str | None = config_option(
        None,
        public_path=("inference", "serving"),
        help=(
            "How a weight-training deployment serves its snapshots: engines, trainer (the "
            "training backend's own serving), none (nothing serves) or a module:Factory runtime. "
            "Default: engines when an engine setting is set, else the training backend's default."
        ),
    )
    inference_serving_config: Mapping[str, Any] = field(
        default_factory=dict,
        metadata=config_metadata(
            "The settings of a module:Factory serving runtime; refused without one.",
            public_path=("inference", "serving_config"),
        ),
    )
    inference_max_loaded_adapters: int = config_option(
        1,
        public_path=("inference", "max_loaded_adapters"),
        help=(
            "Adapters the serving engines keep resident when the snapshots are PEFT adapters. Set at "
            "least the number of training scenarios; one more lets a new adapter load beside the "
            "served one."
        ),
    )
    inference_prefix_sharing: bool = config_option(
        False,
        public_path=("inference", "prefix_sharing"),
        help=(
            "Share the engines' prefix cache among requests on the same snapshot; false turns the"
            " prefix cache off. The engines finish their requests and clear the cache before each"
            " snapshot load."
        ),
    )
    teacher_model_path: str | None = config_option(
        None, public_path=("teacher", "model_path"), help="Teacher engine HF model or directory; frozen SGLang."
    )
    teacher_num_gpus: int = config_option(
        1, public_path=("teacher", "num_gpus"), help="Teacher engine GPUs, also its tensor parallel size."
    )
    teacher_port: int = config_option(
        30001, public_path=("teacher", "port"), help="Teacher engine's internal HTTP port."
    )
    teacher_timeout: float = config_option(
        300.0, public_path=("teacher", "timeout"), help="Deadline of one teacher engine scoring request in seconds."
    )
    teacher_options: Mapping[str, Any] = field(
        default_factory=dict,
        metadata=config_metadata("Native SGLang teacher options.", public_path=("teacher", "options")),
    )
    training_backend: str | None = config_option(
        None,
        public_path=("training", "backend"),
        help=(
            "Managed weight-training backend: slime, tinker, spindle_full, an installed "
            "reef.training_backends entry-point name, or the package.module:ClassName of a "
            "TrainingDeployment class. Default: slime."
        ),
    )
    training_ready_timeout: int = config_option(
        3600,
        public_path=("training", "ready_timeout"),
        help=(
            "Startup deadline in seconds of the inference component and the teacher engine; a "
            "positive integer. With Slime or serving engines, reef.ready-timeout is at least this "
            "value."
        ),
    )
    training_backend_options: Mapping[str, Any] = field(
        default_factory=dict,
        metadata=config_metadata(
            (
                "Options owned by the selected training backend. For Slime, they are its native flags"
                " without the leading dashes."
            ),
            public_path=("training", "options"),
        ),
    )
    #: The OpenAI-compatible provider no-update recipes proxy to (no ``/v1``
    #: suffix), its credential, and the model name to request from it. The
    #: only place the upstream is named: the HTTP service forwards to it, and
    #: the training side derives the model binding it hands to methods and
    #: evaluation rollouts from it. ``upstream_model`` is a provider model
    #: name, unlike ``model_path``, which is local weights for training.
    upstream_url: str | None = config_option(
        None, public_path=("inference", "upstream_url"), help="Upstream provider base URL, without the /v1 suffix."
    )
    upstream_api_key: str | None = config_option(
        None, public_path=("inference", "upstream_api_key"), help="Upstream provider credential."
    )
    upstream_model: str | None = config_option(
        None, public_path=("inference", "upstream_model"), help="Model name requested from the upstream provider."
    )
    #: The provider's API dialect: ``openai`` (default), ``responses``, or ``anthropic``.
    upstream_api: str = config_option(
        "openai",
        public_path=("inference", "upstream_api"),
        help=(
            "Provider API dialect: openai (Chat Completions), responses (OpenAI Responses) or "
            "anthropic (Anthropic Messages)."
        ),
    )
    inference_timeout_s: float = config_option(
        300.0, public_path=("inference", "timeout_s"), help="Inference request timeout in seconds; a positive number."
    )
    train_timeout_s: float | None = config_option(
        None,
        public_path=("training", "timeout_s"),
        help=(
            "Seconds that one call to Slime's training workers or to the Tinker service may take; a "
            "positive number. Unset: 14400 with Slime; Tinker waits as long as its SDK does."
        ),
    )
    #: The jobs of one training component that one process of the training service runs at the same time.
    training_max_jobs: int = config_option(
        1,
        public_path=("training", "max_jobs"),
        help=(
            "The training jobs that one process of the training service runs at once for each "
            "training component. Jobs of one scenario run together only on a component that allows "
            "it, such as Tinker."
        ),
    )
    #: Unset, or a ``TrainingManager`` plugin: ``package.module:ClassName`` or module and config.
    training_manager: str | Mapping[str, Any] | None = config_option(
        None,
        public_path=("training", "manager"),
        help=(
            "A TrainingManager plugin reference: it decides what the scenarios in auto mode train. "
            "Unset: a scenario in auto mode starts a job when job_samples untrained samples wait."
        ),
    )
    inference_handler_factory: str | None = config_option(
        None,
        public_path=("inference", "handler_factory"),
        help=(
            "InferenceHandler subclass as package.module:ClassName. Engine serving accepts only the "
            "built-in handler of inference.backend."
        ),
    )
    inference_handler_config: Mapping[str, Any] = field(
        default_factory=dict,
        metadata=config_metadata(
            (
                "Options of the engines' request handler as a JSON/YAML object: tool_call_parser, "
                "sampling_defaults, chat_template_kwargs and force_reasoning. It does not configure "
                "the engine process."
            ),
            public_path=("inference", "handler_config"),
        ),
    )
    inference_retry_initial_s: float = config_option(
        0.05,
        public_path=("inference", "retry_initial_s"),
        help=(
            "Initial retry delay for a routed request, in seconds. Greater than 0 and at most "
            "inference.retry-max-s."
        ),
    )
    inference_retry_max_s: float = config_option(
        1.0, public_path=("inference", "retry_max_s"), help="Maximum retry delay for a routed request, in seconds."
    )
    inference_retry_timeout_s: float = config_option(
        300.0,
        public_path=("inference", "retry_timeout_s"),
        help="Retry deadline for a routed request, in seconds. Unset takes inference.timeout-s.",
    )
    #: ``git_lfs``, or a ``ConfigurableRepositoryBackendFactory`` plugin: ``package.module:ClassName`` or module
    #: and config.
    artifact_backend: str | Mapping[str, Any] = config_option(
        "git_lfs",
        public_path=("storage", "artifact_backend"),
        help="Artifact backend: git_lfs, or a ConfigurableRepositoryBackendFactory plugin reference.",
    )
    artifact_repository: str = config_option(
        ".reef/artifacts.git",
        public_path=("storage", "artifact_repository"),
        help=(
            "Git LFS artifact repository location, for storage.artifact-backend git_lfs. Startup "
            "refuses another value than the default with a plugin backend."
        ),
    )
    artifact_work_dir: str = config_option(
        ".reef/artifact-work", public_path=("storage", "artifact_work_dir"), help="Artifact working directory."
    )
    artifact_cache_dir: str = config_option(
        ".reef/artifact-cache", public_path=("storage", "artifact_cache_dir"), help="Artifact cache directory."
    )
    #: ``ArtifactSourceReader`` plugins beside the built-in ``hf``, ``file`` and ``s3`` readers of artifact imports.
    artifact_sources: tuple[str | Mapping[str, Any], ...] = config_option(
        (),
        public_path=("storage", "artifact_sources"),
        help="Artifact source reader plugin references, as a JSON/YAML list, beside the hf, file and s3 readers.",
    )
    #: The address at which a task's container reaches this host, for an evaluation with a harness tree.
    evaluation_agent_host: str = config_option(
        "host.docker.internal",
        public_path=("evaluation", "agent_host"),
        help=(
            "The address at which a task's container reaches this host, for an evaluation with a "
            "harness tree on docker. Other environments do not use it."
        ),
    )
    #: The rollouts that one process of the evaluation service runs at once, across evaluations. Each one holds a
    #: sandbox of its task.
    evaluation_max_sandboxes: int = config_option(
        4,
        public_path=("evaluation", "max_sandboxes"),
        help=(
            "The task sandboxes that one process of the evaluation service holds at once, across "
            "all evaluations: one for each rollout. Set it to the provider's quota divided by the"
            " evaluation processes."
        ),
    )
    #: What a stopping process of the evaluation service does with the rollouts that it runs (``ON_STOP_CHOICES`` of
    #: ``reef.evaluation.rollout_worker``): ``requeue`` gives them back to the queue at once, ``finish`` finishes them.
    evaluation_on_stop: str = config_option(
        "requeue",
        public_path=("evaluation", "on_stop"),
        help=(
            "requeue or finish: what a stopping evaluation process does with the rollouts that it "
            "runs. requeue cancels them and gives them back to the queue; finish takes no new rollout"
            " and lets them end."
        ),
    )
    #: With ``evaluation_on_stop`` ``finish``: the seconds from the start of a stop to the cancellation of the rollouts
    #: that still run, which go back to the queue. None waits for every rollout.
    evaluation_finish_timeout_s: float | None = config_option(
        None,
        public_path=("evaluation", "finish_timeout_s"),
        help=(
            "With evaluation.on-stop finish, the seconds that a stopping evaluation process lets "
            "its rollouts finish before it requeues the rest; unset is no limit."
        ),
    )
    #: Unset, or a ``TrialObserver`` plugin: ``package.module:ClassName`` or module and config.
    evaluation_trial_observer: str | Mapping[str, Any] | None = config_option(
        None,
        public_path=("evaluation", "trial_observer"),
        help="A TrialObserver plugin reference that reads each evaluation trial before its directory is removed.",
    )
    #: Unset, or a ``TaskTemplateBuilder`` plugin: ``package.module:ClassName`` or module and config.
    evaluation_task_template_builder: str | Mapping[str, Any] | None = config_option(
        None,
        public_path=("evaluation", "task_template_builder"),
        help=(
            "A TaskTemplateBuilder plugin reference that makes the sandbox templates of each task of "
            "an e2b evaluation. Unset builds them from the task's Dockerfiles."
        ),
    )
    #: Whether a rollout of an e2b evaluation runs a trial only from its task's prepared templates: a task whose
    #: template item failed runs no trial, and a trial builds no missing template.
    evaluation_require_prepared_templates: bool = config_option(
        False,
        public_path=("evaluation", "require_prepared_templates"),
        help=(
            "Run each e2b evaluation rollout only from the templates that its task's template "
            "item prepared; a trial then builds no template. A rollout whose template item failed"
            " reports that error and runs no trial."
        ),
    )
    evaluation_pin_task_files: bool = config_option(
        False,
        public_path=("evaluation", "pin_task_files"),
        help=(
            "Use the task files selected at evaluation acceptance for all attempts of that "
            "evaluation. Startup refuses true with an evaluation.task-template-builder that does not "
            "override build_for_version."
        ),
    )
    agent_record_dir: str = config_option(
        ".reef/agent-record",
        public_path=("storage", "agent_record_dir"),
        help=(
            "Directory of the SQLite scenario storage: records, commits and settings. The SQLite data"
            " store, the SQLite coordination tables and the local segments are also under it."
        ),
    )
    #: ``sqlite``, ``postgres``, or a ``ScenarioStorage`` plugin: ``package.module:ClassName`` or module and config.
    record_backend: str | Mapping[str, Any] = config_option(
        "sqlite",
        public_path=("storage", "record_backend"),
        help="Scenario storage backend: sqlite, postgres, or a ScenarioStorage plugin reference.",
    )
    #: ``sqlite``, ``postgres``, or a ``DataStore`` plugin: ``package.module:ClassName`` or module and config.
    #: Unset follows ``record_backend``: ``postgres`` uses ``record_database_url`` and the schema ``reef_data``, and
    #: a plugin record backend uses ``sqlite``.
    data_store: str | Mapping[str, Any] | None = config_option(
        None,
        public_path=("storage", "data_store"),
        help=(
            "Data store of the deployment's items: sqlite, postgres, or a DataStore plugin reference."
            " Unset follows storage.record-backend; postgres needs storage.record-backend postgres "
            "and uses its URL with the schema reef_data."
        ),
    )
    #: ``sqlite`` or ``postgres``: where the coordination tables (served models, work queues, the control lease) live.
    #: Unset follows a built-in ``record_backend``; a plugin record backend must name it, because every process of
    #: the deployment must share these tables. ``postgres`` uses ``record_database_url``.
    coordination_backend: str | None = config_option(
        None,
        public_path=("storage", "coordination_backend"),
        help=(
            "Coordination tables of the services: sqlite (one host) or postgres "
            "(storage.record-database-url). Unset follows storage.record-backend; a plugin record backend requires "
            "this key."
        ),
    )
    #: How many times a worker takes one work item before the item fails, for every queue. An attempt that ends
    #: because its worker stops, such as at a restart of the service, does not count.
    work_max_attempts: int = config_option(
        3,
        public_path=("storage", "work_max_attempts"),
        help=(
            "Attempts of one work item (a job, a selection, a rollout, a data task) before it fails. "
            "An attempt that ends because its service stops does not count."
        ),
    )
    #: How long control keeps an ended work item, so that its producer reads its result; then control deletes it.
    work_retention_s: float = config_option(
        7 * 24 * 3600.0,
        public_path=("storage", "work_retention_s"),
        help="Seconds that control keeps an ended work item before it deletes it. Default: 7 days.",
    )
    job_directory_ttl_s: float = config_option(
        JOB_DIRECTORY_TTL_SECONDS,
        public_path=("storage", "job_directory_ttl_s"),
        help=(
            "Seconds that the data store keeps a job data directory that no training job used; a job of the same "
            "samples uses it again. Default: 7 days."
        ),
    )
    #: ``local``, or a ``TaskFileStore`` plugin: ``package.module:ClassName`` or module and config.
    task_file_backend: str | Mapping[str, Any] = config_option(
        "local",
        public_path=("storage", "task_file_backend"),
        help="Store of the files of generated tasks: local, or a TaskFileStore plugin reference.",
    )
    #: Unset, ``local``, or a ``SegmentStore`` plugin: ``package.module:ClassName`` or module and config.
    segment_backend: str | Mapping[str, Any] | None = config_option(
        None,
        public_path=("storage", "segment_backend"),
        help=(
            "Store of the segment files that keep trajectory content outside the data store's "
            "database: unset (in the database), local (under storage.agent-record-dir), or a "
            "SegmentStore plugin reference."
        ),
    )
    task_file_dir: str = config_option(
        ".reef/task-files",
        public_path=("storage", "task_file_dir"),
        help="Directory of the generated task files, or the local cache of a plugin store.",
    )
    #: The most bytes of the body of a task archive that the data routes take; a larger task uses a create by
    #: reference. The body goes to a local file, so the limit bounds the disk that one upload uses.
    max_task_archive_bytes: int = config_option(
        DEFAULT_MAX_TASK_ARCHIVE_BYTES,
        public_path=("data", "max_task_archive_bytes"),
        help="The most bytes of the body of a task archive upload (POST and PUT .../data/tasks). Default: 512 MiB.",
    )
    task_import_manifest: str | None = config_option(
        None,
        public_path=("task_import", "manifest"),
        help=(
            "Split manifest of the Harbor tasks that each scenario imports into its samples. Set it "
            "together with task-import.tasks-root."
        ),
    )
    task_import_tasks_root: str | None = config_option(
        None,
        public_path=("task_import", "tasks_root"),
        help=(
            "Directory that holds a Harbor task directory for each task name of task-import.manifest."
            " Set both keys or neither."
        ),
    )
    record_database_url: str | None = field(
        default=None,
        repr=False,
        metadata=config_metadata(
            (
                "PostgreSQL connection URL. Required when storage.record-backend or "
                "storage.coordination-backend is postgres; startup refuses it otherwise."
            ),
            public_path=("storage", "record_database_url"),
        ),
    )
    record_database_schema: str = config_option(
        "reef_records",
        public_path=("storage", "record_database_schema"),
        help=(
            "PostgreSQL schema of the scenario storage. The coordination tables use the schema of "
            "this name with the suffix _coordination."
        ),
    )
    agent_record_retention_max_bytes: int = config_option(
        20 * 1024**3,
        public_path=("storage", "agent_record_retention_max_bytes"),
        help=(
            "Maximum bytes of the record items in the data store, for all owners and scenarios, "
            "archived ones included. Above it, Reef deletes the oldest record items, consumed or "
            "not."
        ),
    )
    allow_implicit_scenario_creation: bool = config_option(
        True,
        public_path=("reef", "allow_implicit_scenario_creation"),
        help=(
            "Let a report or a record import create a scenario that does not exist yet. When false, "
            "such a request gets HTTP 404."
        ),
    )
    scenario_idle_unload_s: float | None = config_option(
        None,
        public_path=("reef", "scenario_idle_unload_s"),
        help="Unload a scenario that is idle for this many seconds. The next request loads it again. Default: never.",
    )
    max_loaded_scenarios: int | None = config_option(
        None,
        public_path=("reef", "max_loaded_scenarios"),
        help="Maximum loaded scenarios. At the limit, the idle scenario with the oldest use unloads. Default: no limit.",
    )
    max_scenarios: int | None = config_option(
        None, public_path=("reef", "max_scenarios"), help="Maximum registered scenarios. Default: no limit."
    )
    #: A ``ScenarioAdmission`` plugin that replaces the built-in limits. Give ``package.module:ClassName``, or
    #: a mapping with module and config.
    scenario_admission: str | Mapping[str, Any] | None = config_option(
        None,
        public_path=("reef", "scenario_admission"),
        help=(
            "ScenarioAdmission plugin reference. The plugin decides whether a scenario is created or "
            "loaded, in place of reef.max-loaded-scenarios and reef.max-scenarios; startup refuses it"
            " with either."
        ),
    )
    #: Deployment-level experiment provider settings, sourced from
    #: ``observability.wandb``.
    wandb_config: Mapping[str, Any] = field(
        default_factory=dict,
        metadata=config_metadata("W&B experiment tracking, as a JSON/YAML object.", path=("observability", "wandb")),
    )
    #: OpenTelemetry record tracing settings, sourced from ``observability.tracing``.
    tracing_config: Mapping[str, Any] = field(
        default_factory=dict,
        metadata=config_metadata(
            "Record tracing over OpenTelemetry, as a JSON/YAML object.", path=("observability", "tracing")
        ),
    )
    training_settings: Mapping[str, Any] = field(
        default_factory=dict,
        metadata=config_metadata(
            (
                "Workload variables that the file's other values reference, such as "
                "${training.config.global_batch_size}. Reef passes none of them to the training "
                "backend."
            ),
            path=("training",),
            public_path=("training", "config"),
        ),
    )
    #: The rule that activates the artifacts that training makes, for any recipe.
    activation_settings: Mapping[str, Any] | None = field(
        default=None,
        metadata=config_metadata(
            (
                "Activation rule of new artifacts as a JSON/YAML object: module (always, never, evaluated"
                " or a package.module:ClassName) and config (review_kinds for always, min_mean_score for "
                "evaluated). Omitted is always."
            ),
            path=("activation",),
        ),
    )
    #: The stage classes that replace a weight-training recipe's own: ``record2sample`` and ``sample2job``.
    pipeline_settings: Mapping[str, Any] | None = field(
        default=None,
        metadata=config_metadata(
            (
                "Pipeline stages as a JSON/YAML object: record2sample and sample2job, each with "
                "config and an optional module, a package.module:ClassName. Without module, the stage"
                " is the recipe's own class."
            ),
            path=("pipeline",),
        ),
    )
    #: The generator service ``reef serve`` starts beside the HTTP service; ``reef.deploy.generator`` parses it.
    generator_settings: Mapping[str, Any] | None = field(
        default=None,
        metadata=config_metadata(
            (
                "The generator service, as a JSON/YAML object. With this section, reef serve starts the "
                "generator before the HTTP service."
            ),
            path=("generator",),
        ),
    )
    #: The flat ``reef`` config section, interpolated; recipes read their own
    #: config fields from it (see the class docstring).
    recipe_settings: Mapping[str, Any] = field(default_factory=dict, repr=False)
    preset_config: Mapping[str, Any] | None = field(default=None, repr=False)
    #: The deployment's ``execution`` and ``executors`` sections, which select each role's executor.
    #: Its keys are a subset of ``{"execution", "executors"}``; a training integration reads it.
    execution_config: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.routing_manifest is not None and (
            not isinstance(self.routing_manifest, str) or not self.routing_manifest
        ):
            raise ValueError("reef.routing_manifest must be a non-empty path")
        RecordRetention(max_bytes=self.agent_record_retention_max_bytes)
        if not isinstance(self.execution_config, Mapping):
            raise ValueError("ServiceSettings.execution_config must be a mapping")
        unknown_execution_keys = set(self.execution_config) - {"execution", "executors"}
        if unknown_execution_keys:
            raise ValueError(
                "ServiceSettings.execution_config holds only the execution and executors sections, got "
                + ", ".join(sorted(map(str, unknown_execution_keys)))
            )
        if self.scenario_idle_unload_s is not None and self.scenario_idle_unload_s <= 0:
            raise ValueError("reef.scenario_idle_unload_s must be positive")
        adapters = self.inference_max_loaded_adapters
        if isinstance(adapters, bool) or not isinstance(adapters, int) or adapters <= 0:
            raise ValueError("inference.max-loaded-adapters must be a positive int")
        if not isinstance(self.inference_prefix_sharing, bool):
            raise ValueError("inference.prefix-sharing must be true or false")
        if self.inference_serving is not None and (
            not isinstance(self.inference_serving, str) or not self.inference_serving.strip()
        ):
            raise ValueError("inference.serving must be a non-empty string")
        if not isinstance(self.inference_serving_config, Mapping):
            raise ValueError("inference.serving-config must be an object")
        # The built-in admission checks its own limits.
        ScenarioLimits(max_loaded_scenarios=self.max_loaded_scenarios, max_scenarios=self.max_scenarios)
        if self.scenario_admission is not None:
            if self.max_loaded_scenarios is not None or self.max_scenarios is not None:
                raise ValueError(
                    "reef.max_loaded_scenarios and reef.max_scenarios configure the built-in admission; "
                    "a reef.scenario_admission plugin reads its own config"
                )
            ScenarioAdmission.reference(self.scenario_admission, setting="reef.scenario_admission")
        if self.artifact_backend != "git_lfs":
            if isinstance(self.artifact_backend, str) and ":" not in self.artifact_backend:
                raise ValueError("storage.artifact_backend must be git_lfs or 'package.module:ClassName'")
            ConfigurableRepositoryBackendFactory.reference(self.artifact_backend, setting="storage.artifact_backend")
            if self.artifact_repository != ".reef/artifacts.git":
                raise ValueError(
                    "storage.artifact_repository requires storage.artifact_backend: git_lfs; "
                    "a plugin backend reads its own config"
                )
        for reader in self.artifact_sources:
            ArtifactSourceReader.reference(reader, setting="storage.artifact_sources")
        if self.record_backend not in ("sqlite", "postgres"):
            if isinstance(self.record_backend, str) and ":" not in self.record_backend:
                raise ValueError("storage.record_backend must be sqlite, postgres, or 'package.module:ClassName'")
            ScenarioStorage.reference(self.record_backend, setting="storage.record_backend")
        if self.training_manager is not None:
            TrainingManager.reference(self.training_manager, setting="training.manager")
        max_jobs = self.training_max_jobs
        if isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs < 1:
            raise ValueError("training.max_jobs must be a positive integer")
        sandboxes = self.evaluation_max_sandboxes
        if isinstance(sandboxes, bool) or not isinstance(sandboxes, int) or sandboxes < 1:
            raise ValueError("evaluation.max_sandboxes must be a positive integer")
        if self.evaluation_on_stop not in ON_STOP_CHOICES:
            raise ValueError(f"evaluation.on_stop must be one of {', '.join(ON_STOP_CHOICES)}")
        check_finish_timeout(self.evaluation_on_stop, self.evaluation_finish_timeout_s, prefix="evaluation.")
        if self.evaluation_trial_observer is not None:
            TrialObserver.reference(self.evaluation_trial_observer, setting="evaluation.trial_observer")
        if not isinstance(self.evaluation_require_prepared_templates, bool):
            raise ValueError("evaluation.require_prepared_templates must be true or false")
        if not isinstance(self.evaluation_pin_task_files, bool):
            raise ValueError("evaluation.pin_task_files must be true or false")
        if self.evaluation_task_template_builder is not None:
            reference = TaskTemplateBuilder.reference(
                self.evaluation_task_template_builder, setting="evaluation.task_template_builder"
            )
            builder = TaskTemplateBuilder.referenced_class(reference)
            if self.evaluation_pin_task_files and not builder.supports_pinned_versions():
                raise ValueError("evaluation.pin_task_files needs a template builder that overrides build_for_version")
        if isinstance(self.work_max_attempts, bool) or not isinstance(self.work_max_attempts, int):
            raise ValueError("storage.work_max_attempts must be an integer")
        if self.work_max_attempts < 1:
            raise ValueError("storage.work_max_attempts must be at least 1")
        if not math.isfinite(self.work_retention_s) or self.work_retention_s <= 0:
            raise ValueError("storage.work_retention_s must be a positive number of seconds")
        if not math.isfinite(self.job_directory_ttl_s) or self.job_directory_ttl_s <= 0:
            raise ValueError("storage.job_directory_ttl_s must be a positive number of seconds")
        if self.coordination_backend not in (None, "sqlite", "postgres"):
            raise ValueError("storage.coordination_backend must be sqlite or postgres")
        if self.coordination_backend is None and self.record_backend not in ("sqlite", "postgres"):
            raise ValueError(
                "storage.record_backend is a plugin, so name where the services share their coordination tables: "
                "set storage.coordination_backend to sqlite (processes on one host) or postgres"
            )
        if self.data_store not in (None, "sqlite", "postgres"):
            if isinstance(self.data_store, str) and ":" not in self.data_store:
                raise ValueError("storage.data_store must be sqlite, postgres, or 'package.module:ClassName'")
            DataStore.reference(self.data_store, setting="storage.data_store")
        if self.data_store == "postgres" and self.record_backend != "postgres":
            raise ValueError(
                "storage.data_store: postgres uses storage.record_database_url, so it needs record_backend: postgres"
            )
        if self.segment_backend not in (None, "local"):
            if isinstance(self.segment_backend, str) and ":" not in self.segment_backend:
                raise ValueError("storage.segment_backend must be local or 'package.module:ClassName'")
            SegmentStore.reference(self.segment_backend, setting="storage.segment_backend")
        if self.task_file_backend != "local":
            if isinstance(self.task_file_backend, str) and ":" not in self.task_file_backend:
                raise ValueError("storage.task_file_backend must be local or 'package.module:ClassName'")
            TaskFileStore.reference(self.task_file_backend, setting="storage.task_file_backend")
        archive_bytes = self.max_task_archive_bytes
        if isinstance(archive_bytes, bool) or not isinstance(archive_bytes, int) or archive_bytes < 1:
            raise ValueError("data.max_task_archive_bytes must be a positive integer")
        if (self.task_import_manifest is None) != (self.task_import_tasks_root is None):
            raise ValueError("task-import needs both manifest and tasks-root")
        if self.task_import_manifest == "" or self.task_import_tasks_root == "":
            raise ValueError("task-import.manifest and task-import.tasks-root must be non-empty paths")
        if "postgres" in (self.record_backend, self.coordination_backend):
            if not isinstance(self.record_database_url, str) or not self.record_database_url:
                raise ValueError(
                    "storage.record_database_url is required for a postgres record or coordination backend"
                )
            postgres_url(self.record_database_url)
            validate_postgres_schema(self.record_database_schema)
        elif self.record_database_url is not None or self.record_database_schema != "reef_records":
            raise ValueError(
                "record_database_url and record_database_schema require storage.record_backend or "
                "storage.coordination_backend: postgres"
            )

    def routed_service(self) -> RoutedServiceSettings:
        """The settings that the controller of a routing deployment reads."""
        return RoutedServiceSettings(
            routing_manifest=self.routing_manifest,
            host=self.host,
            port=self.port,
            has_tokens=bool(self.tokens),
            inference_timeout_seconds=self.inference_timeout_s,
        )


#: ``reef.*`` keys the service consumes under a different field name. The
#: service's vocabulary is ``ServiceSettings``' fields plus these, so the
#: recipe-owned remainder of the section never includes them.
SERVICE_CONFIG_ALIASES: Mapping[str, str] = {"token": "tokens"}


def service_owned_keys() -> frozenset[str]:
    """Every ``reef.*`` key the service layer consumes."""
    non_reef_fields = {
        "activation_settings",
        "generator_settings",
        "pipeline_settings",
        "tracing_config",
        "training_settings",
        "wandb_config",
    }
    return frozenset(
        settings_field.name
        for settings_field in dataclasses.fields(ServiceSettings)
        if settings_field.name not in non_reef_fields
    ) | frozenset(SERVICE_CONFIG_ALIASES)


@lru_cache(maxsize=1)
def service_config_arguments() -> tuple[ConfigArgument, ...]:
    """Public settings derive their types and defaults from ServiceSettings."""
    return (
        *config_arguments(ServiceSettings),
        ConfigArgument(
            "routing_deployment",
            ("reef", "routing_deployment"),
            "object",
            False,
            {},
            "Semantic Router settings as a JSON/YAML object; reef serve owns the router unless the lifecycle is "
            "external.",
            public_path=("reef", "routing_deployment"),
        ),
        ConfigArgument(
            "token",
            ("reef", "token"),
            "str",
            True,
            None,
            "One accepted bearer token. The service also accepts each token of reef.tokens.",
            public_path=("reef", "token"),
        ),
    )
