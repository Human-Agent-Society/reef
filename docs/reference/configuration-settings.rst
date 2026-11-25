.. This page is generated. Do not edit it. Change the ``help`` of the setting, then run:
   python .github/scripts/check_settings_reference.py --write

Settings reference
==================

Every setting of a deployment config, with its type, its default and what it does. Each key is also a command-line flag of ``reef serve`` with the same path, such as ``--inference.model-path``. `Configure a deployment <configuration.rst>`__ gives the layout of the file and the value rules.

A key can use hyphens or underscores, except where a table says otherwise. ``unset`` is a key without a value: Reef then uses the behavior that the description gives. An ``object`` holds keys that another table lists, or the options of a backend. A ``plugin`` is a built-in name, a ``package.module:ClassName`` reference, or an object with ``module`` and ``config``.

Service
-------

The HTTP service, its tokens and its scenario limits.

.. container:: config

   ``reef.host`` : string : 127.0.0.1
       HTTP bind address.

   ``reef.port`` : integer : 8900
       HTTP bind port.

   ``reef.tokens`` : list : []
       Accepted bearer tokens as a JSON/YAML list.

   ``reef.allow-unauthenticated`` : boolean : false
       Serve without bearer tokens on an address that other machines can reach. When false, startup refuses a reef.host that is not loopback unless a token is set.

   ``reef.served-url`` : string : unset
       URL at which a composite recipe's own evaluation calls reach this service (default: the bind address, loopback for a wildcard). Set it when the tasks run on another host; other recipes ignore it.

   ``reef.console-origins`` : list : []
       Exact HTTP(S) origins of the browser consoles that may call the service, as a JSON/YAML list, with no path or wildcard. Empty allows no console.

   ``reef.routing-manifest`` : string : unset
       Path of the manifest that routes requests among external targets, for a record-only recipe. Unset turns routing off; reef.routing-deployment holds the router's settings.

   ``reef.allow-implicit-scenario-creation`` : boolean : true
       Let a report or a record import create a scenario that does not exist yet. When false, such a request gets HTTP 404.

   ``reef.scenario-idle-unload-s`` : number : unset
       Unload a scenario that is idle for this many seconds. The next request loads it again. Default: never.

   ``reef.max-loaded-scenarios`` : integer : unset
       Maximum loaded scenarios. At the limit, the idle scenario with the oldest use unloads. Default: no limit.

   ``reef.max-scenarios`` : integer : unset
       Maximum registered scenarios. Default: no limit.

   ``reef.scenario-admission`` : plugin : unset
       ScenarioAdmission plugin reference. The plugin decides whether a scenario is created or loaded, in place of reef.max-loaded-scenarios and reef.max-scenarios; startup refuses it with either.

   ``reef.routing-deployment`` : object : {}
       Semantic Router settings as a JSON/YAML object; reef serve owns the router unless the lifecycle is external.

   ``reef.token`` : string : unset
       One accepted bearer token. The service also accepts each token of reef.tokens.

   ``reef.run-dir`` : string : .reef/run
       Service log directory.

   ``reef.ready-timeout`` : integer : 30
       Readiness deadline of the HTTP service in seconds; with Slime or Reef's serving engines it is at least training.ready-timeout. When set, it is also the deadline of a standalone local engine.

   ``harness.inference-url`` : string : unset
       URL at which clients reach the inference service: a client's harness layer sends its model calls there (default: the address the client gave). Set it when the inference service has its own address.

**reef.routing-deployment.** The Semantic Router of a deployment with ``reef.routing-manifest``.

.. container:: config

   ``reef.routing-deployment.lifecycle`` : string : managed
       Who runs the router: managed (reef serve) or external (beside --service inference). With external, only sr-management-port, selection-timeout-seconds, preview-max-concurrency and finalization-timeout-seconds apply.

   ``reef.routing-deployment.deployment-id`` : string : unset
       Stable local deployment identifier, required by a managed deployment. Use 1 to 48 lowercase letters, digits or hyphens, starting with a letter or digit.

   ``reef.routing-deployment.restart-policy`` : string : fail_stop
       Failure policy: fail_stop or bounded. With bounded, Reef recreates a failed router container up to max-restarts times.

   ``reef.routing-deployment.max-restarts`` : integer : 2
       Recovery attempts allowed during this explicit deployment, at least 1 with a bounded restart policy. A failed attempt counts, and the count never resets.

   ``reef.routing-deployment.restart-backoff-seconds`` : list : [1.0, 2.0]
       Seconds to wait before each recovery attempt, as a list. Later attempts reuse the last value.

   ``reef.routing-deployment.ready-timeout-seconds`` : number : 120
       Seconds allowed for the router to become ready at startup, at recovery and after a policy update.

   ``reef.routing-deployment.drain-timeout-seconds`` : number : 30
       Seconds allowed for accepted requests to drain at a stop. A policy update also waits this long for requests that still select.

   ``reef.routing-deployment.finalization-timeout-seconds`` : number : 5
       Seconds allowed for pending record writes after drain-timeout-seconds ends and Reef cancels the remaining requests.

   ``reef.routing-deployment.container-stop-timeout-seconds`` : number : 10
       Container stop command deadline in seconds.

   ``reef.routing-deployment.image-build-timeout-seconds`` : number : 3600
       Seconds a managed startup may spend building the Semantic Router image it has not cached.

   ``reef.routing-deployment.sr-management-port`` : integer : 50053
       Host loopback port of the router's authenticated HTTP API (preview and configuration). Must differ from reef.port.

   ``reef.routing-deployment.selection-timeout-seconds`` : integer : 10
       Whole seconds, from 1 to 300, that the router may spend selecting for one request (its preview worker deadline).

   ``reef.routing-deployment.preview-max-concurrency`` : integer : 32
       Router preview workers, from 1 to 4096; more concurrent selections are refused with 429. Changes need a restart.

   ``reef.routing-deployment.sr-cpus`` : number : 4
       Semantic Router container CPU limit.

   ``reef.routing-deployment.sr-memory-bytes`` : integer : 4294967296
       Semantic Router container memory limit in bytes.

Storage
-------

The stores of a deployment. A backend key takes a built-in name, or a plugin with ``module`` and ``config``.

.. container:: config

   ``storage.artifact-backend`` : plugin : git_lfs
       Artifact backend: git_lfs, or a ConfigurableRepositoryBackendFactory plugin reference.

   ``storage.artifact-repository`` : string : .reef/artifacts.git
       Git LFS artifact repository location, for storage.artifact-backend git_lfs. Startup refuses another value than the default with a plugin backend.

   ``storage.artifact-work-dir`` : string : .reef/artifact-work
       Artifact working directory.

   ``storage.artifact-cache-dir`` : string : .reef/artifact-cache
       Artifact cache directory.

   ``storage.artifact-sources`` : plugin list : []
       Artifact source reader plugin references, as a JSON/YAML list, beside the hf, file and s3 readers.

   ``storage.agent-record-dir`` : string : .reef/agent-record
       Directory of the SQLite scenario storage: records, commits and settings. The SQLite data store, the SQLite coordination tables and the local segments are also under it.

   ``storage.record-backend`` : plugin : sqlite
       Scenario storage backend: sqlite, postgres, or a ScenarioStorage plugin reference.

   ``storage.data-store`` : plugin : unset
       Data store of the deployment's items: sqlite, postgres, or a DataStore plugin reference. Unset follows storage.record-backend; postgres needs storage.record-backend postgres and uses its URL with the schema reef_data.

   ``storage.coordination-backend`` : string : unset
       Coordination tables of the services: sqlite (one host) or postgres (storage.record-database-url). Unset follows storage.record-backend; a plugin record backend requires this key.

   ``storage.work-max-attempts`` : integer : 3
       Attempts of one work item (a job, a selection, a rollout, a data task) before it fails. An attempt that ends because its service stops does not count.

   ``storage.work-retention-s`` : number : 604800
       Seconds that control keeps an ended work item before it deletes it. Default: 7 days.

   ``storage.job-directory-ttl-s`` : number : 604800
       Seconds that the data store keeps a job data directory that no training job used; a job of the same samples uses it again. Default: 7 days.

   ``storage.task-file-backend`` : plugin : local
       Store of the files of generated tasks: local, or a TaskFileStore plugin reference.

   ``storage.segment-backend`` : plugin : unset
       Store of the segment files that keep trajectory content outside the data store's database: unset (in the database), local (under storage.agent-record-dir), or a SegmentStore plugin reference.

   ``storage.task-file-dir`` : string : .reef/task-files
       Directory of the generated task files, or the local cache of a plugin store.

   ``storage.record-database-url`` : string : unset
       PostgreSQL connection URL. Required when storage.record-backend or storage.coordination-backend is postgres; startup refuses it otherwise.

   ``storage.record-database-schema`` : string : reef_records
       PostgreSQL schema of the scenario storage. The coordination tables use the schema of this name with the suffix _coordination.

   ``storage.agent-record-retention-max-bytes`` : integer : 21474836480
       Maximum bytes of the record items in the data store, for all owners and scenarios, archived ones included. Above it, Reef deletes the oldest record items, consumed or not.

   ``data.max-task-archive-bytes`` : integer : 536870912
       The most bytes of the body of a task archive upload (POST and PUT .../data/tasks). Default: 512 MiB.

**storage.artifact-backend.config.** With ``module: git_lfs``.

.. container:: config

   ``storage.artifact-backend.config.repository`` : string : required
       The Git repository: a local path, created on first use, or a remote URL.

   ``storage.artifact-backend.config.refs`` : string : reef
       Where the refs live: reef keeps them under refs/reef/; branches-and-tags keeps them on branches and tags under reef/, which a Hugging Face repo needs.

   ``storage.artifact-backend.config.token-env`` : string : unset
       The environment variable of a token that authorizes an https:// repository, such as HF_TOKEN.

**storage.record-backend.config.** With ``module: postgres``.

.. container:: config

   ``storage.record-backend.config.database-url`` : string : required
       PostgreSQL connection URL.

   ``storage.record-backend.config.schema`` : string : reef_records
       Dedicated SQL schema of this deployment.

**storage.data-store.config.** With ``module: postgres``.

.. container:: config

   ``storage.data-store.config.database-url`` : string : required
       PostgreSQL connection URL.

   ``storage.data-store.config.schema`` : string : reef_data
       SQL schema of the data store's tables.

**storage.artifact-sources[].config.** With ``module: reef.artifact.sources:S3Reader``.

.. container:: config

   ``storage.artifact-sources[].config.endpoint-url`` : string : unset
       An S3-compatible endpoint; unset is AWS S3.

Inference
---------

The model or provider that answers requests, and the serving engines of a weight-training deployment.

.. container:: config

   ``inference.url`` : string : unset
       Reserved. Startup refuses a value, because Reef starts or names its engines itself.

   ``inference.model-path`` : string : unset
       Local model directory or Hugging Face repository ID.

   ``inference.backend`` : string : unset
       Managed inference engine: sglang (default), vllm, or none. none trains without an inference engine.

   ``inference.tensor-parallel-size`` : integer : unset
       Tensor parallel GPU count per inference engine (default: 1).

   ``inference.num-gpus`` : integer : unset
       Total inference GPUs, or those of each component with inference.remote.slot-urls; defaults to inference.tensor-parallel-size. A positive multiple of that size; standalone and vLLM serving need them equal.

   ``inference.options`` : object : {}
       Native options of the inference engine, by flag name without the leading dashes. Reef refuses the options that it sets itself.

   ``inference.remote`` : object : {}
       Run the engines that inference.backend names on a remote inference component, in place of a local one.

   ``inference.serving`` : string : unset
       How a weight-training deployment serves its snapshots: engines, trainer (the training backend's own serving), none (nothing serves) or a module:Factory runtime. Default: engines when an engine setting is set, else the training backend's default.

   ``inference.serving-config`` : object : {}
       The settings of a module:Factory serving runtime; refused without one.

   ``inference.max-loaded-adapters`` : integer : 1
       Adapters the serving engines keep resident when the snapshots are PEFT adapters. Set at least the number of training scenarios; one more lets a new adapter load beside the served one.

   ``inference.prefix-sharing`` : boolean : false
       Share the engines' prefix cache among requests on the same snapshot; false turns the prefix cache off. The engines finish their requests and clear the cache before each snapshot load.

   ``inference.upstream-url`` : string : unset
       Upstream provider base URL, without the /v1 suffix.

   ``inference.upstream-api-key`` : string : unset
       Upstream provider credential.

   ``inference.upstream-model`` : string : unset
       Model name requested from the upstream provider.

   ``inference.upstream-api`` : string : openai
       Provider API dialect: openai (Chat Completions), responses (OpenAI Responses) or anthropic (Anthropic Messages).

   ``inference.timeout-s`` : number : 300
       Inference request timeout in seconds; a positive number.

   ``inference.handler-factory`` : string : unset
       InferenceHandler subclass as package.module:ClassName. Engine serving accepts only the built-in handler of inference.backend.

   ``inference.handler-config`` : object : {}
       Options of the engines' request handler as a JSON/YAML object: tool_call_parser, sampling_defaults, chat_template_kwargs and force_reasoning. It does not configure the engine process.

   ``inference.retry-initial-s`` : number : 0.05
       Initial retry delay for a routed request, in seconds. Greater than 0 and at most inference.retry-max-s.

   ``inference.retry-max-s`` : number : 1
       Maximum retry delay for a routed request, in seconds.

   ``inference.retry-timeout-s`` : number : ``inference.timeout-s``
       Retry deadline for a routed request, in seconds. Unset takes inference.timeout-s.

**inference.remote.** The remote inference component that runs the engines.

.. container:: config

   ``inference.remote.url`` : string : required
       Base URL of the inference component: HTTP or HTTPS, without credentials, query or fragment.

   ``inference.remote.control-token-env`` : string : REEF_REMOTE_CONTROL_TOKEN
       Environment variable that holds the component's control token. It must differ from inference.remote.inference-token-env.

   ``inference.remote.inference-token-env`` : string : REEF_REMOTE_INFERENCE_TOKEN
       Environment variable that holds the component's inference token. It must differ from inference.remote.control-token-env.

   ``inference.remote.startup-timeout-s`` : number : 3600
       Deadline of one engine start on the component, in seconds, including the model download.

   ``inference.remote.model-path`` : string : unset
       Model that the component loads: a Hugging Face repository ID or a path on the component's host. Unset takes inference.model-path as written.

   ``inference.remote.slot-urls`` : list : []
       Base URLs of further components, each different from inference.remote.url and from each other. Each one serves one more full-weight checkpoint; adapter artifacts refuse it.

Training
--------

The training backend of a weight-training deployment, and the teacher engine.

.. container:: config

   ``training.ray-address`` : string : unset
       Ray cluster address that Slime's training workers and the inference component join; an in-process backend such as tinker refuses it. Unset: the RAY_ADDRESS variable, else Reef starts a local cluster.

   ``training.ray-namespace`` : string : reef
       Ray namespace of the training component.

   ``training.backend`` : string : unset
       Managed weight-training backend: slime, tinker, spindle_full, an installed reef.training_backends entry-point name, or the package.module:ClassName of a TrainingDeployment class. Default: slime.

   ``training.ready-timeout`` : integer : 3600
       Startup deadline in seconds of the inference component and the teacher engine; a positive integer. With Slime or serving engines, reef.ready-timeout is at least this value.

   ``training.options`` : object : {}
       Options owned by the selected training backend. For Slime, they are its native flags without the leading dashes.

   ``training.timeout-s`` : number : unset
       Seconds that one call to Slime's training workers or to the Tinker service may take; a positive number. Unset: 14400 with Slime; Tinker waits as long as its SDK does.

   ``training.max-jobs`` : integer : 1
       The training jobs that one process of the training service runs at once for each training component. Jobs of one scenario run together only on a component that allows it, such as Tinker.

   ``training.manager`` : plugin : unset
       A TrainingManager plugin reference: it decides what the scenarios in auto mode train. Unset: a scenario in auto mode starts a job when job_samples untrained samples wait.

   ``training.config`` : object : {}
       Workload variables that the file's other values reference, such as ${training.config.global_batch_size}. Reef passes none of them to the training backend.

   ``teacher.model-path`` : string : unset
       Teacher engine HF model or directory; frozen SGLang.

   ``teacher.num-gpus`` : integer : 1
       Teacher engine GPUs, also its tensor parallel size.

   ``teacher.port`` : integer : 30001
       Teacher engine's internal HTTP port.

   ``teacher.timeout`` : number : 300
       Deadline of one teacher engine scoring request in seconds.

   ``teacher.options`` : object : {}
       Native SGLang teacher options.

**training.options.** With ``training.backend: tinker`` or ``spindle_full``. Slime takes its own native flags here.

.. container:: config

   ``training.options.state-dir`` : string : required
       Persistent local directory for Tinker checkpoint manifests.

   ``training.options.learning-rate`` : number : 0.0001
       Adam learning rate.

   ``training.options.kl-coef`` : number : 0
       Frozen-base KL coefficient for loss adapters that support it.

   ``training.options.batch-size`` : integer : 1
       Rollouts per optimizer step; the component cuts each job's data into steps of this size.

   ``training.options.api-key-env`` : string : TINKER_API_KEY
       Environment variable containing the Tinker API key.

   ``training.options.base-url`` : string : unset
       Base URL of the Tinker-compatible service that trains and holds the checkpoints, such as a Spindle deployment; unset is Thinking Machines' hosted Tinker.

   ``training.options.project-id`` : string : unset
       Optional Tinker project ID.

   ``training.options.chat-template-kwargs`` : object : unset
       Chat template arguments with which the component renders a recorded trajectory, such as {enable_thinking: false}; set the serving engine's chat_template_kwargs to the same value.

**training.options.** With ``training.backend: tinker``.

.. container:: config

   ``training.options.lora-rank`` : integer : 32
       Rank of the remote LoRA adapter.

   ``training.options.seed`` : integer : 0
       Initial adapter seed; null lets the service choose (Spindle's multi-LoRA trainer refuses a seed).

**training.options.** With ``training.backend: spindle_full``.

.. container:: config

   ``training.options.base`` : string : required
       hf://<repo>@<commit>: the Hugging Face revision that the Spindle full-parameter engine trains.

   ``training.options.release-repo`` : string : required
       org/name: the Hugging Face repo that each saved checkpoint's weights are exported to, one commit each.

   ``training.options.checkpoint-volume`` : string : spindle-checkpoints
       The Modal Volume where the Spindle deployment saves its checkpoints.

   ``training.options.modal-environment`` : string : unset
       The Modal environment of the Spindle deployment; unset is the profile's default.

   ``training.options.hf-secret`` : string : huggingface-secret
       The Modal Secret whose HF_TOKEN may write release_repo and read the base.

   ``training.options.score-centering`` : boolean : false
       Train importance_sampling jobs as Spindle's cispo loss with score centering, and record the sampler's top-k log probabilities in task rollouts.

   ``training.options.score-centering-top-k`` : integer : 128
       Sampler log probabilities per sampled token that score centering reads.

   ``training.options.score-centering-min-tail-mass`` : number : 1e-06
       Floor of the sampler's and the trainer's tail masses before score centering takes their ratio.

   ``training.options.score-centering-weight-low`` : number : 1
       Lower bound of the score-centered token weight clamp(p / q); 1 with a high bound of 1 is REINFORCE.

   ``training.options.score-centering-weight-high`` : number : 1
       Upper bound of the score-centered token weight clamp(p / q).

Recipe
------

The recipe of the deployment, the rule that decides which artifacts serve, and the pipeline stages.

.. container:: config

   ``recipe.implementation`` : string : recipe
       Recipe of the deployment: recipe (records only), a package.module:ClassName, or the name of a YAML preset under REEF_RECIPE_CONFIG_DIR.

   ``activation`` : object : unset
       Activation rule of new artifacts as a JSON/YAML object: module (always, never, evaluated or a package.module:ClassName) and config (review_kinds for always, min_mean_score for evaluated). Omitted is always.

   ``pipeline`` : object : unset
       Pipeline stages as a JSON/YAML object: record2sample and sample2job, each with config and an optional module, a package.module:ClassName. Without module, the stage is the recipe's own class.

**recipe.config.** Every recipe has ``training-mode``.

.. container:: config

   ``recipe.config.training-mode`` : string : auto
       auto starts a job when job_samples untrained samples wait; manual starts a job only for a POST /reef/train request. A request trains in both modes.

   ``recipe.config.job-parameters`` : object : unset
       Job parameters that the recipe sends with every job, over its default parameters; the training component refuses a parameter it does not take.

**recipe.runtime.** With ``recipe.runtime.type: inference_proxy``.

.. container:: config

   ``recipe.runtime.base-url`` : string : empty
       Inference provider base URL.

   ``recipe.runtime.api`` : string : openai
       Provider API format.

   ``recipe.runtime.api-key`` : string : unset
       Provider credential.

   ``recipe.runtime.api-key-env`` : string : unset
       Environment variable containing the provider credential.

   ``recipe.runtime.timeout-s`` : number : 300
       Inference request timeout in seconds.

**recipe.runtime.** With ``recipe.runtime.type: tinker``.

.. container:: config

   ``recipe.runtime.api-key`` : string : unset
       Tinker API key; prefer api_key_env.

   ``recipe.runtime.api-key-env`` : string : TINKER_API_KEY
       Environment variable holding the Tinker API key.

   ``recipe.runtime.project-id`` : string : unset
       Optional Tinker project ID.

   ``recipe.runtime.base-url`` : string : unset
       Base URL of the Tinker-compatible service; unset is Thinking Machines' hosted Tinker.

   ``recipe.runtime.timeout-s`` : number : 300
       Sampling request timeout in seconds.

   ``recipe.runtime.parameterization`` : string : lora
       lora: each snapshot is a LoRA's manifest; full: Spindle's full-parameter snapshots.

**pipeline.sample2job.config.** Every built-in sample-to-job stage that trains on records.

.. container:: config

   ``pipeline.sample2job.config.buffer-size`` : integer : unset
       The most samples that one automatic job takes, beside the rest of a group; a positive integer, at least job-samples. Omitted takes job-samples; on the generated-task stage it takes 4 when that is more.

   ``pipeline.sample2job.config.job-samples`` : integer : unset
       The fewest samples one job trains, a positive integer. Required for weight training; the generated-task stage and a harness evolution stage default it to 1.

**pipeline.sample2job.config.** The stages ``TraceJobData`` and ``TrainTaskJobData`` of harness evolution.

.. container:: config

   ``pipeline.sample2job.config.evaluation-split`` : boolean : false
       Whether each job names the task paths of the scenario's eval split, which a harness step evaluates on. A harness recipe sets it from evolution.evaluation.imported_tasks.

**pipeline.record2sample.config.** The stage ``TrajectorySamples``.

.. container:: config

   ``pipeline.record2sample.config.trajectory-idle-seconds`` : number : 300
       The seconds without a continuing call after which a trajectory ends and trains.

**pipeline.record2sample.config.** The stage ``DistillSamples``.

.. container:: config

   ``pipeline.record2sample.config.accept-multi-turn-policy-samples`` : boolean : false
       Train a report's trajectory of several model calls as one multi-turn sample; otherwise the report trains only the last call of its trajectory.

   ``pipeline.record2sample.config.realign-threshold`` : integer : 1024
       Tokens a multi-turn sample may realign where a later prompt rewrote an earlier response.

   ``pipeline.record2sample.config.scaffold-tolerance`` : integer : 0
       Tokens at the end of a prompt that the next prompt's history may drop, such as a thinking template's scaffold; the sample keeps them, and the response after them trains.

   ``pipeline.record2sample.config.max-teacher-tokens`` : integer : 0
       The longest teacher sequence the trainer scores; a longer sample does not train. 0 is no limit.

**pipeline.record2sample.config.** The stage ``SFTSamples``.

.. container:: config

   ``pipeline.record2sample.config.target`` : string : demonstration
       The response to train: demonstration (the report's teacher_context) or response (the recorded response).

   ``pipeline.record2sample.config.max-tokens`` : integer : 0
       The maximum tokens in a sample; a longer sample does not train. 0 disables the limit.

**pipeline.record2sample.config.** The stage ``GeneratedTaskSamples``.

.. container:: config

   ``pipeline.record2sample.config.rounds`` : integer : 0
       The rounds of task generation; 0 generates nothing. A round counts when its generation finishes or runs out of attempts.

   ``pipeline.record2sample.config.jobs-per-round`` : integer : 1
       The finished task jobs after which the next round starts; 0 starts it when the last round ends.

   ``pipeline.record2sample.config.source-records-per-round`` : integer : 0
       The source sessions that each round takes, at most; 0 generates from the description only.

   ``pipeline.record2sample.config.source-tag`` : string : empty
       The name=value tag that marks a source session: each of its model calls carries the header x-reef-tag-<name>: <value>. Required when source-records-per-round is above 0.

   ``pipeline.record2sample.config.refill-tasks`` : integer : unset
       Without a training pool, the next round starts only while fewer saved tasks than this wait to train; default 1. Startup refuses it when pool-capacity is above 0.

   ``pipeline.record2sample.config.generator`` : object : unset
       The TaskGenerator class (module) and its config. Required with rounds.

   ``pipeline.record2sample.config.description`` : string : empty
       What the tasks are about; every generation request carries it. Required when rounds is above 0.

   ``pipeline.record2sample.config.generator-url`` : string : empty
       The generator service, ${endpoints.generator}. Required when rounds is above 0.

   ``pipeline.record2sample.config.eval-fraction`` : number : 0
       The fraction of new source groups that go to the eval split, from 0 to 1.

   ``pipeline.record2sample.config.split-seed`` : integer : 0
       The fixed seed of the task store's train/eval split.

   ``pipeline.record2sample.config.generation-attempts`` : integer : 3
       The attempts of a generation whose generator or checks fail to run.

   ``pipeline.record2sample.config.generation-deadline-seconds`` : number : 1800
       The seconds that all attempts of one generation may take.

   ``pipeline.record2sample.config.validation-concurrency`` : integer : 1
       The candidates that one generation checks at once.

   ``pipeline.record2sample.config.pool-capacity`` : integer : 0
       Above 0, tasks train again, and rounds start only while fewer tasks than this are active.

   ``pipeline.record2sample.config.review-interval`` : integer : unset
       The task jobs between two pool reviews; 0 reviews nothing; default 0. Every review key needs pool-capacity above 0.

   ``pipeline.record2sample.config.review-window`` : integer : unset
       The most recent rewards of a task that a review reads, at least 2; default 2.

   ``pipeline.record2sample.config.review-band-low`` : number : unset
       A review retires a task whose pass rate is below this; from 0 to review-band-high; default 0.

   ``pipeline.record2sample.config.review-band-high`` : number : unset
       A review retires a task whose pass rate is above this; from review-band-low to 1; default 1.

   ``pipeline.record2sample.config.review-min-reward-std`` : number : unset
       A review retires a task whose rewards have a smaller population standard deviation; 0 turns this off; default 0.

Harness evolution
-----------------

The keys of ``recipe.config.evolution`` for ``reef.recipe.harness:HarnessEvolutionRecipe``.

**recipe.config.evolution.**

.. container:: config

   ``recipe.config.evolution.method`` : plugin : unset
       The Method that evolves the tree. Required: the training service refuses to start without it.

   ``recipe.config.evolution.adapter`` : string : pi
       The harness adapter the tree renders for: pi, opencode, claude, codex, dsh, hermes, native, python, or an adapter of the reef.harness_adapters entry point.

   ``recipe.config.evolution.seed`` : string : unset
       The directory of the first tree, relative to Reef's working directory; none starts empty. Startup refuses a tree that fails the adapter's validation, a symbolic link, and a file that is not UTF-8 text.

   ``recipe.config.evolution.binary`` : string : unset
       The harness binary; none installs the adapter's pinned one.

   ``recipe.config.evolution.models`` : object : {}
       The method's own models by name: url, model, optional api (openai, responses or anthropic), timeout_s, and api_key or api_key_env. The name served is refused.

   ``recipe.config.evolution.model-metadata`` : object : {}
       Model metadata by exact model name: context_window, a positive token count, and reasoning, a boolean. Only the codex adapter reads it, before provider discovery.

   ``recipe.config.evolution.state-dir`` : string : .reef/harness
       The directory of each scenario's history and job records. A relative path resolves against the working directory of the Reef process.

   ``recipe.config.evolution.max-steps`` : integer : 0
       The most jobs before Reef skips automatic jobs; request, skipped and failed jobs count, and a training request still runs. 0 sets no limit.

   ``recipe.config.evolution.max-failure-streak`` : integer : 0
       Skip automatic jobs after this many rejections in a row. 0 sets no limit.

   ``recipe.config.evolution.max-model-calls-per-step`` : integer : 0
       The most model calls of one job. 0 sets no limit.

   ``recipe.config.evolution.max-runs-per-step`` : integer : 0
       The most runs a method's services.evaluate starts in one job; reused runs and the job's own evaluation do not count. 0 sets no limit.

   ``recipe.config.evolution.executor`` : string : local
       Where tasks run: local, sandbox, e2b or modal.

   ``recipe.config.evolution.sandbox`` : object : unset
       The sandbox settings of the run executor: egress_hosts, limits and env_from for sandbox; the e2b or modal options, forward_ports and env_from for a cloud sandbox.

   ``recipe.config.evolution.agent`` : object : unset
       The coding agent a method runs: adapter, binary, sandbox and timeout_s. Unset runs no agent, and startup refuses a method that needs one.

**recipe.config.evolution.evaluation.**

.. container:: config

   ``recipe.config.evolution.evaluation.tasks`` : list : []
       The evaluation tasks: instructions, or task directories for an adapter that takes a task directory.

   ``recipe.config.evolution.evaluation.task-manifest`` : string : unset
       A split manifest whose eval side names the tasks, at least one, as directories under tasks-root. The adapter needs an npm, uv or git install, or must take task directories.

   ``recipe.config.evolution.evaluation.tasks-root`` : string : unset
       The directory of the task manifest's task directories. Set it together with task-manifest.

   ``recipe.config.evolution.evaluation.imported-tasks`` : boolean : false
       Evaluate each job on the eval split of the scenario's imported tasks. The adapter needs an npm, uv or git install, or must take task directories.

   ``recipe.config.evolution.evaluation.scorer`` : plugin : unset
       The TaskScorer that scores each run. Required unless policy is none.

   ``recipe.config.evolution.evaluation.policy`` : string : compare
       The publication policy: compare (the candidate against the current tree), floor (the candidate alone against floor-score) or none (no evaluation).

   ``recipe.config.evolution.evaluation.statistic`` : string : wins
       What compare compares: wins (pairwise) or mean.

   ``recipe.config.evolution.evaluation.min-margin`` : number : 0
       The margin a compare candidate must exceed: wins minus losses, or the mean score difference. It is at least 0.

   ``recipe.config.evolution.evaluation.floor-score`` : number : 1
       The score every floor run must reach. It must be positive.

   ``recipe.config.evolution.evaluation.repeats`` : integer : 1
       The runs per task and tree.

   ``recipe.config.evolution.evaluation.task-timeout-s`` : number : 600
       The seconds one task run may take.

   ``recipe.config.evolution.evaluation.agent-host`` : string : host.docker.internal
       The host name or address at which an agent in a Harbor task's container reaches Reef. It replaces a loopback host of the served model's URL; the port stays.

   ``recipe.config.evolution.evaluation.forbid-residue`` : boolean : false
       Score a run that leaves files behind 0.

   ``recipe.config.evolution.evaluation.collect-failures`` : boolean : false
       Add the prompts of each job's traces to the evaluation tasks. Startup refuses it when the evaluation tasks are task directories.

   ``recipe.config.evolution.evaluation.collector`` : plugin : unset
       The TaskCollector that picks the prompts. Startup refuses it without collect-failures.

   ``recipe.config.evolution.evaluation.max-collected-tasks`` : integer : 50
       The most prompts collection adds in total; 0 adds none.

   ``recipe.config.evolution.evaluation.max-collected-per-client`` : integer : 5
       The most prompts collection adds per tagged client (the x-reef-tag-client tag, else the session tag); 0 sets no per-client limit. Untagged traffic has no per-client limit.

**recipe.config.evolution.method.config.** With ``method.module: reef.train.harness_backend.agent:AgentMethod``.

.. container:: config

   ``recipe.config.evolution.method.config.prompts`` : string : empty
       The method folder: START.md, an optional RESUME.md, the agent's tree.

   ``recipe.config.evolution.method.config.model`` : string : served
       The model the agent runs on: served, or an evolution.models name.

Evaluation
----------

The evaluation service, and the Harbor tasks that each scenario imports.

.. container:: config

   ``evaluation.agent-host`` : string : host.docker.internal
       The address at which a task's container reaches this host, for an evaluation with a harness tree on docker. Other environments do not use it.

   ``evaluation.max-sandboxes`` : integer : 4
       The task sandboxes that one process of the evaluation service holds at once, across all evaluations: one for each rollout. Set it to the provider's quota divided by the evaluation processes.

   ``evaluation.on-stop`` : string : requeue
       requeue or finish: what a stopping evaluation process does with the rollouts that it runs. requeue cancels them and gives them back to the queue; finish takes no new rollout and lets them end.

   ``evaluation.finish-timeout-s`` : number : unset
       With evaluation.on-stop finish, the seconds that a stopping evaluation process lets its rollouts finish before it requeues the rest; unset is no limit.

   ``evaluation.trial-observer`` : plugin : unset
       A TrialObserver plugin reference that reads each evaluation trial before its directory is removed.

   ``evaluation.task-template-builder`` : plugin : unset
       A TaskTemplateBuilder plugin reference that makes the sandbox templates of each task of an e2b evaluation. Unset builds them from the task's Dockerfiles.

   ``evaluation.require-prepared-templates`` : boolean : false
       Run each e2b evaluation rollout only from the templates that its task's template item prepared; a trial then builds no template. A rollout whose template item failed reports that error and runs no trial.

   ``evaluation.pin-task-files`` : boolean : false
       Use the task files selected at evaluation acceptance for all attempts of that evaluation. Startup refuses true with an evaluation.task-template-builder that does not override build_for_version.

   ``task-import.manifest`` : string : unset
       Split manifest of the Harbor tasks that each scenario imports into its samples. Set it together with task-import.tasks-root.

   ``task-import.tasks-root`` : string : unset
       Directory that holds a Harbor task directory for each task name of task-import.manifest. Set both keys or neither.

Generator
---------

The generator service, which writes and checks Harbor tasks and runs their rollouts.

.. container:: config

   ``generator`` : object : unset
       The generator service, as a JSON/YAML object. With this section, reef serve starts the generator before the HTTP service.

**generator.**

.. container:: config

   ``generator.tasks-root`` : string : required
       The directory generated tasks and Harbor job files live under.

   ``generator.host`` : string : 127.0.0.1
       The generator's bind address.

   ``generator.port`` : integer : 8910
       The generator's bind port.

   ``generator.work-dir`` : string : unset
       Where the rollout runner keeps trials; <tasks-root>/.rollouts by default.

   ``generator.agent`` : object : unset
       The Harbor agent each rollout runs, with {model}, {base_url} and {api_key} placeholders; terminus-2 by default.

   ``generator.agent-host`` : string : unset
       An address of this host the task container can reach, for an agent that runs inside it.

   ``generator.harbor`` : string : unset
       The harbor command line for the oracle check; found on PATH by default.

   ``generator.environment`` : string : docker
       The Harbor environment of the checks and rollouts: docker, or e2b or modal on a host without Docker.

   ``generator.concurrency`` : integer : 2
       Rollouts in flight across all rollout requests.

   ``generator.designer-url`` : string : unset
       A Reef service the designer calls go to instead of this deployment's.

   ``generator.designer-token`` : string : unset
       The token for designer-url.

   ``generator.designer-model`` : string : unset
       The served model the designer asks for; the deployment's by default.

   ``generator.designer-timeout-s`` : number : 1800
       Seconds one designer call may take; inference.timeout-s must allow it too.

   ``generator.designer-api`` : string : unset
       The API the designer asks through: openai (Chat Completions) or responses. By default responses when designer calls go to this deployment and its inference.upstream-api is responses, openai otherwise.

   ``generator.designer-options`` : object : unset
       Extra fields of the designer's request, e.g. {"reasoning_effort": "none"}; with designer-api responses, Responses fields such as {"reasoning": {"effort": "low"}}.

   ``generator.ready-timeout`` : integer : 60
       Seconds reef serve waits for the generator to answer.

Executors
---------

``execution.<role>`` selects how Reef starts the workers of a role: ``training`` (Slime's training workers), ``rollout`` (the serving engines), ``generator`` (the generator service) or ``evolution`` (harness evaluation workers). A selector is a backend (a built-in name or ``package.module:ClassName``), the name of a profile under ``executors``, or an object with these keys. A profile takes the same keys.

**execution.<role>.**

.. container:: config

   ``execution.<role>.backend`` : string : auto
       Built-in executor (auto, uni, mp or ray) or package.module:ClassName. With Slime, execution.training accepts only auto or ray.

   ``execution.<role>.options`` : object : {}
       Backend-owned executor options.

   ``execution.<role>.workers`` : integer : unset
       Number of workers, a positive integer; omitted uses the component requirement. Only execution.evolution takes more than 1: training and rollout refuse it.

**execution.<role>.resources.**

.. container:: config

   ``execution.<role>.resources.cpus-per-worker`` : number : unset
       CPU allocation per worker, a finite number of at least 0. Omitted keeps the component's default, normally 1.

   ``execution.<role>.resources.gpus-per-worker`` : number : unset
       GPU allocation per worker, a finite number of at least 0 and not below the component's declared need. Omitted keeps the component's default, normally 0.

Observability
-------------

Experiment tracking and record tracing.

.. container:: config

   ``observability.wandb`` : object : {}
       W&B experiment tracking, as a JSON/YAML object.

   ``observability.tracing`` : object : {}
       Record tracing over OpenTelemetry, as a JSON/YAML object.

**observability.wandb.** These keys take underscores only.

.. container:: config

   ``observability.wandb.enabled`` : boolean : false
       Turn on W&B experiment tracking.

   ``observability.wandb.project`` : string : reef
       W&B project of the runs.

   ``observability.wandb.entity`` : string : unset
       W&B entity (user or team) of the runs. Reef needs it to give the URL of an offline run.

   ``observability.wandb.group_prefix`` : string : unset
       Prefix of each scenario's W&B group. The group is <group_prefix>/<scenario>; without a prefix it is the scenario name.

   ``observability.wandb.name_prefix`` : string : unset
       Start of each run name in place of the scenario name. A run is named <name_prefix>-segment-<N>.

   ``observability.wandb.tags`` : list : []
       W&B tags of every run.

   ``observability.wandb.mode`` : string : disabled
       W&B mode: online, offline (local files for a later wandb sync) or disabled. A section without it is online when enabled is true.

   ``observability.wandb.directory`` : string : unset
       Directory of the local W&B run files. Reef creates it.

   ``observability.wandb.upload_checkpoints`` : boolean : false
       Upload the files of each committed checkpoint as a W&B artifact.

**observability.tracing.** These keys take underscores only.

.. container:: config

   ``observability.tracing.enabled`` : boolean : false
       Turn on record tracing over OTLP/HTTP.

   ``observability.tracing.endpoint`` : string : unset
       Complete OTLP/HTTP traces URL. When unset, the exporter reads the standard OTEL_EXPORTER_OTLP_ENDPOINT variable.

   ``observability.tracing.authorization`` : string : unset
       Value of the Authorization header that the tracing backend expects, such as Bearer <token>. The startup report masks it. Environment fallback: ``REEF_TRACING_AUTHORIZATION``.

   ``observability.tracing.headers`` : object : unset
       Other request headers for the tracing backend, as a mapping of strings. Authorization is refused here; use observability.tracing.authorization.

   ``observability.tracing.service_name`` : string : reef
       The service.name resource attribute of the spans.
