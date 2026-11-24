Harness evolution keys
======================

The keys of ``recipe.config.evolution``: the method, the adapter, the limits of a job, the run executor, the coding agent, and how Reef evaluates and publishes a candidate tree.

`Evolve your harness <../user-guide/evolve-your-harness.rst>`__ describes what each key changes, and `Write a harness method <../developer-guide/write-a-harness-method.rst>`__ describes the method contract. A key may use underscores or hyphens, and startup refuses an unknown key.

The pipeline stages decide what a job trains on:

- **Scored reports (default).** Every valid scored report is a trace, successful outcomes included. A job starts when ``pipeline.sample2job.config.job_samples`` traces are held, 1 by default.
- **Recorded traffic alone.** Set ``pipeline.record2sample.module: reef.samples.traces:InferenceSamples``. A job then starts every ``job_samples`` requests, with unscored samples.
- **Imported train tasks (offline).** Set ``pipeline.sample2job.module: reef.train.sample2job.traces:TrainTaskJobData``. Each uploaded train task trains once, in upload order. A job holds either traces or train tasks. A job of train tasks rolls out up to ``pipeline.sample2job.config.buffer_size`` of them (default ``job_samples``) on the head tree as the method's traces. Startup refuses this stage without an adapter that plays task directories (an npm ``install``) and ``evolution.evaluation.scorer``.

Reef refuses ``batch_size`` and ``batch_policy`` directly under ``recipe.config``.

.. container:: config

   ``evolution.method``
       required: the ``Method`` that evolves the tree, as ``package.module:ClassName`` or a mapping with ``module`` and ``config`` that the class method ``from_config`` receives; its ``run(job)`` makes each job's candidate

   ``evolution.adapter`` : pi
       ``opencode``, ``claude``, ``codex``, ``dsh`` (DeepSeek Harness), ``hermes`` (Hermes Agent), ``native`` (Reef's own agent, whose tools, hook listeners, control flow graphs, delegate agents and loop code are tree files of the ``native_tool``, ``native_hook``, ``native_graph``, ``native_agent`` and ``native_loop`` parts), ``python`` (a Python program that uv runs, whose modules are tree files of the ``program`` part), or an entry-point adapter, such as ``terminus`` from the Meta-Harness Terminal-Bench campaign

   ``evolution.seed``
       the directory of the first tree: every regular file under it, at its relative path as its tree path; a new scenario's root snapshot holds it, and an existing scenario keeps its own root and head. Startup refuses a seed that fails validation for ``evolution.adapter``, holds a symbolic link, or holds a file that is not UTF-8 text. A relative path resolves against the Reef process's working directory. Unset, the empty tree, unless a recipe subclass sets its own (for example Reefine's starter skill)

   ``evolution.binary``
       a path to the harness binary; unset, backend construction installs the adapter's pinned version through the vendor's channel under ``$REEF_HARNESS_PREFIX`` (default ``~/.local/share/reef-harness``)

   ``evolution.models`` : {}
       the method's own models by name, which it calls as ``services.models[name]``: ``url``, ``model``, optional ``api`` (default ``openai``) and ``timeout_s``, with the credential as a literal ``api_key`` or an ``api_key_env`` variable name; the name ``served`` is refused, since it is the model under test

   ``evolution.model_metadata`` : {}
       optional map of exact model names to ``{context_window: <positive token count>, reasoning: <boolean>}``; Codex uses these values before provider discovery

   ``evolution.state_dir`` : .reef/harness
       the directory of each scenario's harness state; a relative path resolves against the Reef process's working directory. ``<state_dir>/<scenario>/<component>/``, where ``<component>`` is the snapshot component the recipe evolves (``harness`` for ``HarnessEvolutionRecipe``, the component's own name in a composite recipe), holds ``history/``, every candidate, run and decision, which is the score cache; and ``steps/``, one job record per job (the workspace with its ``output/``, and ``calls.json``; ``<step>`` is the scenario step, and a retried job lands in ``<step>-2``). Nothing prunes it. A directory that cannot be created refuses to load the scenario

   ``evolution.max_steps`` : 0
       skip automatic evolve jobs once the scenario's step number passes this; the scenario numbers every commit, activations, train-from operations, imports and syncs included; 0 disables the limit; an instruction from ``POST /reef/train`` still runs past it

   ``evolution.max_failure_streak`` : 0
       stop automatic evolve jobs after this many consecutive rejected jobs, request jobs included; 0 disables the limit; a request from ``POST /reef/train`` still runs while the breaker is open

   ``evolution.max_model_calls_per_step`` : 0
       cap the method's model calls in one job, through ``services.models`` or ``services.calls``; a call past the cap raises ``RuntimeError``; 0 disables the limit

   ``evolution.max_runs_per_step`` : 0
       cap the runs that the method's ``services.evaluate`` runs in one job; reused runs and the evaluation's own runs do not count; 0 disables the limit

   ``evolution.executor`` : local
       ``local`` runs tasks as a plain subprocess; ``sandbox`` uses the host's local sandbox, a bubblewrap jail on Linux or a Seatbelt profile on macOS; ``e2b`` and ``modal`` create a cloud sandbox per run and check provider access, image, binary and namespace support at startup. A cloud sandbox uses remote binaries; Terminus retains its own Harbor execution

   ``evolution.sandbox``
       for ``sandbox``: ``egress_hosts`` and ``limits`` (``cpu_seconds``, ``memory_bytes``, ``processes``, ``file_bytes``; macOS refuses ``memory_bytes``); for ``e2b``: ``e2b_api_key_env`` (default ``E2B_API_KEY``), optional ``e2b_api_key``, ``e2b_template`` and ``forward_ports`` (explicit worker-loopback TCP ports). for ``modal``: optional ``modal_app`` (default ``reef-sandboxes``), ``modal_image``, ``modal_cpu``, ``modal_memory_mb`` and ``forward_ports``, with Modal's credentials from ``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET`` or ``~/.modal.toml``. The cloud sandboxes reject nonempty ``egress_hosts``/``limits``; image resources and outbound internet apply. All accept ``env_from``

   ``execution.evolution.workers`` : 1
       fixed worker-group size (`Stack execution backends <configuration-execution.rst#stack-execution-backends>`__)

   ``execution.evolution.backend`` : auto
       worker placement, independent of the run isolation that ``evolution.executor`` selects

   ``execution.evolution.resources``
       ``cpus_per_worker`` and ``gpus_per_worker``; GPUs select Ray under ``auto``

   ``evolution.agent``
       the coding agent that a method which runs one, such as ``reef.train.harness_backend:AgentMethod``, runs in its jobs (``services.run_agent``, or ``services.agent`` for a method that starts the agent itself); unset, no agent runs, and such a method is refused at startup unless it also runs without an agent (``runs_without_agent``)

   ``evolution.agent.adapter`` : pi
       the agent's harness: any adapter whose prompt is text

   ``evolution.agent.binary``
       the agent's binary; unset, backend construction installs the adapter's pinned version, as for ``evolution.binary``

   ``evolution.agent.sandbox`` : ``REEF_AGENT_SANDBOX``
       ``bwrap`` jails the agent, with the internet but no host port except its gateway's; ``seatbelt`` does the same on macOS, but the host's LAN addresses stay reachable; ``e2b`` or ``modal`` runs it in a cloud sandbox that reaches the gateway through a tunnel, with that provider's options in the section, as in ``evolution.sandbox``; ``none`` runs it with the service's own privileges and must be chosen. Unset, the agent is jailed in the host's local sandbox (bwrap and pasta on Linux, sandbox-exec on macOS) where the host can run it, and off where it cannot

   ``evolution.agent.timeout_s`` : 1800
       seconds one agent run may take

``evolution.evaluation`` decides how Reef evaluates a candidate and when it publishes one: the tasks, the scorer, and the publication policy. It takes one task source: ``tasks``, ``task_manifest`` with ``tasks_root``, or ``imported_tasks``. The task source and ``scorer`` are required unless ``policy`` is ``none``.

.. code:: yaml

   evolution:
     evaluation:
       tasks: ["[fib] Compute fib(90) and reply with the number alone."]
       scorer: harness.scoring:AnswerScorer
       policy: compare
       statistic: wins

.. container:: config

   ``evolution.evaluation.tasks``
       the evaluation tasks: instructions, the text the agent reads, or task directories for an adapter that takes one (``takes_task_directory``)

   ``evolution.evaluation.task_manifest``
       a split manifest written by ``reef.core.tasks``; its eval split names the evaluation's tasks as directories under ``evolution.evaluation.tasks_root``, each passed to the adapter as its path. Each task runs in its own container, and the adapter needs an npm ``install`` (`Task directories <../user-guide/evolve-your-harness.rst#task-directories>`__); startup refuses a manifest that names no eval task

   ``evolution.evaluation.tasks_root``
       the directory the manifest's task names live under; set together with ``task_manifest``

   ``evolution.evaluation.imported_tasks`` : false
       true makes each job evaluate on the task directories of the eval split of the scenario's imported tasks (the ``task-import`` section, or task items whose metadata names the scenario's ``eval`` split) as the job reads them, so tasks that the scenario gets later join the next job. The eval split may be empty when the scenario loads. The same adapter rule as ``task_manifest``. Reef refuses the key ``evolution.evaluation.task_store`` with a message that names this key

   ``evolution.evaluation.scorer``
       a ``TaskScorer`` class, in the same form as ``evolution.method``; ``reef.evaluation.harness:VerifierRewardScorer`` scores a task directory run by its Harbor verifier's reward

   ``evolution.evaluation.policy`` : compare
       the publication policy: ``compare`` scores the candidate and the current tree (the training start) on the same tasks and repeats; ``floor`` scores the candidate alone and publishes it when every task has a run that completed and every run that completed scores at least ``floor_score``; ``none`` runs no task and publishes every candidate that passes validation

   ``evolution.evaluation.statistic`` : wins
       what ``compare`` compares. ``wins`` publishes when the candidate's wins minus losses over the (task, repeat) pairs exceed ``min_margin``. ``mean`` publishes when the candidate's mean score exceeds the current tree's by more than ``min_margin``. A run that gave no answer scores 0. A run that the environment could not run (a failed run) drops its pair under ``wins`` and is left out of the mean under ``mean``

   ``evolution.evaluation.min_margin`` : 0
       the margin a ``compare`` candidate must exceed: in wins minus losses, or in mean score; at least 0

   ``evolution.evaluation.floor_score`` : 1.0
       the score every run that completed must reach under ``floor``; a failed run fails no task, but a task whose runs all failed does not pass; positive

   ``evolution.evaluation.repeats`` : 1
       the runs per task and tree; each repeat is scored, and compared, on its own

   ``evolution.evaluation.task_timeout_s`` : 600
       seconds one evaluation run may take

   ``evolution.evaluation.agent_host`` : host.docker.internal
       the address at which an agent in a task's container reaches Reef; Reef puts it in place of a loopback host (``127.0.0.1``, ``localhost``, ``::1`` or ``0.0.0.0``) of the served model's base URL, and keeps the port and the path; another host stays. Docker Desktop and colima resolve the default. On Linux, name an address of the host that the container reaches, such as the Docker bridge gateway, and make Reef listen there

   ``evolution.evaluation.forbid_residue`` : false
       when true, a run leaving files outside the cleanup whitelist scores 0, as a run that gave no answer

   ``evolution.evaluation.collect_failures`` : false
       when true, prompts of each job's traces become permanent evaluation tasks, so no later candidate can win while it brings a collected failure back; the seed tasks stay the floor. Refused when the evaluation tasks are task directories

   ``evolution.evaluation.collector``
       optional ``TaskCollector`` class, in the same form as ``evolution.method``, only with ``collect_failures``; its ``collect(samples)`` chooses which trace prompts to collect; without it every trace's user prompt is collected, and the caps and the credential and directive screens still apply

   ``evolution.evaluation.max_collected_tasks`` : 50
       the cap on collected tasks; admission stops there so the suite is bounded; 0 collects none

   ``evolution.evaluation.max_collected_per_client`` : 5
       the cap on collected tasks from one tagged client (its ``x-reef-tag-client``, else session, tag); untagged traffic has no identity to count under and meets only ``max_collected_tasks``; 0 disables the cap

The evaluation context is what a run's score depends on besides the tree and the task: the served model and the named models of ``evolution.models``; the adapter, its version and ``evolution.binary``; the scorer setting; ``task_timeout_s`` and ``forbid_residue``; and the run executor, ``evolution.executor`` and ``evolution.sandbox``. The history reuses a scored run only in the context it ran in, and never reuses a failed run.

The served model's binding is merged in when a run renders the tree; it never enters a published tree. The seed is the baseline that the first candidate is measured against. The job records hold the method's raw model traffic and the clients' traces, and the history holds every evaluation run's session log, so treat ``evolution.state_dir`` like the commit history.

The beta Reefine recipe runs its agent on pi with ``evolution.agent``. Its method reads ``trial_timeout_s`` and ``multimodal`` from ``evolution.method.config``; its `README <../../recipes/beta/reefine/README.md>`__ documents them.
