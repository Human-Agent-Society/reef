Harness method API
==================

The classes that a harness method, a task scorer and a task collector use, the history records that a method reads, and the candidate evaluation plugin.

`Write a harness method <../developer-guide/write-a-harness-method.rst>`__ explains the job, the plugin settings (``from_config`` and ``PluginSettings``), and when a job is skipped or fails.

Harness method
--------------

.. code:: python

   from reef.evaluation.harness import TaskScorer, History, TaskCollector
   from reef.harness.adapters import get_adapter
   from reef.harness.adapters.parts import Format, Part, TreeLayout
   from reef.harness.tree import HarnessTree, TreePatch
   from reef.train.harness_backend import (
       AgentExecution, AgentMethod, AgentOutcome, Method, MethodJob, ModelCalls, Services, Trace,
       Workspace,
   )

   class MyMethod(Method):
       def run(self, job: MethodJob) -> None: ...

   class MyScorer(TaskScorer):
       def score(self, task, result) -> float: ...

The method
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Member
     - Contract
   * - ``Method.run(job)``
     - Edit ``job.workspace.harness`` into the job's candidate, or leave it unchanged to skip the job. Say what the job did with ``job.workspace.write_output``.
   * - ``Method.runs_agent``
     - A class attribute, ``False`` by default. ``True`` for a method that calls ``Services.run_agent`` or reads ``Services.agent``. Such a method gets ``evolution.agent`` and is refused at startup without it.
   * - ``Method.runs_without_agent``
     - A class attribute, ``False`` by default. ``True`` for a method that ``runs_agent`` and also runs when the deployment configures no agent: it starts without ``evolution.agent`` and reads ``Services.agent`` as ``None``.
   * - ``Method.from_config(config)``
     - Builds the method from ``evolution.method.config`` alone. A method takes its own settings, such as a time limit, from ``config``, and everything that runs (a model, the coding agent, the evaluation) from ``job.services``. Its constructor receives no service.
   * - ``AgentMethod``
     - A method that is a folder of prompts: ``START.md``, an optional ``RESUME.md``, and the agent's own tree. Its ``config`` has ``prompts``, the folder, and ``model``, ``served`` by default. Each job runs the agent once and reads ``output/summary.md``, ``notes.json`` and ``requires.json``.
   * - ``MethodJob``
     - ``workspace``, the job's ``Workspace``, and ``services``, the job's ``Services``. The backend makes one per job.

The workspace
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Member
     - Contract
   * - ``root``, ``history_dir``, ``step``
     - The fields of a ``Workspace``. ``step`` is 1 or more.
   * - ``harness``, ``material``, ``state``, ``scratch``, ``output``
     - The directory paths.
   * - ``tree()``, ``write_tree(tree)``
     - Read and write the harness tree.
   * - ``traces()``
     - The job's ``Trace`` values.
   * - ``request()``
     - A ``TrainingRequest`` or ``None``.
   * - ``tasks()``, ``evaluation_repeats()``
     - The tasks, and the runs per task and tree: 0 when the deployment configured no scorer.
   * - ``host()``
     - The host facts of ``material/host.json``: ``adapter``, ``parts``, ``config_targets`` and ``tasks_are_directories``.
   * - ``history()``
     - The ``History`` (`History records`_).
   * - ``state_files()``
     - Every file of ``state/`` as a path mapped to its text. ``ValueError`` past 16 MiB or for a file that is not UTF-8 text.
   * - ``write_output(summary="", notes=None, requires=())``
     - Writes ``output/summary.md`` (the commit's ``method_summary``), ``output/notes.json`` (a JSON object, the commit's ``method_notes``) and ``output/requires.json`` (``reef.core.requirements`` items, read on a request job only). Raises ``ValueError`` before writing when ``notes`` or ``requires`` is not valid.
   * - ``Trace``
     - ``directory``, ``source`` (``traffic`` or ``run``), ``task``, ``reply``, ``score``, ``feedback``, ``failure``, ``record`` and ``client``. ``trajectory()`` reads the ATIF trajectory of a traffic trace, and returns ``None`` for a run, whose run files the history keeps.
   * - ``TreeLayout``
     - The parts of an adapter's tree, from the ``evolve`` section of its ``adapter.yaml``. A method gets its adapter's as ``get_adapter(workspace.host()["adapter"]).layout``. ``parts`` maps each part name to a ``Part`` (``name``, ``path``, ``format``, a ``Format``, ``renders``, ``holds`` and ``review``). ``locate(path)`` returns the ``TreeLocation`` of a tree path (``part``, and ``name``, the name in the part's ``{name}``, ``""`` for a part without one), or ``None`` when no part takes the path.

The services
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Member
     - Contract
   * - ``Services.models``
     - A ``ModelBindings``: ``served`` and every ``evolution.models`` entry. ``chat(messages, *, timeout_s=None, **params)`` returns the assistant text, ``complete(body)`` the provider's response. Every call counts against ``evolution.max_model_calls_per_step`` and lands in ``calls.json``.
   * - ``Services.calls``
     - The job's ``ModelCalls``. ``spend()`` counts one call and raises ``RuntimeError`` past the budget. ``record(entry)`` appends one call to ``calls.json``. ``note(kind, text, *, failed=False)`` adds one line to the job's activity.
   * - ``Services.check(tree)``
     - The ``ValidationResult`` of ``tree`` for the adapter: ``passed``, ``problems`` and ``message()``.
   * - ``Services.evaluate(tree, tasks, *, repeats=1)``
     - ``repeats`` ``RunRecord`` values per task of the list ``tasks`` (a single string is refused), task by task. Reuses the scored runs that the history holds. ``RuntimeError`` without a scorer, or when the call needs more runs than ``evolution.max_runs_per_step`` leaves; then no task runs. ``ValueError`` for a tree that fails ``check``.
   * - ``Services.agent``
     - The deployment's coding agent bound for the job's scenario, or ``None`` when no agent runs for the method. It is an ``AgentExecution``: ``descriptor``, the agent's adapter; ``executor``, its isolation bound to the adapter's program; ``binary``; and ``timeout_s``, the limit of one run. A method that needs its own gateway or tools starts the agent with it. ``run_agent`` is the usual way.
   * - ``Services.run_agent(prompt, *, tree, model)``
     - One run of the deployment's coding agent in the workspace, as an ``AgentOutcome``. The agent's own harness ``tree`` is rendered after Reef's protocol rules. Its model calls go to the binding that ``model`` names, through a gateway that serves ``check`` and ``evaluate`` too. ``RuntimeError`` when no agent runs for the method, ``KeyError`` for an unknown ``model``, ``ValueError`` for a tree that does not render.
   * - ``AgentOutcome``
     - ``exit_code`` (``None`` when the agent did not exit by itself), ``timed_out``, ``error`` (why it could not start) and ``seconds``. ``succeeded`` is an exit with status 0.

History records
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Contract
   * - ``History``
     - ``candidates`` (``CandidateRecord`` by tree checksum), ``runs`` and ``decisions`` in file order; ``tree(checksum)``, ``runs_of(checksum)``, ``scores(checksum, tasks, *, repeats=1)``, ``decision(step)`` and ``run_dir(run_id)``.
   * - ``CandidateRecord``
     - ``tree``, ``step``, ``origin`` (``seed``, ``method``, ``proposal`` or ``release``), ``parent``, ``summary`` and ``notes``.
   * - ``RunRecord``
     - ``run_id``, ``tree``, ``task``, ``task_key``, ``repeat``, ``context``, ``step``, ``score`` (a ``RolloutScore``: a finite value, or failed when the environment could not run the task), ``failure``, ``reply``, ``residue``, ``agents``, ``path`` and ``transcript_read``.
   * - ``DecisionRecord``
     - ``step``, ``tree`` (the candidate's checksum), ``head`` (the checksum of the head tree that the job started from), ``selected`` (published; the activation rule decides whether it serves), ``policy`` and ``reason``.

Scoring and publication
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Contract
   * - ``TaskScorer.score(task, result)``
     - A finite float for one finished run; higher is better. ``result`` holds ``exit_code``, ``stdout``, ``stderr``, the parsed ``trajectory`` and ``residue``. The evaluation calls ``score_with_models(task, result, models)``, whose default calls ``score``. A model-based judge overrides ``score_with_models`` to use the job's model bindings, and still implements the abstract ``score``, for example by raising. A scorer that needs a local GPU overrides ``execution_requirements()``.
   * - ``TaskCollector.collect(samples)``
     - The prompts to collect from the job's trace samples, with ``evolution.evaluation.collect_failures``.
   * - ``evaluation.metrics``
     - Under ``policy: compare`` and ``policy: floor``, holds ``candidate_scores`` and ``current_scores``: ``RolloutScores`` (``reef.core.evaluation``) in JSON, one entry per task in task order with the scores of its runs in run order, each a number or ``"failed"``. Read them with ``RolloutScores.from_json``. ``current_scores`` is empty under ``floor``, which runs the candidate alone. Under ``policy: none``, which runs no task, it holds no score.

``reef.evaluation.harness`` holds the publication policies that ``evolution.evaluation.policy`` names: ``ComparePolicy`` (``compare``, with ``statistic`` ``wins`` or ``mean`` and ``min_margin``), ``FloorPolicy`` (``floor``, with ``floor_score``) and ``NoEvaluationPolicy`` (``none``). A harness recipe takes no other policy.

Candidate evaluation
--------------------

.. code:: python

   from reef import CandidateEvaluationPlugin, EvaluationResult, SelectionDecision

A harness recipe's plugin explicitly inherits ``CandidateEvaluationPlugin``, measures a candidate tree, and decides whether the search keeps it. ``CandidateEvaluator`` and ``CandidateSelector`` remain separate abstract capabilities; an object that implements only one is not a plugin. Reef enforces the fixed evaluate-then-decide order and verifies that the decision kept the exact result that it was given.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Member
     - Contract
   * - ``evaluate(candidate)``
     - Returns an ``EvaluationResult``: evaluator name, version, and a metrics mapping.
   * - ``decide(candidate, evaluation)``
     - Returns a ``SelectionDecision`` whose ``selected`` is a boolean, and which must carry the evaluation that it was given.

- A rejected candidate makes no artifact, and the kept tree stays the search's current one.
- An exception is fail-closed: Reef does not publish the candidate.
- Make evaluations idempotent by ``candidate.candidate_id``, because recovery can repeat work whose result was not durably committed.
- Whether a kept tree serves is the `activation rule <python-api-recipe.rst#activation-rule>`__'s decision, not the plugin's.

The harness training component builds its plugin for each job from ``evolution.evaluation.policy`` and takes no other plugin (`Scoring and publication`_).
