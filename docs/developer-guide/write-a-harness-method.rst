Write a harness method
======================

.. page::
   :for: authors of a harness evolution method
   :needs: the loop and the harness tree from `Evolve your harness <../user-guide/evolve-your-harness.rst>`__
   :outcome: a method that runs in Reef's steps, the scorer it needs, and a deployment that names it

A method decides how the harness tree changes. It is one class with one
operation, and a deployment names it in ``evolution.method``. GEPA, ACE and
Meta-Harness in ``recipes/`` are methods.

.. code:: python

   from reef.train.harness_backend import Method, StepContext


   class MyMethod(Method):
       def step(self, context: StepContext) -> None: ...

Reef calls ``step`` once per training step. ``context.workspace`` holds a
copy of the head tree, the component's training start, and the step's
material; ``context.services`` are what Reef does for the method. The tree
that the method leaves is the candidate: Reef checks it and evaluates it
under the publication policy. A selected candidate becomes a new release, and
the deployment's activation rule decides whether that release serves.

.. diagram::
   :caption: Reef writes the workspace, calls step(context) once, and reads back the tree, the state and the output that the method left. Then Reef checks the candidate and evaluates it; a selected candidate becomes a release, which the activation rule activates or leaves inactive.

   <img class="fig-light" src="../assets/harness-method-step-light.svg" alt="One training step from the method's view: the workspace Reef writes, the method and its services, what Reef reads back, then the check, the evaluation and the activation rule">
   <img class="fig-dark" src="../assets/harness-method-step-dark.svg" alt="One training step from the method's view: the workspace Reef writes, the method and its services, what Reef reads back, then the check, the evaluation and the activation rule">

An example: add a tool
----------------------

This method gives a pi agent a new tool. It shows the served model the
failing traces, asks for one pi extension that registers a tool, and keeps
the tool only when it scores better than the head:

.. code:: python

   from reef.core.prompts import untrusted_text
   from reef.harness.tree import TreePatch
   from reef.train.harness_backend import Method, StepContext

   PROMPT = (
       "The agent failed the requests below. Write one pi extension that registers a tool that would have "
       "helped, with pi.registerTool({name, label, description, parameters, execute}). "
       "Reply with the TypeScript file alone."
   )


   class AddTool(Method):
       """Each step: write one tool for the failing traces, and keep it when it scores better."""

       def step(self, context: StepContext) -> None:
           workspace, services = context.workspace, context.services
           failing = [trace for trace in workspace.traces() if trace.score is not None and trace.score < 1.0]
           if not failing:
               workspace.write_output("no failing trace")  # the tree stays the same, so the step skips
               return
           shown = "\n\n".join(f"task: {trace.task}\nreply: {trace.reply}" for trace in failing)
           code = services.models.served.chat([{"role": "user", "content": f"{PROMPT}\n\n{untrusted_text(shown)}"}])
           head = workspace.tree()
           name = f"extensions/tool-{workspace.step}.ts"
           candidate = head.apply(TreePatch({name: code.strip() + "\n"}))
           validation = services.check(candidate)
           if not validation.passed:
               workspace.write_output(f"the tool fails validation: {validation.message()}")
               return
           tasks = workspace.tasks()[:4]
           before = mean_score(services.evaluate(head, tasks))
           after = mean_score(services.evaluate(candidate, tasks))
           if after > before:
               workspace.write_tree(candidate)
           workspace.write_output(f"{name}: {before:.2f} -> {after:.2f}", {"before": before, "after": after})


   def mean_score(records) -> float:
       # A failed run (score.value None) counts as 0 here.
       return sum(record.score.value or 0.0 for record in records) / max(len(records), 1)

It uses each piece a method has:

- **Material**: ``workspace.traces()`` and ``workspace.tasks()``.
- **Services**: a model call, ``check`` before it returns, and ``evaluate``
  to compare trees. A scored run is reused, so the head's scores are
  usually free.
- **Result**: ``write_tree`` for the candidate, ``write_output`` for what
  the step did.

A tool is code from client traffic, so the deployment leaves it inactive
until a person activates it, with the activation rule's ``review_kinds:
[extension]`` (see `Configure a deployment`_). Another
adapter takes code in another form: ``workspace.host()["parts"]`` says what
each part holds.

What Reef does with the step
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - The method leaves
     - Reef
   * - the head tree, unchanged
     - skips the step
   * - a tree that fails validation
     - skips the step and names the problems in ``skipped``; no retry
   * - any other tree
     - evaluates it as the candidate
   * - an exception
     - fails the step; nothing publishes, and the calls stay in the record

The workspace
-------------

Each step has one directory,
``<evolution.state_dir>/<scenario>/<component>/steps/<step>``, kept as the
step record.

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Directory
     - Access
     - Holds
   * - ``harness/``
     - read-write
     - a copy of the head tree; the tree here at the end is the candidate
       (``tree()``, ``write_tree(tree)``)
   * - ``material/``
     - read-only
     - the step's traces (``traces()``), its evaluation tasks and repeats
       (``tasks()``, ``evaluation_repeats()``), and the adapter's parts
       (``host()``)
   * - ``state/``
     - read-write
     - the method's own files, carried to the next step (`Keep state between steps`_)
   * - ``scratch/``
     - read-write
     - temporary files, removed when the step ends
   * - ``output/``
     - read-write
     - what the step did (`What the method says`_)

A ``Trace`` is a client's request (``source`` is ``traffic``) or a run
of an offline step (``run``):

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Value
   * - ``task``
     - the last user prompt, or the run's instruction; ``None`` without one
   * - ``reply``
     - the final assistant text, else ``None``
   * - ``score``
     - the report's or the scorer's score, else ``None``
   * - ``feedback``
     - the report's feedback as text, else ``None``
   * - ``failure``
     - why a run failed, else ``None``
   * - ``record``, ``client``
     - the agent record or run id; the client's tag, else ``untagged``
   * - ``trajectory()``
     - a traffic trace's ATIF trajectory, else ``None``

``host()["parts"]`` gives each part's tree ``path``, ``format``, render path
and what it ``holds``. A file at a path that no part takes fails validation.

The services
------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Service
     - Does
   * - ``models.served``, ``models["<name>"]``
     - the model under test, or an ``evolution.models`` entry:
       ``chat(messages)`` returns the text, ``complete(body)`` the raw
       response
   * - ``check(tree)``
     - the ``ValidationResult``: ``passed``, ``problems``, ``message()``
   * - ``evaluate(tree, tasks, repeats=1)``
     - scored ``RunRecord`` values (``task``, ``repeat``, ``score``,
       ``reply``, ``failure``), run like the step's evaluation. ``score`` is
       a ``RolloutScore``: ``score.value`` is the scorer's value, or 0 when
       the tree gave no answer; it is ``None`` for a failed run, which the
       environment could not run, and then ``failure`` says why
   * - ``run_agent(prompt, tree=, model=)``
     - one run of the deployment's coding agent (`Run a coding agent`_)
   * - ``calls``
     - the step's call meter: ``spend()``, ``record(entry)``, ``note(...)``

- Each model call counts against ``evolution.max_model_calls_per_step`` and
  lands in the step record's ``calls.json``.
- ``evaluate`` refuses the whole call, and runs nothing, without a scorer,
  past ``evolution.max_runs_per_step`` (reused runs are free), for a
  tree that fails ``check`` (``ValueError``), and for one string as ``tasks``
  (``TypeError``).

Run a coding agent
~~~~~~~~~~~~~~~~~~

A method whose class sets ``runs_agent = True`` gets the deployment's coding
agent (``evolution.agent``: adapter, binary, sandbox). The agent works in the
workspace and holds no credential: its model calls go through a loopback
gateway at ``$REEF_AGENT_URL``, which spends from the step's budget. The
same address checks and scores trees: ``GET /tools``, ``POST /check`` with
``{"tree": "harness"}``, and ``POST /evaluate`` with ``{"tree": "harness",
"tasks": [...]}``. ``run_agent`` returns an ``AgentOutcome`` (``exit_code``,
``timed_out``, ``error``, ``seconds``).

A method made of prompts
------------------------

``reef.train.harness_backend:AgentMethod`` runs the coding agent once per
step on a folder of prompts:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Holds
   * - ``START.md``
     - the prompt of a step (required); ``{step}`` and ``{history}`` are
       filled in
   * - ``RESUME.md``
     - the prompt when ``state/`` holds earlier files (optional)
   * - ``rules.md``, ``skills/...``
     - the agent's own harness tree
   * - ``README.md``
     - text for people only

.. code:: yaml

   evolution:
     method:
       module: reef.train.harness_backend:AgentMethod
       config: {prompts: methods/rules-editor, model: proposer}
     agent: {adapter: claude, sandbox: bwrap}

What the method says
--------------------

``workspace.write_output(summary, notes)`` writes ``output/summary.md``
(plain text, shown on the step pages, kept as ``method_summary``) and
``output/notes.json`` (a JSON object, kept as ``method_notes``). Reef never
decides by them.

Keep state between steps
------------------------

Files in ``state/`` (an archive, a playbook, counters) are committed with the
step and written back into the next step's ``state/``, so they survive a
restart. A failed step commits none of its state. The files must be UTF-8
text, at most 16 MiB together.

Read the history
----------------

``workspace.history()`` holds every tree, run and decision, read-only:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Member
     - Gives
   * - ``candidates``
     - each tree's record by checksum: ``step``, ``origin``, ``parent``,
       ``summary``, ``notes``
   * - ``runs``, ``decisions``
     - every ``RunRecord``; every decision (``step``, ``tree``, ``head``,
       ``selected``, ``reason``)
   * - ``tree(checksum)``, ``run_dir(run)``
     - a tree's files; a run's trajectory files
   * - ``scores(checksum, tasks, repeats=1)``
     - the latest score per task and repeat, ``None`` where none ran or the
       latest failed
   * - ``decision(step)``
     - the decision on that step's candidate

.. code:: python

   history = workspace.history()
   decision = history.decision(workspace.step - 1)
   if decision is not None:
       scores = history.scores(decision.tree, workspace.tasks())

A failed attempt reuses its step number, so skip records whose ``step`` is
``workspace.step`` or more.

Evaluation
----------

``evolution.evaluation`` names the tasks, the scorer and the policy (`the
publication policy
<../user-guide/evolve-your-harness.rst#evaluation-the-publication-policy>`__).
Runs that the method already ran through ``services.evaluate`` count:
the evaluation reuses them.

Score a run
~~~~~~~~~~~~~~~~

An ``TaskScorer`` returns a finite score for one finished run;
higher is better:

.. code:: python

   from reef.evaluation.harness import TaskScorer
   from reef.harness.tasks.trajectory import final_assistant_text

   EXPECTED = {"[fib]": "2880067194370816120"}


   class ExactAnswer(TaskScorer):
       def score(self, task, result):
           """1.0 when the final assistant text ends with the task's expected answer."""
           text = final_assistant_text(result.trajectory) or ""
           return 1.0 if text.strip().endswith(EXPECTED[task.split()[0]]) else 0.0

A judge that calls a model overrides ``score_with_models(task, result,
models)``. ``VerifierRewardScorer`` takes a Harbor task's reward.

Collect traffic prompts
~~~~~~~~~~~~~~~~~~~~~~~

With ``collect_failures: true``, a ``TaskCollector`` in
``evolution.evaluation.collector`` returns the prompts of a step's traces to
keep as evaluation tasks. Without one, Reef keeps every trace's prompt.

Configure a deployment
----------------------

.. code:: yaml

   recipe:
     implementation: reef.recipe.harness:HarnessEvolutionRecipe
     config:
       evolution:
         adapter: pi
         seed: methods/seed            # the first tree; it must pass validation
         method: methods.add_tool:AddTool
         evaluation:
           tasks:
             - "[fib] Compute fib(90) exactly. Reply with the integer alone on the last line."
           scorer: methods.scoring:ExactAnswer
           policy: compare
   activation:
     module: always
     config:
       review_kinds: [extension]       # a tool waits for a person
   inference:
     upstream-url: ${REEF_UPSTREAM_URL}
     upstream-model: qwen3-8b

Start ``reef serve -c`` where Reef can import ``methods``.
``tutorials/evolve-your-harness/configs/`` has working examples, and `Harness
evolution keys <../reference/configuration.rst#harness-evolution-keys>`__
lists every key.

Method settings
~~~~~~~~~~~~~~~

A method with settings overrides ``from_config`` with a ``PluginSettings``
dataclass, which refuses an unknown key:

.. code:: python

   from dataclasses import dataclass

   from reef.core.config import config_option
   from reef.core.plugins import PluginSettings


   @dataclass(frozen=True)
   class AddToolSettings(PluginSettings):
       model: str = config_option("served", help="The model binding that writes the tool.")


   class AddTool(Method):
       def __init__(self, settings: AddToolSettings) -> None:
           self.model = settings.model

       @classmethod
       def from_config(cls, config):
           return cls(AddToolSettings.parse(config, plugin_class=cls))

The deployment then writes ``method: {module: methods.add_tool:AddTool,
config: {model: proposer}}``, and ``evolution.models.proposer`` names the
model with ``url``, ``model`` and ``api_key_env``.

Package a method
----------------

A method in this repository is one package under ``recipes/<name>/``:
``method.py`` with the method and its settings, prompts as ``*.md`` files,
``README.md``, and runnable deployments in ``examples/``. It needs no recipe
of its own.

Untrusted input
---------------

- Traces are client data. Wrap their text with
  ``reef.core.prompts.untrusted_text`` before it enters a prompt.
- Trust ``trace.client`` as an identity only when a gateway sets the tag.
- Client text can shape code parts (``extension``, ``native_tool``,
  ``native_hook``). Name them in the ``always`` activation rule's
  ``review_kinds`` (`Configuration
  <../reference/configuration.rst#the-activation-section>`__). A
  ``native_loop`` change always waits for a person.

.. warning::

   Do not put credentials in tree files. Validation refuses them, and a tree
   never needs one: the served model binds at render time, and a named model
   reads its key from ``api_key_env``.
