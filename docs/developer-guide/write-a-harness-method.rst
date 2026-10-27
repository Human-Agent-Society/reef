Write a harness method
======================

.. page::
   :for: authors of a harness evolution method, and anyone reading ``reef/train/harness_backend/``
   :needs: the loop and the tree layout from `Evolve your harness <../user-guide/evolve-your-harness.rst>`__
   :outcome: a method that runs in Reef's steps, the evaluation it needs, and a deployment that names it

A harness method decides how the harness tree changes. At each step, Reef
gives the method a workspace: one directory with a copy of the head tree, the
step's traces, and the method's own files. The method edits the tree in that
directory. Reef then validates the tree, evaluates it under the deployment's
publication policy, records the result in the scenario's history, and
publishes a selected tree as a new release.

A method is one class with one operation:

.. code:: python

   from reef.train.harness_backend import Method, StepResult, Tools, Workspace


   class MyMethod(Method):
       def step(self, workspace: Workspace, tools: Tools) -> StepResult: ...

``step`` edits ``workspace.harness``. The tree in that directory when ``step``
returns is the step's candidate. ``tools`` calls models, checks trees, and
scores trees on tasks. The ``StepResult`` says what the step did. The
deployment names the class in ``evolution.method``. A method never subclasses
the backend.

A small method
--------------

This method makes one model call per step. It shows the served model the
traces that scored below 1, asks for new rules, and writes them to
``rules.md``:

.. code:: python

   from reef.core.prompts import untrusted_text
   from reef.harness.tree import TreePatch
   from reef.train.harness_backend import Method, StepResult, Tools, Workspace

   PROMPT = "Rewrite these agent rules so that the failing requests succeed. Reply with the rules alone."


   class Reflect(Method):
       """One model call per step: rewrite rules.md from the failing traces."""

       def step(self, workspace: Workspace, tools: Tools) -> StepResult:
           failing = [trace for trace in workspace.traces() if trace.score is not None and trace.score < 1.0]
           if not failing:
               return StepResult("no failing trace")  # harness/ is unchanged, so the step is skipped
           tree = workspace.tree()
           shown = "\n\n".join(
               f"task: {trace.task}\nreply: {trace.reply}\nfeedback: {trace.feedback}" for trace in failing
           )
           rules = tree.files.get("rules.md", "")
           prompt = f"{PROMPT}\n\nRules:\n{rules}\n\nFailing requests:\n{untrusted_text(shown)}"
           reply = tools.models.served.chat([{"role": "user", "content": prompt}])
           candidate = tree.apply(TreePatch({"rules.md": reply.strip() + "\n"}))
           validation = tools.check(candidate)
           if not validation.passed:
               return StepResult(f"the new rules fail validation: {validation.message()}")
           workspace.write_tree(candidate)
           return StepResult(f"rewrote rules.md from {len(failing)} failing traces", {"failing": len(failing)})

`Configure a deployment`_ shows the deployment file that runs it, and `Score
an episode`_ shows the scorer it names.

The step
--------

One step runs in this order:

#. Reef reads the head tree from the scenario's releases, chooses the step's
   evaluation tasks, and writes the step's workspace.
#. Reef calls ``step(workspace, tools)``.
#. Reef reads the tree in ``harness/``. An unchanged tree skips the step. A
   tree that fails validation skips the step too, and the step's ``skipped``
   names the problems. There is no retry, so call ``tools.check`` before you
   return.
#. The publication policy evaluates the candidate (see `Evaluation`_). Reef
   records the decision in the history and publishes a selected candidate as
   a new release.

Two kinds of automatic step do not call the method. A recheck compares the
latest publish with the tree it replaced
(``evolution.evaluation.recheck_every``). A client's proposal from
``POST /reef/harness/proposals`` takes the next automatic step, and the
evaluation scores it in place of a method's candidate. A step that answers a
training request always calls the method (see `Answer a training request`_).

If ``step`` raises, the step fails and publishes nothing. The step record
keeps the model calls made before the failure.

The workspace
-------------

Each step gets one directory,
``<evolution.state_dir>/<scenario>/<component>/steps/<step>``.
``<component>`` is the release component that the recipe evolves, such as
``harness`` for ``HarnessEvolutionRecipe`` (see `History
<../user-guide/evolve-your-harness.rst#history>`__). Reef keeps the directory
after the step as the step record.

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Directory
     - Access
     - Contents
   * - ``harness/``
     - read-write
     - a copy of the head tree; the tree here when ``step`` returns is the
       candidate
   * - ``material/``
     - read-only
     - what the step learns from (below)
   * - ``state/``
     - read-write
     - the method's own files, committed with the step (see `Keep state
       between steps`_)
   * - ``scratch/``
     - read-write
     - temporary files; Reef removes the directory when the step ends
   * - ``output/``
     - read-write
     - files the method keeps with the step record

``material/`` holds these files:

- ``traces/<n>/trace.json`` and ``traces/<n>/trajectory.json``: one trace per
  sample of the step's job data, and the trace's ATIF trajectory.
- ``request.json``: the training request the step answers. Only a request
  step has it.
- ``evaluation.json``: the step's evaluation tasks and the episodes per task
  and tree (``repeats``), which ``workspace.tasks()`` and
  ``workspace.evaluation_repeats()`` read.
- ``failures.json``: the failure manifest of the head tree, after the
  first evaluation.
- ``host.json``: the adapter, each tree kind it renders with that kind's
  path pattern and description, its config targets, and whether its tasks
  are directories.

The scenario's history is beside the step directories, at
``workspace.history_dir``. A method reads it and never writes it (see `Read
the history`_).

A Python method reads the directories through ``Workspace``:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Member
     - Value
   * - ``harness``, ``material``, ``state``, ``scratch``, ``output``
     - the paths of the directories
   * - ``step``
     - the step's number, 1 or more
   * - ``tree()``
     - the ``HarnessTree`` that ``harness/`` holds now
   * - ``write_tree(tree)``
     - replace everything in ``harness/`` with ``tree``
   * - ``traces()``
     - the ``Trace`` values of ``material/traces/``, in trace order
   * - ``request()``
     - the ``TrainingRequest`` the step answers, ``None`` on an automatic step
   * - ``tasks()``
     - the step's evaluation tasks, the seed tasks first
   * - ``evaluation_repeats()``
     - the episodes that the evaluation runs per task and tree
       (``evolution.evaluation.repeats``), 0 when the deployment configured
       no scorer
   * - ``host()``
     - the host facts: ``adapter``, ``kinds``, ``config_targets`` and
       ``tasks_are_directories``
   * - ``history()``
     - the scenario's ``History``, read from ``history_dir``

A ``Trace`` has these fields:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Value
   * - ``task``
     - the last user prompt of the request, ``None`` for a request without
       a user prompt
   * - ``reply``
     - the final assistant text, ``None`` when there is none. With the
       inference stages (``InferenceRecord2Record`` and ``InferenceSamples``),
       every model call is a trace, and only the call that ends a run has a
       reply: a call that ends on a tool call has none
   * - ``score``
     - the reported score, ``None`` for unscored traffic
   * - ``feedback``
     - the reported feedback as text (structured feedback as JSON text),
       ``None`` when there is none
   * - ``record``
     - the agent record id of the request
   * - ``client``
     - the client's ``x-reef-tag-client`` tag, else its session tag, else
       ``untagged``
   * - ``directory``
     - the trace's directory; ``trajectory()`` reads its ATIF trajectory

Material is client data. Wrap its text with ``reef.core.prompts.untrusted_text``
before the text enters a model prompt (see `Untrusted input`_).

The tools
---------

``tools`` is the method's way to call a model and to run an episode:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Tool
     - Contract
   * - ``tools.models``
     - the model bindings: ``tools.models.served`` is the model under test,
       and ``tools.models["reflection"]`` is the ``evolution.models``
       entry of that name
   * - ``tools.check(tree)``
     - the ``ValidationResult`` of ``tree`` for the adapter: ``passed``, and
       ``problems``, every problem found
   * - ``tools.evaluate(tree, tasks, *, repeats=1)``
     - ``repeats`` scored episodes of ``tree`` on each task, task by task, as
       ``EpisodeRecord`` values; ``tasks`` is a list of tasks, and a single
       string is refused
   * - ``tools.calls``
     - the step's call meter: ``spend()`` and ``record(entry)`` for a model
       call that does not go through ``tools.models``, and
       ``note(kind, text, *, failed=False)`` for one line of the step's
       activity

Call a model
~~~~~~~~~~~~

``binding.chat(messages, *, timeout_s=None, **params)`` returns the assistant
text. The messages use the OpenAI shape whatever the endpoint's dialect.
``binding.complete(body)`` sends a raw request body and returns the
provider's response.

Every call through ``tools.models`` counts against
``evolution.max_model_calls_per_step`` (0, no limit, by default). A call past
the limit raises ``RuntimeError``. Each call lands in the step record's
``calls.json``: the request, the reply or the error, the provider's response,
the seconds and the token usage. A method can also let a process call a model
by itself, such as a coding agent behind a gateway. Then the gateway calls
``tools.calls.spend()`` before each call and ``tools.calls.record(entry)``
after it, so that the budget and the record stay complete.

Evaluate a tree
~~~~~~~~~~~~~~~

``tools.evaluate`` runs episodes as the step's evaluation does: the same
rendering, worker pool, scorer and episode settings. Each ``EpisodeRecord``
has the ``task``, the ``repeat`` index, the ``score``, and the ``reply``. The
``score`` is ``None`` for an episode that could not run, and then
``failure`` says why. ``episode`` names the episode's record directory in the
history.

The history keeps every episode, and reuses a scored episode of the same
tree, task, repeat and evaluation context. Asking again costs nothing, except
for an episode that could not run: the history does not reuse it, so it runs
again. Ask for more repeats to get new samples. A tree that the method
evaluates joins the history as a candidate of the step.

``evolution.max_episodes_per_step`` (0, no limit, by default) limits the
episodes that ``tools.evaluate`` runs in one step. Reused episodes do not
count. An episode counts when it ran, also in a call whose scorer raised.
``evaluate`` raises ``RuntimeError`` when the deployment configured no
scorer. It also raises ``RuntimeError`` when a call needs more episodes than
the step has left, and then no episode of the call runs. It raises
``ValueError`` when the tree fails ``check``, and ``TypeError`` when
``tasks`` is one string. A refused call adds nothing to the history.

The result
----------

``StepResult`` has three optional fields:

- ``summary``: plain text that says what the step did. The step pages show it,
  and the commit metrics keep it as ``method_summary``.
- ``notes``: a JSON object; ``StepResult`` refuses a value that is not one.
  The commit metrics keep it as ``method_notes``. Reef never reads it to make
  a decision.
- ``requires``: what the change needs on the person's machine, as
  ``reef.core.requirements`` items. Reef reads it on a request step only (see
  `Answer a training request`_).

Reef writes the result to the step record's ``result.json``. The history keeps
``summary`` and ``notes`` with the candidate, whether the step publishes or
not.

Keep state between steps
------------------------

A method keeps what it learns across steps as files in ``workspace.state``:
an archive, a population, a playbook, or counters. Reef commits every file of
``state/`` with the step and writes the files back into the next step's
``state/``. The state lives in the scenario's commit history, so it survives
a restart. Reef commits it whether the candidate is published, rejected or
skipped: a step that skips because the tree did not change still commits its
state. A step that fails commits none of its state, and the previous state
stays.

.. code:: python

   import json


   class Counting(Method):
       def step(self, workspace: Workspace, tools: Tools) -> StepResult:
           path = workspace.state / "counts.json"
           counts = json.loads(path.read_text()) if path.is_file() else {"steps": 0}
           counts["steps"] += 1
           path.write_text(json.dumps(counts))
           ...  # this step's change to workspace.harness
           return StepResult(notes=counts)

The files must be UTF-8 text, at most 16 MiB together. A ``state/`` that
breaks either rule skips the step, and the previous state stays.

Read the history
----------------

The history holds every tree that a step proposed or evaluated, every
episode, and every decision. ``workspace.history()`` returns it as a
``reef.evaluation.harness.History``:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Member
     - Value
   * - ``candidates``
     - each tree's ``CandidateRecord`` by checksum: ``step``, ``origin``
       (``seed``, ``method``, ``proposal`` or ``release``), ``parent``,
       ``summary`` and ``notes``
   * - ``episodes``
     - every ``EpisodeRecord``, in run order
   * - ``decisions``
     - every ``DecisionRecord``: ``step``, ``tree`` (the candidate),
       ``head`` (the head tree that the step started from), ``selected``
       (published, or held for review under ``evolution.publish: review``),
       ``policy`` and ``reason``
   * - ``tree(checksum)``
     - the files of a tree that the history holds
   * - ``scores(checksum, tasks, *, repeats=1)``
     - the tree's scores on ``tasks``, task by task and repeat by repeat: the
       latest episode of each pair, ``None`` where none ran or the latest
       could not run
   * - ``decision(step)``
     - the decision on that step's candidate, ``None`` when the step decided
       none
   * - ``episode_dir(episode)``
     - the episode's record directory: its trajectory files and
       ``episode.json``

The history also keeps the records of a step attempt that did not commit,
such as one whose method, evaluation or commit failed. The next attempt runs
under the same step number, so a record whose ``step`` is ``workspace.step``
or more comes from such an attempt. A method that counts or compares what it
tried skips these records.

A tree's checksum is ``HarnessTree.checksum()``. This code reads what happened
to the previous step's candidate:

.. code:: python

   history = workspace.history()
   decision = history.decision(workspace.step - 1)
   repeats = max(1, workspace.evaluation_repeats())
   if decision is not None:
       selected = decision.selected  # published, or held for review
       scores = history.scores(decision.tree, workspace.tasks(), repeats=repeats)
       head_scores = history.scores(decision.head, workspace.tasks(), repeats=repeats)

GEPA settles its pending candidate this way. Meta-Harness reads its whole
population from the history: every candidate, its scores and its notes.

Answer a training request
-------------------------

In ``manual`` and ``hybrid`` training mode, a step can answer a training
request from ``POST /reef/train`` (see `Manual training
<../reference/http-api.rst#manual-training>`__). Every method takes such a
step. ``workspace.request()`` returns the ``TrainingRequest``: ``id``,
``text``, ``session``, ``release_id``, ``requires``, and ``client``, the
requesting client's report of its machine. The text is client text (see
`Untrusted input`_). In ``manual`` mode the step has no traces. In ``hybrid``
mode it has up to ``job_samples`` traces from the next automatic job data, or
none.

Return ``requires`` when the change needs something on the person's machine
that the request did not name: an ``env`` variable, a ``permission``, a
``service`` or a ``binary``. Reef screens each item for credentials and
instruction overrides. It merges the items that pass after the request's own
items into the step's ``training_request.requires``, and the release manifest
serves them to clients.

A client's proposal and a recheck cannot take a request's step. The
evaluation and ``evolution.publish`` apply as for any other step.

Evaluation
----------

The deployment's ``evolution.evaluation`` section names the evaluation
tasks, the scorer, and the publication policy that decides whether Reef
publishes the candidate: ``compare``, ``floor`` or ``none``. `Evaluation: the
publication policy
<../user-guide/evolve-your-harness.rst#evaluation-the-publication-policy>`__
explains each policy and the score cache, and `Harness evolution keys
<../reference/configuration.rst#harness-evolution-keys>`__ lists every key.

Choose the policy that fits how the method decides. GEPA and Meta-Harness
compare means. ACE and ReasoningBank learn online from traffic with ``none``.
The beta Reefine recipe uses ``floor`` with one health task.

The evaluation reads the score cache that ``tools.evaluate`` fills. So a
candidate that the method already scored with ``tools.evaluate`` on the
step's tasks, ``workspace.evaluation_repeats()`` times each, needs no new
episodes in the evaluation.

Score an episode
~~~~~~~~~~~~~~~~

An ``EpisodeScorer`` (``reef.evaluation.harness``) scores one finished
episode. ``score(task, result)`` returns a finite number; higher is better.
``result`` holds the exit code, stdout, stderr, and the parsed
``trajectory``. An episode that could not run does not reach the scorer.

.. code:: python

   from reef.evaluation.harness import EpisodeScorer
   from reef.harness.episodes.trajectory import final_assistant_text

   EXPECTED = {"[fib]": "2880067194370816120"}


   class ExactAnswer(EpisodeScorer):
       def score(self, task, result):
           """1.0 when the final assistant text ends with the task's expected answer."""
           text = final_assistant_text(result.trajectory) or ""
           return 1.0 if text.strip().endswith(EXPECTED[task.split()[0]]) else 0.0

The ``[fib]`` prefix is a convention of this scorer: Reef passes the task text
through unchanged. A judge that calls a model overrides
``score_with_models(task, result, models)``, which receives the step's model
bindings. The evaluation always calls ``score_with_models``, and its default
calls ``score``. ``score`` stays abstract, so a judge implements it too, for
example by raising an error (see `Scenario model configuration
<../user-guide/scenario-models.rst#harness-evolution>`__).
``VerifierRewardScorer`` scores a Harbor task episode by its verifier's
reward.

Promote traffic prompts
~~~~~~~~~~~~~~~~~~~~~~~

With ``evolution.evaluation.promote_failures: true``, prompts of the step's
traces become permanent evaluation tasks. ``evolution.evaluation.promote``
names a ``Promoter`` (``reef.evaluation.harness``). Its ``promote(samples)``
receives the step's trace samples and returns the prompts to promote. Without
a promoter, Reef promotes the user prompt of every trace. Either way, Reef
removes duplicates, screens the prompts, and limits how many it adds.

Configure a deployment
----------------------

A deployment names ``reef.recipe.harness:HarnessEvolutionRecipe`` and puts
the method in ``evolution.method``:

.. code:: yaml

   recipe:
     implementation: reef.recipe.harness:HarnessEvolutionRecipe
     config:
       evolution:
         adapter: pi
         seed: methods/seed            # holds rules.md, skills/<name>/SKILL.md, ...
         method: methods.reflect:Reflect
         evaluation:
           tasks:
             - "[fib] Compute fib(90) exactly. Reply with the integer alone on the last line."
           scorer: methods.scoring:ExactAnswer
           policy: compare
   inference:
     upstream-url: ${REEF_UPSTREAM_URL}
     upstream-model: qwen3-8b

``reef serve -c`` starts this deployment. Start Reef where it can import the
method package. Give ``-c`` an absolute path, or Reef resolves it against the
working directory. The files in ``tutorials/evolve-your-harness/configs/``
are working examples. `Recipe configuration
<../reference/configuration.rst#recipe-configuration>`__ describes presets and
composite recipes. `Harness evolution keys
<../reference/configuration.rst#harness-evolution-keys>`__ lists every field.

``seed`` is the directory of the first tree. It becomes the base release of a
new scenario, and a scenario that already has releases keeps its head. A
relative path resolves against the directory that ``reef serve`` runs in. The
seed must pass validation for the adapter, or startup fails.

Keep the evaluation tasks short: a ``compare`` step scores each task on the
candidate, and on the head when the history holds no score for it.

Method settings
~~~~~~~~~~~~~~~

A method with settings overrides the class method ``from_config(config)``.
The deployment then uses the mapping form,
``method: {module: methods.reflect:Reflect, config: {model: reflection}}``.
The default ``from_config`` builds the class with no arguments and refuses a
non-empty ``config``. Parse the config into a ``PluginSettings`` dataclass
(``reef.core.plugins``), which refuses an unknown key:

.. code:: python

   from dataclasses import dataclass

   from reef.core.config import config_option
   from reef.core.plugins import PluginSettings


   @dataclass(frozen=True)
   class ReflectSettings(PluginSettings):
       model: str = config_option("served", help="The model binding that writes the rules.")


   class Reflect(Method):
       def __init__(self, settings: ReflectSettings) -> None:
           self.model = settings.model

       @classmethod
       def from_config(cls, config):
           return cls(ReflectSettings.parse(config, plugin_class=cls))

The step then calls ``tools.models[self.model]``. A model other than
``served`` comes from ``evolution.models``, and its key comes from the
environment variable that ``api_key_env`` names:

.. code:: yaml

   evolution:
     method: {module: methods.reflect:Reflect, config: {model: reflection}}
     models:
       reflection:
         url: https://api.openai.com
         model: gpt-4o
         api_key_env: OPENAI_API_KEY

Package a method
----------------

A method in this repository is one package under ``recipes/``:

.. code:: text

   recipes/<name>/
     __init__.py     exports the Method
     method.py       the Method, and its PluginSettings when it takes settings
     *.md            its prompts, read at import
     README.md       what the method does, how it uses the workspace, and how to run it
     examples/       runnable deployments

A method needs no recipe of its own. Subclass ``HarnessEvolutionRecipe`` only
when the method changes more than the tree. SkillClaw's recipe adds a
delivery surface that injects its skill catalog. The beta Reefine recipe runs
a coding agent in a sandbox that it configures. A subclass that reads keys of
its own removes them from the ``evolution`` section in ``_recipe_kwargs``
before it calls the base class.

These packages are complete methods:

- `GEPA <../../recipes/gepa/README.md>`__: reflective prompt evolution. It
  keeps a Pareto archive in ``state/``, scores minibatches with
  ``tools.evaluate``, and settles each candidate from the history.
- `Meta-Harness <../../recipes/meta_harness/README.md>`__: a model reads the
  earlier candidates, their scores and their notes from the history, and
  writes a whole new tree.
- `ACE <../../recipes/ace/README.md>`__ (Agentic Context Engineering): a
  reflector call and a curator call per trace edit a playbook. ``state/``
  holds the playbook, and ``rules.md`` shows it to the agent.
- `ReasoningBank <../../recipes/reasoning_bank/README.md>`__: each trace is a
  success or a failure, by its score or by a judge model. An extractor turns
  it into memory items, each an Agent Skill ``skills/rb-<slug>/SKILL.md``.

Untrusted input
---------------

Treat traces and request text as client-controlled data. If a method puts
their text in a model prompt, fence it first. With ``promote_failures``, a
trace's user prompt can also become an evaluation task. Read ``trace.client``
when a decision depends on who supplied a trace:

.. code:: python

   from reef.core.prompts import untrusted_text

   tagged = [trace for trace in workspace.traces() if trace.client != "untagged"]
   shown = untrusted_text("\n\n".join(f"{trace.task}\n{trace.reply}" for trace in tagged))
   prompt = f"Requests from tagged clients:\n{shown}\n\nWrite one skill that would help."

``untrusted_text`` places text inside a block with random delimiters. Text
inside cannot close the block and pose as the prompt's author. A client can
set its own tag, so trust ``trace.client`` as an identity only if a gateway
sets the tag.

Reef screens prompts before it adds them as evaluation tasks. It skips
prompts that carry credentials or instruction overrides, such as ``ignore
the previous instructions``, a forged ``new system prompt:``, or a
chat-template control token. The step counts these under
``screened_tasks``. ``evolution.evaluation.max_promoted_per_client`` limits
the promoted prompts of each tagged client, and
``evolution.evaluation.max_promoted_tasks`` limits all of them.

Put code-bearing kinds that client text can shape (``code_extension``,
``native_tool`` and ``native_hook``) in ``evolution.review_kinds``, so that a
person reviews them before publication. A ``native_graph`` holds no code and
can publish on the evaluation alone; add it to ``review_kinds`` if every loop
change needs a review. A ``native_loop`` always waits for a person.

.. warning::

   Do not put credentials in tree files. Validation refuses a config file
   with a credential field (a name such as ``apiKey``, ``api_key_env``,
   ``token``, ``secrets`` or ``password``) and any other file that holds a
   credential-shaped literal. This applies to the seed, a method's candidate
   and a client's proposal. The served model binds at render time through the
   adapter's model binding, and a named model in ``evolution.models`` reads
   its key from ``api_key_env``, so a tree never needs one.
