Write a harness method
======================

.. page::
   :for: authors of a harness evolution method
   :needs: the loop and the harness tree from `Evolve your harness <../user-guide/evolve-your-harness.rst>`__
   :outcome: a method that runs in Reef's steps, the scorer it needs, and a deployment that names it

A harness method decides how the harness tree changes. It is one class with
one operation:

.. code:: python

   from reef.train.harness_backend import Method, StepContext, StepResult


   class MyMethod(Method):
       def step(self, context: StepContext) -> StepResult: ...

At each step, Reef calls ``step`` with a ``StepContext`` of two parts:

- ``context.workspace``: the step's directory. It holds a copy of the head
  tree, the step's material, and the method's own files.
- ``context.tools``: the services that Reef runs for the method: model calls,
  the tree check, the evaluation of a tree on tasks, and a run of the
  deployment's coding agent.

The method edits ``workspace.harness``. The tree there when ``step`` returns
is the step's candidate. Reef then validates it, evaluates it, records the
result in the history, and publishes a selected candidate (see `The loop
<../user-guide/evolve-your-harness.rst#the-loop>`__). A method never
subclasses the backend.

A small method
--------------

This method makes one model call per step. It shows the served model the
traces that scored below 1, and writes the reply to ``rules.md``:

.. code:: python

   from reef.core.prompts import untrusted_text
   from reef.harness.tree import TreePatch
   from reef.train.harness_backend import Method, StepContext, StepResult

   PROMPT = "Rewrite these agent rules so that the failing requests succeed. Reply with the rules alone."


   class Reflect(Method):
       """One model call per step: rewrite rules.md from the failing traces."""

       def step(self, context: StepContext) -> StepResult:
           workspace, tools = context.workspace, context.tools
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

`Configure a deployment`_ shows the deployment that runs it, and `Score an
episode`_ shows a scorer.

A method made of prompts
------------------------

A method can also be a folder of prompts that a coding agent runs:
``reef.train.harness_backend:AgentMethod``. Each step runs the deployment's
coding agent (``evolution.agent``) once in the step's workspace, and the tree
it leaves in ``harness/`` is the candidate. The folder holds:

- ``START.md``: the prompt of a step. Required.
- ``RESUME.md``: the prompt of a step whose ``state/`` holds files from an
  earlier step. Optional; ``START.md`` takes its place.
- ``rules.md``, ``skills/<name>/SKILL.md`` and the other tree paths: the
  agent's own harness tree, rendered for the agent's adapter after Reef's
  protocol rules.
- ``README.md``: text for people. The agent does not get it.

In a prompt, ``{step}`` becomes the step's number and ``{history}`` the
history's path from the workspace. Reef's protocol rules
(``reef/train/harness_backend/agent/protocol.md``) tell the agent the rest:
the workspace, the files it writes, and its tools (see `Run a coding
agent`_). After the agent exits, the method reads ``output/summary.md`` (the
step's summary), ``output/notes.json`` (a JSON object added to its notes) and
``output/requires.json`` (a JSON array, its requires). A run that does not
exit 0 leaves the head tree, so the step skips.

.. code:: yaml

   evolution:
     method:
       module: reef.train.harness_backend:AgentMethod
       config:
         prompts: methods/rules-editor   # the folder
         model: proposer                  # served, or an evolution.models entry
     agent:
       adapter: claude
       sandbox: bwrap

What Reef does with the step
----------------------------

After ``step`` returns, Reef reads the tree in ``harness/``:

- An unchanged tree skips the step.
- A tree that fails validation skips the step, and the step's ``skipped``
  metric names the problems. There is no retry, so call ``tools.check``
  before you return.
- Any other tree is the candidate. The deployment's publication policy
  evaluates it.

If ``step`` raises, the step fails and publishes nothing. The step record
keeps the model calls made before the failure.

A recheck and a client's proposal take an automatic step without calling the
method. A step that answers a training request always calls it.

The workspace
-------------

Each step gets one directory,
``<evolution.state_dir>/<scenario>/<component>/steps/<step>``. Reef keeps it
after the step as the step record.

- ``harness/`` (read-write): a copy of the head tree. The tree here when
  ``step`` returns is the candidate.
- ``material/`` (read-only): what the step learns from (below).
- ``state/`` (read-write): the method's own files, committed with the step
  (see `Keep state between steps`_).
- ``scratch/`` (read-write): temporary files, removed when the step ends.
- ``output/`` (read-write): files the method keeps with the step record.

``material/`` holds these files. A Python method reads them through the
``Workspace`` methods in parentheses:

- ``traces/<n>/trace.json`` (``traces()``, in trace order): one ``Trace`` per
  traffic sample of the step's job data, with its ATIF ``trajectory.json``.
  An offline step then adds one ``trace.json`` per train task it rolled out
  on the head tree.
- ``request.json`` (``request()``): the ``TrainingRequest`` that the step
  answers. Only a request step has it; ``request()`` is ``None`` otherwise.
- ``evaluation.json`` (``tasks()``, ``evaluation_repeats()``): the step's
  evaluation tasks, the seed tasks first, and the episodes per task and tree
  (0 when the deployment configured no scorer).
- ``failures.json``: the failure manifest of the head tree, after the first
  evaluation.
- ``host.json`` (``host()``): the ``adapter``; the ``parts`` of its tree,
  each with its tree ``path``, its ``format``, the path it ``renders`` to and
  what it ``holds``; its ``config_targets``; and ``tasks_are_directories``. A
  file at a path that no part takes fails validation.
  ``get_adapter(host()["adapter"]).layout.locate(path)`` returns the part of a
  tree path, or ``None``.

``Workspace`` also has the directory paths (``harness``, ``material``,
``state``, ``scratch``, ``output``), the step number ``step`` (1 or more),
``tree()`` and ``write_tree(tree)`` to read and replace the tree in
``harness/``, and ``history()`` (see `Read the history`_).

A ``Trace`` is a client's request (``source`` is ``traffic``) or an episode
that an offline step ran on a train task (``source`` is ``rollout``):

- ``task``: the last user prompt of the request, or the instruction of the
  rollout's task. ``None`` for a request without a user prompt.
- ``reply``: the final assistant text, else ``None``. With the inference
  stages, every model call is a trace, and a call that ends on a tool call
  has no reply.
- ``score``: the reported score, or the scorer's score for a rollout.
  ``None`` for unscored traffic and for a rollout that could not run.
- ``feedback``: the reported feedback as text (structured feedback as JSON
  text). ``None`` when there is none and for a rollout.
- ``failure``: why the rollout's episode failed, else ``None``.
- ``record``: the agent record id of the request, or the history's episode
  id of the rollout.
- ``client``: the client's ``x-reef-tag-client`` tag, else its session tag,
  else ``untagged``. Always ``untagged`` for a rollout.
- ``directory``: the trace's directory. ``trajectory()`` reads a traffic
  trace's ATIF trajectory, and returns ``None`` for a rollout.

Material is client data. See `Untrusted input`_ before its text enters a
model prompt.

The tools
---------

- ``tools.models``: the model bindings. ``served`` is the model under test,
  and ``tools.models["<name>"]`` is an ``evolution.models`` entry.
  ``chat(messages, *, timeout_s=None, **params)`` takes OpenAI-shaped
  messages and returns the assistant text. ``complete(body)`` sends a raw
  request body and returns the provider's response.
- ``tools.calls``: the step's call meter. ``spend()`` counts one call,
  ``record(entry)`` appends one call to the record, and ``note(kind, text,
  *, failed=False)`` adds one line to the step's activity.
- ``tools.check(tree)``: the ``ValidationResult`` of ``tree`` for the
  adapter, with ``passed``, ``problems`` and ``message()``.
- ``tools.evaluate(tree, tasks, *, repeats=1)``: ``repeats`` scored
  ``EpisodeRecord`` values per task of the list ``tasks``, task by task.
- ``tools.run_agent(prompt, *, tree, model)``: one run of the deployment's
  coding agent in the workspace, as an ``AgentOutcome``. Only a method whose
  class sets ``runs_agent`` gets an agent (see `Run a coding agent`_).

Every call through ``tools.models`` counts against
``evolution.max_model_calls_per_step``, and a call past the limit raises
``RuntimeError``. Each call lands in the step record's ``calls.json``. A
process that calls a model by itself, such as a coding agent behind a
gateway, must call ``tools.calls.spend()`` before each call and
``tools.calls.record(entry)`` after it. Then the budget and the record stay
complete.

``tools.evaluate`` runs episodes as the step's evaluation does: the same
rendering, worker pool and scorer. An ``EpisodeRecord`` has the ``task``, the
``repeat`` index, the ``score``, the ``reply`` and the ``episode`` id. The
``score`` is ``None`` for an episode that could not run, and ``failure`` says
why. The history keeps every episode and reuses a scored episode of the same
tree, task, repeat and evaluation context, so asking again costs nothing. Ask
for more repeats to get new samples. A tree that the method evaluates joins
the history as a candidate of the step.

``tools.evaluate`` refuses a call, runs no episode, and adds nothing to the
history in these cases:

- ``RuntimeError``: the deployment configured no scorer, or the call needs
  more episodes than ``evolution.max_episodes_per_step`` leaves. Reused
  episodes do not count against that limit.
- ``ValueError``: the tree fails ``check``.
- ``TypeError``: ``tasks`` is one string.

Run a coding agent
~~~~~~~~~~~~~~~~~~

``tools.run_agent`` runs the deployment's coding agent once, and returns when
the agent exits or runs past ``evolution.agent.timeout_s``. ``evolution.agent``
names the agent's adapter (any adapter whose prompt is text, ``pi`` by
default), its binary, and the sandbox it runs in; see the `configuration
reference <../reference/configuration.rst#harness-evolution-keys>`__. A
method whose class sets ``runs_agent = True`` gets the agent, and Reef refuses
to start such a method when the deployment configures none.

The agent's working directory is the workspace. It can write ``harness/``,
``state/``, ``scratch/`` and ``output/``, and it reads ``material/`` and the
history. Its own harness is ``tree``, rendered for the agent's adapter after
Reef's protocol rules. The agent holds no credential: its model calls go to
the binding that ``model`` names (``served`` or an ``evolution.models`` entry)
through a loopback gateway at ``$REEF_AGENT_URL``. The gateway spends each
call from the step's budget and records it in ``calls.json``. The same
address serves two tools on the workspace's trees, so a method that is only
prompts can check and score its work:

- ``GET $REEF_AGENT_URL/tools`` lists the tools and their arguments.
- ``POST $REEF_AGENT_URL/check`` with ``{"tree": "harness"}`` answers
  ``passed`` and ``problems``.
- ``POST $REEF_AGENT_URL/evaluate`` with ``{"tree": "harness", "tasks":
  [...], "repeats": 1}`` answers one record per episode, through the score
  cache and the step's episode budget.

A ``tree`` argument is a directory of the workspace, such as ``harness`` or
``scratch/try-2``. A refused call answers 400 with the reason.

``AgentOutcome`` says how the run ended: ``exit_code``, ``timed_out``, or the
``error`` that kept it from starting, and its ``seconds``. The agent's session
log stays in ``output/agent-session.jsonl``. ``run_agent`` raises
``RuntimeError`` when no agent runs for the method, ``KeyError`` when
``model`` names no binding, and ``ValueError`` when ``tree`` does not render
for the agent's adapter.

The result
----------

``StepResult`` has three optional fields:

- ``summary``: plain text that says what the step did. The step pages show
  it, and the commit metrics keep it as ``method_summary``.
- ``notes``: a JSON object, kept as ``method_notes``. ``StepResult`` refuses
  a value that is not one. Reef never reads it to make a decision.
- ``requires``: what the change needs on the person's machine, as
  ``reef.core.requirements`` items. Reef reads it on a request step only (see
  `Answer a training request`_).

Reef writes the result to the step record's ``result.json``. The history keeps
``summary`` and ``notes`` with the candidate, whether the step publishes or
not.

Keep state between steps
------------------------

A method keeps what it learns as files in ``workspace.state``: an archive, a
population, a playbook, or counters. Reef commits every file of ``state/``
with the step and writes the files back into the next step's ``state/``. The
state lives in the scenario's commit history, so it survives a restart.

- A published, rejected or skipped step commits its state.
- A step that fails commits none of its state. The previous state stays.
- The files must be UTF-8 text, at most 16 MiB together. A ``state/`` that
  breaks either rule skips the step, and the previous state stays.

Read the history
----------------

The history holds every tree that a step proposed or evaluated, every
episode, and every decision. ``workspace.history()`` returns it as a
``reef.evaluation.harness.History``. A method reads it and never writes it.

- ``candidates``: each tree's ``CandidateRecord`` by checksum, with
  ``step``, ``origin`` (``seed``, ``method``, ``proposal`` or ``release``),
  ``parent``, ``summary`` and ``notes``.
- ``episodes``: every ``EpisodeRecord``, in run order.
- ``decisions``: every ``DecisionRecord``, with ``step``, ``tree`` (the
  candidate), ``head`` (the tree that the step started from), ``selected``
  (published, or held for review), ``policy`` and ``reason``.
- ``tree(checksum)``: the files of a tree that the history holds.
- ``scores(checksum, tasks, *, repeats=1)``: the tree's scores, task by task
  and repeat by repeat. Each is the latest episode of its pair, ``None``
  where none ran or the latest could not run.
- ``decision(step)``: the decision on that step's candidate, else ``None``.
- ``episode_dir(episode)``: the episode's trajectory files and
  ``episode.json``.

A tree's checksum is ``HarnessTree.checksum()``. The history also keeps the
records of a step attempt that did not commit. The next attempt reuses its
step number, so skip records whose ``step`` is ``workspace.step`` or more when
you count or compare what the method tried.

This code reads what happened to the previous step's candidate:

.. code:: python

   history = workspace.history()
   decision = history.decision(workspace.step - 1)
   repeats = max(1, workspace.evaluation_repeats())
   if decision is not None:
       selected = decision.selected  # published, or held for review
       scores = history.scores(decision.tree, workspace.tasks(), repeats=repeats)
       head_scores = history.scores(decision.head, workspace.tasks(), repeats=repeats)

Answer a training request
-------------------------

In ``manual`` and ``hybrid`` training mode, a step can answer a request from
``POST /reef/train`` (see `Manual training
<../reference/http-api.rst#manual-training>`__). Every method takes such a
step. ``workspace.request()`` returns the ``TrainingRequest``: ``id``,
``text``, ``session``, ``release_id``, ``requires``, and ``client``, the
client's report of its machine. The text is client text (see `Untrusted
input`_). In ``manual`` mode the step has no traces. In ``hybrid`` mode it has
up to ``job_samples`` traces from the next automatic job data, or none.

Return ``requires`` when the change needs something on the person's machine
that the request did not name: an ``env`` variable, a ``permission``, a
``service`` or a ``binary``. Reef screens each item for credentials and
instruction overrides. It adds the items that pass to the request's own
items in the step's ``training_request.requires``, and the release manifest
serves them to clients.

Evaluation
----------

The deployment's ``evolution.evaluation`` section names the tasks, the scorer
and the publication policy. `Evaluation: the publication policy
<../user-guide/evolve-your-harness.rst#evaluation-the-publication-policy>`__
explains the policies. The evaluation reads the episodes that
``tools.evaluate`` stored. A candidate that the method already scored on the
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

Reef passes the task text unchanged; the ``[fib]`` prefix is this scorer's
own convention. The evaluation calls ``score_with_models(task, result,
models)``, whose default calls ``score``. A judge that calls a model
overrides ``score_with_models`` to use the step's model bindings. ``score``
stays abstract, so a judge implements it too, for example by raising.
``VerifierRewardScorer`` scores a Harbor task episode by its verifier's
reward.

Collect traffic prompts
~~~~~~~~~~~~~~~~~~~~~~~

With ``evolution.evaluation.collect_failures: true``, prompts of the step's
traces become permanent evaluation tasks. ``evolution.evaluation.collector``
can name a ``TaskCollector`` (``reef.evaluation.harness``). Its
``collect(samples)`` receives the step's trace samples and returns the
prompts to collect. Without a collector, Reef collects the user prompt of
every trace. Either way, Reef removes duplicates, screens the prompts, and
applies ``max_collected_tasks`` and ``max_collected_per_client``.

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
         seed: methods/seed            # the first tree; it must pass validation
         method: methods.reflect:Reflect
         evaluation:
           tasks:
             - "[fib] Compute fib(90) exactly. Reply with the integer alone on the last line."
           scorer: methods.scoring:ExactAnswer
           policy: compare
   inference:
     upstream-url: ${REEF_UPSTREAM_URL}
     upstream-model: qwen3-8b

Start ``reef serve -c`` where Reef can import the method package.
``tutorials/evolve-your-harness/configs/`` has working examples, and `Harness
evolution keys <../reference/configuration.rst#harness-evolution-keys>`__
lists every key and default.

Method settings
~~~~~~~~~~~~~~~

The default ``from_config`` builds the class with no arguments and refuses a
non-empty ``config``. A method with settings overrides ``from_config`` and
parses ``config`` into a ``PluginSettings`` dataclass
(``reef.core.plugins``), which refuses an unknown key. Scorers and collectors
take settings the same way.

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

The deployment then uses the mapping form. A model other than ``served``
comes from ``evolution.models``, which reads its key from the environment
variable that ``api_key_env`` names:

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

A method in this repository is one package under ``recipes/<name>/``. Its
``__init__.py`` exports the ``Method``. ``method.py`` holds the method and
its ``PluginSettings``. Prompts are ``*.md`` files read at import.
``README.md`` says what the method does, how it uses the workspace, and how
to run it. ``examples/`` holds runnable deployments.

A method needs no recipe of its own. Subclass ``HarnessEvolutionRecipe`` only
when the method changes more than the tree. Such a subclass removes its own
keys from the ``evolution`` section in ``_recipe_kwargs`` before it calls the
base class. `Choose or write a method
<../user-guide/evolve-your-harness.rst#choose-or-write-a-method>`__ lists the
complete methods in this repository.

Untrusted input
---------------

Traces and request text are client-controlled data. Wrap their text with
``reef.core.prompts.untrusted_text`` before it enters a model prompt. The
function puts the text in a block with random delimiters, so the text cannot
close the block and pose as the prompt's author.

A client can set its own tag. Trust ``trace.client`` as an identity only when
a gateway sets the tag.

Client text can shape the parts that hold code (``extension``,
``native_tool`` and ``native_hook``). Name them in ``evolution.review_parts``
so that a person reviews them before publication. A ``native_loop`` change always waits for a
person. `Evaluation: the publication policy
<../user-guide/evolve-your-harness.rst#evaluation-the-publication-policy>`__
describes review and how Reef screens collected prompts.

.. warning::

   Do not put credentials in tree files. Validation refuses a config file
   with a credential field (such as ``apiKey``, ``api_key_env``, ``token``,
   ``secrets`` or ``password``) and any other file with a credential-shaped
   literal. The served model binds at render time, and a named model reads
   its key from ``api_key_env``, so a tree never needs one.
