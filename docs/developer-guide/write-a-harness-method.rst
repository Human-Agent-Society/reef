Write a harness method
======================

A harness method decides which change to try at each step and how to score
an evaluation episode. Reef does the rest: it reads the head tree from the
scenario's git history, applies the method's patch, validates the candidate
tree for the adapter, runs the evaluation, commits the result, and publishes
a selected tree as a new release.

See `Evolve your harness <../user-guide/evolve-your-harness.rst>`__ for the
workflow, the tree layout, and a runnable example.

The parts of a method
---------------------

A method implements up to four classes. Only the proposer is always the
method's own; the other parts have defaults.

+----------------------+-----------------------+--------------------------------------------------+
| Part                 | Base class            | When the method needs it                         |
+======================+=======================+==================================================+
| Proposer             | ``Proposer``          | Always: it returns the change for each step.     |
+----------------------+-----------------------+--------------------------------------------------+
| Episode scorer       | ``EpisodeScorer``     | When no recipe default scores its episodes.      |
+----------------------+-----------------------+--------------------------------------------------+
| Selection policy     | ``SelectionPlugin``   | When it decides publication its own way; the     |
|                      |                       | default is ``score_comparison``.                 |
+----------------------+-----------------------+--------------------------------------------------+
| Promoter             | ``Promoter``          | When it chooses which failing prompts become     |
|                      |                       | evaluation tasks (``evolution.promote_failures``)|
+----------------------+-----------------------+--------------------------------------------------+

``Proposer``, ``Promoter`` and the step's types are in
``reef.train.harness_backend``; ``EpisodeScorer`` is in
``reef.evaluation.harness``, and ``SelectionPlugin`` in ``reef.evaluation``.
A method never subclasses the backend. Everything a step gives the method is
in one ``ProposalContext``, and everything the method gives back is in one
``TreeProposal``, including the state the method keeps between steps.

The deployment file names each class, for example
``propose: methods.mine:MyProposer``. A class with settings uses the mapping
form ``{module: methods.mine:MyProposer, config: {...}}`` and overrides the
class method ``from_config(config)``. The default ``from_config`` builds the
class with no arguments and refuses a non-empty ``config``.

A small method
--------------

This method adds one rule to ``rules.md`` the first time a step's samples
contain a failure, and scores an episode by its final answer. The ``[fib]`` prefix is a
convention of this method: Reef passes the task text through unchanged.

.. code:: python

   from reef.evaluation.harness import EpisodeScorer
   from reef.harness.episodes.trajectory import final_assistant_text
   from reef.harness.tree import TreePatch
   from reef.train.harness_backend import ProposalContext, Proposer, TreeProposal

   RULE = "State the final answer alone on the last line.\n"
   EXPECTED = {"[fib]": "2880067194370816120"}


   class FinalLineRule(Proposer):
       def propose(self, context: ProposalContext) -> TreeProposal | None:
           """Add the rule after a failing sample; once the rules hold it, skip the step."""
           if all((sample.metadata.get("reward") or 0.0) > 0.0 for sample in context.samples):
               return None
           rules = context.tree.files.get("rules.md", "")
           if RULE in rules:
               return None
           return TreeProposal(TreePatch({"rules.md": rules + RULE}))


   class ExactAnswer(EpisodeScorer):
       def score(self, task, result):
           """1.0 when the final assistant text ends with the task's expected answer."""
           text = final_assistant_text(result.trajectory) or ""
           return 1.0 if text.strip().endswith(EXPECTED[task.split()[0]]) else 0.0

`Configure a deployment`_ shows the deployment file that runs it.

Propose a change
----------------

``Proposer.propose(context)`` is called once per step, and returns a
``TreeProposal`` or ``None``.

The context
~~~~~~~~~~~

+----------------------+------------------------------------------------------------------------+
| Field                | Value                                                                  |
+======================+========================================================================+
| ``step``             | This step's number, 1 or more.                                         |
+----------------------+------------------------------------------------------------------------+
| ``tree``             | The head ``HarnessTree``, the tree the patch applies to.               |
|                      | ``tree.files`` maps each path to its text.                             |
+----------------------+------------------------------------------------------------------------+
| ``descriptor``       | The adapter the tree renders for: the kinds and config targets a tree  |
|                      | may hold there.                                                        |
+----------------------+------------------------------------------------------------------------+
| ``samples``          | The step's trace samples (ATIF ``TrajectoryItem`` values) in the order |
|                      | of its job data. Read the reward and feedback from ``sample.metadata`` |
|                      | (``reward`` is absent for unscored traffic) and the original provider  |
|                      | exchanges with ``reef.core.trajectories.recorded_payloads``.           |
+----------------------+------------------------------------------------------------------------+
| ``sources``          | One ``SampleSource`` per sample, in sample order. See                  |
|                      | `Untrusted input`_.                                                    |
+----------------------+------------------------------------------------------------------------+
| ``request``          | The training instruction this step answers, ``None`` for an automatic  |
|                      | step. See `Manual requests`_.                                          |
+----------------------+------------------------------------------------------------------------+
| ``evaluation_tasks`` | The tasks this step's evaluation runs: the seed tasks, then the        |
|                      | promoted ones.                                                         |
+----------------------+------------------------------------------------------------------------+
| ``models``           | The model bindings the proposer may call. See `Call models and run     |
|                      | trials`_.                                                              |
+----------------------+------------------------------------------------------------------------+
| ``manifest``         | The previous step's ``FailureManifest``, ``None`` before the first.    |
+----------------------+------------------------------------------------------------------------+
| ``rejected``         | The latest rejected proposals, oldest first. Each                      |
|                      | ``RejectedProposal`` has ``step``, ``patch`` and ``reason``, so a      |
|                      | method can stop proposing what the evaluation refused.                 |
+----------------------+------------------------------------------------------------------------+
| ``problems``         | Empty on the first call of a step; the validation problems of the      |
|                      | previous patch on the retry.                                           |
+----------------------+------------------------------------------------------------------------+
| ``method_state``     | The state the method committed with the latest step, empty before its  |
|                      | first. See `Keep state between steps`_.                                |
+----------------------+------------------------------------------------------------------------+
| ``last_outcome``     | The ``StepOutcome`` of the latest evaluated step, ``None`` before the  |
|                      | first.                                                                 |
+----------------------+------------------------------------------------------------------------+
| ``trials``           | A ``TrialRunner`` for trial episodes of a tree.                        |
+----------------------+------------------------------------------------------------------------+
| ``agent_host``       | Set only for a proposer that runs a coding agent. See `Run a coding    |
|                      | agent`_.                                                               |
+----------------------+------------------------------------------------------------------------+

The proposal
~~~~~~~~~~~~

A ``TreePatch`` maps a tree path to its new text, or to ``None`` to remove
the file. A path the patch does not name keeps its text:

.. code:: python

   TreePatch({"rules.md": "Answer briefly.\n", "skills/old-skill/SKILL.md": None})

``TreeProposal`` carries the patch and three optional fields:

- ``requires``: what the change needs on the person's machine, as
  ``reef.core.requirements`` items, beyond the request's own. See
  `Manual requests`_.
- ``notes``: a JSON mapping that the commit metrics keep as
  ``proposal_notes``. Reef never reads it to make a decision.
- ``method_state``: the method's new state. ``None`` keeps the committed
  state.

The return value decides what the step does:

+---------------------------------+----------------------------------------------------------------+
| ``propose`` returns             | The step                                                       |
+=================================+================================================================+
| ``TreeProposal(patch)``         | Applies the patch to ``context.tree``, validates the result,   |
|                                 | and evaluates it.                                              |
+---------------------------------+----------------------------------------------------------------+
| ``TreeProposal(TreePatch({}))`` | Skips the evaluation, but commits ``method_state`` and         |
|                                 | ``notes``.                                                     |
+---------------------------------+----------------------------------------------------------------+
| ``None``                        | Skips the step and records nothing for the method.             |
+---------------------------------+----------------------------------------------------------------+

If ``propose`` raises, the step fails and publishes nothing.

Validation
~~~~~~~~~~

Reef validates the patched tree for the adapter before it runs an episode.
Every path must sit on the layout and be of a kind the adapter renders, every
file must pass its kind's content checks, and the tree must render. A patch
that changes no text is a problem too.

A patch with problems gets one retry: Reef calls ``propose`` again with the
same context and ``problems`` set. A second patch with problems skips the
step, and the reason names the problems. The episodes test only a valid tree.

Keep state between steps
------------------------

A method keeps its search state (an archive, a population, a counter) in
``method_state``, a JSON mapping. Reef commits the proposal's
``method_state`` with the step, whatever the step's result, and passes it
back as ``context.method_state`` at the next step. The state lives in the
scenario's commit history, so it survives a restart and needs no files of
the method's own.

The result of an evaluated step reaches the method at its next proposal:
``context.last_outcome`` is a ``StepOutcome`` with ``step``, ``patch``,
``selected``, ``reason``, and the per-task ``candidate_scores`` and
``current_scores`` (``None`` for an episode that could not run). It is the
outcome of the latest step that evaluated the method's own proposal: a
recheck and a client's proposal from ``POST /reef/harness/proposals`` leave
it as it was. A method folds it into its state at the next step. Compare
``last_outcome.step`` with the step the method proposed: a proposal that the
step skipped (one that failed validation, say) has no outcome.

.. code:: python

   class CountingProposer(Proposer):
       def propose(self, context):
           state = dict(context.method_state)
           outcome = context.last_outcome
           if outcome is not None and outcome.step == state.get("proposed_step"):
               state["wins"] = state.get("wins", 0) + int(outcome.selected)
           patch = ...  # this step's change
           return TreeProposal(patch, method_state={**state, "proposed_step": context.step})

GEPA (``recipes/gepa/``) keeps its whole candidate archive this way, and
Meta-Harness (``recipes/meta_harness/``) keeps its population of trees.

Call models and run trials
--------------------------

``context.models.served`` is the model under test. Named bindings, such as
``context.models["teacher"]``, come from ``evolution.models``. Call a
binding with ``binding.chat(messages, *, timeout_s=None, **params) -> str``.
The messages use the OpenAI shape whatever the endpoint's dialect, and the
call returns the assistant text. ``binding.complete(body)`` sends a raw
request body and returns the provider's response.

Every call counts against ``evolution.max_model_calls_per_step`` (0, no cap,
by default). A call past the cap raises ``RuntimeError``. The step record's
``proposer.json`` keeps every call when ``evolution.step_record_dir`` is set.

``context.trials.run(tree, tasks)`` runs one episode of ``tree`` per task,
in task order, and returns a ``ScoredEpisode`` for each, with ``score``
(``None`` when the episode could not run), ``failure`` and ``reply``. Trials
use the evaluation's worker pool, model binding and scorer, so a trial
measures what the evaluation would. A method that probes its candidates
before it settles on a patch, such as GEPA's minibatch check, uses trials
instead of launching episodes itself.

Run a coding agent
------------------

A proposer that runs a coding agent overrides the ``runs_agent`` property to
return ``True``. When the deployment configures ``evolution.proposer_agent``,
``context.agent_host`` is an ``AgentHost``: the adapter ``descriptor`` and
its installed ``binary``, the agent's ``executor`` (its isolation, separate
from the episodes), the step's record directory ``step_dir``, the
``timeout_s`` of the whole run and the ``trial_timeout_s`` of each trial, and
``calls``, the step's budget and record. Without that setting
``agent_host`` is ``None``.

``reef.harness.tree.TreeWorkspace`` gives the agent the head tree as plain
files: ``TreeWorkspace.create(context.tree, root)`` writes them, the agent
edits them with its own tools, and ``workspace.read()`` returns the patch
from the head tree and the files the agent wrote off the layout. For model
calls the agent process makes outside ``context.models``, call
``agent_host.calls.spend()`` before each and ``agent_host.calls.record(entry)``
after it. ``reef.recipe.reefine.agent`` is a complete agent proposer.

Manual requests
---------------

In ``manual`` or ``hybrid`` training mode, the proposer must override
``reads_requests`` to return ``True``. Otherwise the recipe fails at startup
with ``RecipeConfigError``. See `Manual training
<../reference/http-api.rst#manual-training>`__ for the request route.

``context.request`` is a ``TrainingRequest`` with ``id``, ``text``,
``session``, ``release_id``, ``requires``, and ``client``, the requesting
client's report of its machine. The text is client text: see `Untrusted
input`_. In ``manual`` mode, ``samples`` is empty. In ``hybrid`` mode, it
holds up to ``job_samples`` samples from the next automatic job data, and
may be empty.

Return ``requires`` in the proposal when the change needs something on the
person's machine that the request did not name: an ``env`` variable, a
``permission``, a ``service`` or a ``binary``. Reef merges them with the
request's own items into the step's ``training_request`` metrics, and the
release manifest serves them to clients.

Reef evaluates a request's patch and applies the same ``evolution.publish``
policy as for other steps. Pending client proposals and periodic rechecks
cannot take over the step assigned to a request.

Score an episode
----------------

``EpisodeScorer.score(task, result)`` scores one completed episode. Reef
calls it for each side the selection policy evaluates; ``floor`` evaluates
only the candidate. The ``result`` holds the exit code, stdout, stderr, and
the parsed ``trajectory``. Episodes that could not run do not reach the
scorer. A judge that calls a model overrides
``score_with_models(task, result, models)`` instead.
``VerifierRewardScorer`` scores a Harbor task episode by its verifier's
reward.

Selection policies
------------------

``evolution.selection`` names the policy that decides publication:

- ``score_comparison`` (the default) selects a candidate when the task
  pairings it wins exceed the ones it loses by more than
  ``evolution.min_win_margin``.
- ``floor`` runs only the candidate and selects it when every task reaches
  ``evolution.floor_score`` (default ``1.0``). An episode that could not run
  misses the floor. ``current_scores`` is empty.
- ``always`` selects every evaluated candidate.

A custom policy reads ``EvaluationResult.metrics``, which holds the per-task
score lists ``candidate_scores`` and ``current_scores`` in task order; a
score is ``None`` when an episode could not run. The candidate is a
``reef.evaluation.harness.HarnessCandidate``, whose ``method_state`` is the
state the proposal committed. The policy below selects a candidate only when
at least one task improves and none regresses:

.. code:: python

   from reef import CandidateEvaluator, SelectionDecision
   from reef.evaluation import SelectionPlugin


   class ParetoPlugin(SelectionPlugin):
       def __init__(self, candidate_backend: CandidateEvaluator):
           self.candidate_backend = candidate_backend

       @classmethod
       def from_config(cls, config, *, candidate_backend: CandidateEvaluator) -> "ParetoPlugin":
           return cls(candidate_backend)

       def evaluate(self, candidate):
           return self.candidate_backend.evaluate(candidate)

       def decide(self, candidate, evaluation):
           pairs = zip(
               evaluation.metrics["candidate_scores"],
               evaluation.metrics["current_scores"],
               strict=True,
           )
           scores = [(c if c is not None else -1e30, k if k is not None else -1e30) for c, k in pairs]
           selected = all(c >= k for c, k in scores) and any(c > k for c, k in scores)
           return SelectionDecision(
               selected=selected,
               policy="pareto",
               reason="no task regressed and at least one improved" if selected else "a task regressed or none improved",
               evaluation=evaluation,
           )

Set ``selection: my_pkg.policies:ParetoPlugin`` to use it. Reef checks the
class when it loads the configuration and calls ``from_config`` for each
scenario's candidate backend, with the ``config`` of the mapping form. A
``SelectionPlugin`` must both evaluate and decide; keep publication in this
flow so Reef can keep the head when a candidate is rejected.

Promote failing prompts
-----------------------

``evolution.promote`` matters only with ``evolution.promote_failures``. Its
``Promoter`` receives the step's samples and the ``FailureManifest`` through
``promote(samples, *, manifest=None)`` and returns the prompts to add as
permanent evaluation tasks. Reef deduplicates and screens the prompts, then
caps how many it adds. Without a promoter, Reef uses the user prompt of
each failing trace.

Configure a deployment
----------------------

The recipe config names the plugin classes, the tasks, the step size, and
the seed, the directory that holds the first tree:

.. code:: yaml

   recipe:
     implementation: reef.recipe.harness:HarnessEvolutionRecipe
     config:
       evolution:
         adapter: pi
         binary: pi
         batch_size: 1
         seed: methods/seed            # holds rules.md, skills/<name>/SKILL.md, ...
         propose: methods.mine:FinalLineRule
         evaluate: methods.mine:ExactAnswer
         tasks:
           - "[fib] Compute fib(90) exactly. Reply with the integer alone on the last line."
         models:                       # optional extras; each key read via api_key_env
           teacher:
             url: https://api.openai.com
             model: gpt-4o
             api_key_env: OPENAI_API_KEY
   inference:
     upstream-model: qwen3-8b

``seed`` becomes the base release of a new scenario. A relative path
resolves against the directory ``reef serve`` runs in, and the seed must
pass validation for the adapter, or startup fails. A scenario that already
has releases keeps its head: the seed applies only to a new scenario.

The YAML above is a standalone recipe preset. Save it as
``recipes/<name>.yaml`` and set ``REEF_RECIPE_CONFIG_DIR`` to that directory;
there is no default directory. Presets are read as-is, so write literal
values instead of ``${VAR}``. Variable interpolation applies to deployment
configs.

A preset describes the method and model but does not start the serving
processes. ``reef serve -c`` reads a deployment config. The one in
``tutorials/evolve-your-harness/configs/serve.yaml`` is a working example:

.. code:: yaml

   recipe:
     implementation: <name>  # resolves to recipes/<name>.yaml
   reef:
     token: reef-local
     port: 8900
   inference:
     upstream-url: ${REEF_UPSTREAM_URL}
     upstream-api-key: ${REEF_UPSTREAM_API_KEY}
     upstream-model: ${REEF_MODEL}

The tutorial selects the dotted class directly and keeps its recipe settings
in the same versioned deployment file. See `Recipe configuration
<../reference/configuration.rst#recipe-configuration>`__ for both forms.

Keep the ``tasks`` list short: Reef runs them in each evaluation step. Start
Reef where it can import the method package. Give ``-c`` an absolute path,
or Reef resolves it against the working directory. See `Harness evolution
keys <../reference/configuration.rst#harness-evolution-keys>`__ for every
field.

Two trace sources
~~~~~~~~~~~~~~~~~

The deployment's ``pipeline`` section selects the source of each step's
traces. ``pipeline.task2job.config.job_samples`` sets the number of traces
in one step. Without it, ``recipe.config.evolution.batch_size`` (default 1)
gives the number.

- The recipe's default stages train on valid reports with explicit scores. Use
  them when a grader, test, or user action supplies an outcome.
- The inference stages train on recorded inference requests and do not require
  reports. A step starts every ``job_samples`` requests. These samples carry
  no ``reward``, which the proposer must handle. The SkillClaw night example
  judges its own unscored samples later. Name the stages as follows:

  .. code:: yaml

     pipeline:
       record2record:
         module: reef.record2record:InferenceRecord2Record
       record2task:
         module: reef.record2task.traces:InferenceTasks

Package a method
----------------

A method that binds its own defaults or reads settings of its own ships a
``HarnessEvolutionRecipe`` subclass. Its ``evolution_spec()`` returns the
default proposer, scorer, selection and seed, the counterpart of
``training_spec()`` for weight training; a deployment's
``evolution.propose``, ``evolution.evaluate``, ``evolution.selection`` and
``evolution.seed`` replace each one:

.. code:: python

   from dataclasses import dataclass, field
   from pathlib import Path

   from reef.core.plugins import PluginReference
   from reef.recipe.harness import HarnessEvolutionRecipe, HarnessEvolutionSpec

   from .proposer import FinalLineRule
   from .scorer import ExactAnswer


   @dataclass(frozen=True)
   class FinalLineRecipe(HarnessEvolutionRecipe):
       name: str = field(default="final_line", kw_only=True)

       @classmethod
       def evolution_spec(cls) -> HarnessEvolutionSpec:
           return HarnessEvolutionSpec(
               proposer=PluginReference(FinalLineRule),
               scorer=PluginReference(ExactAnswer),
               seed=Path(__file__).with_name("seed"),
           )

Every method package keeps each part in the same file:

.. code:: text

   recipe.py       the HarnessEvolutionRecipe subclass: its name, its config fields, and evolution_spec()
   proposer.py     its Proposer
   scorer.py       its EpisodeScorer, when it scores episodes its own way
   selection.py    its SelectionPlugin, when it selects its own way
   processor.py    its record stages, when it reads records its own way
   seed/           its first tree

``reef/recipe/reefine/``, ``recipes/gepa/`` and ``recipes/meta_harness/``
follow this layout.

Untrusted input
---------------

Treat samples and request text as client-controlled data. If a method puts
sample text in a model prompt, fence it first. With ``promote_failures``, a
failing sample's user prompt may also become an evaluation task. Read
``sources`` when a decision depends on who supplied a sample.

.. code:: python

   import json

   from reef.core.prompts import untrusted_text
   from reef.core.trajectories import recorded_payloads


   class TaggedFailures(Proposer):
       def propose(self, context):
           tagged = [
               sample
               for sample, source in zip(context.samples, context.sources, strict=True)
               if source.client != "untagged"
           ]
           shown = untrusted_text(json.dumps([recorded_payloads(sample) for sample in tagged], default=str))
           prompt = f"Failing requests:\n{shown}\n\nPropose one skill."
           reply = context.models.served.chat([{"role": "user", "content": prompt}])
           ...

``untrusted_text`` places text inside a block with random delimiters. Text
inside cannot close the block and pose as the prompt's author.

Each ``SampleSource`` has ``record`` (the agent record id of the sample's
last request, ``None`` when it has none) and ``client`` (the
``x-reef-tag-client`` header value, else the session tag, else
``untagged``). A client can set its own tag; trust it as an identity only
if a gateway sets it.

Reef screens prompts before it adds them as evaluation tasks. It skips
prompts that carry credentials or instruction overrides, such as ``ignore
the previous instructions``, a forged ``new system prompt:``, or a
chat-template control token. The step counts these under
``screened_tasks``. Promotion is capped by
``evolution.max_promoted_per_client`` for each tagged client and
``evolution.max_promoted_tasks`` overall.

Put code-bearing kinds that client text can shape (``code_extension``,
``native_tool`` and ``native_hook``) in ``evolution.review_kinds`` so a
person reviews them before publication. A ``native_graph`` holds no code and
can publish on the evaluation alone; add it to ``review_kinds`` if every
loop change should be reviewed. A ``native_loop`` always waits for a person.

.. warning::

   Do not put credentials in tree files. Validation refuses a config file
   with a credential field (a name such as ``apiKey``, ``api_key_env``,
   ``token``, ``secrets`` or ``password``) and any other file that holds a
   credential-shaped literal, whether the file comes from the seed, the
   method's proposal or a client's proposal. The served model binds at
   render time through the adapter's model binding, and a named model in
   ``evolution.models`` reads its key from ``api_key_env``, so a tree never
   needs one.
