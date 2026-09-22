Write a harness method
======================

A harness method is the part you write: what edit to try, and how an episode
scored. Reef runs the loop around it: snapshot, apply, run the paired episodes,
record, publish or revert.

`Evolve your harness <../user-guide/evolve-your-harness.rst>`__ is the mechanism this plugs
into, and the runnable example.

A method fills three slots:

.. code:: python

   def propose(nodes, samples, models) -> Mutation | Sequence[Mutation] | StepProposal | None: ...
   def evaluate(task, result) -> float: ...
   class SelectionFactory(CandidatePluginFactory):  # optional
       def build(self, candidate_backend) -> CandidateEvaluationPlugin: ...

``propose`` sees the tree as ``(kind, config)`` pairs, the batch of
ATIF ``TrajectoryItem`` values, and
``models``, its only path to a model. ``models.served`` is the model under
test, ``models["teacher"]`` comes from ``evolution.models``. Read the trajectory
from ``item.trajectory``, reward/feedback from ``item.metadata``, and original
provider exchanges through ``reef.core.trajectories.recorded_payloads``.
Call each binding
as ``binding.chat(messages, *, timeout_s=None, **params) -> str``. The
``messages`` are OpenAI-shaped regardless of the endpoint's dialect, and the
binding returns the assistant text. ``propose`` returns one ``Mutation``
(``create``, ``update``, or
``remove`` on one root-level entry), a sequence applied as one composite
proposal under one result, or ``None`` to skip. It may also return a
``StepProposal(mutations, notes)``: the same mutations plus ``notes``, a JSON
mapping (a plan, a review result, what the method could not honor) that the
step records under the commit metrics key ``proposal_notes`` and never reads;
empty mutations skip the step as ``None`` does. An optional keyword-only
``manifest`` argument receives the previous step's ``FailureManifest``, and an
optional keyword-only ``rejected`` argument receives the recent rejected
proposals, oldest first, each a mapping of ``step``, ``mutations`` (each with
its ``op``, ``id`` and the ``options`` it carried, ``None`` for a remove), and
the result's ``reason``; a method uses it to stop re-proposing what the evaluation
already refused, and can read the refused content rather than only its id.
An optional keyword-only ``entries`` argument receives the tree as entry
options, ``{"id", "name", "config"}`` mappings in tree order, so an ``update``
or ``remove`` can name the entry it targets instead of creating a second one.
An optional keyword-only ``agent_host`` argument receives an ``AgentHost``
when the deployment configured ``evolution.proposer_agent``, else ``None``:
the adapter descriptor and its installed binary, the executor built for the
agent (its isolation, not the episodes'), the step's record directory, the
two timeouts, and ``calls``, the step's model-call budget and record
(``spend()`` and ``record(entry)``) for traffic the agent's own process makes
outside ``models``. ``reef.recipe.reefine.agent`` is the reference user.

An optional keyword-only ``requests`` argument carries a training request in
``training_mode`` ``manual`` and ``hybrid`` (see `Manual training
<../reference/http-api.rst#manual-training>`__ for the route that queues
one). It holds exactly one request mapping, with ``id``, ``text``,
``session``, ``release_id``, ``requires`` and ``untrusted=True``. The parameter
must be declared by name; a ``**kwargs`` catch-all does not count, so a
method never takes a request without being written for it. A deployment in
either mode whose ``propose`` declares no such parameter fails at startup
with ``RecipeConfigError``, so a failure-only method must grow a
``requests`` branch before it runs there.
``samples`` is empty in ``manual``; in ``hybrid`` it carries what an
automatic batch would take next, up to ``batch_size`` and possibly none
(scored traces, or records under ``data.batch_policy: records``), so the
method can answer the request with the failures beside it. A request's
mutations pass through the same evaluation and ``evolution.publish`` policy
as any other step's, and pending agent proposals and periodic rollback
rechecks cannot take the step a request owns.

Reef passes each keyword only to a signature that names it.

``evaluate`` grades one finished episode. Reef calls it for both sides of every
pair. ``result`` carries the exit code, stdout, stderr, and the parsed ``trajectory``. Episodes
that could not run never reach it.

``promote`` is an optional ``Promoter`` subclass or instance and only matters
with ``evolution.promote_failures``. Its ``__call__(samples, *, manifest=None)``
receives the step's trace samples and ``FailureManifest``, and returns the prompts to add to the evaluation as
permanent tasks. Reef dedupes, screens for credentials, and caps what it
returns. Without it every failing trace's user prompt is promoted.

``selection`` defaults to ``score_comparison``: select when the candidate wins
more task comparisons than it loses, by more than ``evolution.min_win_margin``
when that is set. ``floor`` runs the candidate alone and selects it when every
evaluation task scores at least ``evolution.floor_score`` (default ``1.0``; an
episode that could not run missed the floor): a floor is absolute, not a
comparison, so the current release is not run and ``current_scores`` is empty.
``always`` selects every applied mutation.

.. warning::

   The tree refuses credentials outright. A config node holding a literal
   credential (``apiKey``, ``token``, plural and list forms) fails admission at
   seed boot, at every proposal, and when recovered state loads. Tree state
   persists into the commit log, the snapshot metadata, and the published
   artifact. If a workdir from before this admission check already holds a key, resuming
   fails and names the field: rotate the key, then edit the entry out of the
   stored state.

A complete method
~~~~~~~~~~~~~~~~~

Each task string starts with a tag, such as ``[fib]`` in the config below, and
``evaluate`` uses it to look up that task's expected answer. The tag convention
is the method's own; Reef passes the task string through unchanged.

This method adds a rules node the first time a batch contains a failure:

.. code:: python

   from reef.train.cordis_backend import Mutation

   RULE = "State the final answer alone on the last line.\n"
   EXPECTED = {"[fib]": "2880067194370816120"}


   def propose(nodes, samples, models):
       """Add the rule after a failing batch; once it is in the tree, sit out."""
       if all((sample.score or 0.0) > 0.0 for sample in samples):
           return None
       if any(config.get("text") == RULE for kind, config in nodes if kind == "rules"):
           return None
       return Mutation("create", "final-line-rule", {"name": "rules", "config": {"text": RULE}})


   def evaluate(task, result):
       """1.0 when the final assistant text ends with the task's expected answer."""
       text = _final_text(result.trajectory) or ""
       return 1.0 if text.strip().endswith(EXPECTED[task.split()[0]]) else 0.0


   def _final_text(trajectory):
       for event in reversed(trajectory):
           message = event.get("message") if isinstance(event.get("message"), dict) else event
           if message.get("role") == "assistant" and isinstance(message.get("content"), str):
               return message["content"]
       return None

Two batching modes
~~~~~~~~~~~~~~~~~~

Evolution batches in one of two modes, selected by ``data.batch_policy``.
The default, ``reports``, batches every valid explicitly scored report;
use it whenever the deployment has an outcome signal (a
grader, a test result, a user action), because a measured result beats
model self judgment. ``records`` batches recorded inference traffic alone,
every ``batch_size`` requests, so a deployment that only serves still
evolves. Samples batched this way carry ``score=None``, and ``propose``
must handle unscored samples; the SkillClaw night backfills its own
judgment over them and is the worked instance.

Configure it
~~~~~~~~~~~~

The recipe config names the callables, the tasks, and the first-boot tree:

.. code:: yaml

   schema-version: 2
   recipe:
     implementation: reef.recipe.cordis:CordisRecipe
     config:
       batch-size: 1
       max-score: 0.0
       evolution:
         adapter: pi
         binary: pi
         propose: methods.mine:propose
         evaluate: methods.mine:evaluate
         tasks:
           - "[fib] Compute fib(90) exactly. Reply with the integer alone on the last line."
         seed:
           - id: answer-style
             name: skill
             config: {name: answer-style, text: "# answer-style\n\nStarter skill."}
         models:                        # optional extras; each key read via api_key_env
           teacher:
             url: https://api.openai.com
             model: gpt-4o
             api_key_env: OPENAI_API_KEY
   inference:
     upstream-model: qwen3-8b

Preset YAML is read as-is: ``${VAR}`` is **not** interpolated in a preset, only
in a deployment config. Write literal values.

That standalone preset describes the method and model; it does not assemble
the serving processes. Save it as
``recipes/<name>.yaml`` and ``export REEF_RECIPE_CONFIG_DIR=$PWD/recipes``;
there is no default directory. The deployment config is the file ``reef serve
-c`` reads, and ``tutorials/evolve-your-harness/configs/serve.yaml`` is
the one to copy:

.. code:: yaml

   schema-version: 2
   recipe:
     implementation: <name>  # resolves to recipes/<name>.yaml
   reef:
     token: reef-local
     port: 8900
   inference:
     upstream-url: ${REEF_UPSTREAM_URL}
     upstream-api-key: ${REEF_UPSTREAM_API_KEY}
     upstream-model: ${REEF_MODEL}

See `Recipe configuration <../reference/configuration.rst#recipe-configuration>`__.
The tutorial selects the dotted class directly and keeps its recipe settings
in the same versioned deployment file.

Keep the ``tasks`` list short because it sets each step's cost. Start Reef where the method
package is importable, and give ``-c`` an absolute path: Reef resolves a
relative ``-c`` against your working directory. Recovered
tree state always wins over ``seed``. The full field list is in `Harness
evolution keys <../reference/configuration.rst#harness-evolution-keys>`__.

Selection policies
~~~~~~~~~~~~~~~~~~

A policy reads ``EvaluationResult.metrics``, where the mechanism guarantees
``candidate_scores`` and ``current_scores``: per-task score lists in task order,
``None`` marking a could-not-run episode. This one selects only when no task
regressed and at least one improved.

.. code:: python

   from reef import CandidateEvaluationPlugin, CandidateEvaluator, SelectionDecision
   from reef.train.evaluation import CandidatePluginFactory


   class ParetoPlugin(CandidateEvaluationPlugin):
       def __init__(self, candidate_backend: CandidateEvaluator):
           self._candidate_backend = candidate_backend

       def evaluate(self, candidate):
           return self._candidate_backend.evaluate(candidate)

       def decide(self, candidate, evaluation):
           pairs = zip(
               evaluation.metrics["candidate_scores"],
               evaluation.metrics["current_scores"],
               strict=True,
           )
           scores = [(c if c is not None else -1e30, k if k is not None else -1e30) for c, k in pairs]
           selected = all(c >= k for c, k in scores) and any(c > k for c, k in scores)
           return SelectionDecision(
               outcome="select" if selected else "reject",
               policy="pareto",
               policy_version="1",
               reason="no task regressed and at least one improved" if selected else "Pareto failed",
               evaluation=evaluation,
           )


   class ParetoFactory(CandidatePluginFactory):
       def build(self, candidate_backend: CandidateEvaluator) -> CandidateEvaluationPlugin:
           return ParetoPlugin(candidate_backend)

Name it ``selection: my_pkg.policies:ParetoFactory``. Reef constructs the
factory without arguments, then calls ``build`` for each scenario's candidate
backend. A factory instance can also be supplied directly from Python. The
plugin explicitly inherits both evaluation and selection through
``CandidateEvaluationPlugin``; a selector-only object is not a complete plugin.
Publishing outside candidate selection breaks revert.

Untrusted input
~~~~~~~~~~~~~~~

Every sample is client text. It enters the proposer's model prompt, and with
``promote_failures`` it is re-run as an evaluation task, so a method treats it as
data: fence it before it reaches a prompt, and read ``sources`` when a
decision depends on who sent it.

.. code:: python

   import json

   from reef.train.cordis_backend import Mutation, untrusted_text


   def propose(nodes, samples, models, sources):
       tagged = [s for s, p in zip(samples, sources, strict=True) if p["client"] != "untagged"]
       shown = untrusted_text(json.dumps([s.payload for s in tagged], default=str))
       reply = models.served.chat([{"role": "user", "content": f"Failing requests:\n{shown}\n\nPropose one skill."}])
       ...

``untrusted_text`` wraps text in a block whose delimiters carry a fresh random
token, so nothing inside the block can close it and speak as the prompt's
author. ``sources`` is one mapping per sample, in sample order: ``record``
(the agent record id), ``client`` (the ``x-reef-tag-client`` header's value,
else the session tag, else ``untagged``) and ``untrusted`` (always true). A
tag is set by the client, so it names a client only where a gateway sets it.

Reef screens what the method promotes. A prompt that carries a credential or
an instruction override (``ignore the previous instructions``, a forged ``new
system prompt:``, a chat-template control token) is skipped and counted in
the step's ``screened_tasks`` metric; one tagged client holds at most
``evolution.max_promoted_per_client`` promoted tasks, with at most
``evolution.max_promoted_tasks`` promoted tasks in total. A code-bearing mutation
(``code_extension``, ``native_tool``, ``native_hook``) proposed from client
text belongs behind ``evolution.review_kinds``, so a person reads it before
it publishes. A ``native_graph`` carries no code, so a loop change can
publish on the evaluation alone; list the kind in ``review_kinds`` when a person
should read every loop change.
