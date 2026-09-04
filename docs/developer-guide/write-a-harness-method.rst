Write a harness method
======================

A harness method is the part you write: what edit to try, and how an episode
scored. Reef runs the loop around it: snapshot, apply, run the paired episodes,
record, publish or revert.

`Evolve your harness <../user-guide/evolve-your-harness.rst>`__ is the mechanism this plugs
into, and the runnable example.

A method fills three slots:

.. code:: python

   def propose(nodes, samples, models) -> Mutation | Sequence[Mutation] | None: ...
   def evaluate(task, result) -> float: ...
   class Selection:  # optional
       def decide(self, candidate, evaluation) -> SelectionDecision: ...

``propose`` sees the tree as ``(kind, config)`` pairs, the batch of
``TraceSample`` records (recorded request payload, score, source receipt), and
``models``, its only path to a model. ``models.served`` is the model under
test, ``models["teacher"]`` comes from ``evolution.models``. Call each binding
as ``binding.chat(messages, *, timeout_s=None, **params) -> str``. The
``messages`` are OpenAI-shaped regardless of the endpoint's dialect, and the
binding returns the assistant text. ``propose`` returns one ``Mutation``
(``create``, ``update``, or
``remove`` on one root-level entry), a sequence applied as one composite
proposal under one verdict, or ``None`` to skip. An optional keyword-only
``manifest`` argument receives the previous step's ``FailureManifest``, and an
optional keyword-only ``rejected`` argument receives the recent rejected
proposals, oldest first, each a mapping of ``step``, ``mutations`` (``op`` and
``id`` pairs), and the verdict's ``reason``; a method uses it to stop
re-proposing what the gate already refused. Reef passes each keyword only to
a signature that names it.

``evaluate`` grades one finished episode. Reef calls it for both sides of every
pair. ``result`` carries the exit code, stdout, stderr, and the parsed ``trajectory``. Episodes
that could not run never reach it.

``promote`` is optional and only matters with ``evolution.promote_failures``:
it receives the step's trace samples, plus the ``FailureManifest`` when its
signature names ``manifest``, and returns the prompts to add to the gate as
permanent tasks. Reef dedupes, screens for credentials, and caps what it
returns. Without it every failing trace's user prompt is promoted.

``selection`` defaults to ``score_comparison``: select when the candidate wins
more task comparisons than it loses, by more than ``evolution.min_win_margin``
when that is set. ``always`` selects every applied mutation.

.. warning::

   The tree refuses credentials outright. A config node holding a literal
   credential (``apiKey``, ``token``, plural and list forms) fails admission at
   seed boot, at every proposal, and when recovered state loads. Tree state
   persists into the commit log, the snapshot metadata, and the published
   artifact. If a workdir from before this gate already holds a key, resuming
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
The default, ``reports``, batches explicitly scored reports through the
score window; use it whenever the deployment has an outcome signal (a
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

   implementation: reef.train.cordis_backend.recipe:CordisRecipe

   model:
     path: qwen3-8b

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

   data:
     batch_size: 1
     max_score: 0.0

Preset YAML is read as-is: ``${VAR}`` is **not** interpolated in a preset, only
in a deployment config. Write literal values.

That file is a preset, not a deployment config. It has no ``services`` and no
``reef`` section, so ``reef serve -c`` cannot read it. Save it as
``recipes/<name>.yaml`` and ``export REEF_RECIPE_CONFIG_DIR=$PWD/recipes``;
there is no default directory. The deployment config is the file ``reef serve
-c`` reads, and ``tutorials/harness_evolve/serve.yaml`` is
the one to copy:

.. code:: yaml

   reef:
     recipe: <name>                      # resolves to recipes/<name>.yaml
     token: reef-local
     port: 8900                          # the ready probe's ${reef.port} resolves against this
     upstream_url: ${REEF_UPSTREAM_URL}  # interpolated here, unlike the preset
     upstream_api_key: ${REEF_UPSTREAM_API_KEY}
     upstream_model: ${REEF_MODEL}

   services:
     - name: reef
       command: ["${REEF_PYTHON}", "-m", "reef.service"]
       ready: curl -sf http://127.0.0.1:${reef.port}/healthz

See `Recipe configuration <../reference/configuration.rst#recipe-configuration>`__.
``tutorials/harness_evolve/run.sh`` does exactly this.

Keep the ``tasks`` list short because it sets each step's cost. Start Reef where the method
package is importable, and give ``-c`` an absolute path: Reef resolves a
relative ``-c`` against its own repo root, not your working directory. Recovered
tree state always wins over ``seed``. The full field list is in `Harness
evolution keys <../reference/configuration.rst#harness-evolution-keys>`__.

Selection policies
~~~~~~~~~~~~~~~~~~

A policy reads ``EvaluationResult.metrics``, where the mechanism guarantees
``candidate_scores`` and ``current_scores``: per-task score lists in task order,
``None`` marking a could-not-run episode. This one selects only when no task
regressed and at least one improved.

.. code:: python

   from reef import SelectionDecision


   class ParetoSelectionPolicy:
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


   pareto_selection = ParetoSelectionPolicy()

Name it ``selection: my_pkg.policies:pareto_selection``. Publishing outside
candidate selection breaks revert.

Trust boundary
~~~~~~~~~~~~~~

Every sample is client text. It enters the proposer's model prompt, and with
``promote_failures`` it is re-run as a gate task, so a method treats it as
data: fence it before it reaches a prompt, and read ``provenance`` when a
decision depends on who sent it.

.. code:: python

   import json

   from reef.train.cordis_backend import Mutation, untrusted_text


   def propose(nodes, samples, models, provenance):
       tagged = [s for s, p in zip(samples, provenance, strict=True) if p["source"] != "untagged"]
       shown = untrusted_text(json.dumps([s.payload for s in tagged], default=str))
       reply = models.served.chat([{"role": "user", "content": f"Failing requests:\n{shown}\n\nPropose one skill."}])
       ...

``untrusted_text`` wraps text in a block whose delimiters carry a fresh random
token, so nothing inside the block can close it and speak as the prompt's
author. ``provenance`` is one mapping per sample, in sample order: ``record``
(the agent record id), ``source`` (the client's ``x-reef-tag-source`` header,
else its session tag, else ``untagged``) and ``untrusted`` (always true). A
tag is set by the client, so a source is an identity only where a gateway
sets it.

Reef screens what the method promotes. A prompt that carries a credential or
an instruction override (``ignore the previous instructions``, a forged ``new
system prompt:``, a chat-template control token) is skipped and counted in
the step's ``screened_tasks`` metric; one tagged source holds at most
``evolution.max_promoted_per_source`` promoted tasks and the whole ledger at
most ``evolution.max_promoted_tasks``. A code-bearing mutation
(``code_extension``, ``native_tool``, ``native_hook``) proposed from client
text belongs behind ``evolution.review_kinds``, so a person reads it before
it publishes.
