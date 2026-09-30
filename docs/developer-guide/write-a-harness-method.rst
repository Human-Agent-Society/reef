Write a harness method
======================

A harness method decides what change to try and how to score an evaluation
episode. Reef applies the change to a candidate harness, runs the evaluation,
records the result, and follows the selection and publication policies.

See `Evolve your harness <../user-guide/evolve-your-harness.rst>`__ for the
evolution workflow and a runnable example.

A method provides ``propose`` and ``evaluate``. A custom selection plugin is
optional:

.. code:: python

   def propose(nodes, samples, models) -> Mutation | Sequence[Mutation] | StepProposal | None: ...
   def evaluate(task, result) -> float: ...
   class SelectionFactory(CandidatePluginFactory):  # optional
       def build(self, candidate_backend) -> CandidateEvaluationPlugin: ...

Propose a change
~~~~~~~~~~~~~~~~

``propose`` receives three positional arguments:

- ``nodes``: the current tree as ``(kind, config)`` pairs in tree order.
- ``samples``: a batch of ATIF ``TrajectoryItem`` values. Read the trajectory
  from ``item.trajectory``, reward and feedback from ``item.metadata``, and
  original provider exchanges through ``reef.core.trajectories.recorded_payloads``.
- ``models``: bindings for models the method can call. ``models.served`` is
  the model under test; named bindings such as ``models["teacher"]`` come from
  ``evolution.models``. Call a binding with
  ``binding.chat(messages, *, timeout_s=None, **params) -> str``. The messages
  use the OpenAI shape regardless of the endpoint's dialect, and the call
  returns the assistant text.

Return one ``Mutation`` to ``create``, ``update``, or ``remove`` a root-level
entry. A sequence of mutations forms one proposal with one evaluation result.
Return ``None`` to skip the step. ``StepProposal(mutations, notes)`` also
records a JSON mapping of notes under the commit metric ``proposal_notes``;
Reef does not read the notes to make a decision. Empty mutations skip the
step, just as ``None`` does.

Declare optional keyword arguments when the method needs more context:

- ``manifest`` is the previous step's ``FailureManifest``, if there is one.
- ``rejected`` lists recent rejected proposals, oldest first. Each record
  contains ``step``, ``reason``, and ``mutations``. Each mutation includes its
  ``op``, ``id``, and original ``options`` (``None`` for a remove). Use these
  records to inspect a refusal before proposing the same change again.
- ``entries`` contains the tree as ``{"id", "name", "config"}`` mappings in
  tree order. Use an entry's id to update or remove it.
- ``agent_host`` is an ``AgentHost`` when ``evolution.proposer_agent`` is
  configured; otherwise it is ``None``. It provides the agent's adapter,
  installed binary, executor, record directory, timeouts, and ``calls``
  budget and record. The agent executor is separate from episode execution.
  See ``reef.recipe.reefine.agent`` for a use of this argument.

For model calls the agent process makes outside ``models``, use
``agent_host.calls.spend()`` and ``agent_host.calls.record(entry)``. Declare
``agent_host`` by name to receive it; ``**kwargs`` does not activate it.

Manual requests
~~~~~~~~~~~~~~~

In ``manual`` or ``hybrid`` training mode, ``propose`` must explicitly declare
the ``requests`` keyword argument; ``**kwargs`` does not count. Otherwise the
recipe fails at startup with ``RecipeConfigError``. See `Manual training
<../reference/http-api.rst#manual-training>`__ for the request route.

``requests`` contains one mapping with ``id``, ``text``, ``session``,
``release_id``, ``requires``, and ``untrusted=True``. In ``manual`` mode,
``samples`` is empty. In ``hybrid`` mode, it contains up to ``batch_size``
samples from the next automatic batch, and may be empty. These can be scored
traces or records under ``data.batch_policy: records``.

Reef evaluates request mutations and applies the same ``evolution.publish``
policy as for other steps. Pending agent proposals and periodic rollback
rechecks cannot take over the step assigned to a request.

Score an episode
~~~~~~~~~~~~~~~~

``evaluate`` scores one completed episode. Reef calls it for each side the
selection policy evaluates; ``floor`` evaluates only the candidate. The
``result`` contains the exit code, stdout, stderr, and parsed ``trajectory``.
Episodes that could not run do not reach ``evaluate``.

Other method options
~~~~~~~~~~~~~~~~~~~~

``promote`` matters only with ``evolution.promote_failures``. This optional
``Promoter`` subclass or instance receives the step's samples and
``FailureManifest`` through ``__call__(samples, *, manifest=None)``. It
returns prompts to add as permanent evaluation tasks. Reef deduplicates and
screens the prompts for credentials, then caps how many it adds. Without a
custom promoter, it uses the user prompt from each failing trace.

``selection`` defaults to ``score_comparison``. It selects a candidate when
the number of task comparisons it wins exceeds the number it loses by more
than ``evolution.min_win_margin``, if a margin is set. ``floor`` runs only the
candidate and selects it when every task reaches ``evolution.floor_score``
(default ``1.0``). An episode that could not run misses the floor. The
current release does not run under ``floor``, so ``current_scores`` is empty.
``always`` selects every applied mutation.

.. warning::

   Do not put credentials in tree entries. Reef rejects literal credential
   fields (including ``apiKey``, ``token``, and their plural or list forms)
   when loading a seed, applying a proposal, or recovering stored state.
   Tree state is stored in the commit log, snapshot metadata, and published
   artifact. If older stored state contains a credential, recovery fails and
   names the field. Rotate the key and remove it from the stored entry.

A small method example
~~~~~~~~~~~~~~~~~~~~~~

This example uses a ``[fib]`` prefix to look up the expected answer in
``evaluate``. The prefix is a convention of this method; Reef passes the
task string through unchanged.

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

``data.batch_policy`` selects the source of each batch:

- ``reports`` (the default) batches valid reports with explicit scores. Use
  it when a grader, test, or user action supplies an outcome.
- ``records`` batches recorded inference requests, without requiring
  reports. It starts a step every ``batch_size`` requests. These samples
  have ``score=None``, which ``propose`` must handle. The SkillClaw night
  example judges its own unscored samples later.

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

The YAML above is a standalone recipe preset. Save it as
``recipes/<name>.yaml`` and set ``REEF_RECIPE_CONFIG_DIR`` to that directory;
there is no default directory. Presets are read as-is, so write literal
values instead of ``${VAR}``. Variable interpolation applies to deployment
configs.

A preset describes the method and model but does not start the serving
processes. ``reef serve -c`` reads a deployment config. The one in
``tutorials/evolve-your-harness/configs/serve.yaml`` is a working example:

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

The tutorial selects the dotted class directly and keeps its recipe settings
in the same versioned deployment file. See `Recipe configuration
<../reference/configuration.rst#recipe-configuration>`__ for both forms.

Keep the ``tasks`` list short: Reef runs them in each evaluation step. Start
Reef where it can import the method package. Give ``-c`` an absolute path,
or Reef resolves it against the working directory. On restart, recovered
tree state takes precedence over ``seed``. See `Harness evolution keys
<../reference/configuration.rst#harness-evolution-keys>`__ for every field.

Selection policies
~~~~~~~~~~~~~~~~~~

A policy reads ``EvaluationResult.metrics``. It contains per-task score lists
in task order: ``candidate_scores`` and ``current_scores``. A score is
``None`` when an episode could not run. The policy below selects a candidate
only when at least one task improves and none regresses.

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

Set ``selection: my_pkg.policies:ParetoFactory`` to use this policy. Reef
constructs the factory without arguments and calls ``build`` for each
scenario's candidate backend. Python callers can also supply a factory
instance. A ``CandidateEvaluationPlugin`` must both evaluate and select;
a selector alone is insufficient. Keep publication in the candidate
selection flow so Reef can revert a rejected change.

Untrusted input
~~~~~~~~~~~~~~~

Treat samples as client-controlled data. If a method includes sample text in
a model prompt, fence it first. With ``promote_failures``, a failing sample's
user prompt may also become an evaluation task. Read ``sources`` when a
decision depends on who supplied a sample.

.. code:: python

   import json

   from reef.train.cordis_backend import Mutation, untrusted_text


   def propose(nodes, samples, models, sources):
       tagged = [s for s, p in zip(samples, sources, strict=True) if p["client"] != "untagged"]
       shown = untrusted_text(json.dumps([s.payload for s in tagged], default=str))
       reply = models.served.chat([{"role": "user", "content": f"Failing requests:\n{shown}\n\nPropose one skill."}])
       ...

``untrusted_text`` places text inside a block with random delimiters. Text
inside cannot close the block and pose as the prompt's author.

``sources`` contains one mapping per sample, in the same order. Each mapping
has ``record`` (the agent record id), ``client`` (the ``x-reef-tag-client``
header value, falling back to the session tag, then ``untagged``), and
``untrusted=True``. A client can set its own tag; trust it as an identity
only if a gateway sets it.

Reef screens prompts before adding them as evaluation tasks. It skips prompts
containing credentials or instruction overrides, such as ``ignore the
previous instructions``, a forged ``new system prompt:``, or a chat-template
control token. The step counts these under ``screened_tasks``. Promotion is
capped by ``evolution.max_promoted_per_client`` for each tagged client and
``evolution.max_promoted_tasks`` overall.

Put code-bearing mutations derived from client text (``code_extension``,
``native_tool``, and ``native_hook``) behind ``evolution.review_kinds`` so a
person reviews them before publication. A ``native_graph`` has no code and
can publish based on evaluation alone. Add it to ``review_kinds`` if every
loop change should receive human review.
