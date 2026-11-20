Retained harness release evaluation (experimental)
=================================================

This local prototype addresses the harness portion of issue
`#355 <https://github.com/Human-Agent-Society/reef/issues/355>`_. Its Python
interface and export format are proposed in `RFC #514
<https://github.com/Human-Agent-Society/reef/issues/514>`_ and need acceptance
before they become supported contracts. It does not close the whole issue.

What it measures
----------------

``RetainedHarnessEvaluation`` evaluates explicit, retained harness releases
against identical text prompts and an ``EpisodeScorer``. The first release is
the baseline. Evaluation never invokes the trainer, commits a release, or
submits learning feedback. Releases are interleaved for each task and repeat.
The caller must keep the scenario available while constructing the evaluator;
its rendered target snapshots are held in memory after construction.

Missing releases fail before execution; there is no fallback to the served
head. Pending publication targets and weight surfaces are rejected. If a model
binding is supplied, only the adapter's declared provider configuration files
are overlaid. Every other retained file stays byte-for-byte unchanged. This
also supports initial seed releases without a training commit. Re-rendering
historical tool modules is deliberately avoided: JSON key ordering and renderer
changes can otherwise change the executed artifact.

Using an existing scenario
--------------------------

The application provides the already-open ``Scenario``, adapter, scorer,
harness executable and, for model-backed episodes, ``ModelBindings``. Do not
bootstrap a second live dispatcher against a running service's storage merely
to perform evaluation.

.. code-block:: python

   from pathlib import Path
   from reef.scenario.evaluation import (
       EvaluationConditions, EvaluationTask, RetainedHarnessEvaluation,
   )

   evaluation = RetainedHarnessEvaluation(
       scenario,
       [baseline_release_id, previous_release_id, current_release_id],
       [EvaluationTask("old-capability", "Your fixed task prompt")],
       EvaluationConditions(
           suite_version="suite-content-sha256",
           scorer_version="scorer-code-and-config-sha256",
           model_version="immutable-model-revision",
           environment_version="harness-dependencies-fixtures-sampling-sha256",
           repeats=3,
           retain_episodes=True,
           episode_timeout_seconds=60,
       ),
       descriptor=descriptor,
       scorer=scorer,
       binary="/absolute/path/to/harness",
       models=evaluation_models,
   )
   # The budget counts admitted episodes, not model API calls or currency.
   evaluation.run(Path("evaluation-runs/example"), max_new_episodes=3)
   # Same inputs and directory resume the remaining work.
   evaluation.run(Path("evaluation-runs/example"))

Use a direct evaluation model endpoint whose requests are not ingested into
Reef's learning stream. Fixed prompts alone do not isolate evaluation traffic
if the caller points the harness at an ordinary learning proxy. This version
does not provision an isolated model service or prove provider immutability.

Conditions and reuse
--------------------

A manifest pins scenario and release/content identities, retained file checksums,
prompts, conditions, adapter execution settings, executable bytes, Reef Python
source checksums, scorer class and non-secret model-binding configuration.
Scorer, environment and model versions are supplied by the operator. Include
external scorer code/configuration, task fixtures, harness dependencies,
provider/model revision, sampling configuration and applicable seeds. An API
model alias or mutable endpoint cannot guarantee repeatability. Specifying a
seed identity does not make a harness apply that seed.

Changing any recorded condition requires a new output directory and a common
re-evaluation of the compared targets. A finite error outcome is retained just
like a scored outcome; it is not retried until it happens to pass. Completed
results are reused by exact plan position and identity. Malformed/misplaced
result records are rejected. Do not edit result records by hand.

Run files and lifecycle
----------------------

* ``manifest.json``: fixed evaluation conditions and targets.
* ``results/00000000.json`` etc.: atomically replaced per-episode outcomes.
* ``episodes/00000000.json`` etc. (opt-in): normalized worker observations and
  trajectories, with checksums validated on resume. Interrupted writes without
  a completed result are not reusable.
* ``run.json``: exported complete matrix, including explicit unrun slots,
  reported native usage and family-level paired comparisons.
* ``report.md``: coverage, paired deltas and per-task changes relative to both
  baseline and previous release. Higher scores must mean better outcomes.

Only matched valid scores contribute to paired deltas. A score of zero is an
ordinary scored outcome. Execution errors and invalid scores are distinct from
zero and from unrun tasks. Report coverage alongside score changes; an improved
mean over fewer valid tasks is not proof of improvement. Task families are explicit on ``EvaluationTask``. The family report averages
matched repetitions within each task, then gives tasks equal weight. A seeded
2,000-draw percentile bootstrap resamples tasks rather than treating repeated
responses as independent tasks. A single paired task has no interval. These
intervals are descriptive and unadjusted for multiple comparisons; they do not
account for model drift or pretraining contamination. Inspect coverage, not
only deltas: missing pairs can create survivor bias.

.. code-block:: bash

   python -m reef.scenario.evaluation evaluation-runs/example/run.json

The initial runner is serial and POSIX-only (an OS file lock excludes concurrent
writers and is released on process death). ``max_new_episodes`` bounds admission
per invocation; callers can also set a ``threading.Event`` to cancel admission.
The current episode drains under its executor timeout, which kills its process
group on timeout. Cancellation is not an immediate kill. Scorers must bound
their own computation and I/O: the episode timeout does not bound a custom
scorer. Unexpected programming/infrastructure errors abort the run while
preserving earlier result files. Resume regenerates the final export.

A crash after a remote model call but before saving the result can repeat that
call on resume. There is no exactly-once billing guarantee. Reported elapsed
seconds cover episode execution and scoring, not serving throughput. Native input/output token counts are reported only when all observed model
responses include the relevant counter; unsupported formats remain unknown.
These counts exclude scorer calls and provider attempts without a recorded
response. Currency cost remains unknown. No strict API-spend cap is claimed
for the reusable runner.

Run directories are created private, and provider credentials/rendered binding
files are excluded from exports. Opt-in episode records include the worker's
error details and task text; explicit binding credentials and credential-shaped
strings are redacted, but this is not a general private-data classifier. Task
prompts themselves are retained. Choose a protected output location; do not commit private tasks.
The local executor is not a security sandbox; use only trusted harnesses and
artifacts in this prototype.

Remaining issue scope
---------------------

Directory task fixtures, weight targets, enforced token/currency budgets,
parallel workers, hard scorer cancellation, centralized retention/access
policies and model-endpoint isolation remain outside this first implementation.
Existing executor machinery should supply parallel placement in a follow-up rather than a second scheduler.

The contract tests publish three real retained artifacts through Reef's
scenario commit path and launch a deterministic fixture harness through the
actual episode executor. They detect a deliberately regressing third version,
exercise resume/invalidation/error behavior, and check that evaluation does not
change scenario history. This validates infrastructure, not learned model
quality or performance on a live model.

Feedback-driven Agent experiment
--------------------------------

The `BBH experiment <../../tutorials/release_evaluation/README.md>`_ runs the
native Agent with a bounded state-tracking tool, learns one prompt update per
task family from training observations, and validates each candidate before
publication. It then evaluates the frozen baseline and all accepted retained
releases on a disjoint held-out set. Failed or rejected updates are retained in
the experiment's history; the driver does not force a positive learning result.

.. code-block:: bash

   python -m tutorials.release_evaluation.run \
       --api-key-file /private/path/deepseek-key.txt \
       --output /private/path/new-evaluation-directory

The provider transport has a separate, explicit request/output budget. The
example records the dataset commit and checksums, split identities, reflection
input/output, validation decisions, held-out traces and provider usage. It is
a small single-seed case study of measurement infrastructure, not a new RSI
algorithm or a reproduction of the full BBH benchmark.
