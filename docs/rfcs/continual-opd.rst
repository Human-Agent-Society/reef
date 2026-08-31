.. _continual-opd-on-reef--framework:

Continual OPD on Reef: framework
=================================

   A small model serves by default; a verifier flags tasks it handles poorly and
   escalates them to a large model; escalated trajectories are continually
   distilled back into the small model (hard distillation + soft on-policy
   distillation, including cross-tokenizer). The intended result is a lower
   escalation rate and a steadily falling cost per solved task as the small
   model improves.

.. _1-system-framework-four-components:

1. System framework (four components)
-------------------------------------

::

                     ┌─────────────────────────────────────────┐
    task ──▶ Harness │ Router (v1 lives harness-side, ~50 LoC) │
                     └──────┬───────────────────┬──────────────┘
                            │ default           │ escalate
                            ▼                   ▼
               Reef scenario "student"   Reef scenario "teacher"
               recipe: trainable        recipe: `recipe` proxy
               SGLang small model        GLM/Kimi engine or closed API
                            │                   │
                            ▼                   ▼
                   Verifier ◀── both responses (with receipts)
                            │
                            │ POST /reef/report {score, route metadata}
                            ▼
           ┌──────────────── distillation data plane ─────────────────┐
           │ A. teacher trajectories → hard distill SFT               │
           │    (tokenizer-agnostic by construction)                  │
           │ B. student trajectories + teacher scoring → soft OPD     │
           └───────────────────────┬──────────────────────────────────┘
                                   ▼
        bridge → one slime train step → weight hot-swap → next request
                 already samples from the improved model
                                   │
                                   ▼
      artifact version chain (one version per distillation step;
      probe-set regression → rollback)

1. **Router:** v1 lives on the harness side (~50 LoC). The student runs first;
   if the verifier fails it, the task is retried on the teacher. This requires no
   Reef changes: two scenarios, each bound to its own recipe/upstream
   (multi-recipe deployments are natively supported).
2. **Verifier:** the existing report score is the verifier. v1 targets domains
   where verification is free (code via unit tests, math via numeric checks).
3. **Distillation:** layers run from cheapest to most expensive (see §2).
   Escalated samples filling a batch provide the trigger, using Reef's native
   processor training mechanism, so no separate trigger is needed.
4. **Trust evolution:** track per-domain student pass rates; once a domain
   clears the bar, escalation is switched off there (keeping a small audit
   sample; dropping below the floor re-enables it). Thresholds and state
   eventually become versioned artifacts; v1 uses a simple EMA + fixed
   threshold.

.. _2-three-distillation-tiers:

2. Three distillation tiers
---------------------------

+-----------------+----------------+-----------------------+-------------------+
| Tier            | Applies to     | Method                | Depends on        |
+=================+================+=======================+===================+
| L0 hard distill | Any teacher    | Re-tokenize           | Needs a dedicated |
|                 | (incl. closed  | successful teacher    | distillation      |
|                 | models), any   | trajectories with the | recipe; the       |
|                 | tokenizer      | student tokenizer →   | low-level ``sft`` |
|                 |                | SFT                   | loss family       |
|                 |                |                       | remains backend   |
|                 |                |                       | plumbing          |
+-----------------+----------------+-----------------------+-------------------+
| L1 same-vocab   | Large Qwen →   | Teacher scores the    | **Backend         |
| OPD             | small Qwen     | student's own sampled | shipped**: the   |
|                 |                | tokens per-token;     | ``opd`` loss      |
|                 |                | reverse-KL signal     | family            |
|                 |                |                       | (slime-bridge     |
|                 |                |                       | teacher scoring)  |
|                 |                |                       | is in place; the  |
|                 |                |                       | cookbook recipe   |
|                 |                |                       | that demonstrated |
|                 |                |                       | it was removed    |
|                 |                |                       | from the tree     |
|                 |                |                       | (``examples/opd`` |
|                 |                |                       | in git history)   |
+-----------------+----------------+-----------------------+-------------------+
| L2              | GLM/Kimi →     | Project both          | L1 + an alignment |
| cross-tokenizer | Qwen           | tokenizers'           | library           |
| OPD             |                | boundaries onto the   |                   |
|                 |                | byte stream, cut      |                   |
|                 |                | chunks at common      |                   |
|                 |                | boundaries, compare   |                   |
|                 |                | summed log-probs per  |                   |
|                 |                | chunk                 |                   |
|                 |                | (tokenizer-invariant, |                   |
|                 |                | no vocab mapping      |                   |
|                 |                | needed)               |                   |
+-----------------+----------------+-----------------------+-------------------+

The code already provides the required OPD backend. Slime includes teacher
scoring, reverse-KL, an end-to-end ``teacher_log_probs`` pipeline, and frozen
heterogeneous teacher engines. The Reef↔slime bridge now carries it
too: ``opd`` is in the bridge's algorithm whitelist, and after the bridge
receives a batch it calls the teacher for scoring and directly constructs
per-token advantages
(``β·clip(RMSNorm(teacher_logp − rollout_logp), −c, c)``; optional centering
happens before RMS normalization) through the existing channel. This requires
no slime-core changes and leaves the client contract unchanged.

.. _3-roadmap:

3. Roadmap
----------

- **Phase 0 (no code changes):** run the loop with dual scenarios, a harness
  router + verifier + the L0 hard-distillation loop. Deliverables: end-to-end
  engineering validation plus all baselines.
- **Phase 1 (core development, shipped):** L1 same-vocab OPD through the
  bridge, implemented in ``reef/train/slime_backend/reef_adapters/``
  plus one cookbook backend-preparer file (the removed ``examples/opd``
  prototype exercised this end to end; see git history).
- **Phase 2:** cross-tokenizer alignment library + L2; automate trust
  evolution and rollback.

.. _4-first-experiments:

4. First experiments
--------------------

+----+--------------------------------+--------------------------------+
| #  | Experiment                     | Purpose                        |
+====+================================+================================+
| E0 | Three baselines: student-only, | How big the gap is and whether |
|    | **each teacher's pure-run      | it's worth pursuing; the       |
|    | ceiling** (large Qwen,         | teacher-only lines are the     |
|    | GLM/Kimi (one line each),     | ceiling denominator for every  |
|    | oracle-router                  | later experiment and the cost  |
|    |                                | reference for "just use the    |
|    |                                | big model"                     |
+----+--------------------------------+--------------------------------+
| E1 | Run the Phase 0                | Verify escalation rate falls,  |
|    | hard-distillation loop for N   | pass rate rises, general       |
|    | rounds                         | benchmarks don't regress       |
+----+--------------------------------+--------------------------------+
| E2 | Same-vocab OPD vs. hard        | **Go/no-go**: if OPD shows no  |
|    | distillation (matched budget)  | significant gain, shrink the   |
|    | + β sweep                      | plan to router + hard          |
|    |                                | distillation and skip Phase 2  |
+----+--------------------------------+--------------------------------+
| E3 | Cross-tokenizer chunk-KL vs.   | How much soft-signal value the |
|    | hard distillation              | alignment layer recovers       |
+----+--------------------------------+--------------------------------+
| E4 | Verifier FP/FN calibration +   | How to set thresholds; whether |
|    | routing-threshold Pareto sweep | the verifier can be trusted    |
+----+--------------------------------+--------------------------------+

.. _5-long-term-items-backlog-to-be-detailed-later:

5. Long-term items (backlog, to be detailed later)
--------------------------------------------------

Bridge/algorithm registry (#81) · first-class per-token signal channel ·
streaming traces as training records · distribution-level cross-tokenizer
distillation (the teacher server's candidate-scoring primitive is implemented
and awaiting a consumer) · router inside Reef with composite versions (see
the README roadmap) ·
anti-forgetting replay · learned verifier/router · LoRA-adapter distillation
steps.

.. _6-main-risks-one-mitigation-each:

6. Main risks (one mitigation each)
-----------------------------------

Student learning to exploit verifier blind spots → prefer executable checks +
always-on audit sampling + probe-set rollback; hard-distillation distribution
shift → move to on-policy L1/L2 quickly; catastrophic forgetting → replay +
standing general-benchmark monitoring; teacher-scoring throughput → batch
scoring inside the bridge + independently scalable teacher engines.
