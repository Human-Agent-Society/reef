# PPO/RLHF-Style Adaptive KL Controller: Implementation Handoff

Status: implementation handoff; design is intentionally opt-in and not yet an
approval to change the default training behavior.

Audience: an engineer or coding agent working in the Reef repository and, if
necessary, the exact Slime checkout used by the experiment.

Scope: the first implementation phase only covers a PPO/RLHF-style,
reference-policy KL penalty applied on the reward/advantage path. It does not
cover the other KL-like quantities already present in Reef.

## 1. Context and decision history

Reef issue [#466](https://github.com/Human-Agent-Society/reef/issues/466)
discusses an adaptive-KL direction. The current decision is to make the first
implementation small and testable:

```text
rollout policy + frozen reference policy
    -> reference-policy KL estimate
    -> observed batch KL
    -> bounded adaptive beta update
    -> reward shaping for the next training batch
```

Reef PR [#533](https://github.com/Human-Agent-Society/reef/pull/533) added
important distillation infrastructure. It is relevant background, but it is
not an implementation of this proposal. In particular, #533 does not provide
an `AdaptiveKLController`, a target-KL controller, a dynamically updated
scalar `beta`, or PPO/RLHF reward-side KL shaping. Its distillation backend is
useful future infrastructure for OPD-specific divergence experiments.

There are also related Slime discussions, including
[Slime #2387](https://github.com/THUDM/slime/issues/2387). They should be
consulted for the exact upstream runtime and current maintainer direction, but
they must not be treated as proof that the required reward-side hook already
exists in the pinned checkout.

The implementation agent must record the exact Slime commit or package version
used by the experiment. Do not assume that the current upstream main branch,
the Reef source tree, and the server's runtime expose the same interfaces.

## 2. Exact problem definition

For a sampled response token `t`, let:

- `pi_theta` be the policy that generated or is being trained on the sample;
- `pi_ref` be a frozen reference policy;
- `r_task,t` be the task reward in the convention already used by the
  selected training path;
- `k_t` be the reference-policy KL contribution computed by the actual Slime
  implementation and masked to valid response tokens;
- `beta` be the reward-side KL coefficient.

The intended reward shaping is:

```text
r_shaped,t = r_task,t - beta * k_t
```

The shaped reward is then consumed by the existing advantage and policy-update
pipeline. The controller does not replace PPO clipping, does not replace the
policy objective, and does not add a second policy-loss term.

The controller observes a batch-level statistic:

```text
observed_kl = masked_mean(k_t)
```

The exact estimator, sign convention, token mask, and data-parallel reduction
must be confirmed from the pinned Slime runtime before implementation. The
formula above is the contract, not permission to guess the upstream tensor
names or signs.

The coefficient update is applied after a successful training step and is used
by the next eligible batch. It must not retroactively change the reward or
advantage of the batch that was just trained.

## 3. Goals

The first phase should:

1. Provide a small, independently unit-testable adaptive-KL controller.
2. Apply it only to PPO/RLHF-style reference-policy reward shaping.
3. Keep the controller disabled by default.
4. Make the coefficient bounded, finite, restartable, and observable.
5. Keep the reference policy loaded for the whole run when this mode is
   enabled.
6. Preserve Reef's training/checkpoint/restart semantics.
7. Support a telemetry-only and a shadow mode before changing rewards.
8. Make it possible to compare fixed-beta and adaptive-beta runs under the
   same data and reference-policy identity.

## 4. Non-goals and explicit exclusions

Do not include any of the following in the first implementation:

- OPD adaptive divergence or teacher-KL control;
- SAO `ppo_kl` or asynchronous policy-lag correction;
- TTTD's frozen-base KL advantage correction;
- OpenClawRL's `kl_loss_coef` final-loss term;
- GRPO reward-side KL;
- a controller shared by multiple loss families;
- adaptive PPO clipping ranges;
- PID control or an unconstrained dual optimizer;
- using held-out evaluation scores as a direct training reward or as the
  controller's observed KL;
- changing default behavior for existing recipes;
- silently modifying an external Slime checkout without recording the change.

If implementation discovery shows that the pinned Slime runtime cannot expose
the required reward-side reference-policy tensors safely, stop at the
controller plus telemetry/shadow-mode integration and report the missing
upstream hook. Do not emulate reference KL with `ppo_kl`.

## 5. KL terminology: do not merge these quantities

### 5.1 Reference-policy KL — this proposal

```text
current/rollout policy versus a frozen reference policy
```

It is used as a reward-side penalty:

```text
r_shaped = r_task - beta * reference_kl
```

This is the only KL that the first-phase controller may update.

### 5.2 `ppo_kl` — existing old-policy/current-policy quantity

In the existing policy-loss paths, `ppo_kl` is typically derived from the
rollout/old-policy log probability and the current-policy log probability. It
is used to form an importance ratio and clipping or masking decisions. For
example, SAO documents a convention equivalent to:

```text
ppo_kl = log pi_rollout - log pi_current
ratio = exp(-ppo_kl)
```

This measures policy lag or update divergence. It is not the frozen-reference
KL required here and must not drive this controller.

### 5.3 Teacher divergence / OPD KL

This is student-versus-teacher divergence in the distillation backend added in
#533. It can be forward KL, reverse KL, JSD, top-K divergence, or another
teacher-specific estimator. It is not the PPO/RLHF reward-side reference KL.

### 5.4 TTTD KL

TTTD uses a frozen-base/reference signal to modify advantages. That is a
recipe-specific advantage correction, not this controller's reward-shaping
contract.

### 5.5 OpenClawRL `kl_loss_coef`

OpenClawRL has a fixed coefficient that adds a reference-related term directly
to the final loss. That is a loss-side regularizer. It is not the reward-side
`beta` implemented here.

The implementation must use names and metric keys that make these distinctions
visible. In particular, never label a reference-policy reward KL as
`ppo_kl`.

## 6. Current Reef code map

The following locations are known entry points. They are starting points for
inspection, not a substitute for checking the exact runtime:

| Area | Current location | Relevance |
| --- | --- | --- |
| Loss-family contract | `docs/developer-guide/loss-families.rst` | Explains driver-side specs, worker-side objectives, and lifecycle hooks. |
| Step signal | `reef/train/algos/signals.py` | Backend-neutral signal carrying loss family, advantages, metrics, and scheduling. |
| Slime algorithm base | `reef/train/slime_backend/algorithm.py` | Shared lifecycle and loss-family integration. |
| Actor/reference construction | `reef/train/slime_backend/reef_adapters/ray_train_groups.py` | Current `with_ref` decision depends on `kl_coef` or `use_kl_loss`. |
| Worker metrics | `reef/train/slime_backend/reef_adapters/worker_hooks.py` | Existing per-step metric collection and draining. |
| Bridge lifecycle | `reef/train/slime_backend/reef_adapters/bridge.py` | Training, checkpoint, publication, durable metrics, and restart boundaries. |
| SAO objective | `recipes/sao/slime/objective.py` | Existing `ppo_kl`; explicitly out of scope. |
| TTTD objective | `recipes/tttd/slime/objective.py` | Existing frozen-base KL advantage correction; out of scope. |
| OpenClawRL objective | `recipes/openclawrl/slime/objective.py` | Existing `kl_loss_coef`; out of scope. |
| Distillation backend | `reef/train/slime_backend/distill/` | #533 infrastructure; not this controller. |
| Historical RFC policy | `docs/rfcs/README.rst` | New RFCs belong in GitHub Issues, not a new file under `docs/rfcs/`. |

The current actor construction has an important implication: reference-model
availability is decided during initialization. If adaptive reward KL starts at
zero but is intended to become positive later, the reference actor still must
be loaded from the beginning. Do not rely on changing `kl_coef` after actor
initialization to create a reference model that was never constructed.

## 7. Selected design

### 7.1 Configuration

Use an explicitly namespaced, opt-in configuration. Exact flag names may follow
the existing Slime convention after discovery, but the semantics should be
equivalent to:

```text
adaptive_kl_enabled: false
adaptive_kl_mode: off | telemetry | shadow | reward
adaptive_kl_target: positive finite target KL
adaptive_kl_initial_beta: positive finite coefficient
adaptive_kl_min_beta: positive finite lower bound
adaptive_kl_max_beta: finite upper bound, >= min_beta
adaptive_kl_adaptation_rate: finite positive eta
adaptive_kl_ema_decay: value in [0, 1)
adaptive_kl_max_update_ratio: bounded multiplicative change per update
adaptive_kl_cooldown_steps: non-negative integer
adaptive_kl_spike_threshold: optional finite threshold
adaptive_kl_reference_policy_id: required stable identity
```

Do not overload an unrelated `kl_loss_coef`, `ppo_kl`, or teacher-divergence
flag. If the upstream reward-side path already calls its coefficient
`kl_coef`, introduce a clear adapter/config alias and document that it is the
reward-side coefficient only.

Modes:

- `off`: no controller and no behavior change;
- `telemetry`: compute and log the reference-KL signals without changing
  rewards;
- `shadow`: compute the beta that would have been used and log hypothetical
  shaped-reward statistics, while training uses the fixed baseline beta;
- `reward`: use the updated beta for the next eligible batch.

Only `reward` changes training behavior, and it must remain opt-in.

### 7.2 Controller state

The controller state must be serializable and versioned. At minimum:

```json
{
  "schema_version": 1,
  "beta": 0.0,
  "target_kl": 0.0,
  "ema_kl": null,
  "step": 0,
  "cooldown_remaining": 0,
  "reference_policy_id": "...",
  "loss_family": "ppo_rlhf_reference_reward",
  "controller_version": "..."
}
```

The exact serialization format may be a Reef marker/checkpoint payload or a
Slime checkpoint sidecar, depending on the discovered ownership boundary. It
must satisfy all of these properties:

- save and restore are deterministic;
- an incompatible schema fails loudly;
- a different reference-policy identity cannot silently reuse the old beta;
- beta remains within configured bounds after restore;
- rejected or failed candidates do not become the active controller state;
- a retry of the same training job does not apply the update twice.

### 7.3 Update rule

Use a conservative multiplicative update as the first implementation:

```text
ema_kl = decay * ema_kl + (1 - decay) * observed_kl

raw_beta_next = beta * exp(eta * (ema_kl / target_kl - 1))
beta_next = clip(raw_beta_next,
                  beta * (1 / max_update_ratio),
                  beta * max_update_ratio)
beta_next = clip(beta_next, min_beta, max_beta)
```

Equivalent implementations are acceptable if they preserve the same bounded
semantics. A controller must:

- increase beta when the smoothed KL is above target;
- decrease beta when the smoothed KL is below target;
- apply lower and upper absolute bounds;
- apply a per-update ratio bound;
- honor cooldown and spike handling;
- leave state unchanged on invalid or failed steps.

Do not implement PID or a free-form dual ascent update in phase one. Those can
be evaluated later after telemetry establishes the controller's behavior.

The first implementation should not change beta within a multi-microbatch
optimizer step unless the actual upstream reward path requires it. Prefer one
controller update per completed logical training step, not one update per
microbatch or per data-parallel rank.

### 7.4 Failure and numerical behavior

The controller must be conservative under bad input:

- non-finite KL, loss, beta, or target: do not update;
- empty valid-token mask: do not update and report the reason;
- failed rollout preparation: do not update;
- failed optimizer step: do not update;
- failed checkpoint or publication: keep the pending candidate state
  transactional and do not activate it;
- KL spike above the configured threshold: log the spike, optionally enter
  cooldown, and do not make an uncontrolled large update;
- beta overflow/underflow: clip or fail closed with a clear metric/error;
- reference-policy identity mismatch: stop rather than silently continue.

Training should fail closed for a misconfigured `reward` mode. It should not
silently fall back to an unrelated loss-side KL term.

### 7.5 Transaction and restart behavior

Reef can train a candidate, checkpoint it, publish it, or reject it. The
controller state must follow the same candidate lifecycle:

```text
active state + batch
    -> pending candidate state after successful train
    -> durable candidate checkpoint/marker
    -> active state only after successful publication/commit
    -> discard pending state on rejection or failed candidate
```

The implementation agent must identify the exact atomic boundary in the current
bridge before coding. It is not acceptable to store beta only in process
memory if a restart can cause the model and beta to disagree.

On restart:

1. recover the model/checkpoint marker;
2. recover the matching controller state;
3. verify reference-policy identity and controller schema;
4. restore beta and EMA before accepting the next rollout;
5. emit a restore metric containing the source checkpoint and state version.

### 7.6 Telemetry

Reuse existing worker/bridge metric plumbing where possible. Names should
remain clearly scoped, for example:

```text
adaptive_kl/mode
adaptive_kl/observed_kl
adaptive_kl/smoothed_kl
adaptive_kl/target_kl
adaptive_kl/beta
adaptive_kl/beta_next
adaptive_kl/beta_update_ratio
adaptive_kl/valid_token_fraction
adaptive_kl/finite_loss
adaptive_kl/update_applied
adaptive_kl/update_reason
adaptive_kl/reference_policy_id
adaptive_kl/controller_state_version
adaptive_kl/restore_source
```

Keep these separate from existing `train/ppo_kl`, rollout `kl`, teacher
divergence, and loss-side KL metrics. If a metric is an estimator rather than
the mathematical KL itself, say so in the metric documentation.

At minimum, log both the current beta and the beta that will be used for the
next batch. This makes off-by-one and failed-step bugs visible.

## 8. Integration plan

### Phase 0: audit the exact upstream runtime

Before modifying code, inspect the pinned Slime version and answer all of the
following with file paths, symbols, and tests or call sites:

1. Where is reward-side reference KL computed?
2. Which tensor is `ref_log_probs` and what is its shape/mask?
3. What estimator and sign convention are used?
4. Where is `kl_coef` consumed: Reef driver, Slime driver, or worker?
5. Where are rewards shaped relative to `compute_advantages_and_returns`?
6. Where is observed KL reduced across data-parallel ranks?
7. What signal proves a logical training step succeeded?
8. Which checkpoint owns actor state, optimizer state, reference identity, and
   any reward-shaping state?
9. In what order are checkpoint and controller state restored?
10. Can the next rollout observe the new beta without changing the current
    batch?
11. Does the existing `with_ref` initialization remain valid when beta starts
    at zero but adaptive mode is enabled?
12. Are there existing tests for reward-side KL that can be extended?

If any answer is unknown, continue discovery rather than guessing.

### Phase 1: implement the pure controller

Add a small dependency-light controller module in the most appropriate shared
Reef layer discovered during Phase 0. It should not import Torch, Ray, Slime,
or a recipe-specific objective. It should expose operations equivalent to:

```python
state = controller.state_dict()
controller = AdaptiveKLController.from_state_dict(state, config)
decision = controller.observe(
    observed_kl=...,
    training_step_succeeded=True,
    loss_is_finite=True,
)
```

The pure module should own validation, EMA, bounds, cooldown, update reasons,
and serialization. It should not own rollout transport or checkpoint I/O.

### Phase 2: connect the existing reward path

Use the actual upstream hook found in Phase 0. The integration must:

1. load the reference model whenever adaptive reward mode is enabled;
2. compute the existing reference-policy KL using the upstream estimator;
3. shape rewards before the existing advantage calculation;
4. calculate observed KL using the same valid-token convention;
5. update the controller only after a successful logical train step;
6. pass the next beta to the next batch only;
7. preserve all existing sample masks and reward scaling conventions.

Do not implement a parallel reward pipeline if the upstream one can be safely
parameterized. Do not use `ppo_kl` as a substitute.

### Phase 3: bridge and durable state

Integrate with the existing bridge lifecycle and worker metrics. The pending
controller state should travel with the candidate checkpoint/marker until the
candidate is committed. Rejection, failed publication, and retry must leave the
previous active state intact.

If the correct ownership boundary is in Slime rather than Reef, keep Reef's
integration narrow and document the required upstream change rather than
copying Slime internals into Reef.

### Phase 4: telemetry-only validation

Run the controller in `telemetry` mode on a fixed-beta baseline. Verify:

- reference model identity is stable;
- observed KL is finite and aggregated once;
- masks and token counts are correct;
- metrics line up between worker and bridge;
- no rewards, advantages, or weights change relative to baseline.

### Phase 5: shadow mode

Run `shadow` mode with the same seed/data/checkpoint as the fixed-beta run.
Compare hypothetical beta, shaped reward, KL, token counts, throughput, and
failure handling. This is the first place to tune target and bounds.

### Phase 6: opt-in reward mode

Run small, reproducible experiments with the feature explicitly enabled. Keep a
simple rollback switch that returns to `off` or fixed beta without deleting
controller state or checkpoints.

## 9. Candidate files to add or modify

These are hypotheses to validate during Phase 0, not a pre-approved patch
list:

```text
reef/train/slime_backend/<shared adaptive-KL controller module>
reef/train/slime_backend/reef_adapters/<bridge or state integration>
reef/train/slime_backend/reef_adapters/worker_hooks.py       # only if needed
<selected PPO/RLHF recipe spec/objective>                    # only if needed
tests/<pure controller tests>
tests/<integration/contract tests>
docs/adaptive-kl-ppo-rlhf-implementation.md                 # this document
```

Do not modify `recipes/sao/`, `recipes/tttd/`, `recipes/openclawrl/`, or the
distillation backend merely to make the first phase compile. If the chosen
PPO/RLHF path is not yet represented by a dedicated Reef recipe, report that
gap and propose the smallest owner rather than silently attaching to SAO or
OpenClawRL.

## 10. Tests and acceptance criteria

### 10.1 Pure controller tests

Cover at least:

- above-target KL increases beta;
- below-target KL decreases beta;
- exact-target KL leaves beta stable within tolerance;
- absolute min/max bounds;
- per-update ratio bound;
- EMA behavior and first-observation initialization;
- cooldown behavior;
- spike behavior;
- NaN/Inf/negative/zero-invalid inputs;
- failed training step does not update state;
- failed/non-finite loss does not update state;
- state round-trip is deterministic;
- incompatible schema is rejected;
- reference-policy identity mismatch is rejected;
- repeated restore does not double-apply an update.

### 10.2 Integration tests

Cover at least:

- reward shaping occurs before advantage calculation;
- the current batch uses beta-before-update;
- the next batch uses beta-after-update;
- valid-token masks are honored;
- data-parallel observed KL is reduced once, not once per rank;
- reference actor is present in adaptive mode even if initial beta is zero;
- telemetry mode produces metrics but leaves training tensors unchanged;
- shadow mode does not change model updates;
- candidate rejection restores the prior active controller state;
- restart restores model and controller state together;
- `ppo_kl` and `adaptive_kl/observed_kl` remain distinct metrics;
- existing off-by-default recipe tests remain unchanged.

### 10.3 Experiment acceptance

Before proposing default enablement, provide paired fixed-beta and adaptive-
beta runs using the same model, data, reference policy, seed policy, and
evaluation setup. Report:

- task reward and success rate;
- reference-policy KL distribution and target tracking;
- beta trajectory and update reasons;
- reward/advantage statistics;
- response length and truncation;
- throughput and memory;
- NaN/Inf/failure counts;
- checkpoint/restart behavior;
- any regression on protected tasks.

The first implementation is accepted only if it is opt-in, reproducible,
rollbackable, and shows no silent mixing of KL families. Improvement in task
quality is an experiment question, not an assumption baked into the controller.

## 11. Experiment matrix

Start with a small matrix:

| Run | Mode | Beta | Purpose |
| --- | --- | --- | --- |
| A | `off` | existing fixed baseline | Regression baseline. |
| B | `telemetry` | existing fixed beta | Validate estimator and metrics only. |
| C | `shadow` | fixed beta for training | Validate hypothetical controller trajectory. |
| D | `reward` | adaptive, conservative bounds | First behavior-changing run. |
| E | `reward` | adaptive, alternate target | Sensitivity check. |

Do not compare runs that use different reference-policy checkpoints or silently
different reward scaling. Store configuration, exact code commit, exact Slime
commit, controller state, and metrics with every run.

## 12. Relationship to evaluation and Reef #466

This phase is a training-side mechanism. It does not feed held-out evaluation
scores into the beta update. Reef-level evaluation, protected regression checks,
publish/pause/rollback, and candidate gating remain separate concerns.

The intended relationship is:

```text
training-side controller: keeps reference KL near a configured target
evaluation-side policy: decides whether the resulting candidate is acceptable
```

An evaluation failure may reject a candidate and therefore discard its pending
controller state, but evaluation score must not be used as `observed_kl` or as a
replacement for the training reward.

## 13. Instructions for the server-side coding agent

Give the coding agent this document together with the Reef checkout and the
exact Slime runtime. Require the agent to follow this order:

1. Read this entire document before editing anything.
2. Inspect the current git status and preserve unrelated user changes.
3. Identify the exact Slime commit/package and report it.
4. Locate the real reward-side reference-KL path, estimator, masks, reduction,
   and advantage boundary.
5. Locate checkpoint save/load and candidate commit/rejection boundaries.
6. Produce a short discovery report with file paths and symbols.
7. Stop and ask for review if the only available signal is `ppo_kl`, teacher
   divergence, TTTD KL, or a final-loss `kl_loss_coef`.
8. Implement and test the pure controller first.
9. Add telemetry-only integration.
10. Add shadow mode.
11. Add opt-in reward mode only after the previous tests pass.
12. Keep all unrelated recipes and the default path unchanged.
13. Run focused CPU/unit/contract tests after each stage, then the relevant
    integration tests.
14. Run `git diff --check` and report every modified file.

The agent must not:

- invent paths, flags, or tensor semantics without citing the runtime source;
- change all KL coefficients through one shared controller;
- convert an existing loss-side KL into reward shaping by renaming a variable;
- enable the feature by default;
- update beta on failed or rejected candidates;
- hide a required upstream Slime change;
- delete or reset unrelated worktree files.

The final implementation report must contain:

```text
Exact Reef commit/base:
Exact Slime commit/package:
Discovery report:
Modified files:
New configuration and default values:
Controller state format and ownership:
Reward-side estimator and sign convention:
Checkpoint/restart/rollback behavior:
Tests run and results:
Telemetry examples:
Experiment commands/configs:
Known limitations:
Rollback procedure:
Any required upstream Slime PR or issue:
```

## 14. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Wrong KL family is wired in | Require source-level estimator audit and separate metric names. |
| Reference actor is absent when beta becomes positive | Construct `with_ref` whenever adaptive mode is enabled, independent of initial beta. |
| Beta oscillates or explodes | EMA, absolute bounds, per-update ratio bounds, cooldown, finite checks. |
| Failed steps advance controller state | Update only on successful logical steps and persist transactionally. |
| Restart mixes model and beta | Store/restore controller state with the candidate checkpoint and verify identity. |
| Data-parallel ranks over-count KL | Reduce at one explicit aggregation point and test rank invariance. |
| Reward scaling changes silently | Reuse the existing reward path and record shaped/unshaped statistics. |
| Adaptive mode regresses protected tasks | Keep Reef evaluation and rejection separate and retain rollback. |
| Upstream Slime API differs from assumptions | Pin and report the exact runtime; stop at a documented missing-hook boundary. |

## 15. Explicit stop conditions

Stop implementation and return a discovery report instead of guessing if any of
the following is true:

- no safe reference-policy reward-shaping hook exists in the pinned Slime
  runtime;
- `ref_log_probs` or the reference-policy identity cannot be tied to the
  rollout safely;
- the only available KL is `ppo_kl` or a teacher/loss-side divergence;
- successful-step and candidate-commit boundaries cannot be identified;
- controller state cannot be restored atomically with model state;
- data-parallel observed KL cannot be aggregated without double counting;
- adding the feature would require changing a non-target recipe;
- the implementation would change default behavior.

In any stop case, do not broaden the scope to OPD, SAO, GRPO, TTTD, or
OpenClawRL. Report the smallest upstream interface or Reef design decision
needed to continue.
