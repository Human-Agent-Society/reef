# SDPO: reinforcement learning via self-distillation

This recipe implements the feedback-conditioned self-teacher of
[arXiv:2601.20802](https://arxiv.org/abs/2601.20802). The reference used for
the implementation is `lasgroup/SDPO` at
`7c457fc1b1f636ae794eb0362ba37d4743b06fbc`. Code and numerical tests
have passed focused CPU checks; there are no Reef GPU training results yet.

## Data flow

1. Sample several responses to the same question with the current published
   Reef artifact. Keep every inference receipt, response and artifact version.
2. Grade the whole group before reporting. For each attempt,
   `prepare_group()` chooses the first successful *other* response (if any)
   and/or that attempt's environment feedback. It rejects mixed versions and
   duplicate receipts. Even a group with no usable feedback retains all
   attempts with zero weights, preserving the reference's batch denominator.
3. Send each prepared report's `payload()` through `ReefClient.report()` to the
   same scenario. A report references one inference. Within a group containing
   feedback, attempts lacking a successful sibling or environment feedback
   have an empty teacher context and a zero sample weight. Their original
   response masks remain intact for diagnostic metrics.
4. `SDPOProcessor` renders the original request with the added context using
   the served tokenizer, then appends the **exact recorded response token IDs**.
   The teacher scores that sequence. The student scores the original request.
5. The `sdpo` Slime loss family selects student top-K IDs in a no-gradient
   forward pass, scores the feedback-conditioned teacher at those IDs, and
   minimizes JSD or reverse KL on the K shared IDs plus the remaining tail
   probability. The teacher moves toward the actor by EMA before each step.
   The teacher's packing schedule selects the corresponding student IDs;
   matching never requires copying prompt tokens into Python.

The default family settings follow the paper's no-rich-feedback configuration:
JSD with teacher mixture weight 0.5, top-K 100 plus tail, EMA rate 0.05 and
per-token importance weights clipped at 2. For LiveCodeBench-style rich
feedback, use `--sdpo-divergence=reverse`, `--sdpo-top-k=20`, and
`--sdpo-teacher-update-rate=0.01`. Leave Slime's
`--calculate-per-token-loss` disabled: the pinned reference uses one response
per microbatch, takes its token mean, then averages over all responses,
including inactive responses as zero. A global mean over active tokens would
give longer responses more weight and change the gradient scale.
The reference disables Qwen3 thinking in both prompts; this recipe defaults
`enable_thinking` to false for the teacher, and the example sends the same
setting in the student's `chat_template_kwargs`.
`batch_size` must equal `--global-batch-size` and counts attempted responses,
including zero-weight attempts and groups.

## Report construction

```python
from recipes.sdpo.preparer import SDPOAttempt, prepare_group

attempts = [
    SDPOAttempt("question-1", "receipt-1", "version-7", "wrong", 0.0, "Test 2 failed"),
    SDPOAttempt("question-1", "receipt-2", "version-7", "correct", 1.0),
]
for prepared in prepare_group(attempts):
    client.report(scenario, prepared.payload())
```

Call `prepare_group()` only after all attempts for that question have been
graded. The score is metadata; the loss distils teacher distributions and
does not use it as a policy-gradient reward. The harness must keep public
training feedback separate from held-out tests. On-policy training should
wait for the published version before sampling the next group.
An oversized teacher sequence fails explicitly: dropping one report would
leave the fixed-size group incomplete. Increase both the recipe's teacher
limit and the trainer's sequence window, or shorten feedback and retry the
complete group. SDFT retains its existing overflow-discard behavior.

## Current validation boundary

The tests cover group selection, report references, response-token alignment,
settings and CPU top-K-plus-tail loss/gradient parity. A real Slime worker run,
EMA checkpoint restoration, SciKnowEval/ToolAlpaca/LiveCodeBench v6 campaigns,
GRPO controls and test-time `discovery@k` remain to be run on a suitable GPU
host. The current moving teacher is re-seeded from the actor after restart;
do not treat an interrupted training run as a continuous EMA run.

To compare the worker loss and gradients directly with the author's pinned
functions, check out `lasgroup/SDPO` at the commit above and run:

```bash
SDPO_REFERENCE_ROOT=/path/to/SDPO python -m pytest tests/reef_service/test_sdpo_reference.py -q
```

This optional CPU test reads the committed source at the pin and executes
the author's numerical functions without importing the verl GPU stack. It
covers forward/reverse KL, JSD, several top-K sizes, inactive responses and
clipped token importance weights. It does not run the author's trainer.
