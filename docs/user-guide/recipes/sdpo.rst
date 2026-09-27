SDPO: learn from feedback on the model's own attempts
======================================================

Self-Distillation Policy Optimization (`arXiv:2601.20802
<https://arxiv.org/abs/2601.20802>`__) uses an on-policy response twice.
The student reads the original question; an EMA teacher reads that question
plus a successful sibling response or environment feedback. The teacher
scores the student's exact response tokens, and the student minimizes a
divergence to those conditional next-token distributions.

``recipes/sdpo/`` contains the group preparer, report processor and Slime
loss family. The preparer requires one question and one artifact version per
group. It selects a successful sibling and formats optional execution
feedback. The processor uses ``TeacherContextReport`` and appends the
recorded response token IDs to the teacher prompt. Inactive attempts in a
group remain in the batch with zero sample weight and unchanged response masks.

The ``sdpo`` loss family uses student top-K IDs and a tail probability
bucket. Its default is top-K 100, JSD, EMA rate 0.05 and clipped per-token
importance sampling; rich feedback experiments use top-K 20, reverse KL
and EMA rate 0.01. Leave Slime's ``--calculate-per-token-loss`` disabled:
the pinned reference averages per-response token means over the full batch,
including inactive responses as zero. The student top-K IDs require an extra
no-gradient forward pass before the teacher pass.

The report code and current validation boundary are documented in
`the recipe README <../../../recipes/sdpo/README.md>`__. No Reef GPU
training result or paper performance comparison has been recorded yet.
