"""OpenClaw-RL example: the harness side of the bundled recipe.

The method itself (next-state binary reward, arXiv:2603.10165) lives
entirely in the reef package — the ``openclawrl`` recipe's processor
correlates sessions from recorded traffic and judges turns with its
private PRM worker (``recipes/openclawrl/processor.py``).

This directory keeps the harness side: the tide task stream (``run.sh``,
``harness/``, ``harbor-tasks/``) driving a real Hermes agent through 72
GSM8K homework sessions, with the student simulated by ``user_sim/``.
"""
