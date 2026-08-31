"""reef-eval: autoresearch and continual-learning evaluation on the Harbor
task standard.

One primitive, the episode (one Harbor trial): a task run under an agent
and scored by an isolated verifier. The judge's score for every submission
along the way is recorded as ``trace`` rows beside it.

Two regimes on top. Autoresearch measures learning within one open-ended
episode (the anytime curve of judge-scored submissions); streams measure
what carries into later tasks (a :class:`Stream` of episodes under one
carried agent state).

:class:`Lab` runs episodes into an append-only results store,
:class:`Stream` sequences them with carried state, and
:mod:`reef_eval.metrics` turns the store into curves.
"""

from reef_eval import metrics
from reef_eval.budget import Budget
from reef_eval.executors import FakeExecutor, HarborExecutor, LocalExecutor
from reef_eval.lab import Lab
from reef_eval.stream import Stream
from reef_eval.targets import tasks
from reef_eval.types import EpisodeResult, EpisodeSpec, Row, TracePoint

__version__ = "0.1.0"

__all__ = [
    "Lab",
    "Stream",
    "tasks",
    "Budget",
    "HarborExecutor",
    "LocalExecutor",
    "FakeExecutor",
    "EpisodeSpec",
    "EpisodeResult",
    "TracePoint",
    "Row",
    "metrics",
]
