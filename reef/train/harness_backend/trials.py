"""Trial episodes a proposer runs while it builds its patch.

``EpisodeTrials`` runs a tree's episodes through the step's evaluator: the
same rendering with the served model binding, the same worker pool, and the
same scorer as the evaluation, so a probe measures what the evaluation would.
Each trial notes itself in the step's activity and keeps its trajectory under
the step record's ``trials/`` directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from reef.evaluation.harness import HarnessEpisodeEvaluator, ScoredEpisode
from reef.harness.tree import HarnessTree
from reef.train.harness_backend.contract import ProposerCalls, TrialRunner

TRIALS_DIR = "trials"


class EpisodeTrials(TrialRunner):
    """The trials of one step: episodes of proposed trees, on the evaluation's own machinery.

    ``evaluator`` is the step's evaluator, ``calls`` the step's activity and
    budget, and ``record_dir`` the step's record directory (``None`` with the
    record off). Trial ``n`` of the step keeps task ``i``'s trajectory in
    ``trials/<n>-<i>``.
    """

    def __init__(self, evaluator: HarnessEpisodeEvaluator, calls: ProposerCalls, record_dir: Path | None) -> None:
        self.evaluator = evaluator
        self.calls = calls
        self.record_dir = record_dir
        self.trial_count = 0

    def run(self, tree: HarnessTree, tasks: Sequence[str]) -> tuple[ScoredEpisode, ...]:
        self.trial_count += 1
        files = self.evaluator.render(tree)
        keep = None if self.record_dir is None else self.record_dir / TRIALS_DIR
        pairings = [
            (files, task, None if keep is None else keep / f"{self.trial_count}-{index}")
            for index, task in enumerate(tasks)
        ]
        self.calls.note("trial", f"trial {self.trial_count}: {len(pairings)} episode(s)")
        return tuple(self.evaluator.run_episodes(pairings))


__all__ = ["EpisodeTrials"]
