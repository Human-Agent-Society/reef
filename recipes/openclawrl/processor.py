"""The OpenClaw-RL record-to-task and task-to-batch stages: one judged turn, one sample.

A computed-feedback recipe. The correlation and the judge are
:class:`~recipes.openclawrl.judging.OpenClawRLJudging`, which runs in the
record-to-record stage :class:`~recipes.openclawrl.judging.OpenClawRLRecord2Record`.
This file validates the tensors (``make_sample``), shapes the batch
(``make_batch``), and writes the PRM record file. Rewards pass through raw
(upstream ``--n-samples-per-prompt 1``; no grouping, no normalization).
"""

from __future__ import annotations

import collections
import json
import logging
import pathlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Self

from recipes.openclawrl.judging import OpenClawRLRecord2Record
from recipes.openclawrl.turns import TurnJudgment, validate_teacher_cands
from reef.core.config import config_option
from reef.core.plugins import NoSettings, PluginConfigError
from reef.core.records_types import AgentRecord
from reef.core.trajectories import source_record_id, trajectory_reward
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record
from reef.record2task import JudgedTasks, Record2Task
from reef.record2task.samples import make_policy_trajectory
from reef.task2batch import Task2Batch
from reef.task2batch.base import BatchSettings
from reef.train.types import TrainDataItem, TrainingBatch, TrajectoryItem, policy_row_violation

logger = logging.getLogger(__name__)


class OpenClawRLTasks(JudgedTasks):
    """One judged turn as one sample; a turn whose tensors the top-K select loss cannot train on retires.

    The capture must cover every response token, and every trained sample
    carries at least one teacher candidate. The stage takes no config.
    """

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls()

    def make_sample(self, record: AgentRecord, judgment: TurnJudgment) -> TrajectoryItem | None:
        # 1. A declined judgment never trains. (Narrow on the score itself:
        #    it is what step 2 needs, and a property would not narrow it.)
        if judgment.score is None:
            return None
        # 2. The policy tensors must satisfy the bridge contract, and the
        #    top-K capture must cover every response token.
        sample = make_policy_trajectory(record, judgment.score)
        if (
            policy_row_violation(
                sample.training.get("tokens", []),
                sample.training.get("loss_mask", []),
                sample.training.get("rollout_log_probs", []),
            )
            is not None
        ):
            return None
        if not sample.training.get("topk_indices", []) or len(sample.training.get("topk_indices", [])) != len(
            sample.training.get("loss_mask", [])
        ):
            return None
        # 3. Every trained sample carries at least one teacher candidate —
        #    upstream constructs one per sample by design (RL-only turns get
        #    the anchor), so a candidate-less judgment here means candidate
        #    construction failed and the turn retires instead of training a
        #    shape the reference never produces.
        if not judgment.teacher_cands:
            return None
        validated = validate_teacher_cands(judgment.teacher_cands, sample)
        if validated is None:
            return None
        return sample.with_training(extras={**sample.training.get("extras", {}), "teacher_cands": list(validated)})


@dataclass(frozen=True)
class OpenClawRLBatchSettings(BatchSettings):
    """The config of ``OpenClawRLBatches``."""

    batch_size: int = config_option(
        16, help="The judged turns in one batch. Must equal the Slime driver's --global-batch-size."
    )
    # Upstream's OPENCLAW_PRM_RECORD_FILE.
    prm_record_file: str = config_option(
        "", help="The file that gets one JSON line for each batch: the judged population. Empty disables it."
    )


class OpenClawRLBatches(Task2Batch):
    """Judged turns into batches, and the PRM record file.

    Config: :class:`OpenClawRLBatchSettings`. The PRM record file reports
    what the judging of ``record2record`` saw, so the stage needs the
    scenario's ``OpenClawRLRecord2Record``.
    """

    def __init__(
        self, context: PipelineContext, settings: OpenClawRLBatchSettings, record2record: OpenClawRLRecord2Record
    ) -> None:
        super().__init__(context, settings.batch_size)
        # What the judge actually saw goes to the PRM record file, one line
        # per batch. Upstream persists this per turn; without it a batch mean
        # is the only visible number. A file, not a log line: reef configures
        # no logging, so anything below WARNING is discarded.
        record_file = settings.prm_record_file.strip()
        self.record_path = pathlib.Path(record_file) if record_file else None
        self.openclawrl_judging = record2record.openclawrl_judging

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2task: Record2Task,
        environ: Mapping[str, str],
    ) -> Self:
        settings = OpenClawRLBatchSettings.parse(config, plugin_class=cls)
        if not isinstance(record2record, OpenClawRLRecord2Record):
            raise PluginConfigError(
                f"{cls.__name__} needs OpenClawRLRecord2Record as record2record, not {type(record2record).__name__}"
            )
        return cls(context, settings, record2record)

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        samples = tuple(task for task in tasks if isinstance(task, TrajectoryItem))
        if len(samples) != len(tasks):
            raise TypeError("OpenClaw-RL tasks must be trajectories")
        self.write_record_line(samples, batch_id)
        return TrainingBatch(
            batch_id,
            tuple(replace(sample, source_agent_record_ids=(source_record_id(sample),)) for sample in samples),
        )

    def write_record_line(self, samples: tuple[TrajectoryItem, ...], batch_id: str) -> None:
        """Append this batch's judged population to the PRM record file.

        One line per batch: what the judges returned since the previous one,
        the reward mix that reached the trainer, and how correlation is doing
        overall. A batch mean alone cannot distinguish "the judges disagree"
        from "the judges never ran", and neither can it show which next
        states are being judged.
        """
        record = {
            "batch_id": batch_id,
            "scenario": self.scenario,
            "samples": len(samples),
            "rewards": dict(sorted(collections.Counter(f"{trajectory_reward(s):+.0f}" for s in samples).items())),
            "judged": dict(sorted(self.openclawrl_judging.judged.items())),
            # Cumulative, unlike the per-batch counters above: these answer
            # "is correlation working" for the run as a whole.
            "correlation": asdict(self.openclawrl_judging.index.stats),
        }
        self.openclawrl_judging.judged.clear()
        if self.record_path is None:
            return
        try:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            with self.record_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError:
            # Diagnostics must never take a training step down with them.
            logger.warning("openclawrl: could not append to %s", self.record_path, exc_info=True)
