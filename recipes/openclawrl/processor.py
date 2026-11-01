"""The OpenClaw-RL record-to-sample and sample-to-job stages: one judged turn, one sample.

A computed-feedback recipe. The correlation and the judge are
:class:`~recipes.openclawrl.judging.OpenClawRLJudging`, which runs on the PRM
worker of :class:`OpenClawRLSamples`. This file makes each judged turn's
sample with its teacher candidates (``make_sample``), shapes the data
(``make_job_data``), and writes the PRM record file. Rewards pass through raw
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

from recipes.openclawrl.judging import OpenClawRLJudging, OpenClawRLJudgingSettings
from recipes.openclawrl.turns import TurnJudgment
from reef.core.chat_rendering import TeacherRequest, trajectory_tools
from reef.core.config import config_option
from reef.core.plugins import PluginConfigError
from reef.core.records_types import AgentRecord
from reef.core.trajectories import make_trajectory, source_record_id, trajectory_reward
from reef.execution.jobs import BackgroundJobWorker
from reef.pipeline_context import PipelineContext
from reef.record2sample import JudgedSamples, Record2Sample
from reef.sample2job import Sample2Job, Sample2JobSettings
from reef.train.types import JobData, Sample, TrajectorySample

logger = logging.getLogger(__name__)


class OpenClawRLSamples(JudgedSamples):
    """Sessions correlated, turns judged on a PRM worker, and one judged turn as one sample.

    The sample is the turn's call as an ATIF trajectory, whose reply trains, with its teacher candidates in
    ``extra.reef.teacher_candidates`` (``reef.core.chat_rendering.teacher_candidates``). Every trained sample carries
    at least one teacher candidate; a declined or candidate-less judgment retires the turn. The training component
    renders the sample and its candidates, and the actor computes the student's top-K. Config:
    :class:`~recipes.openclawrl.judging.OpenClawRLJudgingSettings`. Without a ``worker``, the stage starts one when
    the config names a PRM judge. ``worker`` and ``clients`` (PRM judge, renderer) replace the configured ones in
    tests.
    """

    def __init__(
        self,
        settings: OpenClawRLJudgingSettings,
        *,
        worker: BackgroundJobWorker | None = None,
        clients: tuple[Any, Any] | None = None,
    ) -> None:
        #: The judging that the batches read for the PRM record file.
        self.openclawrl_judging = OpenClawRLJudging(settings, clients)
        if worker is None and self.openclawrl_judging.prm is not None:
            worker = BackgroundJobWorker(
                self.openclawrl_judging, thread_name="openclawrl-judging", concurrency=settings.prm_concurrency
            )
        super().__init__(self.openclawrl_judging, worker, max_staleness=settings.max_staleness)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(OpenClawRLJudgingSettings.parse(config, plugin_class=cls))

    def make_sample(self, record: AgentRecord, judgment: TurnJudgment) -> TrajectorySample | None:
        # 1. A declined judgment never trains. (Narrow on the score itself.)
        if judgment.score is None:
            return None
        # 2. Every trained sample carries at least one teacher candidate — upstream constructs one per sample by
        #    design (RL-only turns get the anchor), so a candidate-less judgment means candidate construction failed
        #    and the turn retires instead of training a shape the reference never produces.
        if not judgment.teacher_cands:
            return None
        # 3. The turn's call, whose reply trains; the training component renders it and each candidate, and the
        #    actor computes the student's top-K from the weights the step starts from.
        sample = make_trajectory((record,), judgment.score)
        tools = trajectory_tools(sample.trajectory)
        candidates = [
            {"hint": str(candidate.get("hint", "")), **TeacherRequest(candidate.get("messages"), tools).as_json()}
            for candidate in judgment.teacher_cands
        ]
        return sample.with_metadata(teacher_candidates=candidates)


@dataclass(frozen=True)
class OpenClawRLJobSettings(Sample2JobSettings):
    """The config of ``OpenClawRLJobData``: the buffer, the job size, and the PRM record file."""

    # Upstream's OPENCLAW_PRM_RECORD_FILE.
    prm_record_file: str = config_option(
        "", help="The file that gets one JSON line for each job: the judged population. Empty disables it."
    )


class OpenClawRLJobData(Sample2Job):
    """Judged turns into job data, and the PRM record file.

    Config: :class:`OpenClawRLJobSettings`. The PRM record file reports
    what the judging of ``record2sample`` saw, so the stage needs the
    scenario's ``OpenClawRLSamples``.
    """

    def __init__(
        self, context: PipelineContext, settings: OpenClawRLJobSettings, record2sample: OpenClawRLSamples
    ) -> None:
        super().__init__(context, settings.buffer_size, settings.job_samples)
        # What the judge actually saw goes to the PRM record file, one line
        # per job. Upstream persists this per turn; without it a job mean
        # is the only visible number. A file, not a log line: reef configures
        # no logging, so anything below WARNING is discarded.
        record_file = settings.prm_record_file.strip()
        self.record_path = pathlib.Path(record_file) if record_file else None
        self.openclawrl_judging = record2sample.openclawrl_judging

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2sample: Record2Sample,
        environ: Mapping[str, str],
    ) -> Self:
        settings = OpenClawRLJobSettings.parse(config, plugin_class=cls)
        if not isinstance(record2sample, OpenClawRLSamples):
            raise PluginConfigError(
                f"{cls.__name__} needs OpenClawRLSamples as record2sample, not {type(record2sample).__name__}"
            )
        return cls(context, settings, record2sample)

    def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> JobData:
        turns = tuple(sample for sample in samples if isinstance(sample, TrajectorySample))
        if len(turns) != len(samples):
            raise TypeError("OpenClaw-RL samples must be trajectories")
        self.write_record_line(turns, data_id)
        return JobData(
            data_id,
            tuple(replace(sample, source_agent_record_ids=(source_record_id(sample),)) for sample in turns),
        )

    def write_record_line(self, samples: tuple[TrajectorySample, ...], data_id: str) -> None:
        """Append this job's judged population to the PRM record file.

        One line per job: what the judges returned since the previous one,
        the reward mix that reached the trainer, and how correlation is doing
        overall. A job mean alone cannot distinguish "the judges disagree"
        from "the judges never ran", and neither can it show which next
        states are being judged.
        """
        record = {
            "data_id": data_id,
            "scenario": self.scenario,
            "samples": len(samples),
            "rewards": dict(sorted(collections.Counter(f"{trajectory_reward(s):+.0f}" for s in samples).items())),
            "judged": dict(sorted(self.openclawrl_judging.judged.items())),
            # Cumulative, unlike the per-job counters above: these answer
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
