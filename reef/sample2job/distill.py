"""Job data of distillation samples; ``reef.record2sample.distill`` makes the samples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.core.job_data import JobData, Sample
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record
from reef.record2sample import Record2Sample
from reef.sample2job.base import Sample2Job, Sample2JobSettings


class DistillJobData(Sample2Job):
    """Put the samples into one job's data as they come, in selection order."""

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2sample: Record2Sample,
        environ: Mapping[str, str],
    ) -> Self:
        settings = Sample2JobSettings.parse(config, plugin_class=cls)
        return cls(context, settings.buffer_size, settings.job_samples)

    def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> JobData:
        return JobData(data_id, samples)
