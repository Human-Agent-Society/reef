"""The dataset recipe's job stage: one job, with no samples, for each training instruction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.core.plugins import NoSettings
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record
from reef.record2sample import Record2Sample
from reef.sample2job.base import Sample2Job
from reef.train.types import JobData, Sample


class DatasetJobData(Sample2Job):
    """Make the job of a training instruction: no samples, and the recipe's parameters under the instruction's.

    The instruction's ``parameters`` name the job's ``dataset``, which the
    recipe's ``job_parameters`` can also set; the training component checks
    it. The stage runs in ``manual`` only and takes no samples.
    """

    supported_training_modes = frozenset({"manual"})
    reads_instructions = True

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
        NoSettings.parse(config, plugin_class=cls)
        return cls(context)

    def __init__(self, context: PipelineContext) -> None:
        # The trainer requires a job size; this stage makes a job per instruction, in manual only, so the
        # size never makes data ready (an empty buffer holds no sample).
        super().__init__(context, None, 1)

    def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> JobData:
        return JobData(data_id, ())

    def job_parameters(self, data: JobData) -> Mapping[str, Any]:
        """The recipe's parameters, which every kind shares, with the instruction's ``parameters`` over them."""
        defaults = self.context.job_parameters.get("trajectory") or {}
        overrides = {} if data.request is None else data.request.parameters
        return {**defaults, **overrides}
