"""The dataset recipe's job stage: one job, with no samples, for each training request."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.core.plugins import NoSettings
from reef.core.values import deep_merge
from reef.pipeline_context import PipelineContext
from reef.record2sample import Record2Sample
from reef.sample2job.base import Sample2Job
from reef.train.types import JobData, Sample


class DatasetJobData(Sample2Job):
    """Make the job of a training request: no samples, and the recipe's parameters under the request's.

    The request's ``parameters`` name the job's ``dataset``, which the
    recipe's ``job_parameters`` can also set; the training component checks
    it. The stage runs in ``manual`` only and takes no samples.
    """

    supported_training_modes = frozenset({"manual"})
    reads_requests = True

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2sample: Record2Sample,
        environ: Mapping[str, str],
    ) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls(context)

    def __init__(self, context: PipelineContext) -> None:
        # The trainer requires a job size; this stage makes a job per request, in manual only, so the
        # size never makes data ready (an empty buffer holds no sample).
        super().__init__(context, None, 1)

    def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> JobData:
        return JobData(data_id, ())

    def job_parameters(self, data: JobData) -> Mapping[str, Any]:
        """The recipe's parameters, which every kind shares, with the request's ``parameters`` merged in."""
        defaults = self.context.job_parameters.get("trajectory") or {}
        overrides = {} if data.request is None else data.request.parameters
        return deep_merge(defaults, overrides)
