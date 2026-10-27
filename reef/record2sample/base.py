"""The record-to-sample contract: each record set from the record-to-record stage becomes samples.

A sample is a ``Sample``: a ``TrajectorySample`` that trains directly, or a
``TaskSample`` that the algorithm rolls out. The training pipeline pushes each
record set that the record stage forms (``push``) and pulls the sample results
(``pull``). It gives each result back to the record stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from reef.core.job_data import Sample, TrajectorySample
from reef.core.plugins import NoSettings, Plugin, PluginConfigError
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record, RecordSet, SampleStatus


@dataclass(frozen=True)
class SampleResult:
    """The samples made from one record set, and what the record stage does with the set.

    Values: a ``READY`` result holds one or more samples. A ``SKIPPED`` result holds no sample.
    """

    status: SampleStatus
    samples: tuple[Sample, ...] = ()

    def __post_init__(self) -> None:
        if (self.status is SampleStatus.READY) != bool(self.samples):
            raise ValueError("a READY result holds one or more samples, and a SKIPPED result holds no sample")

    @classmethod
    def ready(cls, samples: tuple[Sample, ...]) -> SampleResult:
        """The set trains as ``samples``."""
        return cls(SampleStatus.READY, samples)

    @classmethod
    def skipped(cls) -> SampleResult:
        """The set cannot train. The record stage ends its records."""
        return cls(SampleStatus.SKIPPED)


class Record2Sample(ABC, Plugin):
    """The record-to-sample stage: record sets in, sample results out.

    The pipeline pushes each record set one time. ``pull`` returns each pushed
    set's result exactly one time, by set ID, as soon as it is ready. Both run
    on the trainer thread and must not block on network or model calls. A
    stage with slow work runs it on a worker that it owns, and ``close`` stops
    the worker. Raise on a broken training contract. Most stages make the
    samples of a set at once: they inherit ``ImmediateRecord2Sample``. A
    deployment names a stage class in ``pipeline.record2sample`` and gives its
    config. The class parses the config into its settings dataclass and builds
    the stage in ``from_config``.
    """

    def uploaded_sample(self, trajectory: TrajectorySample) -> TrajectorySample:
        """The training sample of an uploaded ATIF trajectory; the default is the trajectory as it was uploaded.

        The default fits a backend that reads ATIF trajectories as they are. A stage whose backend trains
        token-level data, and whose sample-to-job stage accepts uploaded trajectories, overrides it to render the
        trajectory as its backend trains it, such as ``SFTSamples``. The pipeline calls it at upload, from a request thread, to refuse a
        trajectory that cannot train, and again when it reads the upload. It must be pure: the same trajectory
        gives the same sample, and the stage's state does not change. A trajectory that cannot become a sample
        raises ``ValueError`` with the reason.
        """
        return trajectory

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> Self:
        """Build the stage of one scenario from its config.

        ``record2record`` is the scenario's record-to-record stage, built first.
        ``environ`` holds the environment the deployment was read with.
        """
        raise PluginConfigError(f"{cls.__name__} cannot be built from a deployment file: it has no from_config")

    @classmethod
    def scenario_state_dirs(cls, config: Mapping[str, Any], scenario: str) -> tuple[Path, ...]:
        """The directories under which a stage built from ``config`` keeps a scenario's state, for archiving."""
        return ()

    @abstractmethod
    def push(self, record_set: RecordSet) -> None:
        """Take one record set to make its samples."""

    @abstractmethod
    def pull(self) -> tuple[tuple[Hashable, SampleResult], ...]:
        """Return the results that are ready since the last call, by set ID."""

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe state that the pipeline adds to its status."""
        return {}

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Return numeric state that the pipeline adds to its metrics."""
        return {}

    def close(self) -> None:
        """Release resources that the stage owns. Calling it more than one time is safe."""
        return


class ImmediateRecord2Sample(Record2Sample):
    """A record-to-sample stage that makes the samples of a set when the pipeline pushes it.

    A subclass implements ``make_samples``. A subclass with its own ``__init__``
    calls ``super().__init__()``.
    """

    def __init__(self) -> None:
        self.results: list[tuple[Hashable, SampleResult]] = []

    @abstractmethod
    def make_samples(self, record_set: RecordSet) -> SampleResult:
        """Return the samples of ``record_set``, or a skip."""

    def push(self, record_set: RecordSet) -> None:
        self.results.append((record_set.set_id, self.make_samples(record_set)))

    def pull(self) -> tuple[tuple[Hashable, SampleResult], ...]:
        results, self.results = tuple(self.results), []
        return results


class NoSamples(Record2Sample):
    """The record-to-sample stage of a pipeline that trains nothing. Its record stage, ``NoRecordSets``, forms no set."""

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

    def push(self, record_set: RecordSet) -> None:
        raise TypeError(f"{type(self).__name__} makes no samples: pair it with NoRecordSets")

    def pull(self) -> tuple[tuple[Hashable, SampleResult], ...]:
        return ()
