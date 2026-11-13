"""The record-to-sample stage: records in, samples out, for one pipeline.

``Record2Sample`` is the interface. ``NoSamples`` is the default of a pipeline that trains nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

from reef.core.errors import inference_incomplete
from reef.core.job_data import JobOutcome, Sample, TrajectorySample
from reef.core.plugins import NoSettings, Plugin, PluginConfigError
from reef.core.records_types import AgentRecord, RequestType
from reef.pipeline_context import PipelineContext
from reef.record2sample.sets import RecordSet, RecordSetUnit, SingleRecordSet, SingleSetUnit
from reef.record2sample.staleness import ServedArtifacts, lags_behind


class SampleStatus(Enum):
    """What the stage made of a record set. The status settles all records of the set together."""

    #: The set has one or more samples. Its records wait in one record set unit until the pipeline stores them.
    READY = "ready"
    #: The set has no sample and cannot train. The stage ends its records.
    SKIPPED = "skipped"


@dataclass(frozen=True)
class SampleResult:
    """The samples made from one record set, and what the stage does with the set.

    Values: a ``READY`` result holds one or more samples. A ``SKIPPED`` result holds no sample. The constructor
    checks both.
    Purpose: ``Record2Sample.make_samples`` returns it, and ``Record2Sample.settle`` reads it.
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
        """The set cannot train. The stage ends its records."""
        return cls(SampleStatus.SKIPPED)


class Record2Sample(ABC, Plugin):
    """Turn records into samples for one pipeline, and decide which samples train together.

    The pipeline pushes every record that it reads, in append order (``push``). The stage collects the records that
    train together into record sets and forms each one (``form``). No record is in two record sets of one stage:
    ``form`` refuses a set that holds a record of another set. Before it makes samples, ``form`` rejects a set if
    any inference response is incomplete (``inference_incomplete``): it reports an engine abort, or a stream whose
    delivery did not complete. The whole set then ends, including reports that reference that inference.
    Otherwise ``form`` makes the set's samples (``make_samples``) and settles the result
    (``settle``): a set that has samples waits in a record set unit, and a set without samples ends. When the
    pipeline looks for job data, it first advances the stage to the current time (``advance``): the stage takes in
    background work that ended, such as judgments, and forms the sets that are due. The pipeline takes the units
    that became ready (``take_ready_units``), saves the samples of each one (``unit_samples``) as data items,
    and the stage ends it (``consume``). The stage never calls the pipeline.

    The stage also decides when its stored samples are stale (``is_stale``). No other part of Reef checks how far
    training data lags behind the weights. The samples service asks the stage about each group of its stored samples
    that no job trained, against the artifacts that the scenario served (``TrainingPipeline.mark_stale``), and marks the
    stale ones in the store. Control's automatic jobs leave them out.

    A record that the stage ends, for example a record that can no longer train, appears in
    ``releasable_record_ids``. The samples service stores those IDs, and the trainer's next commit names them, so the
    record never trains after a restart. Once that commit is durable, the samples service releases them. A restart
    before that commit reads the records again, so the stage rebuilds what it decided. The default bookkeeping keeps
    the ended records in ``terminal`` until their release; a stage with its own record lifecycle overrides the
    lifecycle methods.

    Only the samples service feeds a stage (``push``, ``advance``, ``job_ended``), and a stage that holds records or
    runs background work keeps that state there. Control asks only ``uploaded_sample``, which depends on its argument
    and the stage's config.

    ``push``, ``advance``, and ``make_samples`` run on the trainer thread and must not block on network or model
    calls. A stage with slow work runs it on a worker that it owns, forms sets when the work ends, and ``close``
    stops the worker. Raise on a broken training contract. The stage keeps only memory that a replay of the
    retained records rebuilds, as pipeline memory does.

    A stage can also make samples from no record set, such as generated tasks (``made_samples``), and render an
    uploaded trajectory into its training sample (``uploaded_sample``).

    A deployment names a stage class in ``pipeline.record2sample`` and gives its config. The class parses the config
    into its settings dataclass and builds the stage in ``from_config``.
    """

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        """Build the stage of one scenario from its config.

        ``environ`` holds the environment the deployment was read with.
        """
        raise PluginConfigError(f"{cls.__name__} cannot be built from a deployment file: it has no from_config")

    def __init__(self) -> None:
        #: The record set units that became ready since the last ``take_ready_units``, by unit ID.
        self.ready_units: dict[Hashable, RecordSetUnit] = {}
        #: The samples of each set that has samples and has not ended, by set ID.
        self.set_samples: dict[Hashable, tuple[Sample, ...]] = {}
        #: Records that ended and that no live set references, until their release.
        self.terminal: set[str] = set()
        #: The set ID of each record that a formed set holds, until the record's release.
        self.set_of_record: dict[str, Hashable] = {}

    @abstractmethod
    def push(self, record: AgentRecord) -> None:
        """Read one record. The stage must not block on network or model calls."""

    @abstractmethod
    def make_samples(self, record_set: RecordSet) -> SampleResult:
        """Return the samples of ``record_set``, a set that this stage formed, or a skip."""

    def form(self, record_set: RecordSet) -> None:
        """Make the samples of ``record_set`` and settle them. Raises ``ValueError`` if a record of it is in another
        set.

        A set with the ID of an earlier set takes its records again, as a retry does. An incomplete inference
        rejects the whole set before sample creation. Replaying retained records applies the same check.
        """
        record_ids = record_set.record_ids()
        taken = [
            record_id
            for record_id in record_ids
            if self.set_of_record.get(record_id, record_set.set_id) != record_set.set_id
        ]
        if taken:
            raise ValueError(f"records {taken!r} are in another record set and cannot join {record_set.set_id!r}")
        self.set_of_record.update(dict.fromkeys(record_ids, record_set.set_id))
        for record in record_set.records():
            if record.request_type is not RequestType.INFERENCE:
                continue
            response = record.payload.get("response")
            if isinstance(response, Mapping) and inference_incomplete(response):
                self.reject(record_set)
                return
        self.settle(record_set, self.make_samples(record_set))

    def advance(self, now: float) -> None:
        """Take in the background work that ended by ``now``, a monotonic time, and form the sets that are due."""
        return

    def context_changed(self, context: PipelineContext) -> None:
        """The pipeline's context changed: its training mode or admitted report type."""
        return

    def settle(self, record_set: RecordSet, result: SampleResult) -> None:
        """Hold the samples of a READY set and accept it, or end a SKIPPED one."""
        if result.status is SampleStatus.READY:
            self.set_samples[record_set.set_id] = result.samples
            self.accept(record_set)
        else:
            self.reject(record_set)

    def accept(self, record_set: RecordSet) -> None:
        """Make a set that has samples ready, as a record set unit by itself."""
        self.ready_units[record_set.set_id] = SingleSetUnit.of(record_set)

    def take_ready_units(self) -> tuple[RecordSetUnit, ...]:
        """Return the units that became ready since the last call, in the order that they first became ready."""
        ready, self.ready_units = tuple(self.ready_units.values()), {}
        return ready

    def unit_samples(self, unit: RecordSetUnit) -> tuple[tuple[Hashable, tuple[Sample, ...]], ...]:
        """The samples of the ready ``unit``: each set ID of it, in unit order, with the samples of that set."""
        return tuple((set_id, self.set_samples[set_id]) for set_id in unit.set_ids())

    def reject(self, record_set: RecordSet) -> None:
        """End a set that cannot train."""
        self.terminal.update(record_set.record_ids())
        self.end_set(record_set.set_id)

    def end_set(self, set_id: Hashable) -> None:
        """Forget the samples of the set ``set_id``, which ended: the pipeline stored them, or they never train."""
        self.set_samples.pop(set_id, None)

    # ------------------------------------------------------------- staleness

    #: Whether the samples of the stage can go stale. A stage whose samples do not depend on the artifact that
    #: produced them, such as fixed SFT targets, harness traces or generated tasks, sets it to False: the samples
    #: service then never asks ``is_stale`` and never reads the stage's stored samples to judge them.
    goes_stale: bool = True

    def is_stale(self, samples: Sequence[Sample], served: ServedArtifacts) -> bool:
        """Whether ``samples``, the stored samples of one group that no job trained, may no longer train now that the
        scenario serves ``served``.

        The samples service calls it only when ``goes_stale`` is True, and marks every sample of a stale group in the
        store; the stage's state does not change. The default is on-policy: the samples are stale when the data of
        one of them is of an artifact other than the one that the scenario serves now, or one that it serves beside
        it (``lags_behind(samples, served, 0)``). A task never goes stale. A stage whose method accepts a bounded lag
        passes its own bound to ``lags_behind``. The result may depend on the stage's config: the samples service
        judges again when the config changes.
        """
        return lags_behind(samples, served, 0)

    def consume(self, unit: RecordSetUnit) -> None:
        """End ``unit``, whose samples the pipeline stored: its records are releasable."""
        self.terminal |= unit.consumed_ids()
        for set_id in unit.set_ids():
            self.end_set(set_id)

    # ------------------------------------------------------------- lifecycle

    def releasable_record_ids(self) -> frozenset[str]:
        """Return the IDs of records that the stage ended and that no live set references."""
        return frozenset(self.terminal)

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        """Forget released records. Stored records do not change."""
        self.terminal -= agent_record_ids
        for record_id in agent_record_ids:
            self.set_of_record.pop(record_id, None)

    def background_jobs_running(self) -> bool:
        """Return True while background jobs run. A finished job can form sets without a new record."""
        return False

    def next_due(self) -> float | None:
        """Return the monotonic time at which ``advance`` can form a set without a new record, such as the end of a
        trajectory's idle time; None when no set waits for time alone. The default is None."""
        return None

    # ------------------------------------------------------- other samples

    def uploaded_sample(self, trajectory: TrajectorySample) -> TrajectorySample:
        """The training sample of an uploaded ATIF trajectory; the default is the trajectory as it was uploaded.

        The default fits a backend that reads ATIF trajectories as they are. A stage whose backend trains
        token-level data, and whose sample-to-job stage accepts uploaded trajectories, overrides it to render the
        trajectory as its backend trains it, such as ``SFTSamples``. The pipeline calls it when it reads the upload,
        and skips a trajectory that it refuses. It must be pure: the same trajectory gives the same sample, and the
        stage's state does not change. A trajectory that cannot become a sample raises ``ValueError`` with the
        reason.
        """
        return trajectory

    def made_samples(self) -> tuple[tuple[str, Sample], ...]:
        """Return the samples that the stage made from no record set since the last call; the default makes none.

        Each pair is ``(name, sample)``. ``name`` is non-empty, and a name names the same sample every time it is
        returned. The pipeline stores each sample as ``<name>``, or ``<component>:<name>`` for a trainer of a
        scenario of several, with no group and no consumed record. A sample that the store holds already changes
        nothing, so a stage may return a sample again, for example after a restart.
        """
        return ()

    def job_ended(self, outcome: JobOutcome) -> None:
        """A training job ended and the commit of its end is durable; the default does nothing.

        The samples service calls it, also when control runs in another process. A stage that made tasks reads their
        results from ``outcome``. A trainer that starts with no running job calls it again for the last job that
        ended, before it reads a record, so a stage that starts in that process takes it as the end of the last job
        that may have trained its samples.
        """
        return

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe state that the pipeline adds to its status."""
        return {}

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Return numeric state that the pipeline adds to its metrics."""
        return {}

    def close(self) -> None:
        """Release resources that the stage owns. Calling it more than one time is safe."""
        return


class NoSamples(Record2Sample):
    """Make no sample: the default of a pipeline that trains nothing. No config.

    The stage reads each record and keeps nothing. The pipeline then makes no job data, and the records stay stored
    for audit.
    """

    goes_stale = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls()

    def push(self, record: AgentRecord) -> None:
        return

    def make_samples(self, record_set: RecordSet) -> SampleResult:
        raise TypeError(f"{type(self).__name__} forms no record set")


class InferenceRecordSamples(Record2Sample, ABC):
    """Form each inference record as one set, in arrival order, and end every other record untouched.

    A subclass implements ``make_samples`` for a ``SingleRecordSet`` of one inference record.
    """

    def push(self, record: AgentRecord) -> None:
        # A record that is not an inference ends at once.
        if record.request_type is not RequestType.INFERENCE:
            self.terminal.add(record.agent_record_id)
            return
        self.form(SingleRecordSet(set_id=record.agent_record_id, record=record))
