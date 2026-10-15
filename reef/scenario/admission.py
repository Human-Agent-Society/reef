"""Service-local admission and durable scheduler control operations."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace
from threading import Lock
from typing import Literal, cast

from reef.core.errors import ReefError, ScenarioBusy
from reef.core.records_types import AgentRecord
from reef.storage.admissions import PendingTrainingRequest, ScenarioAdmissionStore
from reef.storage.records import RecordConflict

ScenarioControlOperation = Literal["start", "reclaim", "cleanup"]


class ScenarioAdmissionError(ReefError):
    """A scenario has no matching, usable scheduler assignment."""


@dataclass(frozen=True)
class ServiceScenarioRecord:
    """One ready scenario admitted to business requests in this service process."""

    scenario_id: int
    tenant_id: int
    scenario_name: str
    generation: int = 1
    idle_since: float | None = None
    training_mode: str | None = None


@dataclass(frozen=True)
class ScenarioControlRecord:
    """Persist a control operation independently of the process's admitted set."""

    scenario_id: int
    tenant_id: int
    scenario_name: str
    operation: ScenarioControlOperation = "start"
    complete: bool = False
    generation: int = 1
    error: str | None = None
    startup_failed: bool = False
    training_mode: str | None = None
    # Reject stale commands and bind legacy records to the scheduler's generation once.
    generation_bound: bool = True
    accepts_training: bool = False
    pending_training: PendingTrainingRequest | None = None

    @classmethod
    def from_value(cls, value: object, scenario_name: str) -> ScenarioControlRecord:
        """Read current operations and translate the prior lifecycle format at this boundary."""
        if not isinstance(value, dict):
            raise ScenarioAdmissionError("stored scenario assignment must be an object")
        scenario_id = value.get("scenario_id")
        tenant_id = value.get("tenant_id")
        generation = value.get("generation", 1)
        generation_bound = "generation" in value and value.get("generation_bound", True)
        error = value.get("error")
        failed = value.get("startup_failed", False)
        mode = value.get("training_mode")
        # Older manual/hybrid records predate the explicit backend capability.
        accepts_training = value.get("accepts_training", mode in ("manual", "hybrid"))
        if "operation" in value:
            operation = value["operation"]
            complete = value.get("complete")
        else:
            # Legacy state strings are not retained in new operation records.
            state = value.get("state")
            cleanup_complete = value.get("cleanup_complete")
            reclaim_complete = value.get("reclaim_complete", False)
            idle_since = value.get("idle_since")
            if (
                state
                not in ("starting", "admitted", "reclaim_pending", "reclaimed", "cleanup_pending", "deleted", "failed")
                or not isinstance(cleanup_complete, bool)
                or not isinstance(reclaim_complete, bool)
                or not isinstance(failed, bool)
                or (
                    idle_since is not None
                    and (
                        isinstance(idle_since, bool)
                        or not isinstance(idle_since, int | float)
                        or not math.isfinite(idle_since)
                    )
                )
            ):
                raise ScenarioAdmissionError("stored scenario assignment has an invalid legacy lifecycle")
            if state in ("starting", "admitted"):
                operation = "start"
                complete = state == "admitted"
            elif state in ("reclaim_pending", "reclaimed"):
                operation = "reclaim"
                complete = reclaim_complete
            else:
                operation = "cleanup"
                complete = cleanup_complete
                failed = failed or state == "failed"
        if (
            any(
                isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= 2**63 - 1
                for number in (scenario_id, tenant_id, generation)
            )
            or value.get("scenario_name") != scenario_name
            or operation not in ("start", "reclaim", "cleanup")
            or not isinstance(complete, bool)
            or not isinstance(failed, bool)
            or not isinstance(generation_bound, bool)
            or (error is not None and not isinstance(error, str))
            or mode not in (None, "auto", "manual", "hybrid")
            or not isinstance(accepts_training, bool)
        ):
            raise ScenarioAdmissionError(
                "stored scenario assignment is invalid; legacy tenant IDs require an explicit mapping"
            )
        return cls(
            scenario_id=cast(int, scenario_id),
            tenant_id=cast(int, tenant_id),
            scenario_name=scenario_name,
            operation=cast(ScenarioControlOperation, operation),
            complete=complete,
            generation=cast(int, generation),
            error=error,
            startup_failed=failed,
            training_mode=mode,
            generation_bound=generation_bound,
            accepts_training=accepts_training,
            pending_training=PendingTrainingRequest.from_value(value.get("pending_training"), scenario_name),
        )


@dataclass(frozen=True)
class ScenarioControlResult:
    """Project local readiness and operation results onto the scheduler's existing wire contract."""

    scenario_id: int
    tenant_id: int
    scenario_name: str
    state: Literal["starting", "admitted", "reclaim_pending", "reclaimed", "cleanup_pending", "deleted", "failed"]
    cleanup_complete: bool = False
    error: str | None = None
    startup_failed: bool = False
    generation: int = 1
    idle_since: float | None = None
    reclaim_complete: bool = False
    training_mode: str | None = None
    generation_bound: bool = True
    wake_request_id: str | None = None
    wake_generation: int | None = None


class ScenarioAdmissions:
    """Admit ready instances and retain control operations for retries and recovery."""

    def __init__(self, storage: ScenarioAdmissionStore) -> None:
        self.lock = Lock()
        self.admitted_by_name: dict[str, ServiceScenarioRecord] = {}
        self.operations_by_name: dict[str, ScenarioControlRecord] = {}
        self.stopping = False
        self.storage = storage
        self.request_counts: dict[tuple[str, int], int] = {}
        for name in storage.scenario_names():
            value = storage.read(name)
            if isinstance(value, dict) and value.get("pending_training") is not None:
                self.read_operation(name)

    def read_operation(self, scenario_name: str) -> ScenarioControlRecord | None:
        """Read under the table lock; reading persisted operations never admits an instance."""
        current = self.operations_by_name.get(scenario_name)
        if current is not None:
            return current
        value = self.storage.read(scenario_name)
        if value is None:
            return None
        current = ScenarioControlRecord.from_value(value, scenario_name)
        if isinstance(value, dict) and "operation" not in value:
            self.save_operation(current)
        elif (
            not current.complete
            or current.operation == "start"
            or (current.pending_training is not None and current.operation != "cleanup")
        ):
            self.operations_by_name[scenario_name] = current
        return current

    def save_operation(self, current: ScenarioControlRecord) -> None:
        """Persist before exposing a result; caller holds the table lock."""
        self.storage.write(current.scenario_name, asdict(current))
        if (
            current.complete
            and current.operation != "start"
            and self.storage.durable
            and (current.pending_training is None or current.operation == "cleanup")
        ):
            self.operations_by_name.pop(current.scenario_name, None)
        else:
            self.operations_by_name[current.scenario_name] = current

    def result(self, operation: ScenarioControlRecord) -> ScenarioControlResult:
        """Build a control response under the lock without maintaining a scenario state field."""
        admitted = self.admitted_by_name.get(operation.scenario_name)
        state: Literal["starting", "admitted", "reclaim_pending", "reclaimed", "cleanup_pending", "deleted", "failed"]
        if admitted is not None:
            state = "admitted"
        elif operation.operation == "start":
            state = "starting"
        elif operation.operation == "reclaim":
            state = "reclaimed" if operation.complete else "reclaim_pending"
        elif operation.complete:
            state = "failed" if operation.startup_failed else "deleted"
        else:
            state = "cleanup_pending"
        return ScenarioControlResult(
            scenario_id=operation.scenario_id,
            tenant_id=operation.tenant_id,
            scenario_name=operation.scenario_name,
            state=state,
            cleanup_complete=operation.operation == "cleanup" and operation.complete,
            error=operation.error,
            startup_failed=operation.startup_failed,
            generation=operation.generation,
            idle_since=None if admitted is None else admitted.idle_since,
            reclaim_complete=operation.operation == "reclaim" and operation.complete,
            training_mode=operation.training_mode,
            generation_bound=operation.generation_bound,
            wake_request_id=(
                None if operation.pending_training is None else operation.pending_training.item.agent_record_id
            ),
            wake_generation=None if operation.pending_training is None else operation.pending_training.generation,
        )

    def get(self, scenario_name: str) -> ServiceScenarioRecord | None:
        """Return only a currently admitted instance's metadata."""
        with self.lock:
            return self.admitted_by_name.get(scenario_name)

    def get_operation(self, scenario_name: str) -> ScenarioControlRecord | None:
        with self.lock:
            return self.read_operation(scenario_name)

    def status(self, scenario_name: str) -> ScenarioControlResult | None:
        with self.lock:
            operation = self.read_operation(scenario_name)
            return None if operation is None else self.result(operation)

    def require_result(self, scenario_name: str) -> ScenarioControlResult:
        """Return the result of a known control operation."""
        with self.lock:
            operation = self.read_operation(scenario_name)
            if operation is None:
                raise ScenarioAdmissionError("scenario has no recorded control operation")
            return self.result(operation)

    def begin(
        self, scenario_id: int, tenant_id: int, scenario_name: str, generation: int = 1
    ) -> ScenarioControlRecord:
        """Authorize construction without adding the scenario to the admitted set."""
        if any(
            isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= 2**63 - 1
            for number in (scenario_id, tenant_id, generation)
        ):
            raise ValueError("scenario_id, tenant_id and generation must be positive integers")
        if not scenario_name or "/" in scenario_name or scenario_name in (".", ".."):
            raise ValueError("scenario_name must be a non-empty name, not a path")
        with self.lock:
            if self.stopping:
                raise ScenarioAdmissionError("the service is stopping")
            current = self.read_operation(scenario_name)
            if current is not None:
                if current.scenario_id == scenario_id:
                    if current.tenant_id != tenant_id:
                        raise ScenarioAdmissionError("scenario tenant does not match its assignment")
                    if generation < current.generation:
                        raise ScenarioAdmissionError("stale scenario operation generation")
                    if current.operation == "cleanup" and current.complete:
                        raise ScenarioAdmissionError("scenario lifecycle already completed")
                    if current.operation == "reclaim" and current.complete and generation > current.generation:
                        current = replace(
                            current, operation="start", complete=False, generation=generation, error=None
                        )
                    elif not current.generation_bound and current.operation == "start":
                        current = replace(current, generation=generation, generation_bound=True)
                    elif generation != current.generation:
                        raise ScenarioAdmissionError("a new execution round requires completed reclamation")
                    if current.operation != "start" or scenario_name in self.admitted_by_name:
                        return current
                    current = replace(current, complete=False)
                    self.save_operation(current)
                    return current
                if scenario_id < current.scenario_id or current.operation != "cleanup" or not current.complete:
                    raise ScenarioAdmissionError("scenario name belongs to another lifecycle")
            current = ScenarioControlRecord(scenario_id, tenant_id, scenario_name, generation=generation)
            self.save_operation(current)
            return current

    def require(self, scenario_name: str, *, starting_id: int | None = None) -> ServiceScenarioRecord:
        """Require local readiness, or a matching construction authorization for control callers."""
        with self.lock:
            admitted = self.admitted_by_name.get(scenario_name)
            if admitted is not None:
                return admitted
            operation = self.read_operation(scenario_name)
            if operation is None:
                raise ScenarioAdmissionError(f"scenario {scenario_name!r} requires scheduler admission")
            if operation.operation == "start" and not operation.complete and operation.scenario_id == starting_id:
                return ServiceScenarioRecord(
                    operation.scenario_id,
                    operation.tenant_id,
                    scenario_name,
                    generation=operation.generation,
                    training_mode=operation.training_mode,
                )
            raise ScenarioAdmissionError(f"scenario {scenario_name!r} is {self.result(operation).state}")

    def admit(
        self, scenario_name: str, scenario_id: int, *, training_mode: str | None = None, accepts_training: bool = False
    ) -> ScenarioControlResult:
        """Publish readiness only after construction succeeds and the operation is committed."""
        with self.lock:
            operation = self.read_operation(scenario_name)
            if operation is None or operation.scenario_id != scenario_id or operation.operation != "start":
                raise ScenarioAdmissionError("scenario lifecycle does not match its start operation")
            operation = replace(
                operation, complete=True, training_mode=training_mode, accepts_training=accepts_training, error=None
            )
            self.save_operation(operation)
            self.admitted_by_name[scenario_name] = ServiceScenarioRecord(
                scenario_id, operation.tenant_id, scenario_name, operation.generation, training_mode=training_mode
            )
            return self.result(operation)

    def pending_wakes(self) -> tuple[ScenarioControlRecord, ...]:
        """Return only retained instructions; sleeping scenarios without work need no polling."""
        with self.lock:
            return tuple(
                record
                for record in self.operations_by_name.values()
                if record.pending_training is not None and record.operation != "cleanup"
            )

    def stage_training(self, item: AgentRecord, tenant_id: int) -> AgentRecord:
        """Persist the cold instruction in the same atomic file as its lifecycle identity."""
        with self.lock:
            current = self.read_operation(item.scenario)
            if current is None or current.tenant_id != tenant_id or self.stopping:
                raise ScenarioAdmissionError("scenario has no matching scheduler assignment")
            pending = current.pending_training
            if pending is not None:
                if pending.item.agent_record_id == item.agent_record_id:
                    if pending.item.payload != item.payload:
                        raise RecordConflict("record ID already names a different instruction")
                    return pending.item
                raise ScenarioBusy("scenario already has an instruction awaiting admission")
            if current.operation != "reclaim" or not current.complete:
                raise ScenarioAdmissionError("scenario is awaiting admission or cleanup")
            self.save_operation(replace(current, pending_training=PendingTrainingRequest(item, current.generation)))
            return item

    def finish_training_delivery(self, scenario_name: str, record_id: str) -> None:
        with self.lock:
            current = self.read_operation(scenario_name)
            if current is not None and current.pending_training is not None:
                if current.pending_training.item.agent_record_id != record_id:
                    raise ScenarioAdmissionError("pending instruction identity changed")
                self.save_operation(replace(current, pending_training=None))

    def advance_control(
        self, scenario_name: str, scenario_id: int, tenant_id: int | None, generation: int
    ) -> ScenarioControlRecord:
        """Reject old messages before they can alter a newer execution round."""
        with self.lock:
            current = self.read_operation(scenario_name)
            if (
                current is None
                or current.scenario_id != scenario_id
                or (tenant_id is not None and current.tenant_id != tenant_id)
            ):
                raise ScenarioAdmissionError("scenario lifecycle does not match its assignment")
            if (
                isinstance(generation, bool)
                or not isinstance(generation, int)
                or not current.generation <= generation <= 2**63 - 1
            ):
                raise ScenarioAdmissionError("stale scenario operation generation")
            if generation > current.generation or not current.generation_bound:
                current = replace(current, generation=generation, generation_bound=True)
                self.save_operation(current)
                admitted = self.admitted_by_name.get(scenario_name)
                if admitted is not None:
                    self.admitted_by_name[scenario_name] = replace(admitted, generation=generation)
            return current

    def fail_start(self, scenario_name: str, scenario_id: int, error: str) -> ScenarioControlResult:
        with self.lock:
            current = self.read_operation(scenario_name)
            if current is None or current.scenario_id != scenario_id:
                raise ScenarioAdmissionError("scenario lifecycle does not match its assignment")
            current = replace(current, operation="cleanup", complete=False, error=error, startup_failed=True)
            self.save_operation(current)
            self.admitted_by_name.pop(scenario_name, None)
            return self.result(current)

    def finish_cleanup(
        self, scenario_name: str, scenario_id: int, *, error: str | None = None
    ) -> ScenarioControlResult:
        with self.lock:
            current = self.read_operation(scenario_name)
            if current is None or current.scenario_id != scenario_id or current.operation == "start":
                raise ScenarioAdmissionError("scenario lifecycle does not match its release operation")
            current = replace(current, complete=error is None, error=error)
            self.save_operation(current)
            return self.result(current)

    def set_training_mode(self, scenario_name: str, training_mode: str) -> None:
        with self.lock:
            current = self.admitted_by_name.get(scenario_name)
            operation = self.read_operation(scenario_name)
            if current is None or operation is None:
                raise ScenarioAdmissionError("scenario requires scheduler admission")
            self.save_operation(replace(operation, training_mode=training_mode))
            self.admitted_by_name[scenario_name] = replace(current, training_mode=training_mode, idle_since=None)

    def observe_idle(self, scenario_name: str, idle: bool) -> ScenarioControlResult | None:
        with self.lock:
            current = self.admitted_by_name.get(scenario_name)
            operation = self.read_operation(scenario_name)
            if operation is None:
                return None
            if current is not None:
                if self.request_counts.get((scenario_name, current.scenario_id), 0):
                    idle = False
                timestamp = time.time() if idle and current.idle_since is None else current.idle_since
                self.admitted_by_name[scenario_name] = replace(current, idle_since=timestamp if idle else None)
            return self.result(operation)

    def touch(self, scenario_name: str) -> None:
        with self.lock:
            current = self.admitted_by_name.get(scenario_name)
            if current is not None:
                self.admitted_by_name[scenario_name] = replace(current, idle_since=None)

    def acquire_request(self, scenario_name: str, tenant_id: int) -> int:
        with self.lock:
            current = self.admitted_by_name.get(scenario_name)
            if current is None or current.tenant_id != tenant_id:
                raise ScenarioAdmissionError("request tenant has no admitted scenario with this name")
            key = (scenario_name, current.scenario_id)
            self.request_counts[key] = self.request_counts.get(key, 0) + 1
            return current.scenario_id

    def release_request(self, scenario_name: str, scenario_id: int) -> None:
        with self.lock:
            key = (scenario_name, scenario_id)
            count = self.request_counts.get(key, 0)
            if count < 1:
                raise ScenarioAdmissionError("request has no matching lifecycle lease")
            if count == 1:
                self.request_counts.pop(key)
            else:
                self.request_counts[key] = count - 1

    def begin_cleanup(
        self,
        scenario_name: str,
        scenario_id: int,
        inactive_before: float | None = None,
        *,
        reclaim: bool = False,
        training_mode: str | None = None,
    ) -> bool:
        """Commit release intent and remove admission atomically with request and idle checks."""
        with self.lock:
            operation = self.read_operation(scenario_name)
            if operation is None or operation.scenario_id != scenario_id:
                raise ScenarioAdmissionError("scenario lifecycle does not match its assignment")
            current = self.admitted_by_name.get(scenario_name)
            if self.request_counts.get((scenario_name, scenario_id), 0):
                if reclaim:
                    if current is not None:
                        self.admitted_by_name[scenario_name] = replace(current, idle_since=None)
                    return False
                raise ScenarioBusy(f"cannot delete scenario {scenario_name!r} while a request is in flight")
            if (
                reclaim
                and current is not None
                and (current.idle_since is None or inactive_before is None or current.idle_since > inactive_before)
            ):
                return False
            operation = replace(
                operation,
                operation="reclaim" if reclaim else "cleanup",
                complete=False,
                training_mode=training_mode or operation.training_mode,
            )
            self.save_operation(operation)
            self.admitted_by_name.pop(scenario_name, None)
            return True

    def stop(self) -> None:
        with self.lock:
            self.stopping = True
