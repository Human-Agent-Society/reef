"""The training status that ``GET /reef/status`` reports.

Reading the status is also the only health check that Reef runs. It warns
about a ready batch that the training thread has not drained. It also finds a
training thread that ended without recording an error.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from reef.dispatcher.errors import TRAINING_THREAD_SOURCE, TrainingErrors, error_text
from reef.dispatcher.training_loop import DispatchedTrainingLoop, in_flight_job_marker
from reef.recipe.base import Recipe
from reef.scenario.registry import ScenarioRegistry

logger = logging.getLogger(__name__)


class TrainingStatus:
    """Build the training status from the registry, the training thread and the recorded errors."""

    def __init__(
        self, recipe: Recipe, registry: ScenarioRegistry, errors: TrainingErrors, training_loop: DispatchedTrainingLoop
    ) -> None:
        self.recipe = recipe
        self.registry = registry
        self.errors = errors
        self.training_loop = training_loop

    def build(self) -> Mapping[str, Any]:
        """Assemble the training status block, checking training health as it goes.

        Reading the status records state so a repeated read does not repeat
        a log line, so callers must expect a status read to have effects.
        """
        preload_errors = self.registry.preload_errors
        storage_status, last_drain = self.training_loop.drain_state()
        # One scenario's status failing stops the sweep: the recorded error goes
        # out with whatever was collected before it.
        scenarios: dict[str, dict[str, Any]] = {}
        failed = False
        for scenario_name in self.registry.training_status_scenario_names:
            try:
                block = self.scenario_status(scenario_name, storage_status, last_drain)
            except Exception as exc:
                failed = True
                if self.errors.set_status_build_error(f"{scenario_name}: {error_text(exc)}"):
                    logger.exception("failed to build training status for scenario %r", scenario_name)
                break
            if block is not None:
                scenarios[scenario_name] = block
        if not failed:
            self.errors.set_status_build_error(None)
        errors = self.training_errors()
        return {
            "error": "\n".join(errors) or None,
            "last_drain_at": last_drain,
            "loaded_scenarios": len(self.registry.loaded_scenarios()),
            "preload_errors": preload_errors,
            "scenarios": scenarios,
            "serving": self.serving_status(),
            "training_job": self.training_job_view(),
        }

    def training_job_view(self) -> dict[str, Any] | None:
        """The weight job the training runtime holds out, for an operator; ``None`` when none is out.

        ``owner`` names the scenario whose job it is, the one a delete refuses.
        """
        try:
            marker = in_flight_job_marker(self.recipe.training_runtime)
        except Exception as exc:
            return {"error": error_text(exc)}
        if marker is None:
            return None
        return {
            "status": marker.get("status"),
            "training_job_id": marker.get("training_job_id", marker.get("job_id")),
            "owner": marker.get("scenario"),
        }

    def scenario_status(
        self,
        scenario_name: str,
        storage_status: Mapping[str, Any] | None,
        last_drain: float | None,
    ) -> dict[str, Any] | None:
        current = self.registry.get_optional(scenario_name)
        if current is None:
            return None
        runtime = current.runtime
        stepping_trainer = current.trainer
        batch_ready = stepping_trainer.batch_ready()
        if batch_ready:
            self.training_loop.warn_if_undrained(scenario_name, last_drain)
        # The stepping trainer's status is read once and reported again in its component block.
        stepping_pipeline_status = stepping_trainer.pipeline_status()
        processor = dict(stepping_pipeline_status)
        if "buffered_requests" in processor:
            # Include instructions still unread in storage alongside the buffered ones.
            processor["pending_instructions"] = stepping_trainer.pending_instructions()
        block: dict[str, Any] = {
            **current.commit_status,
            # A version is current only after Reef commits its head
            # and reopens admission. The backend may report it
            # earlier while the update is still being published.
            "current_runtime_load_id": (runtime.current_runtime_load_id() if runtime is not None else None),
            "checkpoint_storage": storage_status,
            "batch_ready": batch_ready,
            "training_mode": stepping_trainer.pipeline.training_mode,
            "processor": processor,
            "inference_admission": runtime.inference_admission_status if runtime is not None else None,
        }
        if (
            runtime is not None
            and current.training_runtime is not None
            and current.training_runtime.concurrent_training_scenarios
        ):
            block["adapter_runtime_load_id"] = runtime.serving_adapter_runtime_load_id(scenario_name)
        if len(current.trainers) > 1:
            # Each component's trainer commits on its own; report each one beside the scenario-wide step.
            components: dict[str, Any] = {}
            for bound in current.trainers:
                last = current.last_commit_for(bound.component)
                components[str(bound.component)] = {
                    "batch_ready": bound.trainer.batch_ready(),
                    "training_mode": bound.trainer.pipeline.training_mode,
                    "processor": dict(
                        stepping_pipeline_status
                        if bound.trainer is stepping_trainer
                        else bound.trainer.pipeline_status()
                    ),
                    "last_committed_step": (
                        None
                        if last is None
                        else {
                            "step": last.step,
                            "recorded_at": last.recorded_at,
                            "metrics": None if last.metrics is None else dict(last.metrics),
                        }
                    ),
                }
            block["components"] = components
        return block

    def training_errors(self) -> list[str]:
        """Every recorded training error, noticing a training thread that died silently."""
        training_errors, status_build_error = self.errors.snapshot()
        training_scenario = self.registry.training_scenario_name or "<unbound>"
        thread_error_key = (training_scenario, TRAINING_THREAD_SOURCE)
        if thread_error_key not in training_errors and self.training_loop.stopped_unexpectedly():
            error = "RuntimeError: training thread stopped unexpectedly"
            logger.error("%s: %s", training_scenario, error)
            self.errors.record(training_scenario, error)
            training_errors[thread_error_key] = error
        errors = [f"{scenario}: {training_errors[(scenario, source)]}" for scenario, source in sorted(training_errors)]
        if status_build_error is not None:
            errors.append(status_build_error)
        return errors

    def serving_status(self) -> dict[str, Any]:
        """Runtime-wide serving state for the deployment's recipe."""
        try:
            status = self.recipe.serving_status()
        except Exception as exc:
            status = {"error": error_text(exc)}
        return {} if status is None else {self.recipe.name: status}
