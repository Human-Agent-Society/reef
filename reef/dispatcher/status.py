"""The training status that ``GET /reef/status`` reports: each training scenario's commits, job data and job."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from reef.dispatcher.errors import TrainingErrors, error_text
from reef.recipe.base import Recipe
from reef.scenario.registry import ScenarioRegistry
from reef.train.jobs import JobLoopState

logger = logging.getLogger(__name__)


class TrainingStatus:
    """Build the training status from the registry and the recorded errors."""

    def __init__(self, recipe: Recipe, registry: ScenarioRegistry, errors: TrainingErrors) -> None:
        self.recipe = recipe
        self.registry = registry
        self.errors = errors

    def build(self) -> Mapping[str, Any]:
        """Assemble the training status block, checking training health as it goes.

        Reading the status records state so a repeated read does not repeat
        a log line, so callers must expect a status read to have effects.
        """
        preload_errors = self.registry.preload_errors
        # One scenario's status failing stops the sweep: the recorded error goes
        # out with whatever was collected before it.
        scenarios: dict[str, dict[str, Any]] = {}
        failed = False
        for scenario_name in self.registry.training_status_scenario_names:
            try:
                block = self.scenario_status(scenario_name)
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
            "loaded_scenarios": len(self.registry.loaded_scenarios()),
            "preload_errors": preload_errors,
            "scenarios": scenarios,
            "serving": self.serving_status(),
        }

    def scenario_status(self, scenario_name: str) -> dict[str, Any] | None:
        current = self.registry.get_optional(scenario_name)
        if current is None:
            return None
        runtime = current.runtime
        stepping_trainer = current.trainer
        job_data_ready = stepping_trainer.job_data_ready()
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
            "job_data_ready": job_data_ready,
            "training_mode": stepping_trainer.pipeline.training_mode,
            "processor": processor,
            "inference_admission": runtime.inference_admission_status if runtime is not None else None,
        }
        if stepping_trainer.job_loop is not None:
            # The job the trainer waits on, and where its next job starts.
            state = JobLoopState.from_dict(stepping_trainer.state)
            job = state.to_dict()["job"]
            block["training_job"] = (
                None if job is None else {**job, "backend": stepping_trainer.job_loop.component.name}
            )
            block["finished_jobs"] = state.finished_jobs
        if runtime is not None and runtime.per_scenario_adapters:
            block["adapter_runtime_load_id"] = runtime.serving_adapter_runtime_load_id(scenario_name)
        if len(current.trainers) > 1:
            # Each component's trainer commits on its own; report each one beside the scenario-wide step.
            components: dict[str, Any] = {}
            for component_trainer in current.trainers:
                last = current.last_commit_for(component_trainer.component)
                components[str(component_trainer.component)] = {
                    "job_data_ready": component_trainer.trainer.job_data_ready(),
                    "training_mode": component_trainer.trainer.pipeline.training_mode,
                    "processor": dict(
                        stepping_pipeline_status
                        if component_trainer.trainer is stepping_trainer
                        else component_trainer.trainer.pipeline_status()
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
        """Every recorded training error."""
        training_errors, status_build_error = self.errors.snapshot()
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
