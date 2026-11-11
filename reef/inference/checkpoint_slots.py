"""Serve several full-weight checkpoints of one scenario at once.

A ``FullWeightServingRuntime`` holds one checkpoint in its engines. ``CheckpointSlotsRuntime`` composes several
of them, each with its own engines, so that a scenario serves its active checkpoint and other checkpoints beside
it, for example for a traffic split. This module starts no engine and imports no engine library.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Collection, Mapping, Sequence

from reef.artifact.artifact import Artifact
from reef.core.errors import ReefError
from reef.core.served_models import ServedArtifact
from reef.core.values import JsonValue
from reef.inference.interfaces import InferenceHandler, InferenceRuntime, full_weight_name
from reef.inference.serving import FullWeightServingRuntime

logger = logging.getLogger(__name__)


class CheckpointSlotsRuntime(InferenceRuntime):
    """Serve several full-weight checkpoints of one scenario at once, each in the engines of its own slot.

    Values: ``slots`` holds two or more ``FullWeightServingRuntime`` values that bind no checkpoint at
    construction, with distinct HTTP(S) base URLs, one non-empty model path and one request timeout. Each slot
    keeps its own engines, health watcher and recovery, and holds at most one checkpoint. ``bindings`` maps the
    artifact ID of each bound checkpoint to the route of its last verified load. The route's URL is the base URL of
    the checkpoint's slot, and no two bindings share a slot. ``active_artifact_id`` is the key in ``bindings`` of
    the checkpoint that the last successful ``activate_checkpoint`` selected. It is None before that call, and
    again after ``stop_serving`` released that checkpoint. ``opened`` is whether every slot's engines opened.
    ``stopped`` is whether ``shutdown`` ran; from then on the runtime opens no slot and binds no checkpoint.

    A checkpoint keeps its slot until ``stop_serving`` releases it, also while the slot recovers. The runtime
    never replaces a bound checkpoint with another one: while every slot is bound, a further artifact is refused.
    A released slot keeps its engines, paused, for a later checkpoint. A switch between two bound checkpoints
    loads nothing. ``open``, each load, each release and ``shutdown`` hold one lock, so they run one at a time.

    The runtime serves one training scenario, and the serving worker runs one work at a time on it
    (``serving_slots`` is 1). It answers only from the checkpoints that it has loaded, so Reef evaluates no
    candidate on it (``serves_any_artifact`` is False). It takes over no engine of an earlier owner (``adopt``).

    Purpose: a runtime factory builds one over the full-weight runtimes of several inference components. The
    inference worker opens and drives it. Other processes build it without opening it and read ``handler_for``.
    Building it touches no engine.
    """

    def __init__(self, slots: Sequence[FullWeightServingRuntime]) -> None:
        configured = tuple(slots)
        if not all(isinstance(slot, FullWeightServingRuntime) for slot in configured):
            raise TypeError("each checkpoint slot is a FullWeightServingRuntime")
        if len(configured) < 2:
            raise ValueError("checkpoint slots need at least two runtimes; one FullWeightServingRuntime serves alone")
        if len({slot.base_url for slot in configured}) != len(configured):
            raise ValueError("checkpoint slots need distinct base URLs, because each slot has its own engines")
        if len({(slot.model_path, slot.inference_timeout_s) for slot in configured}) != 1:
            raise ValueError("checkpoint slots serve one model with one request timeout")
        for slot in configured:
            if slot.served_artifact_id is not None:
                raise ValueError(f"the checkpoint slot at {slot.base_url!r} already binds a checkpoint")
            try:
                # Each verified load needs a route. Without one, nothing could reach or release the checkpoint.
                ServedArtifact("checkpoint", slot.base_url, full_weight_name(slot.model_path, slot.base_version))
            except ValueError as error:
                raise ValueError(f"the checkpoint slot at {slot.base_url!r} can have no route: {error}") from error
        super().__init__(base_url=configured[0].base_url, inference_timeout_s=configured[0].inference_timeout_s)
        self.slots = configured
        self.slot_by_url = {slot.base_url: slot for slot in configured}
        self.bindings: dict[str, ServedArtifact] = {}
        self.active_artifact_id: str | None = None
        self.opened = False
        self.stopped = False
        self.lock = threading.RLock()

    @property
    def inference_handler(self) -> InferenceHandler:
        """The first slot's handler. Each slot's URL has its own handler (``handler_for``)."""
        return self.slots[0].inference_handler

    @property
    def model_path(self) -> str:
        return self.slots[0].model_path

    @property
    def served_artifact_capacity(self) -> int:
        """The number of slots: each one holds one checkpoint."""
        return len(self.slots)

    def handler_for(self, url: str) -> InferenceHandler | None:
        """The handler of the slot at ``url``, or None for any other URL. The runtime need not be open."""
        slot = self.slot_by_url.get(url)
        return None if slot is None else slot.inference_handler

    def set_owner_sequence(self, sequence: int) -> None:
        """Give every slot's engines the serving-owner sequence, before the slots open."""
        for slot in self.slots:
            slot.set_owner_sequence(sequence)

    def apply_serving_config(self, scenario: str, override: Mapping[str, JsonValue]) -> None:
        """Apply ``override`` to every slot, so that each checkpoint of ``scenario`` serves with the same fields."""
        for slot in self.slots:
            slot.apply_serving_config(scenario, override)

    def open(self) -> None:
        """Open every slot's engines, one slot after another. A second call opens only the slots that are not open.

        A slot that cannot open raises. The slots before it stay open, paused and unbound, and no checkpoint binds
        until a later call has opened every slot. The caller shuts the runtime down when it stops trying
        (``shutdown``), also after a failed call, because the open slots run their health watchers. ``ReefError``
        after ``shutdown``.
        """
        with self.lock:
            if self.stopped:
                raise ReefError("the checkpoint slots shut down, so they open no engines any more")
            for slot in self.slots:
                slot.open()
            self.opened = True

    def bind(self, artifact: Artifact) -> str:
        """Make a slot serve ``artifact`` and return the runtime load ID of the load that serves it.

        The slot is the one that binds the artifact already, which loads nothing while its binding holds, or else
        the first slot without a binding. ``ReefError`` when the runtime is not open or shut down, or when every
        slot binds another artifact; then no binding changes. A load that fails raises. It leaves a new slot paused
        and without a binding, so the slot can load this artifact again or another one. It leaves a slot that held
        the artifact with its binding, which the slot restores.
        """
        artifact_id = artifact.artifact_id
        with self.lock:
            if self.stopped:
                raise ReefError("the checkpoint slots shut down, so they load no checkpoint any more")
            if not self.opened:
                raise ReefError(
                    "the checkpoint slots are not open: only the inference worker, which opens them, changes the "
                    "engines"
                )
            binding = self.bindings.get(artifact_id)
            if binding is not None:
                slot = self.slot_by_url[binding.url]
            else:
                bound_urls = {bound.url for bound in self.bindings.values()}
                free = [slot for slot in self.slots if slot.base_url not in bound_urls]
                if not free:
                    raise ReefError(
                        f"each of the {len(self.slots)} checkpoint slots serves another artifact "
                        f"({', '.join(sorted(self.bindings))}), so artifact {artifact_id!r} has no slot; a slot "
                        "becomes free when no route names its artifact any more and the inference worker stops "
                        "serving it"
                    )
                slot = free[0]
            version = slot.activate_checkpoint(artifact)
            self.bindings[artifact_id] = ServedArtifact(
                artifact_id, slot.base_url, full_weight_name(slot.model_path, version)
            )
            return version

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        """Serve ``artifact`` (``bind``) and make it the active checkpoint; return its runtime load ID.

        The checkpoints that serve beside it keep their slots. A raise changes no other binding and leaves the
        active checkpoint as it was. The runtime serves one scenario, so it ignores ``scenario``.
        """
        with self.lock:
            version = self.bind(artifact)
            self.active_artifact_id = artifact.artifact_id
            return version

    def serve_artifact(self, artifact: Artifact, *, scenario: str) -> ServedArtifact | None:
        """Serve ``artifact`` (``bind``) beside the active checkpoint, which stays active, and return its route.

        None when the binding became uncertain after the load, as ``served_artifact`` says.
        """
        with self.lock:
            self.bind(artifact)
            return self.served_artifact(artifact.artifact_id, scenario)

    def served_model_name(self, scenario: str) -> str:
        """The route name of the active checkpoint's last verified load, or ``model_path`` while none is active."""
        with self.lock:
            if self.active_artifact_id is None:
                return self.model_path
            return self.bindings[self.active_artifact_id].model_name

    def served_artifact(self, artifact_id: str, scenario: str) -> ServedArtifact | None:
        """The route that the slot of ``artifact_id`` gives it now. None for an artifact without a binding, and
        while the slot must restore the binding."""
        with self.lock:
            binding = self.bindings.get(artifact_id)
            if binding is None:
                return None
            return self.slot_by_url[binding.url].served_artifact(artifact_id, scenario)

    def stop_serving(self, served: ServedArtifact) -> None:
        """Release the slot of ``served`` when ``served`` is the binding of a checkpoint.

        ``served`` must equal the route of the checkpoint's last verified load. Any other value changes nothing:
        the route of an earlier load, or of an artifact without a binding. The caller knows that no published
        route uses ``served`` (``InferenceRuntime.stop_serving``), so the active checkpoint is released like any
        other one, and then no checkpoint is active. The release also applies while the slot recovers. The slot's
        engines pause and stay allocated (``release_checkpoint``); the binding goes only after the pause. A pause
        that raises keeps the binding and the slot, so that a later call releases it.
        """
        with self.lock:
            if self.bindings.get(served.artifact_id) != served:
                return
            self.slot_by_url[served.url].release_checkpoint()
            self.bindings.pop(served.artifact_id)
            if self.active_artifact_id == served.artifact_id:
                self.active_artifact_id = None

    def release_unreferenced_artifacts(self, retained: Collection[ServedArtifact]) -> None:
        """Release the slot of each binding whose URL and model name no route of ``retained`` has (``stop_serving``).

        The active checkpoint is released like any other one. A binding that a route of ``retained`` uses stays,
        also one that waits for its scheduled stop. A release that raises keeps its binding and ends the call; the
        next call tries it again, and then the bindings after it.
        """
        used = {(route.url, route.model_name) for route in retained}
        with self.lock:
            for binding in tuple(self.bindings.values()):
                if (binding.url, binding.model_name) not in used:
                    self.stop_serving(binding)

    def shutdown(self) -> None:
        """Close admission and shut every slot down. Afterwards the runtime opens no slot and binds no checkpoint.

        The call waits for an ``open``, a load or a release that runs, so no slot opens after its own shutdown.
        Each slot's shutdown runs, also after another one raised, so that every health watcher stops. The first
        failure is raised after the last slot, and each later failure is logged. A slot's engines outlive the
        runtime when the slot's own shutdown leaves them running. A second call does nothing, also after a call
        that raised.
        """
        self.close_admission()
        failure: Exception | None = None
        with self.lock:
            if self.stopped:
                return
            self.stopped = True
            for slot in self.slots:
                try:
                    slot.shutdown()
                except Exception as error:
                    if failure is None:
                        failure = error
                    else:
                        logger.exception("the checkpoint slot at %s did not shut down either", slot.base_url)
        if failure is not None:
            raise failure


__all__ = ["CheckpointSlotsRuntime"]
