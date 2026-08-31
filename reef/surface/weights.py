from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactRef, LiveWeightArtifactRef
from reef.core.artifact_ref import WeightVersionSpan, parse_weight_version_spans
from reef.core.errors import ReefError
from reef.surface.adapter import adapter_name
from reef.surface.base import ServingRuntime, Surface, WeightRuntime

logger = logging.getLogger(__name__)


class WeightVersionMismatch(ReefError):
    """The engine generated with different weights than the request froze.

    Raised when a live artifact's ``weight_version`` does not match the
    version the serving engine reports for the response, or when the engine
    does not report one at all. Reef cannot record the exchange's serving
    version with confidence, so the completed response is rejected explicitly
    and never becomes a training record.
    """


def artifact_weight_version(artifact: Artifact | ArtifactRef) -> str | None:
    """The serving weight version an artifact was published under, if any."""
    ref = artifact.ref if isinstance(artifact, Artifact) else artifact
    if isinstance(ref, LiveWeightArtifactRef):
        return ref.weight_version
    metadata = artifact.metadata if isinstance(artifact, Artifact) else None
    version = None if metadata is None else metadata.get("weight_version")
    return version if isinstance(version, str) and version else None


class WeightLoader:
    """Restore weight checkpoints and recover the live serving head.

    ``scenario`` binds the loader to one scenario of a runtime that serves a
    separate adapter per scenario: recovery then checks that scenario's
    resident adapter rather than the engine-global weight version, which
    every other scenario's publication advances.
    """

    def __init__(self, scenario: str | None = None) -> None:
        self._scenario = scenario

    def recover(
        self,
        current: ArtifactRef | None,
        checkpoint: ArtifactRef,
        runtime: ServingRuntime | None,
    ) -> ArtifactRef:
        if current is None or current.version == checkpoint.version:
            return checkpoint
        if not isinstance(current, LiveWeightArtifactRef):
            # A staged-but-never-published artifact (``local:``) has no
            # durable bytes address; keep the committed step/state
            # continuity and fall serving back to the last checkpoint.
            logger.info(
                "committed artifact %r is not durable; serving falls back to checkpoint %r",
                current.version,
                checkpoint.version,
            )
            return checkpoint
        # Weight versions are session scoped, so a restarted engine never
        # matches the recovered one; serve the checkpoint so the recorded
        # serving version remains exact.
        if isinstance(runtime, WeightRuntime):
            served = self._served_version(runtime)
            if served is None and self._scenario is not None and self._per_scenario(runtime):
                # An adapter runtime that holds nothing for this scenario
                # cannot serve its live head; only the checkpoint is exact.
                logger.warning(
                    "recovered scenario %r at weight version %r but the engine holds no adapter for it; "
                    "serving falls back to checkpoint %r",
                    self._scenario,
                    current.weight_version,
                    checkpoint.version,
                )
                return checkpoint
            if served is not None and served != current.weight_version:
                logger.warning(
                    "recovered at weight version %r but the serving engine reports %r; "
                    "serving falls back to checkpoint %r, and the next training step republishes a live head",
                    current.weight_version,
                    served,
                    checkpoint.version,
                )
                return checkpoint
        return current

    def _served_version(self, runtime: WeightRuntime) -> str | None:
        if self._scenario is not None and self._per_scenario(runtime):
            return runtime.serving_adapter_version(self._scenario)  # type: ignore[attr-defined]
        return runtime.serving_weight_version()

    @staticmethod
    def _per_scenario(runtime: WeightRuntime) -> bool:
        return callable(getattr(runtime, "serving_adapter_version", None))

    def load(self, artifact: Artifact, runtime: ServingRuntime | None) -> str:
        if not isinstance(runtime, WeightRuntime):
            raise ReefError("weight rollback requires a training runtime")
        if artifact.local_path is None:
            raise ReefError("weight rollback requires a materialized checkpoint")
        weight_version = runtime.restore_checkpoint(artifact)
        if not isinstance(weight_version, str) or not weight_version:
            raise TypeError("restore_checkpoint must return a non-empty weight version")
        return weight_version


class WeightInferenceHooks:
    """Address weight-backed requests and verify the serving weight version.

    ``adapter_name`` names the one adapter a shared-slot LoRA runtime serves.
    ``scenario`` instead derives the name per request on a runtime that
    serves one adapter per scenario: ``adapter_name(scenario,
    weight_version)`` of the frozen artifact, so the recorded ``lora_path``
    proves which of that scenario's publications answered. An artifact with
    no weight version (nothing published yet) samples the frozen base, which
    is exactly what a fresh zero-initialised adapter computes.
    """

    def __init__(self, adapter_name: str | None = None, *, scenario: str | None = None) -> None:
        if adapter_name is not None and not adapter_name:
            raise ValueError("adapter_name must be a non-empty name or None")
        if adapter_name is not None and scenario is not None:
            raise ValueError("a weight surface serves either one shared adapter or per-scenario adapters")
        self._adapter_name = adapter_name
        self._scenario = scenario

    def prepare_request(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        payload = self._address_adapter(payload, self._served_adapter(artifact))
        if not isinstance(artifact.ref, LiveWeightArtifactRef):
            return payload
        if payload.get("stream") is True:
            # Streaming and return_meta_info are mutually exclusive on the
            # engine; streamed exchanges are recorded unverified.
            return payload
        return {**payload, "return_meta_info": True}

    def _served_adapter(self, artifact: Artifact) -> str | None:
        if self._scenario is None:
            return self._adapter_name
        version = artifact_weight_version(artifact)
        return None if version is None else adapter_name(self._scenario, version)

    def _address_adapter(self, payload: dict[str, Any], served: str | None) -> dict[str, Any]:
        """Name the served adapter, refusing a request that names another one."""
        requested = payload.get("lora_path")
        if served is None:
            if requested is not None and self._scenario is not None:
                raise ReefError(
                    f"scenario {self._scenario!r} has published no adapter yet; "
                    f"the request asked for lora_path {requested!r}"
                )
            return payload
        if requested is not None and requested != served:
            raise ReefError(
                f"this deployment serves adapter {served!r}; the request asked for lora_path {requested!r}"
            )
        return {**payload, "lora_path": served}

    def verify_response(self, artifact: Artifact, path: str, response: Mapping[str, Any]) -> None:
        if not isinstance(artifact.ref, LiveWeightArtifactRef):
            return
        frozen = artifact.ref.weight_version
        reported = reported_weight_version(response)
        spans = reported_weight_version_spans(response)
        if spans:
            final = spans[-1].weight_version
            if reported != final:
                raise WeightVersionMismatch(
                    f"response reports final weight version {reported!r} but token weight-version spans end at {final!r}"
                )
            frozen_order = _canonical_weight_version_order(frozen)
            first_order = _canonical_weight_version_order(spans[0].weight_version)
            if frozen_order is not None:
                span_orders = [_canonical_weight_version_order(span.weight_version) for span in spans]
                if any(order is None or order[0] != frozen_order[0] for order in span_orders):
                    raise WeightVersionMismatch(
                        f"response token weight versions are incompatible with canonical frozen weight version {frozen!r}"
                    )
                if first_order is None:
                    raise WeightVersionMismatch(
                        "canonical frozen weight versions require canonical response token weight versions"
                    )
                if first_order[1] < frozen_order[1]:
                    raise WeightVersionMismatch(
                        f"response token weight versions begin at {spans[0].weight_version!r}, which cannot follow frozen "
                        f"weight version {frozen!r}"
                    )
            # An admitted request can still be waiting for its first decode
            # when a weight update pauses the scheduler. Exact spans are therefore
            # authoritative even when every token uses a version newer than
            # the artifact frozen before the upstream request began.
            return
        if reported is None:
            raise WeightVersionMismatch(
                f"request froze weight version {frozen!r} but the engine response reports no weight_version "
                f"(path {path}); the upstream cannot confirm which weights produced this response"
            )
        if reported != frozen:
            raise WeightVersionMismatch(
                f"request froze weight version {frozen!r} but the engine reported final weight version "
                f"{reported!r} without token-level weight-version spans (path {path}); "
                "Reef cannot verify which weights produced this response"
            )


def create_weight_surface(adapter_name: str | None = None, *, scenario: str | None = None) -> Surface:
    """Build weight loading and inference capabilities.

    ``scenario`` selects per-scenario adapter routing on a runtime whose
    training slot is shared by several scenarios.
    """
    return Surface(
        loader=WeightLoader(scenario),
        inference=WeightInferenceHooks(adapter_name, scenario=scenario),
    )


def reported_weight_version(response: Mapping[str, Any]) -> str | None:
    """Extract the engine-reported weight version from a provider response.

    Native replies keep ``meta_info`` at the top level; OpenAI replies use
    top-level ``metadata`` or per-choice ``meta_info`` when the request sets
    ``return_meta_info``.
    """
    for key in ("meta_info", "metadata"):
        meta_info = response.get(key)
        if isinstance(meta_info, Mapping) and (version := meta_info.get("weight_version")) is not None:
            return str(version)
    choices = response.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            meta_info = choice.get("meta_info")
            if isinstance(meta_info, Mapping):
                version = meta_info.get("weight_version")
                if version is not None:
                    return str(version)
    # Token-native provider facades keep engine provenance in the private
    # training block so OpenAI and Anthropic client envelopes can stay
    # provider-compatible. A multi-version response has no single training
    # weight_version; in that case the final exact span is authoritative.
    training = response.get("training")
    if isinstance(training, Mapping):
        version = training.get("weight_version")
        if version is not None:
            return str(version)
        spans = training.get("weight_version_spans")
        if isinstance(spans, list) and spans and isinstance(spans[-1], Mapping):
            final = spans[-1].get("weight_version")
            if final is not None:
                return str(final)
    return None


def reported_weight_version_spans(response: Mapping[str, Any]) -> tuple[WeightVersionSpan, ...]:
    """Extract and validate contiguous response-token weight-version spans."""
    training = response.get("training")
    raw = training.get("weight_version_spans") if isinstance(training, Mapping) else None
    if raw is None:
        return ()
    response_length = training.get("response_length") if isinstance(training, Mapping) else None
    expected_length = (
        response_length if isinstance(response_length, int) and not isinstance(response_length, bool) else None
    )
    spans = parse_weight_version_spans(
        raw,
        field_name="response token weight-version spans",
        response_length=expected_length,
    )
    _validate_weight_version_transitions(spans)
    return spans


def _validate_weight_version_transitions(spans: tuple[WeightVersionSpan, ...]) -> None:
    """Enforce the scheduler transition policy at the weight surface."""
    seen_versions: set[str] = set()
    for index, span in enumerate(spans):
        version = span.weight_version
        if index:
            previous = spans[index - 1].weight_version
            if version == previous:
                raise ValueError("adjacent response token weight-version spans must be coalesced")
            previous_order = _canonical_weight_version_order(previous)
            current_order = _canonical_weight_version_order(version)
            if (
                previous_order is not None
                and current_order is not None
                and (current_order[0] != previous_order[0] or current_order[1] <= previous_order[1])
            ):
                raise ValueError("response token weight versions must advance monotonically in one incarnation")
            if version in seen_versions:
                raise ValueError("response token weight versions cannot return to an earlier version")
        seen_versions.add(version)


def _canonical_weight_version_order(value: str) -> tuple[str, int] | None:
    """Parse Reef's canonical serving token while accepting opaque versions."""
    incarnation, separator, sequence = value.rpartition(":")
    if (
        not separator
        or not incarnation
        or ":" in incarnation
        or not sequence.isascii()
        or not sequence.isdecimal()
        or str(int(sequence)) != sequence
    ):
        return None
    return incarnation, int(sequence)
