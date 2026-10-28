from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactRef, ArtifactValidator
from reef.core.artifact_ref import RuntimeLoadSpan, parse_runtime_load_spans
from reef.core.errors import ReefError
from reef.core.evaluation import ArtifactKind
from reef.surface.adapter import adapter_name
from reef.surface.base import (
    AcceptAnyArtifact,
    AdapterWeightRuntime,
    ArtifactLoader,
    ComponentSurface,
    InferenceHooks,
    ServingRuntime,
    Surface,
    WeightRuntime,
)

logger = logging.getLogger(__name__)


class RuntimeLoadMismatch(ReefError):
    """The engine's response does not prove which weights produced it.

    Raised when a durable training response reports no ``runtime_load_id``,
    or reports token runtime-load-ID spans that break the weight surface's
    transition policy. Reef cannot record the exchange's serving version with
    confidence, so the completed response is rejected explicitly and never
    becomes a training record.
    """


class WeightLoader(ArtifactLoader):
    """Serve published weight releases on a ``WeightRuntime``, and recover the serving head.

    ``scenario`` binds the loader to one scenario of a runtime that serves a
    separate adapter per scenario: a durable release of that scenario serves
    under the version of its resident adapter rather than the engine-global
    runtime load ID, which every other scenario's load advances.
    """

    def __init__(self, scenario: str | None = None) -> None:
        self._scenario = scenario

    def activate(self, artifact: Artifact, runtime: ServingRuntime | None, *, source: Artifact | None = None) -> str:
        """Serve the final release on the runtime before traffic reaches it; without a weight runtime, its release ID."""
        if isinstance(runtime, WeightRuntime):
            return runtime.activate_checkpoint(artifact, scenario=self._scenario)
        return artifact.ref.release_id

    def recover(
        self,
        current: ArtifactRef | None,
        checkpoint: ArtifactRef,
        runtime: ServingRuntime | None,
    ) -> ArtifactRef:
        if current is not None and current.release_id != checkpoint.release_id:
            # A staged-but-never-published artifact (``local:``) has no
            # durable bytes address; keep the committed step/state
            # continuity and fall serving back to the last checkpoint.
            logger.info(
                "committed artifact %r is not durable; serving falls back to checkpoint %r",
                current.release_id,
                checkpoint.release_id,
            )
        return checkpoint

    def served_runtime_load_id(self, artifact: Artifact, runtime: ServingRuntime | None) -> str | None:
        """Return the runtime load ID that serves the committed ``artifact``, or None when it is unknown.

        The runtime names it: every activation of a release, at a commit, a
        rollback or a restart, loads it under a version of its own.
        """
        if self._scenario is not None:
            return scenario_adapter_version(self._scenario, runtime)
        if not isinstance(runtime, WeightRuntime):
            return None
        return runtime.current_runtime_load_id()


def scenario_adapter_version(scenario: str, runtime: ServingRuntime | None) -> str | None:
    """The version of the adapter that ``runtime`` serves for ``scenario``, or None when it serves none.

    A durable release of the scenario is served under this version. A restart
    publishes each scenario's checkpoint under a new runtime load ID, as it
    publishes full weights, so the runtime, not the release, names the version.
    """
    return runtime.serving_adapter_runtime_load_id(scenario) if isinstance(runtime, AdapterWeightRuntime) else None


class WeightInferenceHooks(InferenceHooks):
    """Address weight-backed requests to the adapter that serves them.

    Without ``scenario``, requests name no adapter: the runtime serves full
    weights. With ``scenario``, ``runtime`` serves one adapter per scenario,
    and each request names ``adapter_name(scenario, runtime_load_id)`` for the
    version ``runtime`` serves the scenario. A scenario the runtime serves no
    adapter for samples the base model. The request service checks the version
    a durable training response reports, so these hooks verify no response
    themselves.
    """

    def __init__(self, *, scenario: str | None = None, runtime: AdapterWeightRuntime | None = None) -> None:
        if (scenario is None) != (runtime is None):
            raise ValueError("per-scenario adapters need both the scenario and the runtime that serves them")
        self._scenario = scenario
        self._runtime = runtime

    def prepare_request(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._address_adapter(payload, self._served_adapter(artifact))

    def _served_adapter(self, artifact: Artifact) -> str | None:
        if self._scenario is None:
            return None
        version = scenario_adapter_version(self._scenario, self._runtime)
        # A scenario served no adapter samples the base model, as a release without weights does.
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
        return


#: The component name a weight surface binds.
WEIGHTS_COMPONENT = "weights"


def create_weight_surface(
    *,
    scenario: str | None = None,
    kind: ArtifactKind,
    runtime: AdapterWeightRuntime | None = None,
    validator: ArtifactValidator | None = None,
) -> Surface:
    """Build weight loading and inference capabilities as the ``weights`` component.

    ``kind`` is the kind of the weights, ``model`` or ``model_adapter``: the
    serving runtime's ``served_weight_kind``. ``scenario`` selects
    per-scenario adapter routing on ``runtime``, which several scenarios
    share. ``validator`` admits published weights; the default accepts any.
    """
    if kind not in ("model", "model_adapter"):
        raise ValueError(f"the weights component is a model or a model_adapter, not {kind!r}")
    return Surface(
        components={
            WEIGHTS_COMPONENT: ComponentSurface(
                validator=AcceptAnyArtifact() if validator is None else validator,
                loader=WeightLoader(scenario),
                inference=WeightInferenceHooks(scenario=scenario, runtime=runtime),
                kind=kind,
            )
        }
    )


def reported_runtime_load_id(response: Mapping[str, Any]) -> str | None:
    """Extract the engine-reported runtime load ID from a provider response.

    Native replies keep ``meta_info`` at the top level; OpenAI replies use
    top-level ``metadata`` or per-choice ``meta_info`` when the request sets
    ``return_meta_info``.
    """
    for key in ("meta_info", "metadata"):
        meta_info = response.get(key)
        if isinstance(meta_info, Mapping) and (version := meta_info.get("runtime_load_id")) is not None:
            return str(version)
    choices = response.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            meta_info = choice.get("meta_info")
            if isinstance(meta_info, Mapping):
                version = meta_info.get("runtime_load_id")
                if version is not None:
                    return str(version)
    # Token-native provider facades keep the producing engine version in the
    # private training block so OpenAI and Anthropic client envelopes can stay
    # provider-compatible. A multi-version response has no single training
    # runtime_load_id; in that case the final exact span is authoritative.
    training = response.get("training")
    if isinstance(training, Mapping):
        version = training.get("runtime_load_id")
        if version is not None:
            return str(version)
        spans = training.get("runtime_load_spans")
        if isinstance(spans, list) and spans and isinstance(spans[-1], Mapping):
            final = spans[-1].get("runtime_load_id")
            if final is not None:
                return str(final)
    return None


def reported_runtime_load_spans(response: Mapping[str, Any]) -> tuple[RuntimeLoadSpan, ...]:
    """Extract and validate contiguous response-token runtime-load-ID spans."""
    training = response.get("training")
    raw = training.get("runtime_load_spans") if isinstance(training, Mapping) else None
    if raw is None:
        return ()
    response_length = training.get("response_length") if isinstance(training, Mapping) else None
    expected_length = (
        response_length if isinstance(response_length, int) and not isinstance(response_length, bool) else None
    )
    spans = parse_runtime_load_spans(
        raw,
        field_name="response token runtime-load-ID spans",
        response_length=expected_length,
    )
    _validate_runtime_load_id_transitions(spans)
    return spans


def _validate_runtime_load_id_transitions(spans: tuple[RuntimeLoadSpan, ...]) -> None:
    """Enforce the runtime-load-ID transition policy of a response's token spans."""
    seen_versions: set[str] = set()
    for index, span in enumerate(spans):
        version = span.runtime_load_id
        if index:
            previous = spans[index - 1].runtime_load_id
            if version == previous:
                raise ValueError("adjacent response token runtime-load-ID spans must be coalesced")
            previous_order = _canonical_runtime_load_id_order(previous)
            current_order = _canonical_runtime_load_id_order(version)
            if (
                previous_order is not None
                and current_order is not None
                and (current_order[0] != previous_order[0] or current_order[1] <= previous_order[1])
            ):
                raise ValueError("response token runtime load IDs must advance monotonically in one incarnation")
            if version in seen_versions:
                raise ValueError("response token runtime load IDs cannot return to an earlier version")
        seen_versions.add(version)


def _canonical_runtime_load_id_order(value: str) -> tuple[str, int] | None:
    """Parse Reef's canonical serving token while accepting opaque runtime load IDs."""
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
