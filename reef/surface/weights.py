from __future__ import annotations


from reef.artifact.artifact import Artifact, ArtifactValidator
from reef.core.evaluation import ArtifactKind
from reef.surface.base import (
    AcceptAnyArtifact,
    AdapterWeightRuntime,
    ArtifactLayout,
    ArtifactLoader,
    ComponentSurface,
    ServingRuntime,
    Surface,
    WeightRuntime,
)


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


#: The component name a weight surface binds.
WEIGHTS_COMPONENT = "weights"


def create_weight_surface(
    *,
    scenario: str | None = None,
    kind: ArtifactKind,
    runtime: AdapterWeightRuntime | None = None,
    validator: ArtifactValidator | None = None,
    layout: ArtifactLayout | None = None,
) -> Surface:
    """Build weight loading and inference capabilities as the ``weights`` component.

    ``kind`` is the kind of the weights, ``model`` or ``model_adapter``: the
    serving runtime's ``served_weight_kind``. ``scenario`` selects
    per-scenario adapter routing on ``runtime``, which several scenarios
    share. ``validator`` admits published weights; the default accepts any.
    ``layout`` checks an imported directory; ``None`` takes no import.
    """
    if kind not in ("model", "model_adapter"):
        raise ValueError(f"the weights component is a model or a model_adapter, not {kind!r}")
    return Surface(
        components={
            WEIGHTS_COMPONENT: ComponentSurface(
                validator=AcceptAnyArtifact() if validator is None else validator,
                loader=WeightLoader(scenario),
                kind=kind,
                layout=layout,
            )
        }
    )
