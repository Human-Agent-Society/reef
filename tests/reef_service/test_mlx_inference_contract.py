"""MLX receiver reports the version its engine serves."""

from reef.runtime.interfaces import InferenceRuntime
from reef.train.mlx_backend.inference import MLXInferenceBackend


class ResidentRuntime(InferenceRuntime):
    def __init__(self) -> None:
        super().__init__(base_url="mlx://local")
        self.resident_version = "mlx-1"
        self.mark_published()

    def serving_runtime_load_id(self) -> str:
        return self.resident_version

    @property
    def inference_handler(self) -> MLXInferenceBackend:
        return MLXInferenceBackend(self)


def test_receiver_reports_resident_version_before_publication() -> None:
    runtime = ResidentRuntime()
    backend = MLXInferenceBackend(runtime)
    assert backend.runtime_load_ids() == ("mlx-1",)

    runtime.resident_version = "mlx-2"
    assert runtime.current_runtime_load_id() == "mlx-1"
    assert backend.runtime_load_ids() == ("mlx-2",)

    runtime.mark_published()
    assert runtime.current_runtime_load_id() == "mlx-2"
    assert backend.runtime_load_ids() == ("mlx-2",)
