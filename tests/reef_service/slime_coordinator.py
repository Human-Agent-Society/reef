"""Explicit test assembly of Reef coordination and the two native adapters."""

from unittest.mock import patch

from reef.inference.sglang.backend import SGLangInferenceBackend
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.publication import BackendWeightPublisher
from reef.runtime.scheduler import TrainingCoordinator
from reef.train.slime_backend.reef_adapters.bridge import SlimeTrainingBackend


class FixtureWeightPublisher(BackendWeightPublisher):
    """Preserve explicit fake-group versions in older wire contract fixtures.

    Allocation with canonical, advancing IDs is tested separately against the
    unmodified TrainingCoordinator in test_training_coordinator.py.
    """

    def next_runtime_load_id(self):
        return self.training._group.next_runtime_load_id()


class FixtureInferenceBackend(SGLangInferenceBackend):
    def initialize_version(self, runtime_load_id):
        # These fixtures begin with the selected identity already loaded.
        pass


def build_slime_coordinator(actor_group, inference, **kwargs) -> TrainingCoordinator:
    training = SlimeTrainingBackend(actor_group, **kwargs)
    training.context.runtime_load_id = training.current_runtime_load_id()
    receiver = FixtureInferenceBackend(RayExecutor.from_workers([inference]))
    with patch("reef.runtime.scheduler.BackendWeightPublisher", FixtureWeightPublisher):
        return TrainingCoordinator(training, receiver)
