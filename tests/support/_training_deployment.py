"""Importable CPU stand-in for an optional in-process training integration."""

import json
from dataclasses import dataclass
from pathlib import Path

from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent

from reef.core.config import config_option
from reef.runtime.deployment import TrainingComponentFactory
from reef.train.deployment import InProcessTrainingDeployment


@dataclass(frozen=True)
class Options:
    inference_timeout_s: float = config_option(300.0)
    train_timeout_s: float | None = config_option(None)
    batch_size: int = config_option(1)
    lora_rank: int = config_option(8)
    learning_rate: float = config_option(0.00001)
    checkpoint_layers: bool = config_option(False)
    label: str = config_option("001")
    marker: str = config_option("")

    def __post_init__(self):
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be positive")


class LocalRuntime(StubTrainingComponent):
    def __init__(self, config, model_path):
        super().__init__()
        self.config = dict(config)
        self.received_model_path = model_path
        self.closed = False
        if config["marker"]:
            Path(config["marker"]).write_text(json.dumps({"model_path": model_path, **config}))

    def close(self):
        self.closed = True
        if self.config["marker"]:
            Path(self.config["marker"] + ".closed").touch()


class LocalFactory(TrainingComponentFactory):
    kind = "test-local"

    def config_type(self):
        return Options

    def __call__(self, config, model_path, environ):
        return LocalRuntime(config, model_path)


component_factory = LocalFactory()


def build_local_serving(config, model_path, recipe_config, environ):
    """The deployment's own serving runtime (``inference.serving: trainer``)."""
    return StubInferenceRuntime(model_path=model_path)


class LocalDeployment(InProcessTrainingDeployment):
    """A training backend that serves its own releases by default, as Tinker does."""

    component_type = "tests.support._training_deployment:component_factory"
    default_serving = "trainer"

    def trainer_serving_config(self, settings):
        return {"type": "tests.support._training_deployment:build_local_serving"}
