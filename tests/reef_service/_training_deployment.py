"""Importable CPU stand-in for an optional in-process training integration."""

import json
from dataclasses import dataclass
from pathlib import Path

from reef_service.runtime_stubs import StubTrainingRuntime

from reef.core.config import config_option
from reef.runtime.registry import RuntimeFactory
from reef.runtime.settings import TrainingRuntimeSettings
from reef.train.deployment import InProcessTrainingDeployment


@dataclass(frozen=True)
class Options(TrainingRuntimeSettings):
    lora_rank: int = config_option(8)
    learning_rate: float = config_option(0.00001)
    checkpoint_layers: bool = config_option(False)
    label: str = config_option("001")
    marker: str = config_option("")

    def __post_init__(self):
        super().__post_init__()
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be positive")


class LocalRuntime(StubTrainingRuntime):
    def __init__(self, config, model_path):
        super().__init__(max_staleness=config["max_staleness"])
        self.config = dict(config)
        self.received_model_path = model_path
        self.closed = False
        if config["marker"]:
            Path(config["marker"]).write_text(json.dumps({"model_path": model_path, **config}))

    def shutdown(self):
        self.closed = True
        if self.config["marker"]:
            Path(self.config["marker"] + ".closed").touch()


class LocalFactory(RuntimeFactory):
    kind = "test-local"

    def config_type(self):
        return Options

    def __call__(self, config, model_path, recipe_config, environ):
        training = LocalRuntime(config, model_path)
        return training, training.inference


runtime_factory = LocalFactory()


class LocalDeployment(InProcessTrainingDeployment):
    runtime_type = "reef_service._training_deployment:runtime_factory"
