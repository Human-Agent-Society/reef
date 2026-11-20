"""Importable CPU stand-in for an optional in-process training integration."""

import json
from dataclasses import dataclass
from pathlib import Path

from tests.support.runtime_stubs import FULL_WEIGHTS, StubInferenceRuntime, StubTrainingComponent

from reef.artifact import DefaultStart
from reef.core.config import config_option
from reef.inference.registry import RuntimeFactory
from reef.train.component import TrainingBackendFacts, TrainingComponentFactory
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

    def facts(self, config, model_path, environ):
        return LocalFacts()

    def __call__(self, config, model_path, environ):
        return LocalRuntime(config, model_path)


component_factory = LocalFactory()


class LocalServingFactory(RuntimeFactory):
    """Build the deployment's own serving runtime (``inference.serving: trainer``)."""

    kind = "tests.support.training_deployment:LocalServingFactory"

    def __call__(self, config, model_path, recipe_config, environ):
        return StubInferenceRuntime(model_path=model_path)


class LocalDeployment(InProcessTrainingDeployment):
    """A training backend that serves its own snapshots by default, as Tinker does."""

    component_type = "tests.support.training_deployment:component_factory"
    default_serving = "trainer"

    def trainer_serving_config(self, settings):
        return {"type": LocalServingFactory.kind}


class LocalFacts(TrainingBackendFacts):
    """The facts of ``LocalRuntime``, which are those of a ``StubTrainingComponent`` with its defaults."""

    @property
    def name(self):
        return "stub"

    @property
    def artifact_format(self):
        return FULL_WEIGHTS

    @property
    def shares_scenarios(self):
        return False

    @property
    def concurrent_jobs(self):
        return False

    @property
    def default_start_inputs(self):
        return {"model": "stub"}

    def default_start(self):
        return DefaultStart(inputs=self.default_start_inputs)

    def parameters_schema(self, kind):
        return None

    @property
    def start_files(self):
        return ()

    def check_root_start(self, root):
        return None

    def check_import_start(self, artifact_dir):
        return None


class FactsFactory(LocalFactory):
    """``LocalFactory`` that counts the components that it built in ``built``."""

    kind = "test-local-facts"

    def __init__(self):
        self.built = 0

    def __call__(self, config, model_path, environ):
        self.built += 1
        return super().__call__(config, model_path, environ)


facts_component_factory = FactsFactory()


class FactsOnlyFactory(LocalFactory):
    """``LocalFactory`` in a process that must build no component: only ``facts`` works."""

    kind = "test-local-facts-only"

    def __call__(self, config, model_path, environ):
        raise AssertionError("this process builds no training component")


facts_only_component_factory = FactsOnlyFactory()


class FactsOnlyDeployment(LocalDeployment):
    """``LocalDeployment`` on a backend whose factory builds no component."""

    component_type = "tests.support.training_deployment:facts_only_component_factory"


class FactsDeployment(LocalDeployment):
    """``LocalDeployment`` on a backend that states its facts from configuration."""

    component_type = "tests.support.training_deployment:facts_component_factory"
