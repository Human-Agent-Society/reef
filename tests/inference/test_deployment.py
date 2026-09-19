"""Inference factory discovery stays independent of trainer construction."""

from types import SimpleNamespace

import pytest

from reef.core.errors import DeployConfigError
from reef.inference import deployment
from reef.runtime.deployment import InferenceService


class Receiver(InferenceService):
    connection_protocol = "test-weights-v1"

    def __init__(self, config):
        self.config = config

    def start(self, resources):
        raise RuntimeError("factory construction must not allocate")

    def prepare_weight_transfer(self, connection):
        pass

    def backend(self, connection):
        return self

    def check_health(self):
        pass

    def poll(self):
        pass

    def close(self):
        pass


class UnregisteredReceiver:
    def __init__(self, config):
        self.receiver = Receiver(config)

    def __getattr__(self, name):
        return getattr(self.receiver, name)


def test_dotted_receiver_factory_receives_native_options_without_allocating():
    options = {"model": "small", "tensor_parallel_size": 2}
    receiver = deployment.inference_service_for(f"{__name__}:Receiver", options)
    assert isinstance(receiver, Receiver)
    assert receiver.config is options


def test_receiver_factory_requires_explicit_service_inheritance():
    with pytest.raises(DeployConfigError, match="must return an InferenceService"):
        deployment.inference_service_for(f"{__name__}:UnregisteredReceiver", {})


def test_installed_receiver_definition_is_selected_by_name(monkeypatch):
    def installed(*, group, name):
        assert group == "reef.inference_backends"
        assert name == "custom"
        return (SimpleNamespace(value=f"{__name__}:Receiver"),)

    monkeypatch.setattr(deployment, "entry_points", installed)
    assert isinstance(deployment.inference_service_for("custom", {}), Receiver)


@pytest.mark.parametrize("reference", ["builtins:dict", "builtins:None", "missing_inference_backend:create"])
def test_invalid_receiver_definition_fails_before_allocation(reference):
    with pytest.raises(DeployConfigError):
        deployment.inference_service_for(reference, {})


@pytest.mark.parametrize("count, reason", [(0, "unknown"), (2, "ambiguous")])
def test_receiver_name_must_resolve_exactly_once(monkeypatch, count, reason):
    monkeypatch.setattr(
        deployment, "entry_points", lambda **kwargs: (SimpleNamespace(value=f"{__name__}:Receiver"),) * count
    )
    with pytest.raises(DeployConfigError, match=reason):
        deployment.inference_service_for("custom", {})
