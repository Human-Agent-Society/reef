"""Runtime kinds: bundled names, dotted factory references, registration, and the training pair."""

from __future__ import annotations

import pytest
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent

from reef.inference.null import NullRuntimeFactory
from reef.runtime.deployment import (
    RuntimeConfigError,
    RuntimeFactory,
    RuntimeRegistry,
    register_runtime_kind,
    runtime_factory_for,
    runtime_kinds,
    runtime_pair,
)

#: A value a dotted runtime kind can name that is not a factory.
NOT_A_FACTORY = 7


class PairFactory(RuntimeFactory):
    """A factory reef does not bundle: it builds a training component and its runtime."""

    kind = "test_pair"

    def __call__(self, config, model_path, recipe_config, environ):
        return StubTrainingComponent(), StubInferenceRuntime(model_path=model_path)


def build_proxy(config, model_path, recipe_config, environ):
    return StubInferenceRuntime(model_path=model_path)


class ClosingRuntime(StubInferenceRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.shut_down = False

    def shutdown(self) -> None:
        self.shut_down = True


def test_bundled_kinds_resolve_lazily_to_their_factories():
    assert set(runtime_kinds()) >= {"inference_proxy", "none", "remote_serving", "tinker"}
    assert isinstance(runtime_factory_for("none"), NullRuntimeFactory)
    assert runtime_factory_for("none") is runtime_factory_for("none")
    assert runtime_factory_for("unknown") is None


def test_a_dotted_kind_names_a_factory_class_or_a_callable():
    built = RuntimeRegistry().build({"type": f"{__name__}:PairFactory"}, model_path="model", environ={})
    pair = runtime_pair(built)
    assert pair is not None and pair[1].model_path == "model"
    proxy = RuntimeRegistry().build({"type": f"{__name__}:build_proxy"}, model_path="model", environ={})
    assert isinstance(proxy, StubInferenceRuntime)


@pytest.mark.parametrize(
    ("kind", "match"),
    [
        (":factory", "must be 'package.module:factory_name'"),
        ("tests.runtime.missing_module:factory", "cannot import runtime kind"),
        (f"{__name__}:missing", "cannot import runtime kind"),
        (f"{__name__}:NOT_A_FACTORY", "is not callable"),
    ],
)
def test_an_invalid_dotted_kind_is_a_configuration_error(kind, match):
    with pytest.raises(RuntimeConfigError, match=match):
        RuntimeRegistry().build({"type": kind}, model_path="model", environ={})


@pytest.mark.parametrize("config", [{}, {"type": ""}, {"type": 3}])
def test_a_runtime_section_needs_a_type(config):
    with pytest.raises(RuntimeConfigError, match=r"runtime\.type must be a non-empty string"):
        RuntimeRegistry().build(config, model_path="model", environ={})


def test_registration_refuses_a_bundled_or_registered_kind():
    class Shadow(RuntimeFactory):
        kind = "none"

        def __call__(self, config, model_path, recipe_config, environ):
            raise AssertionError("unused")

    with pytest.raises(ValueError, match="'none' is already registered"):
        register_runtime_kind(Shadow)
    # An injected factory shadows a bundled kind for one registry only.
    registry = RuntimeRegistry({"none": build_proxy})
    assert isinstance(registry.build({"type": "none"}, model_path="model", environ={}), StubInferenceRuntime)
    assert "none" in registry.names


def test_runtime_pair_accepts_only_a_component_and_runtime_and_closes_anything_else():
    component, runtime = StubTrainingComponent(), ClosingRuntime()
    assert runtime_pair((component, runtime)) == (component, runtime)
    assert not component.closed and not runtime.shut_down
    assert runtime_pair((runtime, component)) is None
    assert component.closed and runtime.shut_down
    alone = ClosingRuntime()
    assert runtime_pair(alone) is None
    assert alone.shut_down
