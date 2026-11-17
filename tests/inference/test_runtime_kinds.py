"""Runtime kinds: bundled names, factory class references, and injected factories."""

from __future__ import annotations

import pytest
from tests.support.runtime_stubs import StubInferenceRuntime

from reef.core.errors import RuntimeConfigError
from reef.core.plugins import PluginConfigError
from reef.inference.null import NullRuntimeFactory
from reef.inference.registry import RuntimeFactory, RuntimeRegistry, runtime_factory_for

#: A value a runtime kind reference can name that is not a factory class.
NOT_A_FACTORY = 7


class ProxyFactory(RuntimeFactory):
    """A factory reef does not bundle: it builds a proxy runtime."""

    kind = "test_proxy"

    def __call__(self, config, model_path, recipe_config, environ):
        return StubInferenceRuntime(model_path=model_path)


def build_proxy(config, model_path, recipe_config, environ):
    return StubInferenceRuntime(model_path=model_path)


def test_bundled_kinds_resolve_lazily_to_their_factories():
    assert set(RuntimeRegistry().names) == {"inference_proxy", "none", "remote_serving", "tinker"}
    assert isinstance(runtime_factory_for("none"), NullRuntimeFactory)
    with pytest.raises(PluginConfigError, match="unknown runtime type 'unknown'; use one of: inference_proxy, none"):
        runtime_factory_for("unknown")


def test_a_reference_kind_names_a_factory_class():
    built = RuntimeRegistry().build({"type": f"{__name__}:ProxyFactory"}, model_path="model", environ={})
    assert isinstance(built, StubInferenceRuntime) and built.model_path == "model"


@pytest.mark.parametrize(
    ("kind", "match"),
    [
        (":factory", "must be 'package.module:ClassName'"),
        ("tests.runtime.missing_module:factory", "cannot import runtime type"),
        (f"{__name__}:missing", "cannot import runtime type"),
        (f"{__name__}:NOT_A_FACTORY", "must name a class that inherits RuntimeFactory"),
        (f"{__name__}:build_proxy", "must name a class that inherits RuntimeFactory"),
        ("reef.inference.registry:RuntimeFactory", "names an abstract class"),
    ],
)
def test_an_invalid_reference_kind_is_a_configuration_error(kind, match):
    with pytest.raises(RuntimeConfigError, match=match):
        RuntimeRegistry().build({"type": kind}, model_path="model", environ={})


@pytest.mark.parametrize("config", [{}, {"type": ""}, {"type": 3}])
def test_a_runtime_section_needs_a_type(config):
    with pytest.raises(RuntimeConfigError, match=r"runtime\.type must be a non-empty string"):
        RuntimeRegistry().build(config, model_path="model", environ={})


def test_an_injected_factory_shadows_a_bundled_kind_for_one_registry():
    registry = RuntimeRegistry({"none": ProxyFactory()})
    assert isinstance(registry.build({"type": "none"}, model_path="model", environ={}), StubInferenceRuntime)
    assert "none" in registry.names
