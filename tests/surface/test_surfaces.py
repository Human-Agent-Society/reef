from __future__ import annotations

import pytest

from reef.core.errors import ReefError
from reef.recipe.base import Recipe
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.surface import (
    AdapterWeightRuntime,
    ArtifactActivator,
    ArtifactLoader,
    ComponentSurface,
    InferenceHooks,
    LeasingInferenceHooks,
    RequestSkillLayer,
    SkillLayer,
    Surface,
    WeightInferenceHooks,
    WeightLoader,
    WeightRuntime,
    create_harness_surface,
    create_skill_surface,
    create_weight_surface,
)
from reef.surface.base import AcceptAnyArtifact
from reef.surface.skills import SkillValidator
from reef.surface.weights import reported_runtime_load_id, reported_runtime_load_spans


class _InjectingModule(RequestSkillLayer):
    """Test stand-in for a method-owned injecting layer module.

    Mirrors the shape of skillclaw's catalog module: reads its layer's
    files and prepends a system message to chat payloads.
    """

    layer = "skills"

    def prepare_request(self, files, path, payload):
        del path
        text = files.get("SKILL.md", "").strip()
        if not text:
            return payload
        return {**payload, "messages": [{"role": "system", "content": text}, *payload["messages"]]}


class _PullOnlyModule(SkillLayer):
    """Test stand-in for a pull-only layer module: never transforms."""

    layer = "skills"


def test_recipe_default_surface_is_noop() -> None:
    surface = Recipe().build_surface("s")
    assert type(surface) is Surface
    assert surface.loader is None
    assert surface.inference is None
    assert surface.files is None


def test_surface_factories_return_composed_surface_instances() -> None:
    assert type(create_weight_surface()) is Surface
    assert type(create_harness_surface()) is Surface
    assert type(create_skill_surface([_PullOnlyModule()])) is Surface


def test_optional_surface_capabilities_are_not_claimed_by_basic_implementations() -> None:
    loader = WeightLoader()
    hooks = WeightInferenceHooks()
    assert isinstance(loader, ArtifactLoader)
    # Weight surfaces activate: a runtime serving remote snapshots binds the published head there.
    assert isinstance(loader, ArtifactActivator)
    assert isinstance(hooks, InferenceHooks)
    assert not isinstance(hooks, LeasingInferenceHooks)
    assert isinstance(_InjectingModule(), RequestSkillLayer)
    assert not isinstance(_PullOnlyModule(), RequestSkillLayer)


def test_request_injection_requires_declaring_the_optional_layer_capability() -> None:
    class UndeclaredLayer(SkillLayer):
        layer = "skills"

        def prepare_request(self, files, path, request):
            raise AssertionError("undeclared request hook must not run")

    assert create_skill_surface([UndeclaredLayer()]).inference is None


def test_default_component_validator_accepts_anything() -> None:
    validator = ComponentSurface().validator
    assert isinstance(validator, AcceptAnyArtifact)
    validator.validate(object())  # type: ignore[arg-type]
    # A record-only surface has no components, so nothing to admit.
    Recipe().build_surface("s").validate(object())  # type: ignore[arg-type]


def _skill_artifact(tmp_path, text: str = "Always check units."):
    from reef.artifact import Artifact

    d = tmp_path / "skill-artifact" / "skills"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(text, encoding="utf-8")
    return Artifact.local(d.parent)


def test_skill_surface_hands_each_module_its_layer_files(tmp_path) -> None:
    artifact = _skill_artifact(tmp_path)
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    hooks = create_skill_surface([_InjectingModule()]).inference
    assert hooks is not None
    out = hooks.prepare_request(artifact, "/v1/chat/completions", payload)
    assert out["messages"][0] == {"role": "system", "content": "Always check units."}
    assert out["messages"][1] == {"role": "user", "content": "hi"}
    assert payload["messages"][0]["role"] == "user"


def test_skill_surface_has_no_inference_hooks_for_pull_only_layers() -> None:
    layer = _PullOnlyModule()
    surface = create_skill_surface([layer])
    assert surface.inference is None


def test_skill_validator_rejects_empty_and_stray_files(tmp_path) -> None:
    from reef.artifact import Artifact

    validator = SkillValidator((_PullOnlyModule(),))
    d = tmp_path / "bad-artifact"
    d.mkdir()
    with pytest.raises(ReefError):
        validator.validate(Artifact.local(d))
    (d / "skills").mkdir()
    (d / "skills" / "SKILL.md").write_text("  ", encoding="utf-8")
    with pytest.raises(ReefError):
        validator.validate(Artifact.local(d))
    (d / "skills" / "SKILL.md").write_text("Real content.", encoding="utf-8")
    validator.validate(Artifact.local(d))
    (d / "stray.md").write_text("outside every layer", encoding="utf-8")
    with pytest.raises(ReefError):
        validator.validate(Artifact.local(d))


def test_skill_validator_ignores_repository_bookkeeping(tmp_path) -> None:
    from reef.artifact import Artifact

    root = tmp_path / "artifact"
    (root / "skills").mkdir(parents=True)
    (root / "skills" / "SKILL.md").write_text("rule", encoding="utf-8")
    (root / "reef-artifact.json").write_text("{}", encoding="utf-8")
    (root / ".gitattributes").write_text("* text", encoding="utf-8")
    layer = _PullOnlyModule()
    surface = create_skill_surface([layer])
    assert surface.files is not None
    assert surface.files.read_files(Artifact.local(root)) == {"skills/SKILL.md": "rule"}
    SkillValidator((layer,)).validate(Artifact.local(root))


def test_default_prepare_request_is_identity(tmp_path) -> None:
    payload = {"model": "m"}
    assert _weight_hooks().prepare_request(_checkpoint_artifact(tmp_path), "/v1/chat/completions", payload) is payload


def _checkpoint_artifact(tmp_path):
    from reef.artifact import Artifact

    d = tmp_path / "checkpoint-artifact"
    d.mkdir(exist_ok=True)
    return Artifact.local(d)


def _release_without_files():
    from reef.artifact import Artifact, ArtifactRef

    return Artifact(ArtifactRef(content_id="artifact:release", release_id="release:1", parent_release_id=None), None)


def _weight_hooks(surface: Surface | None = None):
    hooks = (surface or create_weight_surface()).inference
    assert hooks is not None
    return hooks


def _weight_loader(surface: Surface | None = None):
    loader = (surface or create_weight_surface()).loader
    assert loader is not None
    return loader


def test_weight_surface_addresses_every_request_to_the_served_adapter(tmp_path) -> None:
    # The engine applies an adapter only when the request names it, so a
    # LoRA deployment must name it on every request — including the
    # checkpoint and streaming paths that skip version echoing.
    surface = create_weight_surface(adapter_name="reef_lora")
    payload = {"model": "m", "messages": []}
    checkpoint = _weight_hooks(surface).prepare_request(
        _checkpoint_artifact(tmp_path), "/v1/chat/completions", payload
    )
    assert checkpoint == {**payload, "lora_path": "reef_lora"}
    streaming = _weight_hooks(surface).prepare_request(
        _checkpoint_artifact(tmp_path), "/v1/chat/completions", {**payload, "stream": True}
    )
    assert streaming == {**payload, "stream": True, "lora_path": "reef_lora"}
    assert "lora_path" not in payload


def test_weight_surface_refuses_a_request_naming_another_adapter() -> None:
    surface = create_weight_surface(adapter_name="reef_lora")
    with pytest.raises(ReefError, match="serves adapter 'reef_lora'"):
        _weight_hooks(surface).prepare_request(
            _release_without_files(), "/v1/chat/completions", {"lora_path": "other"}
        )
    # Naming the served adapter explicitly is redundant, not wrong.
    out = _weight_hooks(surface).prepare_request(
        _release_without_files(), "/v1/chat/completions", {"lora_path": "reef_lora"}
    )
    assert out["lora_path"] == "reef_lora"


def test_weight_surface_without_an_adapter_leaves_requests_alone(tmp_path) -> None:
    payload = {"model": "m", "lora_path": "whatever"}
    assert _weight_hooks().prepare_request(_checkpoint_artifact(tmp_path), "/v1/chat/completions", payload) is payload


def test_the_reported_version_is_read_from_every_response_shape() -> None:
    assert reported_runtime_load_id({"metadata": {"runtime_load_id": "wv-7"}}) == "wv-7"
    # Native /generate shape: top-level meta_info.
    assert reported_runtime_load_id({"meta_info": {"runtime_load_id": "wv-7"}}) == "wv-7"
    # OpenAI chat shape: per-choice meta_info.
    assert reported_runtime_load_id({"choices": [{"meta_info": {"runtime_load_id": "wv-7"}}]}) == "wv-7"
    assert reported_runtime_load_id({"choices": [{"message": {"content": "hi"}}]}) is None


def test_mixed_token_versions_from_an_in_place_update_are_valid_spans() -> None:
    response = {
        "choices": [{"meta_info": {"runtime_load_id": "engine:7"}}],
        "training": {
            "response_length": 3,
            "runtime_load_id": None,
            "runtime_load_spans": [
                {"start": 0, "end": 1, "runtime_load_id": "engine:6"},
                {"start": 1, "end": 3, "runtime_load_id": "engine:7"},
            ],
        },
    }

    assert [span.runtime_load_id for span in reported_runtime_load_spans(response)] == ["engine:6", "engine:7"]


def test_the_reported_version_of_an_anthropic_response_is_its_final_training_span() -> None:
    response = {
        "type": "message",
        "content": [{"type": "text", "text": "done"}],
        "training": {
            "response_length": 2,
            "runtime_load_id": None,
            "runtime_load_spans": [
                {"start": 0, "end": 1, "runtime_load_id": "engine:6"},
                {"start": 1, "end": 2, "runtime_load_id": "engine:7"},
            ],
        },
    }

    assert reported_runtime_load_id(response) == "engine:7"


@pytest.mark.parametrize(
    "runtime_load_ids",
    [
        ("engine:6", "engine:7", "engine:6"),
        ("engine:7", "engine:6"),
        ("engine:6", "replacement:7"),
    ],
)
def test_impossible_runtime_load_histories_are_rejected(runtime_load_ids) -> None:
    response = {
        "choices": [{"meta_info": {"runtime_load_id": runtime_load_ids[-1]}}],
        "training": {
            "response_length": len(runtime_load_ids),
            "runtime_load_spans": [
                {"start": index, "end": index + 1, "runtime_load_id": runtime_load_id}
                for index, runtime_load_id in enumerate(runtime_load_ids)
            ],
        },
    }

    with pytest.raises(ValueError, match=r"advance monotonically|return to an earlier"):
        reported_runtime_load_spans(response)


def test_record_only_surface_has_no_inference_hooks() -> None:
    assert Surface().inference is None


def test_inference_injects_and_records_the_post_transform_request(tmp_path) -> None:
    import asyncio

    from reef.artifact import InMemoryRepositoryBackend
    from reef.dispatcher import Dispatcher
    from reef.runtime.interfaces import InferenceHandler
    from reef.service.app import RequestService

    class SkillRecipe(Recipe):
        def build_surface(self, scenario):
            return create_skill_surface([_InjectingModule()])

    class RecordingBackend(InferenceHandler):
        def __init__(self) -> None:
            self.seen = None

        async def inference(self, artifact, path, payload):
            del artifact, path
            self.seen = payload
            return {"ok": True}

    bootstrap = tmp_path / "bootstrap"
    (bootstrap / "skills").mkdir(parents=True)
    (bootstrap / "skills" / "SKILL.md").write_text("Always check units.", encoding="utf-8")

    recipe = SkillRecipe(name="skill_delivery")
    dispatcher = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(bootstrap, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        scenario_storage=SQLiteScenarioStorage(None),
    )
    service = RequestService(dispatcher)
    backend = RecordingBackend()

    async def run() -> None:
        response, item = await service.infer_with_data(
            {"x-reef-scenario": "smoke"},
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            "/v1/chat/completions",
            backend,
        )
        assert response == {"ok": True}
        assert backend.seen["messages"][0] == {"role": "system", "content": "Always check units."}
        recorded = item.payload["messages"]
        assert recorded[0] == {"role": "system", "content": "Always check units."}
        assert recorded[1] == {"role": "user", "content": "hi"}

    asyncio.run(run())
    scenario = dispatcher.get_or_create_scenario("smoke")
    assert scenario.surface is scenario.surface


class _StubTrainingRuntime(WeightRuntime):
    """Satisfies WeightRuntime; only the version probe does anything."""

    def __init__(self, served: str | None, published: str | None = None):
        self._served = served
        self._published = published

    @property
    def base_url(self) -> str:
        return "http://engine.invalid"

    def serving_runtime_load_id(self) -> str | None:
        return self._served

    def current_runtime_load_id(self) -> str | None:
        return self._published

    def restore_checkpoint(self, artifact, *, scenario=None) -> str:
        raise AssertionError("recover must not restore a checkpoint")

    def activate_checkpoint(self, artifact, *, scenario=None) -> str:
        return artifact.ref.release_id


class _StubAdapterRuntime(_StubTrainingRuntime, AdapterWeightRuntime):
    """Serves one adapter version per scenario, from ``adapters``."""

    def __init__(self, served: str | None, adapters: dict[str, str]):
        super().__init__(served)
        self._adapters = adapters

    def serving_adapter_runtime_load_id(self, scenario):
        return self._adapters.get(scenario)


def _checkpoint_ref(version: str = "checkpoint:step-7"):
    from reef.artifact import ArtifactRef

    return ArtifactRef(content_id="artifact:ckpt", release_id=version, parent_release_id=None)


def test_a_durable_release_serves_under_the_version_that_reef_acknowledged(tmp_path) -> None:
    from reef.artifact import Artifact

    # A rollback or a restart loaded the release again, under a new version.
    restored = Artifact.local(tmp_path, metadata={"runtime_load_id": "e:1"})
    runtime = _StubTrainingRuntime("e:6", published="e:5")
    assert create_weight_surface().served_runtime_load_id(restored, runtime) == "e:5"
    # A release that no version published, such as the initial one, has no served version.
    assert create_weight_surface().served_runtime_load_id(Artifact.local(tmp_path), runtime) is None
    # A per-scenario adapter serves a durable release under the version the engine holds for the scenario.
    adapters = _StubAdapterRuntime("e:6", {"math": "e:4"})
    surface = create_weight_surface(scenario="math", runtime=adapters)
    assert surface.served_runtime_load_id(restored, adapters) == "e:4"
    assert surface.served_runtime_load_id(Artifact.local(tmp_path), adapters) is None
    # An engine that holds no adapter for the scenario leaves the served version unknown.
    assert surface.served_runtime_load_id(restored, _StubAdapterRuntime("e:6", {})) is None


def test_recover_serves_the_checkpoint_when_the_head_is_not_durable() -> None:
    local = _release_without_files().ref
    checkpoint = _checkpoint_ref()
    assert _weight_loader().recover(local, checkpoint, _StubTrainingRuntime("session:1")) is checkpoint
    assert _weight_loader().recover(local, checkpoint, None) is checkpoint


def test_recover_short_circuits_when_already_at_the_head() -> None:
    surface = create_weight_surface()
    checkpoint = _checkpoint_ref()
    assert _weight_loader(surface).recover(None, checkpoint, _StubTrainingRuntime("wv-1")) is checkpoint
    same = _checkpoint_ref(checkpoint.release_id)
    assert _weight_loader(surface).recover(same, checkpoint, _StubTrainingRuntime("nope")) is checkpoint


def test_weight_hooks_route_each_scenario_to_its_own_adapter_revision(tmp_path) -> None:
    from reef.artifact import Artifact
    from reef.surface import adapter_name

    hooks = WeightInferenceHooks(scenario="math", runtime=_StubAdapterRuntime("engine:9", {"math": "engine:7"}))
    published = Artifact.local(tmp_path, metadata={"runtime_load_id": "engine:3"})
    served = hooks.prepare_request(published, "/v1/chat/completions", {"messages": []})
    # The release serves under the revision the runtime holds for the scenario.
    assert served == {"messages": [], "lora_path": adapter_name("math", "engine:7")}
    with pytest.raises(ReefError, match="asked for lora_path"):
        hooks.prepare_request(published, "/v1/chat/completions", {"lora_path": adapter_name("code", "engine:7")})


def test_weight_hooks_sample_the_base_before_a_scenario_publishes(tmp_path) -> None:
    hooks = WeightInferenceHooks(scenario="math", runtime=_StubAdapterRuntime("engine:9", {}))
    checkpoint = _checkpoint_artifact(tmp_path)
    assert "lora_path" not in hooks.prepare_request(checkpoint, "/v1/chat/completions", {"messages": []})
    with pytest.raises(ReefError, match="published no adapter yet"):
        hooks.prepare_request(checkpoint, "/v1/chat/completions", {"lora_path": "anything"})


def test_weight_hooks_refuse_both_a_shared_and_a_scenario_adapter() -> None:
    with pytest.raises(ValueError, match="either one shared adapter or per-scenario"):
        WeightInferenceHooks("shared", scenario="math", runtime=_StubAdapterRuntime(None, {}))
    with pytest.raises(ValueError, match="both the scenario and the runtime"):
        WeightInferenceHooks(scenario="math")
