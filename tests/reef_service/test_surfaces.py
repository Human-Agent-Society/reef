from __future__ import annotations

import pytest

from reef.core.errors import ReefError
from reef.recipe.base import Recipe
from reef.scenario import AcceptAnyArtifact
from reef.surface import (
    SkillLayer,
    Surface,
    WeightInferenceHooks,
    WeightLoader,
    WeightRuntime,
    RuntimeLoadMismatch,
    create_harness_surface,
    create_skill_surface,
    create_weight_surface,
)
from reef.surface.skills import SkillValidator


class _InjectingModule(SkillLayer):
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


def test_default_validate_accepts_anything() -> None:
    validator = Recipe().build_artifact_validator()
    assert isinstance(validator, AcceptAnyArtifact)
    validator.validate(object())  # type: ignore[arg-type]


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


def _live_artifact(runtime_load_id: str = "wv-1"):
    from reef.artifact import Artifact, LiveWeightArtifactRef

    return Artifact(
        LiveWeightArtifactRef(
            content_id="artifact:live",
            release_id=f"live:proc:{runtime_load_id}:1",
            parent_release_id=None,
            runtime_load_id=runtime_load_id,
        ),
        None,
    )


def _weight_hooks(surface: Surface | None = None):
    hooks = (surface or create_weight_surface()).inference
    assert hooks is not None
    return hooks


def _weight_loader(surface: Surface | None = None):
    loader = (surface or create_weight_surface()).loader
    assert loader is not None
    return loader


def test_weight_surface_asks_live_requests_to_echo_the_serving_version() -> None:
    artifact = _live_artifact("wv-1")
    payload = {"model": "m", "messages": []}
    out = _weight_hooks().prepare_request(artifact, "/v1/chat/completions", payload)
    assert out == {**payload, "return_meta_info": True}
    assert "return_meta_info" not in payload


def test_weight_surface_skips_injection_for_checkpoints_and_streams(tmp_path) -> None:
    surface = create_weight_surface()
    payload = {"model": "m", "messages": []}
    assert (
        _weight_hooks(surface).prepare_request(_checkpoint_artifact(tmp_path), "/v1/chat/completions", payload)
        is payload
    )
    streaming = {**payload, "stream": True}
    assert _weight_hooks(surface).prepare_request(_live_artifact(), "/v1/chat/completions", streaming) is streaming


def test_weight_surface_addresses_every_request_to_the_served_adapter(tmp_path) -> None:
    # The engine applies an adapter only when the request names it, so a
    # LoRA deployment must name it on every request — including the
    # checkpoint and streaming paths that skip version echoing.
    surface = create_weight_surface(adapter_name="reef_lora")
    payload = {"model": "m", "messages": []}
    live = _weight_hooks(surface).prepare_request(_live_artifact("wv-1"), "/v1/chat/completions", payload)
    assert live == {**payload, "lora_path": "reef_lora", "return_meta_info": True}
    checkpoint = _weight_hooks(surface).prepare_request(
        _checkpoint_artifact(tmp_path), "/v1/chat/completions", payload
    )
    assert checkpoint == {**payload, "lora_path": "reef_lora"}
    streaming = _weight_hooks(surface).prepare_request(
        _live_artifact(), "/v1/chat/completions", {**payload, "stream": True}
    )
    assert streaming["lora_path"] == "reef_lora"
    assert "lora_path" not in payload


def test_weight_surface_refuses_a_request_naming_another_adapter() -> None:
    surface = create_weight_surface(adapter_name="reef_lora")
    with pytest.raises(ReefError, match="serves adapter 'reef_lora'"):
        _weight_hooks(surface).prepare_request(_live_artifact(), "/v1/chat/completions", {"lora_path": "other"})
    # Naming the served adapter explicitly is redundant, not wrong.
    out = _weight_hooks(surface).prepare_request(_live_artifact(), "/v1/chat/completions", {"lora_path": "reef_lora"})
    assert out["lora_path"] == "reef_lora"


def test_weight_surface_without_an_adapter_leaves_requests_alone(tmp_path) -> None:
    payload = {"model": "m", "lora_path": "whatever"}
    assert _weight_hooks().prepare_request(_checkpoint_artifact(tmp_path), "/v1/chat/completions", payload) is payload


def test_weight_surface_accepts_a_matching_reported_version() -> None:
    surface = create_weight_surface()
    artifact = _live_artifact("wv-7")
    _weight_hooks(surface).verify_response(artifact, "/v1/chat/completions", {"metadata": {"runtime_load_id": "wv-7"}})
    # Native /generate shape: top-level meta_info.
    _weight_hooks(surface).verify_response(artifact, "/generate", {"meta_info": {"runtime_load_id": "wv-7"}})
    # OpenAI chat shape with return_meta_info: per-choice meta_info.
    _weight_hooks(surface).verify_response(
        artifact, "/v1/chat/completions", {"choices": [{"meta_info": {"runtime_load_id": "wv-7"}}]}
    )


def test_weight_surface_accepts_exact_mixed_token_versions_from_an_in_place_update() -> None:
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

    _weight_hooks().verify_response(_live_artifact("engine:6"), "/v1/chat/completions", response)


def test_weight_surface_reads_anthropic_provenance_from_private_training_spans() -> None:
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

    _weight_hooks().verify_response(_live_artifact("engine:6"), "/v1/messages", response)


def test_weight_surface_rejects_inconsistent_mixed_token_versions() -> None:
    response = {
        "choices": [{"meta_info": {"runtime_load_id": "engine:8"}}],
        "training": {
            "response_length": 2,
            "runtime_load_spans": [
                {"start": 0, "end": 1, "runtime_load_id": "engine:6"},
                {"start": 1, "end": 2, "runtime_load_id": "engine:7"},
            ],
        },
    }
    with pytest.raises(RuntimeLoadMismatch, match="runtime-load-ID spans end"):
        _weight_hooks().verify_response(_live_artifact("engine:6"), "/v1/chat/completions", response)


def test_weight_surface_accepts_when_weight_update_precedes_the_first_decode() -> None:
    response = {
        "choices": [{"meta_info": {"runtime_load_id": "engine:7"}}],
        "training": {
            "response_length": 2,
            "runtime_load_spans": [
                {"start": 0, "end": 2, "runtime_load_id": "engine:7"},
            ],
        },
    }

    _weight_hooks().verify_response(_live_artifact("engine:6"), "/v1/chat/completions", response)


@pytest.mark.parametrize(
    "runtime_load_ids",
    [
        ("engine:6", "engine:7", "engine:6"),
        ("engine:7", "engine:6"),
        ("engine:6", "replacement:7"),
    ],
)
def test_weight_surface_rejects_impossible_runtime_load_histories(runtime_load_ids) -> None:
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
        _weight_hooks().verify_response(_live_artifact("engine:6"), "/v1/chat/completions", response)


def test_weight_surface_rejects_token_history_older_than_the_frozen_head() -> None:
    response = {
        "choices": [{"meta_info": {"runtime_load_id": "engine:5"}}],
        "training": {
            "response_length": 1,
            "runtime_load_spans": [{"start": 0, "end": 1, "runtime_load_id": "engine:5"}],
        },
    }

    with pytest.raises(RuntimeLoadMismatch, match="cannot follow frozen"):
        _weight_hooks().verify_response(_live_artifact("engine:6"), "/v1/chat/completions", response)


def test_weight_surface_rejects_opaque_history_for_a_canonical_frozen_head() -> None:
    response = {
        "choices": [{"meta_info": {"runtime_load_id": "opaque-version"}}],
        "training": {
            "response_length": 1,
            "runtime_load_spans": [{"start": 0, "end": 1, "runtime_load_id": "opaque-version"}],
        },
    }

    with pytest.raises(RuntimeLoadMismatch, match="incompatible with canonical"):
        _weight_hooks().verify_response(_live_artifact("engine:6"), "/v1/chat/completions", response)


def test_weight_surface_skips_validation_for_checkpoints(tmp_path) -> None:
    _weight_hooks().verify_response(_checkpoint_artifact(tmp_path), "/v1/chat/completions", {"provider": "ok"})


def test_weight_surface_rejects_a_mismatched_reported_version() -> None:
    surface = create_weight_surface()
    artifact = _live_artifact("wv-7")
    with pytest.raises(RuntimeLoadMismatch, match=r"wv-7.*wv-8|wv-8.*wv-7"):
        _weight_hooks(surface).verify_response(
            artifact, "/v1/chat/completions", {"choices": [{"meta_info": {"runtime_load_id": "wv-8"}}]}
        )
    with pytest.raises(RuntimeLoadMismatch):
        _weight_hooks(surface).verify_response(artifact, "/generate", {"meta_info": {"runtime_load_id": "wv-8"}})


def test_weight_surface_rejects_an_engine_that_reports_no_version() -> None:
    surface = create_weight_surface()
    artifact = _live_artifact("wv-7")
    with pytest.raises(RuntimeLoadMismatch, match="reports no runtime_load_id"):
        _weight_hooks(surface).verify_response(
            artifact, "/v1/chat/completions", {"choices": [{"message": {"content": "hi"}}]}
        )
    with pytest.raises(RuntimeLoadMismatch, match="reports no runtime_load_id"):
        _weight_hooks(surface).verify_response(
            artifact, "/generate", {"meta_info": {"finish_reason": {"type": "stop"}}}
        )


def test_record_only_surface_has_no_inference_hooks() -> None:
    assert Surface().inference is None


def test_inference_injects_and_records_the_post_transform_request(tmp_path) -> None:
    import asyncio

    from reef.artifact import InMemoryRepositoryBackend
    from reef.dispatcher import Dispatcher
    from reef.recipe import RecipeRegistry
    from reef.runtime.inference import InferenceBackend
    from reef.service.app import RequestService

    class SkillRecipe(Recipe):
        def build_surface(self, scenario):
            return create_skill_surface([_InjectingModule()])

    class RecordingBackend(InferenceBackend):
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
    registry = RecipeRegistry({"skill_delivery": recipe})
    dispatcher = Dispatcher(
        registry,
        InMemoryRepositoryBackend.factory(bootstrap, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        agent_record_dir=None,
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
    scenario = dispatcher.get_or_create_scenario("smoke", None)
    assert scenario.surface is scenario.surface


class _StubTrainingRuntime:
    """Satisfies WeightRuntime; only the version probe does anything."""

    def __init__(self, served: str | None):
        self._served = served

    @property
    def base_url(self) -> str:
        return "http://engine.invalid"

    def serving_runtime_load_id(self) -> str | None:
        return self._served

    def restore_checkpoint(self, artifact) -> str:
        raise AssertionError("recover must not restore a checkpoint")


def _checkpoint_ref(version: str = "checkpoint:step-7"):
    from reef.artifact import ArtifactRef

    return ArtifactRef(content_id="artifact:ckpt", release_id=version, parent_release_id=None)


def test_recover_keeps_the_live_head_the_engine_still_holds() -> None:
    assert isinstance(_StubTrainingRuntime("wv-1"), WeightRuntime)
    live = _live_artifact("wv-1").ref
    got = _weight_loader().recover(live, _checkpoint_ref(), _StubTrainingRuntime("wv-1"))
    assert got is live


def test_recover_falls_back_when_the_engine_cannot_serve_the_live_head() -> None:
    # A restarted engine never names the recovered version: issue #138.
    live = _live_artifact("dead-session:2").ref
    checkpoint = _checkpoint_ref()
    got = _weight_loader().recover(live, checkpoint, _StubTrainingRuntime("live-session:1"))
    assert got is checkpoint


def test_recover_leaves_unprobeable_runtimes_alone() -> None:
    live = _live_artifact("wv-1").ref
    assert _weight_loader().recover(live, _checkpoint_ref(), None) is live
    assert _weight_loader().recover(live, _checkpoint_ref(), _StubTrainingRuntime(None)) is live


def test_recover_short_circuits_when_already_at_the_head() -> None:
    surface = create_weight_surface()
    checkpoint = _checkpoint_ref()
    assert _weight_loader(surface).recover(None, checkpoint, _StubTrainingRuntime("wv-1")) is checkpoint
    same = _checkpoint_ref(checkpoint.release_id)
    assert _weight_loader(surface).recover(same, checkpoint, _StubTrainingRuntime("nope")) is checkpoint


def test_weight_hooks_route_each_scenario_to_its_own_adapter_revision() -> None:
    from reef.surface import adapter_name

    hooks = WeightInferenceHooks(scenario="math")
    live = _live_artifact(runtime_load_id="engine:7")
    served = hooks.prepare_request(live, "/v1/chat/completions", {"messages": []})
    assert served["lora_path"] == adapter_name("math", "engine:7")
    assert served["return_meta_info"] is True
    with pytest.raises(ReefError, match="asked for lora_path"):
        hooks.prepare_request(live, "/v1/chat/completions", {"lora_path": adapter_name("code", "engine:7")})


def test_weight_hooks_sample_the_base_before_a_scenario_publishes(tmp_path) -> None:
    hooks = WeightInferenceHooks(scenario="math")
    checkpoint = _checkpoint_artifact(tmp_path)
    assert "lora_path" not in hooks.prepare_request(checkpoint, "/v1/chat/completions", {"messages": []})
    with pytest.raises(ReefError, match="published no adapter yet"):
        hooks.prepare_request(checkpoint, "/v1/chat/completions", {"lora_path": "anything"})


def test_weight_hooks_refuse_both_a_shared_and_a_scenario_adapter() -> None:
    with pytest.raises(ValueError, match="either one shared adapter or per-scenario"):
        WeightInferenceHooks("shared", scenario="math")


def test_weight_loader_recovers_against_the_scenarios_own_adapter_version(tmp_path) -> None:
    class Runtime(_StubTrainingRuntime):
        def __init__(self):
            super().__init__("engine:9")  # some other scenario published last

        def serving_adapter_runtime_load_id(self, scenario):
            return "engine:4" if scenario == "math" else None

    checkpoint = _checkpoint_artifact(tmp_path).ref
    live = _live_artifact(runtime_load_id="engine:4").ref
    assert WeightLoader(scenario="math").recover(live, checkpoint, Runtime()) == live
    # A scenario whose adapter the engine no longer holds falls back.
    assert WeightLoader(scenario="code").recover(live, checkpoint, Runtime()) == checkpoint
    # The unscoped loader still compares against the global version.
    assert WeightLoader().recover(live, checkpoint, Runtime()) == checkpoint
