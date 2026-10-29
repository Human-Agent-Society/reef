"""Public configuration contracts for the managed Semantic Router container and its settings."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
import yaml
from tests.reef_service.routing.routing_fakes import DOMAIN_LABELS, write_domain_classifier

from reef.service.deploy.routing.renderer import ManagedRoutingRender, generate
from reef.service.deploy.routing.settings import RoutingDeploymentSettings, routing_deployment_settings
from reef.service.deploy.service_config import normalize_service_config, service_config_from_mapping
from reef.service.routing import RoutingSnapshot

pytestmark = pytest.mark.unit
LABELS = {
    "io.reef.routing.deployment": "test-routing",
    "io.reef.routing.owner": "test-owner",
    "io.reef.routing.generation": "test-generation",
}


@pytest.fixture
def deployment_manifest(tmp_path: Path) -> Path:
    """Two OPD Scenarios; target names that look like placeholders must stay literal."""
    manifest = {
        "schema_version": 2,
        "automatic_model": "auto",
        "deployment": {"semantic_router_image": "example/router@sha256:" + "a" * 64},
        "targets": {
            "AUTOMATIC_MODEL": {
                "provider_model": "provider-student",
                "revision": "s1",
                "url": "http://127.0.0.1:8101",
            },
            "REEF_TOKEN": {"provider_model": "provider-teacher", "revision": "t1", "url": "http://127.0.0.1:8102"},
            "math-student": {"provider_model": "math-student", "revision": "s1", "url": "http://127.0.0.1:8103"},
            "retired-teacher": {
                "provider_model": "math-teacher",
                "revision": "t0",
                "url": "http://127.0.0.1:8104",
                "enabled": False,
            },
        },
        "scenarios": {
            "code-opd": {
                "default_target": "AUTOMATIC_MODEL",
                "targets": ["AUTOMATIC_MODEL", "REEF_TOKEN"],
                "rules": [
                    {"name": "teacher_requested", "keywords": ["reef_teacher_required"], "target": "REEF_TOKEN"}
                ],
            },
            "math-opd": {"default_target": "math-student", "targets": ["math-student", "retired-teacher"]},
        },
    }
    RoutingSnapshot.from_mapping(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def render(manifest: Path, output: Path, **settings: object) -> RoutingSnapshot:
    policy = RoutingDeploymentSettings(deployment_id="test-routing", **settings)
    return generate(manifest, output, ManagedRoutingRender(policy, "test-routing", "test-owner", "test-generation"))


def test_each_scenario_becomes_a_native_recipe_behind_a_private_entrypoint(
    deployment_manifest: Path, tmp_path: Path
) -> None:
    output = tmp_path / "generated"
    render(deployment_manifest, output)
    router = yaml.safe_load((output / "router.yaml").read_text())

    assert "recipe_selection" not in router["global"]["router"]
    assert router["global"]["router"]["skip_processing"] == {"enabled": False}
    assert router["global"]["router"]["auto_model_names"] == ["auto"]
    # No deployment-wide default model to fall back on outside a Scenario.
    assert router["providers"] == {"models": router["providers"]["models"]}
    assert {model["name"]: model["provider_model_id"] for model in router["providers"]["models"]} == {
        "AUTOMATIC_MODEL": "provider-student",
        "REEF_TOKEN": "provider-teacher",
        "math-student": "math-student",
    }
    # Previews never call a model: without an inference listener, each model is metadata alone, with no backend,
    # endpoint or credential, so the router's hot reload accepts catalog changes.
    assert router["listeners"] == []
    for model in router["providers"]["models"]:
        assert set(model) == {"name", "provider_model_id", "api_format"}
    assert router["entrypoints"] == [
        {"model_names": ["reef-entrypoint/code-opd"], "recipe": "code-opd"},
        {"model_names": ["reef-entrypoint/math-opd"], "recipe": "math-opd"},
    ]
    recipes = {recipe["name"]: recipe for recipe in router["recipes"]}
    assert all(set(recipe) == {"name", "routing"} for recipe in recipes.values())
    teacher, student = recipes["code-opd"]["routing"]["decisions"]
    assert (teacher["name"], teacher["modelRefs"][0]["model"]) == ("teacher_requested", "REEF_TOKEN")
    assert teacher["rules"]["conditions"] == [{"type": "keyword", "name": "teacher_requested"}]
    assert (student["name"], student["modelRefs"][0]["model"]) == ("default", "AUTOMATIC_MODEL")
    assert student["rules"] == {"operator": "AND", "conditions": []}
    assert teacher["priority"] > student["priority"]
    for decision in (teacher, student):
        # One static target and no decision-marker plugins: REEF reads the rule from the preview reply.
        assert decision["algorithm"] == {"type": "static"}
        assert "plugins" not in decision and len(decision["modelRefs"]) == 1
    assert recipes["code-opd"]["routing"]["signals"]["keywords"][0]["keywords"] == ["reef_teacher_required"]
    # The disabled teacher is listed by the Scenario but never referenced by its recipe.
    [math_default] = recipes["math-opd"]["routing"]["decisions"]
    assert math_default["modelRefs"] == [{"model": "math-student", "use_reasoning": False}]
    assert "x-reef-sr-decision" not in (output / "router.yaml").read_text()


def test_keyword_rules_outrank_the_catch_all_at_any_rule_count(tmp_path: Path) -> None:
    rules = [{"name": f"rule-{index}", "keywords": [f"keyword-{index}"], "target": "teacher"} for index in range(1200)]
    manifest = {
        "schema_version": 2,
        "automatic_model": "auto",
        "deployment": {"semantic_router_image": "example/router@sha256:" + "a" * 64},
        "targets": {
            "student": {"provider_model": "student", "revision": "s1", "url": "http://127.0.0.1:8101"},
            "teacher": {"provider_model": "teacher", "revision": "t1", "url": "http://127.0.0.1:8102"},
        },
        "scenarios": {"coding": {"default_target": "student", "targets": ["student", "teacher"], "rules": rules}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    render(path, tmp_path / "out")
    router = yaml.safe_load((tmp_path / "out" / "router.yaml").read_text())
    *keyword, catch_all = router["recipes"][0]["routing"]["decisions"]
    priorities = [decision["priority"] for decision in keyword]
    assert priorities == sorted(priorities, reverse=True) and len(set(priorities)) == len(priorities)
    assert min(priorities) > catch_all["priority"] == 0


@pytest.mark.parametrize("field", ["target", "automatic_model"])
def test_private_entrypoint_names_are_reserved_in_the_manifest(deployment_manifest: Path, field: str) -> None:
    manifest = json.loads(deployment_manifest.read_text())
    if field == "target":
        manifest["targets"]["Reef-Entrypoint/x"] = manifest["targets"].pop("math-student")
        manifest["scenarios"]["math-opd"] = {"default_target": "Reef-Entrypoint/x", "targets": ["Reef-Entrypoint/x"]}
    else:
        manifest["automatic_model"] = "reef-entrypoint/auto"
    with pytest.raises(ValueError, match="reserved prefix"):
        RoutingSnapshot.from_mapping(manifest)


def test_one_container_publishes_only_the_router_http_api_on_loopback(
    deployment_manifest: Path, tmp_path: Path
) -> None:
    output = tmp_path / "managed"
    snapshot = render(deployment_manifest, output, sr_management_port=15053, sr_cpus=2.5)
    compose = yaml.safe_load((output / "compose.yaml").read_text())
    assert set(compose["services"]) == {"semantic-router"}
    router = compose["services"]["semantic-router"]
    assert router["image"] == "example/router@sha256:" + "a" * 64
    assert router["labels"] == compose["networks"]["default"]["labels"] == LABELS
    assert router["restart"] == "no" and router.get("network_mode") is None
    # Only the HTTP API on host loopback; the router's gRPC and metrics ports stay inside the container.
    assert router["ports"] == ["127.0.0.1:15053:8080"]
    assert router["volumes"] == ["./router-config:/app/config"]
    assert router["command"] == ["/app/config/config.yaml"]
    assert router["environment"] == ["REEF_SR_MANAGEMENT_TOKEN", "REEF_SR_SELECTION_TOKEN"]
    assert router["cpus"] == 2.5 and router["mem_limit"] == 4 * 1024**3
    # /ready needs ready.read; the healthcheck names the token variable, never its value.
    assert router["healthcheck"]["test"][0] == "CMD-SHELL"
    assert '"$$REEF_SR_MANAGEMENT_TOKEN"' in router["healthcheck"]["test"][1]
    assert router["healthcheck"]["test"][1].endswith("-H @- http://127.0.0.1:8080/ready")
    mounted = yaml.safe_load((output / "router-config" / "config.yaml").read_text())
    assert mounted == yaml.safe_load((output / "router.yaml").read_text())
    assert RoutingSnapshot.load(output / "manifest.yaml").version == snapshot.version
    assert {path.name for path in output.iterdir()} == {
        "manifest.yaml",
        "router.yaml",
        "compose.yaml",
        "router-config",
    }
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for path in output.rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)
    with pytest.raises(ValueError, match="new or empty"):
        render(deployment_manifest, output)


def test_selection_and_configuration_use_separate_least_privilege_roles(
    deployment_manifest: Path, tmp_path: Path
) -> None:
    output = tmp_path / "managed"
    render(deployment_manifest, output, selection_timeout_seconds=7, preview_max_concurrency=48)
    services = yaml.safe_load((output / "router.yaml").read_text())["global"]["services"]
    assert services["management_api"] == {
        "bind_address": "0.0.0.0",
        "port": 8080,
        "remote_exposure": True,
        "auth": {
            "mode": "bearer",
            "tokens": [
                {"env": "REEF_SR_MANAGEMENT_TOKEN", "role": "reef-policy"},
                {"env": "REEF_SR_SELECTION_TOKEN", "role": "reef-selection"},
            ],
            "roles": {
                "reef-policy": ["config.read", "config.write", "ready.read"],
                "reef-selection": ["classify.invoke"],
            },
        },
    }
    assert services["api"]["routing_preview"] == {"request_timeout_seconds": 7, "max_concurrency": 48}


def test_snapshot_tracks_router_settings_but_not_the_output_directory(
    deployment_manifest: Path, tmp_path: Path
) -> None:
    first = render(deployment_manifest, tmp_path / "first")
    moved = render(deployment_manifest, tmp_path / "moved")
    wider = render(deployment_manifest, tmp_path / "wider", preview_max_concurrency=64)
    slower = render(deployment_manifest, tmp_path / "slower", selection_timeout_seconds=20)
    assert first.version == moved.version
    # Preview worker limits are part of the frozen configuration; changing one needs a restart.
    assert len({first.version, wider.version, slower.version}) == 3


def test_renderer_does_not_follow_output_symlinks(deployment_manifest: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "generated"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        render(deployment_manifest, link)
    assert list(outside.iterdir()) == []


def policy_manifest(path: Path, classifier: dict[str, object] | None, **domain_rule: object) -> Path:
    """Two Scenarios with every keyword option and, given ``classifier``, one domain rule last in each."""
    keyword_rules: list[dict[str, object]] = [
        {"name": "teacher_requested", "keywords": ["reef_teacher_required"], "target": "teacher"},
        {"name": "any_words", "keywords": ["urgenttrigger", "escalatetrigger"], "match": "any", "target": "teacher"},
        {"name": "both_words", "keywords": ["redflag", "escalate"], "match": "all", "target": "teacher"},
        {"name": "exact_case", "keywords": ["CaseSensitiveGate"], "case_sensitive": True, "target": "teacher"},
    ]
    deployment: dict[str, object] = {"semantic_router_image": "example/router@sha256:" + "a" * 64}
    code_rules, math_rules = list(keyword_rules), []
    if classifier is not None:
        deployment["domain_classifier"] = classifier
        code_rules.append({"name": "math_request", "domains": ["math"], "target": "teacher", **domain_rule})
        math_rules.append({"name": "math_request", "domains": ["math", "physics"], "target": "math-teacher"})
    manifest = {
        "schema_version": 2,
        "automatic_model": "auto",
        "deployment": deployment,
        "targets": {
            name: {"provider_model": name, "revision": "r1", "url": f"http://127.0.0.1:{port}"}
            for port, name in enumerate(["student", "teacher", "math-student", "math-teacher"], start=8101)
        },
        "scenarios": {
            "code-opd": {"default_target": "student", "targets": ["student", "teacher"], "rules": code_rules},
            "math-opd": {
                "default_target": "math-student",
                "targets": ["math-student", "math-teacher"],
                "rules": math_rules,
            },
        },
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.mark.parametrize("code_rule_name", ["math_request", "algebra_request"])
def test_keyword_options_and_domain_rules_become_native_signals_in_rule_order(
    tmp_path: Path, code_rule_name: str
) -> None:
    classifier = write_domain_classifier(tmp_path / "vela-domain")
    output = tmp_path / "generated"
    snapshot = render(policy_manifest(tmp_path / "manifest.json", classifier, name=code_rule_name), output)
    router = yaml.safe_load((output / "router.yaml").read_text())
    recipes = {recipe["name"]: recipe["routing"] for recipe in router["recipes"]}
    keyword_signals = [
        ("teacher_requested", "OR", ["reef_teacher_required"], False),
        ("any_words", "OR", ["urgenttrigger", "escalatetrigger"], False),
        ("both_words", "AND", ["redflag", "escalate"], False),
        ("exact_case", "OR", ["CaseSensitiveGate"], True),
    ]
    assert recipes["code-opd"]["signals"] == {
        "keywords": [
            {"name": name, "operator": operator, "keywords": keywords, "case_sensitive": case_sensitive}
            for name, operator, keywords, case_sensitive in keyword_signals
        ],
        "domains": [{"name": code_rule_name, "mmlu_categories": ["math"]}],
    }
    decisions = recipes["code-opd"]["decisions"]
    assert [(decision["name"], decision["rules"]["conditions"]) for decision in decisions] == [
        ("teacher_requested", [{"type": "keyword", "name": "teacher_requested"}]),
        ("any_words", [{"type": "keyword", "name": "any_words"}]),
        ("both_words", [{"type": "keyword", "name": "both_words"}]),
        ("exact_case", [{"type": "keyword", "name": "exact_case"}]),
        (code_rule_name, [{"type": "domain", "name": code_rule_name}]),
        ("default", []),
    ]
    # Earlier rules outrank later ones whatever their kind; every decision keeps one static target.
    priorities = [decision["priority"] for decision in decisions]
    assert priorities == sorted(priorities, reverse=True) and len(set(priorities)) == len(priorities)
    for decision in decisions:
        assert decision["algorithm"] == {"type": "static"} and len(decision["modelRefs"]) == 1
    math_signal = {"name": "math_request", "mmlu_categories": ["math", "physics"]}
    assert recipes["math-opd"]["signals"] == {"domains": [math_signal]}
    catalog = router["global"]["model_catalog"]
    assert catalog["system"] == {"domain_classifier": "/models/reef-domain"}
    assert catalog["modules"]["classifier"]["domain"] == {
        "enabled": True,
        "model_ref": "domain_classifier",
        "model_id": "/models/reef-domain",
        "category_mapping_path": "/models/reef-domain/category_mapping.json",
        "use_cpu": True,
        "threshold": 0.5,
        "fallback_category": "other",
    }
    # Every other learned module stays disabled.
    assert catalog["modules"]["prompt_guard"]["enabled"] is False
    assert catalog["modules"]["feedback_detector"]["enabled"] is False
    assert catalog["modules"]["classifier"]["pii"] == {"model_ref": "", "model_id": "", "pii_mapping_path": ""}
    assert router["global"]["router"]["learning"] == {"enabled": False}
    # The router configuration names only the container path; the host directory appears in Compose alone.
    assert str(classifier["model_directory"]) not in (output / "router.yaml").read_text()
    service = yaml.safe_load((output / "compose.yaml").read_text())["services"]["semantic-router"]
    assert service["volumes"] == ["./router-config:/app/config"] + [
        {
            "type": "bind",
            "source": f"{classifier['model_directory']}/{name}",
            "target": f"/models/reef-domain/{name}",
            "read_only": True,
            "bind": {"create_host_path": False},
        }
        for name in sorted(["config.json", "tokenizer.json", "model.safetensors", "category_mapping.json"])
    ]
    assert service["environment"] == [
        "REEF_SR_MANAGEMENT_TOKEN",
        "REEF_SR_SELECTION_TOKEN",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "HF_DATASETS_OFFLINE=1",
    ]
    assert snapshot.domain_classifier is not None
    assert RoutingSnapshot.load(output / "manifest.yaml").domain_classifier == snapshot.domain_classifier


def test_classifier_mount_paths_keep_literal_dollars_for_compose(tmp_path: Path) -> None:
    classifier = write_domain_classifier(tmp_path / "${REEF_TEST_COMPONENT}-$literal")
    output = tmp_path / "generated"
    snapshot = render(policy_manifest(tmp_path / "manifest.json", classifier), output)
    service = yaml.safe_load((output / "compose.yaml").read_text())["services"]["semantic-router"]
    assert snapshot.domain_classifier is not None
    assert snapshot.domain_classifier.model_directory == classifier["model_directory"]
    for volume in service["volumes"][1:]:
        name = Path(volume["target"]).name
        assert volume["source"] == f"{classifier['model_directory']}/{name}".replace("$", "$$")


def test_a_manifest_without_a_classifier_keeps_every_learned_module_disabled(tmp_path: Path) -> None:
    output = tmp_path / "generated"
    render(policy_manifest(tmp_path / "manifest.json", None), output)
    catalog = yaml.safe_load((output / "router.yaml").read_text())["global"]["model_catalog"]
    assert set(catalog) == {"embeddings", "modules"}
    assert catalog["modules"]["classifier"]["domain"] == {"model_ref": "", "model_id": "", "category_mapping_path": ""}
    service = yaml.safe_load((output / "compose.yaml").read_text())["services"]["semantic-router"]
    assert service["volumes"] == ["./router-config:/app/config"]
    assert service["environment"] == ["REEF_SR_MANAGEMENT_TOKEN", "REEF_SR_SELECTION_TOKEN"]


def break_classifier(directory: Path, fault: str, classifier: dict[str, object]) -> dict[str, object]:
    """Apply one fault to a written stand-in classifier and return the manifest value to render with."""
    files = dict(json.loads(json.dumps(classifier["files"])))
    if fault == "missing_directory":
        return {**classifier, "model_directory": str(directory) + "-missing"}
    if fault == "symlinked_directory":
        link = directory.parent / "linked-domain"
        link.symlink_to(directory, target_is_directory=True)
        return {**classifier, "model_directory": str(link)}
    if fault == "symlinked_file":
        copy = directory.parent / "config-copy.json"
        copy.write_bytes((directory / "config.json").read_bytes())
        (directory / "config.json").unlink()
        (directory / "config.json").symlink_to(copy)
    elif fault == "fifo":
        (directory / "tokenizer.json").unlink()
        os.mkfifo(directory / "tokenizer.json")
    elif fault == "directory_as_file":
        (directory / "model.safetensors").unlink()
        (directory / "model.safetensors").mkdir()
    elif fault == "missing_file":
        (directory / "category_mapping.json").unlink()
    elif fault == "changed_weights":
        (directory / "model.safetensors").write_text("other weights")
    elif fault == "oversized_labels":
        text = json.dumps({"id2label": {"0": "other"}, "padding": "x" * (1024 * 1024)})
        (directory / "config.json").write_text(text)
        files["config.json"] = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    return {**classifier, "files": files}


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("missing_directory", "existing directory without symlinks"),
        ("symlinked_directory", "existing directory without symlinks"),
        ("symlinked_file", "config.json cannot be read as a regular file without a symlink"),
        ("fifo", "tokenizer.json must be a regular file"),
        ("directory_as_file", "model.safetensors must be a regular file"),
        ("missing_file", "category_mapping.json cannot be read"),
        ("changed_weights", "model.safetensors does not match its sha256 digest"),
        ("oversized_labels", "config.json is larger than"),
    ],
)
def test_classifier_files_are_checked_on_the_host_before_any_file_is_written(
    tmp_path: Path, fault: str, message: str
) -> None:
    directory = tmp_path / "vela-domain"
    classifier = break_classifier(directory, fault, write_domain_classifier(directory))
    output = tmp_path / "generated"
    with pytest.raises(ValueError, match=message):
        render(policy_manifest(tmp_path / "manifest.json", classifier), output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("labels", "mapping_labels", "rule", "message"),
    [
        (DOMAIN_LABELS, DOMAIN_LABELS[::-1], {}, "same order"),
        (DOMAIN_LABELS, DOMAIN_LABELS[:-1], {}, "same order"),
        (tuple(label for label in DOMAIN_LABELS if label != "other"), None, {}, "label 'other'"),
        ((*DOMAIN_LABELS, "math"), None, {}, "repeats a label"),
        (DOMAIN_LABELS, None, {"domains": ["astronomy"]}, "does not have"),
        (DOMAIN_LABELS, None, {"domains": ["Math"]}, "does not have"),
        # The router reports an unconfigured predicted label under its own name, which would match this rule.
        (DOMAIN_LABELS, None, {"name": "physics", "domains": ["chemistry"]}, "named like a classifier label"),
    ],
    ids=["reordered", "shorter", "no_fallback", "repeated", "unknown", "other_case", "rule_named_like_label"],
)
def test_domain_labels_must_agree_between_the_classifier_files_and_the_rules(
    tmp_path: Path,
    labels: tuple[str, ...],
    mapping_labels: tuple[str, ...] | None,
    rule: dict[str, object],
    message: str,
) -> None:
    classifier = write_domain_classifier(tmp_path / "vela-domain", labels, mapping_labels=mapping_labels)
    output = tmp_path / "generated"
    with pytest.raises(ValueError, match=message):
        render(policy_manifest(tmp_path / "manifest.json", classifier, **rule), output)
    assert not output.exists()


def test_managed_settings_parse_at_the_service_boundary() -> None:
    default = service_config_from_mapping({"reef": {"recipe": "recipe"}})
    # Non-routing deployments construct default settings and need no identity.
    assert default.routing_deployment == RoutingDeploymentSettings()
    config = {
        "reef": {
            "recipe": "recipe",
            "routing_deployment": {
                "deployment-id": "test-routing",
                "restart_policy": "bounded",
                "restart_backoff_seconds": [0.1, 0.2],
                "sr_management_port": 19000,
                "preview_max_concurrency": 64,
            },
        },
    }
    settings = service_config_from_mapping(config).routing_deployment
    assert settings.require_identity() == "test-routing"
    assert settings.restart_backoff_seconds == (0.1, 0.2)
    assert (settings.sr_management_port, settings.preview_max_concurrency) == (19000, 64)
    with pytest.raises(ValueError, match="deployment_id"):
        RoutingDeploymentSettings().require_identity()


def test_unknown_routing_settings_are_named_without_their_values() -> None:
    with pytest.raises(ValueError, match="unknown config fields: client_token") as error:
        routing_deployment_settings({"deployment_id": "valid", "client-token": "secret-value"})
    assert "secret-value" not in str(error.value)


@pytest.mark.parametrize(
    "settings",
    [
        {"deployment_id": "../outside"},
        {"deployment_id": "valid", "typo": 1},
        {"restart_policy": "always"},
        {"restart_policy": "bounded", "max_restarts": 0},
        {"restart_backoff_seconds": [-1]},
        {"restart_backoff_seconds": [True]},
        {"restart_backoff_seconds": [float("nan")]},
        {"ready_timeout_seconds": 0},
        {"drain_timeout_seconds": -1},
        {"sr_cpus": 0},
        {"sr_memory_bytes": -1},
        {"sr_management_port": 65536},
        {"sr_management_port": 0},
        {"selection_timeout_seconds": 0},
        {"selection_timeout_seconds": 301},
        {"selection_timeout_seconds": 1.5},
        {"preview_max_concurrency": 0},
        {"deployment-id": "valid", "deployment_id": "valid"},
    ],
)
def test_invalid_routing_settings_fail_before_launch(settings: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        normalize_service_config({"reef": {"routing_deployment": settings}})


def test_directly_made_routing_values_keep_their_definitions() -> None:
    # The public parser turns the backoff list into a tuple; a frozen value never keeps a caller's list.
    with pytest.raises(ValueError, match="tuple"):
        RoutingDeploymentSettings(restart_backoff_seconds=[1.0])
    policy = RoutingDeploymentSettings(deployment_id="test-routing")
    for deployment_id, owner_id, generation in (("other", "owner", "one"), ("test-routing", "", "one")):
        with pytest.raises(ValueError):
            ManagedRoutingRender(policy, deployment_id, owner_id, generation)
    # Candidate validation renders with placeholder labels, not registry identities.
    assert ManagedRoutingRender(policy, "test-routing", "validation", "validation").owner_id == "validation"


def test_the_documented_validate_command_reads_a_public_configuration_without_docker(
    deployment_manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from reef.service.deploy.routing import cli

    def no_docker(*args: object, **kwargs: object) -> None:
        pytest.fail("validate must not use Docker")

    monkeypatch.setattr(cli.DockerComposeRuntime, "ensure_available", no_docker)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("REEF_TOKEN", "validate-test-token")
    config = tmp_path / "reef.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "reef": {
                    "host": "127.0.0.1",
                    "tokens": ["${REEF_TOKEN:?}"],
                    "routing_manifest": str(deployment_manifest),
                    "routing_deployment": {"deployment_id": "validate-test"},
                },
                "recipe": {"implementation": "recipe"},
                "inference": {"upstream_url": "http://127.0.0.1:8101", "upstream_model": "provider-student"},
            }
        )
    )
    cli.main(["validate", "--config", str(config)])
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True and result["docker_mutations"] == 0
    assert not (tmp_path / "state").exists()
