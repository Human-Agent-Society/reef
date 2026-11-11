"""Ownership, command lifetime and event contracts for managed routing."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import yaml
from tests.reef_service.routing.routing_fakes import routing_manifest

from reef.service.deploy.routing.containers import (
    BUILD_RECIPE_LABEL,
    IMAGE_BUILD_RECIPE,
    IMAGE_CACHE_TAG,
    IMAGE_IDENTITY_FORMAT,
    REVISION_LABEL,
    SEMANTIC_ROUTER_BUILD_CONTEXT,
    SEMANTIC_ROUTER_REVISION,
    ContainerCommandError,
    ContainerEvent,
    ContainerInventory,
    ContainerState,
    DockerComposeRuntime,
    DockerEventStream,
    GuardedCommand,
    image_build_arguments,
)
from reef.service.deploy.routing.registry import DeploymentOwnership, DeploymentRegistry, RoutingOwnershipError
from reef.service.deploy.routing.renderer import (
    MANAGEMENT_TOKEN_ENV,
    SELECTION_TOKEN_ENV,
    ManagedRoutingRender,
    generate,
)
from reef.service.deploy.routing.settings import RoutingDeploymentSettings

pytestmark = pytest.mark.unit


@pytest.fixture
def ownership(tmp_path: Path) -> DeploymentOwnership:
    return DeploymentOwnership("test-routing", "a" * 32, "first-generation", tmp_path)


class FakeDocker(DockerComposeRuntime):
    """The real runtime over an in-memory Docker: ``run`` answers CLI commands without starting any.

    ``containers`` maps container IDs to Compose services and ``networks`` holds network IDs, all labeled
    as ``ownership``'s; ``foreign`` and ``paused`` change what container inspection reports, and
    ``image_labels`` is the image inspection output. ``container rm`` and ``network rm`` delete the resource
    they name; they and every other command without an answer here return empty output. ``commands`` logs
    every CLI command in order.
    """

    def __init__(self, ownership: DeploymentOwnership) -> None:
        super().__init__()
        self.commands: list[list[str]] = []
        self.container_labels = {**ownership.labels, "com.docker.compose.project": ownership.project_name}
        self.containers: dict[str, str] = {"student-router": "semantic-router"}
        self.networks = {"routing-network"}
        self.foreign = False
        self.paused = False
        self.image_labels = json.dumps({"org.opencontainers.image.revision": SEMANTIC_ROUTER_REVISION})

    def run(self, arguments: Sequence[str], timeout_seconds: float = 5, *, mutation: bool = False) -> str:
        self.commands.append(list(arguments))
        if arguments[:2] == ["image", "inspect"]:
            return self.image_labels
        if arguments[:2] == ["container", "ls"]:
            return "\n".join(self.containers)
        if arguments[:2] == ["network", "ls"]:
            return "\n".join(self.networks)
        if arguments[:2] == ["container", "inspect"]:
            records = []
            for resource_id in arguments[2:]:
                labels = {**self.container_labels, "com.docker.compose.service": self.containers[resource_id]}
                if self.foreign:
                    labels["io.reef.routing.owner"] = "foreign"
                records.append(
                    {
                        "Id": resource_id,
                        "Config": {"Labels": labels},
                        "State": {"Running": True, "Paused": self.paused, "Health": {"Status": "healthy"}},
                    }
                )
            return json.dumps(records)
        if arguments[:2] == ["network", "inspect"]:
            return json.dumps([{"Id": name, "Labels": self.container_labels} for name in arguments[2:]])
        if arguments[:2] == ["container", "rm"]:
            self.containers.pop(arguments[-1])
        if arguments[:2] == ["network", "rm"]:
            self.networks.remove(arguments[-1])
        return ""


def test_registry_lock_is_stable_across_run_directories(tmp_path: Path) -> None:
    first = DeploymentRegistry("shared", root=tmp_path)
    second = DeploymentRegistry("shared", root=tmp_path)
    with first:
        generation = first.prepare("one", tmp_path / "first-run")
        with pytest.raises(RoutingOwnershipError, match="already controlled"):
            second.acquire()
        assert first.lock_descriptor is not None
        assert not os.get_inheritable(first.lock_descriptor)
        assert stat.S_IMODE(first.directory.stat().st_mode) == 0o700
        assert stat.S_IMODE((first.directory / "owner.json").stat().st_mode) == 0o600
    with second:
        resumed = second.prepare("two", tmp_path / "different-run")
        assert resumed.owner_id == generation.owner_id
        assert resumed.project_name != generation.project_name
        assert second.load() == resumed


def test_registry_rejects_symlink_and_foreign_owner_state(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "linked").symlink_to(target, target_is_directory=True)
    with pytest.raises(RoutingOwnershipError):
        DeploymentRegistry("linked", root=tmp_path).acquire()
    with DeploymentRegistry("safe", root=tmp_path) as registry:
        registry.prepare("one", tmp_path / "generation")
        path = registry.directory / "owner.json"
        path.chmod(0o644)
        with pytest.raises(RoutingOwnershipError, match="private"):
            registry.load()


def test_registry_cannot_mutate_without_lock_or_adopt_new_owner(tmp_path: Path) -> None:
    registry = DeploymentRegistry("safe", root=tmp_path)
    with pytest.raises(RoutingOwnershipError, match="lock"):
        registry.prepare("one", tmp_path)
    with registry:
        registry.prepare("one", tmp_path)
        with pytest.raises(RoutingOwnershipError, match="stable"):
            registry.save(DeploymentOwnership("safe", "b" * 32, "two", tmp_path))


@pytest.mark.parametrize("deployment_id", ["../outside", "", "a/b", "Uppercase", "-bad", "x" * 49])
def test_registry_rejects_invalid_identity(deployment_id: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        DeploymentRegistry(deployment_id, root=tmp_path)


def test_cleanup_validates_before_any_mutation_and_is_idempotent(ownership: DeploymentOwnership) -> None:
    runtime = FakeDocker(ownership)
    runtime.foreign = True
    with pytest.raises(RoutingOwnershipError):
        runtime.remove(ownership, 0.1)
    assert not any(command[0] in ("stop", "unpause") or "rm" in command for command in runtime.commands)
    runtime.foreign = False
    runtime.remove(ownership, 0.1)
    runtime.remove(ownership, 0.1)
    removals = [command for command in runtime.commands if command[:2] == ["container", "rm"]]
    # The verified container goes with its anonymous volumes (the image's /app/models); nothing else does.
    assert removals == [["container", "rm", "--force", "--volumes", "student-router"]]
    assert not runtime.containers and not runtime.networks
    assert all("down" not in command and command[:1] != ["volume"] for command in runtime.commands)


def test_a_second_container_in_the_owned_group_is_refused(ownership: DeploymentOwnership) -> None:
    runtime = FakeDocker(ownership)
    runtime.containers["ingress"] = "envoy"
    with pytest.raises(ContainerCommandError, match="unexpected managed routing container"):
        runtime.inspect(ownership)


def test_paused_owned_containers_unpause_before_stop(ownership: DeploymentOwnership) -> None:
    runtime = FakeDocker(ownership)
    runtime.paused = True
    runtime.remove(ownership, 1)
    mutation_commands = [command for command in runtime.commands if command[0] in ("unpause", "stop")]
    assert [command[0] for command in mutation_commands] == ["unpause", "stop"]


def test_start_refuses_unreconciled_existing_resources(ownership: DeploymentOwnership) -> None:
    runtime = FakeDocker(ownership)
    with pytest.raises(ContainerCommandError, match="reconcile"):
        runtime.start(ownership.configuration_directory / "compose.yaml", ownership, 5)
    assert all(command[0] != "compose" for command in runtime.commands)


def test_start_inspects_the_image_from_the_generated_compose(ownership: DeploymentOwnership) -> None:
    manifest_path = ownership.configuration_directory / "input.yaml"
    manifest = routing_manifest("http://127.0.0.1:8101")
    manifest_path.write_text(yaml.safe_dump(manifest))
    output = ownership.configuration_directory / "generated"
    generate(
        manifest_path,
        output,
        ManagedRoutingRender(
            RoutingDeploymentSettings(deployment_id=ownership.deployment_id),
            ownership.deployment_id,
            ownership.owner_id,
            ownership.generation,
        ),
    )
    runtime = FakeDocker(ownership)
    runtime.containers.clear()
    runtime.networks.clear()
    runtime.start(output / "compose.yaml", ownership, 5)
    image = yaml.safe_load(manifest_path.read_text())["deployment"]["semantic_router_image"]
    assert ["image", "inspect", "--format", "{{json .Config.Labels}}", image] in runtime.commands
    assert any(command[0] == "compose" for command in runtime.commands)


@pytest.mark.parametrize(
    "labels",
    [
        {"org.opencontainers.image.revision": SEMANTIC_ROUTER_REVISION},
        {},
        {"org.opencontainers.image.revision": "f" * 40},
        {"org.opencontainers.image.revision": SEMANTIC_ROUTER_REVISION[:12]},
        {"org.opencontainers.image.revision": None},
        None,
        [],
    ],
)
def test_start_checks_the_digest_image_revision_before_creating_resources(
    ownership: DeploymentOwnership, labels: dict[str, str | None] | list[object] | None
) -> None:
    runtime = FakeDocker(ownership)
    runtime.containers.clear()
    runtime.networks.clear()
    runtime.image_labels = json.dumps(labels)
    image = "example/router@sha256:" + "a" * 64
    compose = ownership.configuration_directory / "compose.yaml"
    compose.write_text(yaml.safe_dump({"services": {"semantic-router": {"image": image}}}))
    if labels == {"org.opencontainers.image.revision": SEMANTIC_ROUTER_REVISION}:
        runtime.start(compose, ownership, 5)
        create = next(command for command in runtime.commands if command[0] == "compose")
        assert create[-5:] == ["up", "--detach", "--no-build", "--pull", "never"]
        assert runtime.commands.index(create) > next(
            index for index, command in enumerate(runtime.commands) if command[:2] == ["image", "inspect"]
        )
    else:
        with pytest.raises(ContainerCommandError, match="revision") as error:
            runtime.start(compose, ownership, 5)
        assert not error.value.outcome_unknown
        assert not any(command[0] == "compose" for command in runtime.commands)
    assert [command for command in runtime.commands if command[:2] == ["image", "inspect"]] == [
        ["image", "inspect", "--format", "{{json .Config.Labels}}", image]
    ]
    assert not runtime.mutation_pending(ownership)


@pytest.mark.parametrize("failure", ["missing-image", "malformed-labels", "timeout"])
def test_image_inspection_failure_never_submits_container_creation(
    ownership: DeploymentOwnership, failure: str
) -> None:
    class UnavailableImage(FakeDocker):
        """A Docker whose image inspection fails as ``failure`` names; other commands are unchanged."""

        def run(self, arguments: Sequence[str], timeout_seconds: float = 5, *, mutation: bool = False) -> str:
            if arguments[:2] == ["image", "inspect"]:
                if failure == "malformed-labels":
                    return "secret-marker not JSON"
                raise ContainerCommandError(failure)
            return super().run(arguments, timeout_seconds, mutation=mutation)

    runtime = UnavailableImage(ownership)
    runtime.containers.clear()
    runtime.networks.clear()
    compose = ownership.configuration_directory / "compose.yaml"
    compose.write_text(
        yaml.safe_dump({"services": {"semantic-router": {"image": "example/router@sha256:" + "a" * 64}}})
    )
    with pytest.raises(ContainerCommandError) as error:
        runtime.start(compose, ownership, 5)
    assert not error.value.outcome_unknown
    assert "secret-marker" not in str(error.value)
    assert not any(command[0] == "compose" for command in runtime.commands)
    assert not runtime.mutation_pending(ownership)


#: A ``docker`` for the image tests. It answers from ``docker-state.json`` beside it and logs each call, with the
#: names of the router token variables it received, to ``docker-calls.jsonl``. ``image ls`` lists ``listed`` for a
#: tag lookup and ``inventory`` for ``--all``; either fails as the daemon would when ``listing_fails`` or
#: ``inventory_fails`` says. A build records its PID and the mode of its metadata file's directory, prints progress
#: to stdout and stderr, then hangs, fails or writes ``metadata`` (a JSON object, raw text, or nothing for None) as
#: ``build`` says.
FAKE_DOCKER_PROGRAM = """
import json, os, sys, time
from pathlib import Path

here = Path(__file__).resolve().parent
state = json.loads((here / "docker-state.json").read_text())
arguments = sys.argv[1:]
tokens = sorted(name for name in os.environ if name.startswith("REEF_SR_"))
with (here / "docker-calls.jsonl").open("a") as calls:
    calls.write(json.dumps({"arguments": arguments, "tokens": tokens}) + "\\n")
if arguments[:2] == ["image", "ls"]:
    inventory = "--all" in arguments
    if state["inventory_fails" if inventory else "listing_fails"]:
        print("Cannot connect to the Docker daemon", file=sys.stderr)
        sys.exit(1)
    print("\\n".join(state["inventory" if inventory else "listed"]))
elif arguments[:2] == ["image", "inspect"]:
    identity = state["images"].get(arguments[-1])
    if identity is None:
        print("Error: No such image", file=sys.stderr)
        sys.exit(1)
    print(identity if isinstance(identity, str) else json.dumps(identity))
elif arguments[:2] == ["buildx", "version"]:
    sys.exit(state["buildx_status"])
elif arguments[:2] == ["buildx", "build"]:
    (here / "build.pid").write_text(str(os.getpid()))
    metadata_file = Path(arguments[arguments.index("--metadata-file") + 1])
    (here / "metadata-directory-mode").write_text(oct(metadata_file.parent.stat().st_mode & 0o777))
    print("#1 [internal] load build definition secret-marker", flush=True)
    print("#1 DONE 0.1s", file=sys.stderr, flush=True)
    if state["build"] == "hang":
        time.sleep(60)
    if state["build"] == "fail":
        sys.exit(3)
    metadata = state["metadata"]
    if metadata is not None:
        metadata_file.write_text(metadata if isinstance(metadata, str) else json.dumps(metadata))
else:
    sys.exit(64)
"""
CACHED_ID = "sha256:" + "c" * 64
#: The digests in a build's metadata. Docker's classic image store names the image by its config digest, the
#: containerd image store by its manifest digest.
CONFIG_ID = "sha256:" + "b" * 64
MANIFEST_ID = "sha256:" + "d" * 64
#: An image of another startup's build, which the shared cache tag may name meanwhile.
OTHER_ID = "sha256:" + "9" * 64
BUILD_METADATA = {
    "containerimage.config.digest": CONFIG_ID,
    "containerimage.digest": MANIFEST_ID,
    "image.name": "secret-marker other metadata is ignored",
}
RECIPE_LABELS = {REVISION_LABEL: SEMANTIC_ROUTER_REVISION, BUILD_RECIPE_LABEL: IMAGE_BUILD_RECIPE}


def fake_docker(directory: Path, **state: object) -> DockerComposeRuntime:
    """The real runtime over ``FAKE_DOCKER_PROGRAM`` in ``directory``, with ``state`` over a missing image that a
    build loads into a classic image store."""
    directory.mkdir(exist_ok=True)
    defaults = {
        "listing_fails": False,
        "listed": [],
        "inventory_fails": False,
        "inventory": [CACHED_ID, CONFIG_ID],
        "images": {CONFIG_ID: {"id": CONFIG_ID, "labels": RECIPE_LABELS}},
        "buildx_status": 0,
        "build": "ok",
        "metadata": BUILD_METADATA,
    }
    (directory / "docker-state.json").write_text(json.dumps({**defaults, **state}))
    executable = directory / "docker"
    executable.write_text(f"#!{sys.executable}\n{FAKE_DOCKER_PROGRAM}")
    executable.chmod(0o700)
    return DockerComposeRuntime(str(executable))


def docker_calls(directory: Path) -> list[dict[str, list[str]]]:
    path = directory / "docker-calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def never_stop() -> bool:
    return False


def wait_until_gone(pid: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    pytest.fail("the image build outlived its command")


def test_the_build_recipe_and_cache_tag_name_the_pinned_source_and_every_build_option() -> None:
    assert SEMANTIC_ROUTER_BUILD_CONTEXT == (
        "https://github.com/vllm-project/semantic-router.git#" + SEMANTIC_ROUTER_REVISION
    )
    assert re.fullmatch(r"[0-9a-f]{16}", IMAGE_BUILD_RECIPE)
    tag = IMAGE_CACHE_TAG
    assert tag == f"reef-semantic-router:{SEMANTIC_ROUTER_REVISION[:12]}-{IMAGE_BUILD_RECIPE}"
    # The recipe hashes the build's own arguments, so changing any build option changes the recipe and the tag.
    template = json.dumps(image_build_arguments("{metadata_file}", "{recipe}", "{tag}")).encode()
    recipe = IMAGE_BUILD_RECIPE
    assert recipe == hashlib.sha256(template).hexdigest()[:16]


def test_a_cached_image_with_this_recipes_labels_is_used_without_a_build(tmp_path: Path) -> None:
    docker = tmp_path / "docker"
    identity = {"id": CACHED_ID, "labels": {**RECIPE_LABELS, "maintainer": "other labels stay allowed"}}
    runtime = fake_docker(docker, listed=[CACHED_ID], images={IMAGE_CACHE_TAG: identity})
    log = tmp_path / "build.log"
    assert runtime.resolve_image(log, 30, never_stop) == CACHED_ID
    # One listing and one inspection that returns the ID and labels together; no Buildx, Git or network.
    assert [call["arguments"] for call in docker_calls(docker)] == [
        ["image", "ls", "--quiet", "--no-trunc", "--filter", f"reference={IMAGE_CACHE_TAG}"],
        ["image", "inspect", "--format", IMAGE_IDENTITY_FORMAT, IMAGE_CACHE_TAG],
    ]
    assert not log.exists()


@pytest.mark.parametrize(
    ("store", "metadata"),
    [
        ("classic", BUILD_METADATA),
        ("containerd", BUILD_METADATA),
        ("legacy_classic", {"containerimage.digest": CONFIG_ID}),
    ],
    ids=["classic", "containerd", "legacy_classic"],
)
def test_a_missing_image_is_built_from_the_pinned_commit_and_bound_to_its_own_local_id(
    tmp_path: Path, store: str, metadata: dict[str, str]
) -> None:
    docker = tmp_path / "docker"
    # The classic image store names the built image by its config digest; the containerd image store by its
    # manifest digest, and there the config digest names no image. Meanwhile another startup's build moved the
    # shared cache tag to its own image, which stays local: this build never reads the tag.
    # Older classic exporters emit only containerimage.digest, containing the config ID.
    built = MANIFEST_ID if store == "containerd" else CONFIG_ID
    images = {
        built: {"id": built, "labels": RECIPE_LABELS},
        OTHER_ID: {"id": OTHER_ID, "labels": RECIPE_LABELS},
        IMAGE_CACHE_TAG: {"id": OTHER_ID, "labels": RECIPE_LABELS},
    }
    runtime = fake_docker(docker, inventory=[CACHED_ID, OTHER_ID, built, built], images=images, metadata=metadata)
    runtime.use_environment({MANAGEMENT_TOKEN_ENV: "management-secret", SELECTION_TOKEN_ENV: "selection-secret"})
    log = tmp_path / "build.log"
    assert runtime.resolve_image(log, 30, never_stop) == built
    calls = docker_calls(docker)
    assert [call["arguments"][:2] for call in calls] == [
        ["image", "ls"],
        ["buildx", "version"],
        ["buildx", "build"],
        ["image", "ls"],
        ["image", "inspect"],
    ]
    build = calls[2]["arguments"]
    metadata_file = build[build.index("--metadata-file") + 1]
    assert build == [
        "buildx",
        "build",
        "--load",
        "--progress",
        "plain",
        "--metadata-file",
        metadata_file,
        "-f",
        "src/vllm-sr/Dockerfile",
        "--label",
        f"org.opencontainers.image.revision={SEMANTIC_ROUTER_REVISION}",
        "--label",
        f"io.reef.semantic-router.build={IMAGE_BUILD_RECIPE}",
        "-t",
        IMAGE_CACHE_TAG,
        f"https://github.com/vllm-project/semantic-router.git#{SEMANTIC_ROUTER_REVISION}",
    ]
    # The router tokens never reach the build, and its metadata file was private and is gone.
    assert calls[2]["tokens"] == []
    assert (docker / "metadata-directory-mode").read_text() == "0o700"
    assert not Path(metadata_file).parent.exists()
    # A complete listing of local image IDs finds the build's own ID, which one inspection then checks.
    assert calls[3]["arguments"] == ["image", "ls", "--all", "--quiet", "--no-trunc"]
    assert calls[4]["arguments"] == ["image", "inspect", "--format", IMAGE_IDENTITY_FORMAT, built]
    # Plain progress from both streams, in a log only its owner can read.
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    assert "load build definition" in log.read_text() and "#1 DONE" in log.read_text()


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (
            {
                "listed": [CACHED_ID],
                "images": {IMAGE_CACHE_TAG: {"id": CACHED_ID, "labels": {**RECIPE_LABELS, REVISION_LABEL: "f" * 40}}},
            },
            "does not carry",
        ),
        (
            {
                "listed": [CACHED_ID],
                "images": {
                    IMAGE_CACHE_TAG: {"id": CACHED_ID, "labels": {**RECIPE_LABELS, BUILD_RECIPE_LABEL: "0" * 16}}
                },
            },
            "does not carry",
        ),
        ({"listed": [CACHED_ID], "images": {IMAGE_CACHE_TAG: {"id": CACHED_ID, "labels": None}}}, "does not carry"),
        (
            {"listed": [CACHED_ID], "images": {IMAGE_CACHE_TAG: "secret-marker not JSON"}},
            "invalid Docker image identity",
        ),
        ({"listed": [CACHED_ID], "images": {IMAGE_CACHE_TAG: {"id": "abc", "labels": RECIPE_LABELS}}}, "invalid"),
        ({"images": {CONFIG_ID: {"id": OTHER_ID, "labels": RECIPE_LABELS}}}, "does not have the ID"),
        ({"images": {CONFIG_ID: {"id": CONFIG_ID, "labels": {}}}}, "does not carry"),
        ({"metadata": None}, "no valid image digests"),
        ({"metadata": "secret-marker {"}, "no valid image digests"),
        ({"metadata": "[]"}, "no valid image digests"),
        ({"metadata": {"containerimage.config.digest": CONFIG_ID}}, "no valid image digests"),
        ({"metadata": {**BUILD_METADATA, "containerimage.config.digest": "sha256:abc"}}, "no valid image digests"),
        ({"metadata": {**BUILD_METADATA, "containerimage.digest": None}}, "no valid image digests"),
        ({"inventory": [CACHED_ID, OTHER_ID]}, "no local image has its digests"),
        ({"inventory": []}, "no local image has its digests"),
        ({"inventory": [CONFIG_ID, MANIFEST_ID]}, "cannot tell which one"),
        ({"inventory": [CONFIG_ID, "secret-marker"]}, "invalid Docker image listing"),
    ],
    ids=[
        "cached_other_revision",
        "cached_other_recipe",
        "cached_without_labels",
        "cached_malformed_inspection",
        "cached_malformed_id",
        "built_other_id",
        "built_without_labels",
        "metadata_missing",
        "metadata_not_json",
        "metadata_not_an_object",
        "metadata_without_image_digest",
        "metadata_with_malformed_config_digest",
        "metadata_with_null_image_digest",
        "built_but_no_local_image_has_its_digests",
        "built_but_no_local_images",
        "both_digests_name_local_images",
        "malformed_inventory",
    ],
)
def test_an_image_without_this_recipes_identity_is_refused_and_never_replaced(
    tmp_path: Path, state: dict[str, object], message: str
) -> None:
    docker = tmp_path / "docker"
    runtime = fake_docker(docker, **state)
    with pytest.raises(ContainerCommandError, match=message) as error:
        runtime.resolve_image(tmp_path / "build.log", 30, never_stop)
    assert "secret-marker" not in str(error.value)
    calls = [call["arguments"] for call in docker_calls(docker)]
    builds = [arguments for arguments in calls if arguments[:2] == ["buildx", "build"]]
    # A cached image that does not match is refused, not rebuilt over.
    assert len(builds) == (0 if state.get("listed") else 1)
    if builds:
        # A build's image is found by its own digests only, never by the shared cache tag.
        assert ["image", "inspect", "--format", IMAGE_IDENTITY_FORMAT, IMAGE_CACHE_TAG] not in calls


def test_a_failed_inventory_after_a_build_is_a_daemon_failure_not_a_missing_image(tmp_path: Path) -> None:
    docker = tmp_path / "docker"
    runtime = fake_docker(docker, inventory_fails=True)
    with pytest.raises(ContainerCommandError, match="Docker image failed with status 1") as error:
        runtime.resolve_image(tmp_path / "build.log", 30, never_stop)
    assert "no local image" not in str(error.value)
    # Nothing is inspected, by digest or tag, and nothing is built again.
    calls = [call["arguments"][:2] for call in docker_calls(docker)]
    assert calls == [["image", "ls"], ["buildx", "version"], ["buildx", "build"], ["image", "ls"]]


@pytest.mark.parametrize(
    ("state", "message"), [({"listing_fails": True}, "Docker image failed"), ({"buildx_status": 1}, "Buildx")]
)
def test_a_failed_listing_is_not_a_missing_image_and_a_build_needs_buildx(
    tmp_path: Path, state: dict[str, object], message: str
) -> None:
    docker = tmp_path / "docker"
    runtime = fake_docker(docker, **state)
    with pytest.raises(ContainerCommandError, match=message):
        runtime.resolve_image(tmp_path / "build.log", 30, never_stop)
    assert not any(call["arguments"][:2] == ["buildx", "build"] for call in docker_calls(docker))
    assert not (tmp_path / "build.log").exists()


@pytest.mark.parametrize(
    ("build", "timeout_seconds", "outcome"),
    [("fail", 30, "failed with status 3"), ("hang", 1.5, "did not finish within 1.5 seconds")],
)
def test_a_failed_or_timed_out_build_names_its_log_and_never_echoes_its_output(
    tmp_path: Path, build: str, timeout_seconds: float, outcome: str
) -> None:
    docker = tmp_path / "docker"
    runtime = fake_docker(docker, build=build)
    log = tmp_path / "build.log"
    started = time.monotonic()
    with pytest.raises(ContainerCommandError) as error:
        runtime.resolve_image(log, timeout_seconds, never_stop)
    assert time.monotonic() - started < 15
    assert outcome in str(error.value) and str(log) in str(error.value)
    assert "secret-marker" not in str(error.value) and "secret-marker" in log.read_text()
    wait_until_gone(int((docker / "build.pid").read_text()))
    # No image was inspected or Compose command run after the failed build.
    assert [call["arguments"][:2] for call in docker_calls(docker)][-1] == ["buildx", "build"]


def test_a_stop_request_ends_the_build_and_its_process_group(tmp_path: Path) -> None:
    docker = tmp_path / "docker"
    runtime = fake_docker(docker, build="hang")
    started = time.monotonic()
    with pytest.raises(InterruptedError):
        runtime.resolve_image(tmp_path / "build.log", 60, lambda: (docker / "build.pid").exists())
    assert time.monotonic() - started < 15
    wait_until_gone(int((docker / "build.pid").read_text()))
    assert [call["arguments"][:2] for call in docker_calls(docker)][-1] == ["buildx", "build"]


@pytest.mark.parametrize("output", [-1, True, 1.5])
def test_a_command_output_must_be_an_open_descriptor(monkeypatch: pytest.MonkeyPatch, output: object) -> None:
    monkeypatch.setattr(os, "pipe", lambda: pytest.fail("a malformed output allocated a pipe"))
    with pytest.raises(ValueError, match="descriptor"):
        GuardedCommand([sys.executable], 1, output=output)  # type: ignore[arg-type]


def test_unknown_mutation_never_echoes_command_output() -> None:
    runtime = DockerComposeRuntime(sys.executable)
    with pytest.raises(ContainerCommandError) as caught:
        runtime.run(["-c", "print('secret-value'); raise SystemExit(7)"], mutation=True)
    assert caught.value.outcome_unknown
    assert "secret-value" not in str(caught.value)


@pytest.mark.parametrize(
    "make",
    [
        lambda: ContainerState("router-0", "semantic-router", "false", "healthy"),
        lambda: ContainerState("router-0", "semantic-router", True, 1),
        lambda: ContainerInventory([ContainerState("router-0", "semantic-router", True, "healthy")]),
        lambda: ContainerInventory((), ["network-0"]),
        lambda: ContainerEvent("router-0", None),
        lambda: ContainerCommandError("failed", outcome_unknown=1),
        lambda: DockerComposeRuntime(""),
    ],
)
def test_container_values_refuse_what_their_definitions_exclude(make: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        make()


@pytest.mark.parametrize(
    ("command", "timeout_seconds"),
    [
        ("docker", 1),
        (b"docker", 1),
        ([], 1),
        ([""], 1),
        (["", "ps"], 1),
        ([sys.executable, 1], 1),
        ([sys.executable], math.inf),
        ([sys.executable], 0),
    ],
)
def test_a_malformed_command_is_refused_before_any_pipe_exists(
    monkeypatch: pytest.MonkeyPatch, command: object, timeout_seconds: float
) -> None:
    monkeypatch.setattr(os, "pipe", lambda: pytest.fail("a malformed command allocated a pipe"))
    with pytest.raises(ValueError):
        GuardedCommand(command, timeout_seconds)


@pytest.mark.parametrize(
    ("command", "environment", "error"),
    [([sys.executable, "embedded\0null"], None, ValueError), ([sys.executable], {"REEF_TEST": 1}, TypeError)],
)
def test_a_command_popen_refuses_closes_both_ends_of_its_owner_pipe(
    monkeypatch: pytest.MonkeyPatch, command: list[str], environment: dict[str, object] | None, error: type
) -> None:
    owner_pipe: list[tuple[int, int]] = []
    real_pipe = os.pipe

    def first_pipe() -> tuple[int, int]:
        descriptors = real_pipe()
        if not owner_pipe:
            owner_pipe.append(descriptors)
        return descriptors

    monkeypatch.setattr(os, "pipe", first_pipe)
    with pytest.raises(error):
        GuardedCommand(command, 1, environment=environment)
    monkeypatch.undo()
    for descriptor in owner_pipe[0]:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_an_empty_argument_after_the_executable_reaches_the_command() -> None:
    output = DockerComposeRuntime(sys.executable).run(["-c", "import sys; print(sys.argv[1:])", ""])
    assert output.strip() == "['']"


def test_container_values_keep_every_observation() -> None:
    """Readiness, not the value, decides what an observation means; duplicates and unknown strings stay."""
    stopped = ContainerState("router-0", "other-service", False, "unknown-health")
    assert ContainerInventory((stopped, stopped)).containers == (stopped, stopped)
    assert ContainerEvent("", "stream_disconnected").container_id == ""
    assert ContainerEvent("router-0", "").action == ""


def test_command_timeout_is_bounded() -> None:
    runtime = DockerComposeRuntime(sys.executable)
    started = time.monotonic()
    with pytest.raises(ContainerCommandError) as caught:
        runtime.run(["-c", "import time; time.sleep(30)"], timeout_seconds=0.1, mutation=True)
    assert caught.value.outcome_unknown
    assert time.monotonic() - started < 5


def test_command_loses_owner_and_reaps_child(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    command = GuardedCommand(
        [
            sys.executable,
            "-c",
            f"import os,time; open({str(pid_path)!r},'w').write(str(os.getpid())); time.sleep(30)",
        ],
        30,
    )
    deadline = time.monotonic() + 5
    while not pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pid_path.exists()
    child_pid = int(pid_path.read_text())
    command.close()
    assert command.process.returncode != 0
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


@pytest.mark.parametrize("ending", ["timeout", "owner_loss"])
def test_a_failed_command_retires_a_descendant_that_ignores_sigterm(tmp_path: Path, ending: str) -> None:
    """The CLI exits on SIGTERM, but the descendant it started ignores it and must still be gone afterwards.

    The descendant holds a FIFO open for writing. End of file there shows that it exited, even before
    anything reaps it.
    """
    fifo = tmp_path / "descendant.fifo"
    os.mkfifo(fifo)
    reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    descendant = (
        "import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"held = os.open({str(fifo)!r}, os.O_WRONLY); os.write(held, str(os.getpid()).encode()); time.sleep(30)"
    )
    cli = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {descendant!r}]); time.sleep(30)"
    command = GuardedCommand([sys.executable, "-c", cli], 2 if ending == "timeout" else 30)
    descendant_pid = None
    try:
        deadline = time.monotonic() + 5
        while descendant_pid is None and time.monotonic() < deadline:
            try:
                written = os.read(reader, 32)
            except BlockingIOError:
                written = b""
            if written:
                descendant_pid = int(written)
            else:
                time.sleep(0.01)
        assert descendant_pid is not None
        if ending == "timeout":
            command.process.wait(timeout=10)
        command.close()
        assert command.process.returncode != 0
        exited = False
        deadline = time.monotonic() + 5
        while not exited and time.monotonic() < deadline:
            try:
                exited = os.read(reader, 1) == b""
            except BlockingIOError:
                time.sleep(0.02)
        assert exited, "a descendant outlived its failed command"
        descendant_pid = None
    finally:
        if descendant_pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(descendant_pid, signal.SIGKILL)
        os.close(reader)
        command.close()
        for stream in (command.process.stdout, command.process.stderr):
            if stream is not None:
                stream.close()


def test_owner_sigkill_closes_lease_without_inheriting_lock(tmp_path: Path) -> None:
    command_pid = tmp_path / "command.pid"
    controller_pid = tmp_path / "controller.pid"
    program = f"""
import os, sys, time
from reef.service.deploy.routing.containers import GuardedCommand
from reef.service.deploy.routing.registry import DeploymentRegistry
from pathlib import Path
registry = DeploymentRegistry("killed", root=Path({str(tmp_path)!r}))
registry.acquire()
command = GuardedCommand([sys.executable, "-c", "import os,time; open({str(command_pid)!r}, 'w').write(str(os.getpid())); time.sleep(30)"], 30)
Path({str(controller_pid)!r}).write_text(str(os.getpid()))
time.sleep(30)
"""
    controller = subprocess.Popen([sys.executable, "-c", program])
    try:
        deadline = time.monotonic() + 5
        while not command_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert command_pid.exists()
        controller.kill()
        controller.wait(timeout=3)
        with DeploymentRegistry("killed", root=tmp_path):
            pass
        child_pid = int(command_pid.read_text())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            pytest.fail("Docker command survived loss of its owner")
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=3)


def test_events_surface_disconnect_and_keep_container_identity() -> None:
    event = {"Actor": {"ID": "old-container"}, "Action": "die"}
    stream = DockerEventStream([sys.executable, "-c", f"print({json.dumps(event)!r})"])
    try:
        first = stream.poll(timeout_seconds=5)
        assert first is not None and first.container_id == "old-container" and first.action == "die"
        disconnected = stream.poll(timeout_seconds=5)
        assert disconnected is not None and disconnected.action == "stream_disconnected"
    finally:
        stream.close()


def test_interrupted_mutation_requires_matching_explicit_acknowledgement(ownership: DeploymentOwnership) -> None:
    runtime = FakeDocker(ownership)
    marker = ownership.configuration_directory / "docker-mutation.json"
    marker.write_text(json.dumps({**ownership.labels, "operation_id": "c" * 32, "operation": "compose"}))
    marker.chmod(0o600)
    # Safe cleanup removes observed owned resources, but an empty snapshot is
    # not proof that an interrupted create can no longer reach the daemon.
    with pytest.raises(ContainerCommandError, match="remains unresolved"):
        runtime.remove(ownership, 0)
    assert not runtime.containers and not runtime.networks
    assert runtime.pending_operation_id(ownership) == "c" * 32
    with pytest.raises(ContainerCommandError, match="unresolved"):
        runtime.start(ownership.configuration_directory / "compose.yaml", ownership, 5)
    with pytest.raises(ContainerCommandError, match="changed"):
        runtime.acknowledge_unknown(ownership, "d" * 32)
    assert marker.exists()
    runtime.acknowledge_unknown(ownership, "c" * 32)
    assert not marker.exists()


def test_mutation_marker_precedes_cli_and_survives_failure(ownership: DeploymentOwnership) -> None:
    class FailingDocker(FakeDocker):
        """A Docker whose every mutation fails with an unknown outcome, after checking its marker exists."""

        def run(self, arguments: Sequence[str], timeout_seconds: float = 5, *, mutation: bool = False) -> str:
            if mutation:
                assert self.mutation_pending(ownership)
                raise ContainerCommandError("unknown result", outcome_unknown=True)
            return super().run(arguments, timeout_seconds, mutation=mutation)

    runtime = FailingDocker(ownership)
    with pytest.raises(ContainerCommandError, match="unknown result"):
        runtime.mutate(["compose", "up"], ownership)
    assert runtime.pending_operation_id(ownership) is not None


def test_successful_mutation_clears_its_own_marker(ownership: DeploymentOwnership) -> None:
    runtime = FakeDocker(ownership)
    runtime.mutate(["compose", "up"], ownership)
    assert not runtime.mutation_pending(ownership)


def test_status_can_read_owner_without_acquiring_running_controller_lock(tmp_path: Path) -> None:
    registry = DeploymentRegistry("shared", root=tmp_path)
    with registry:
        ownership = registry.prepare("one", tmp_path)
        observer = DeploymentRegistry("shared", root=tmp_path)
        assert observer.read_owner() == ownership
        with pytest.raises(RoutingOwnershipError):
            observer.prepare("two", tmp_path)


def test_sigkill_during_mutation_blocks_immediate_replacement(ownership: DeploymentOwnership, tmp_path: Path) -> None:
    command_pid = tmp_path / "mutation-child.pid"
    program = f"""
import sys
from pathlib import Path
from reef.service.deploy.routing.containers import DockerComposeRuntime
from reef.service.deploy.routing.registry import DeploymentOwnership
ownership =DeploymentOwnership({ownership.deployment_id!r}, {ownership.owner_id!r}, {ownership.generation!r}, Path({str(ownership.configuration_directory)!r}))
DockerComposeRuntime(sys.executable).mutate(["-c", "import os,time; open({str(command_pid)!r}, 'w').write(str(os.getpid())); time.sleep(30)"], ownership, 30)
"""
    controller = subprocess.Popen([sys.executable, "-c", program])
    try:
        deadline = time.monotonic() + 5
        while not command_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert command_pid.exists()
        controller.kill()
        controller.wait(timeout=3)
        runtime = FakeDocker(ownership)
        runtime.containers.clear()
        runtime.networks.clear()
        # No polling delay or empty Docker snapshot can override the saved
        # unknown outcome. The next controller fails before attempting create.
        with pytest.raises(ContainerCommandError, match="unresolved"):
            runtime.start(tmp_path / "compose.yaml", ownership, 5)
        assert runtime.pending_operation_id(ownership) is not None
        assert not any(command[0] == "compose" for command in runtime.commands)
        child_pid = int(command_pid.read_text())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            pytest.fail("mutation command survived its owner")
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=3)


def test_docker_event_filters_prevent_healthcheck_exec_noise_overflow(
    ownership: DeploymentOwnership,
    tmp_path: Path,
) -> None:
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        f"""#!{sys.executable}
import json, sys
filters = [item[6:] for item in sys.argv if item.startswith("event=")]
# Simulate a long startup's per-probe exec events while nobody polls.
for index in range(3000):
    action = ("exec_create", "exec_start", "exec_die")[index % 3]
    if not filters or action in filters:
        print(json.dumps({{"Actor": {{"ID": "router"}}, "Action": action}}))
for action in ("health_status: unhealthy", "die"):
    if not filters or action.split(":")[0] in filters:
        print(json.dumps({{"Actor": {{"ID": "router"}}, "Action": action}}))
"""
    )
    fake_docker.chmod(0o700)
    stream = DockerComposeRuntime(str(fake_docker)).events(ownership)
    try:
        assert isinstance(stream, DockerEventStream)
        stream.command.process.wait(timeout=5)
        first = stream.poll(timeout_seconds=5)
        second = stream.poll(timeout_seconds=5)
        assert first is not None and first.action == "health_status: unhealthy"
        assert second is not None and second.action == "die"
        final = stream.poll(timeout_seconds=5)
        assert final is not None and final.action == "stream_disconnected"
    finally:
        stream.close()


@pytest.mark.parametrize("malformation", ["bad-json", "missing-state", "missing-id", "bad-config"])
def test_invalid_inspect_responses_have_typed_sanitized_errors(
    ownership: DeploymentOwnership,
    malformation: str,
) -> None:
    class MalformedDocker(FakeDocker):
        """A Docker whose container inspection reply is malformed as ``malformation`` names."""

        def run(self, arguments: Sequence[str], timeout_seconds: float = 5, *, mutation: bool = False) -> str:
            encoded = super().run(arguments, timeout_seconds, mutation=mutation)
            if arguments[:2] != ["container", "inspect"]:
                return encoded
            if malformation == "bad-json":
                return "secret-marker definitely not valid JSON"
            records = json.loads(encoded)
            if malformation == "missing-state":
                records[0].pop("State")
            elif malformation == "missing-id":
                records[0].pop("Id")
            else:
                records[0]["Config"] = "secret-marker"
            return json.dumps(records)

    with pytest.raises(ContainerCommandError) as caught:
        MalformedDocker(ownership).inspect(ownership)
    assert "secret-marker" not in str(caught.value)


def test_cli_invalid_utf8_is_typed_and_preserves_mutation_uncertainty() -> None:
    runtime = DockerComposeRuntime(sys.executable)
    with pytest.raises(ContainerCommandError, match="UTF-8") as caught:
        runtime.run(["-c", "import sys; sys.stdout.buffer.write(bytes([255]))"], mutation=True)
    assert caught.value.outcome_unknown


@pytest.mark.parametrize("encoded", [b"secret-marker invalid JSON", b"\xff"])
def test_corrupt_private_state_fails_closed_with_ownership_error(tmp_path: Path, encoded: bytes) -> None:
    with DeploymentRegistry("state", root=tmp_path) as registry:
        ownership = registry.prepare("generation", tmp_path)
        (registry.directory / "owner.json").write_bytes(encoded)
        with pytest.raises(RoutingOwnershipError) as caught:
            registry.read_owner()
        assert "secret-marker" not in str(caught.value)
    mutation = tmp_path / "docker-mutation.json"
    mutation.write_bytes(encoded)
    mutation.chmod(0o600)
    with pytest.raises(RoutingOwnershipError) as caught:
        DockerComposeRuntime().pending_operation_id(ownership)
    assert "secret-marker" not in str(caught.value)
