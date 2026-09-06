import copy
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest

from recipes.meta_harness.examples.terminal_bench import runner_resources as resources


def receipts():
    old = {
        "snapshot_id": "old:default",
        "manifest_sha256": "a" * 64,
        "manifest": {"runtime": "pinned"},
        "source_archive_sha256": "b" * 64,
        "verifier_compat": "tb2-torch-gloo-cleanup-v1",
    }
    new = {
        **old,
        "snapshot_id": "new:default",
        "resource_transition": {
            "protocol": resources.PROTOCOL,
            "original_snapshot_id": old["snapshot_id"],
            "verification": {
                "cpu_count": 2,
                "memory_total_bytes": 4090 * 2**20,
                "stress_rss_bytes": 1800 * 2**20,
                "model_credentials_present": False,
            },
            "model_calls": 0,
            "cleanup_confirmed": True,
        },
    }
    return old, new


@pytest.mark.parametrize("field", ["manifest", "manifest_sha256", "verifier_compat", "source_archive_sha256"])
def test_resources_cannot_change_pinned_runtime_identity(field):
    old, new = receipts()
    resources.validate_receipt(old, new)
    new[field] = "changed"
    with pytest.raises(ValueError, match="identity"):
        resources.validate_receipt(old, new)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cpu_count", 4),
        ("memory_total_bytes", 512 * 2**20),
        ("stress_rss_bytes", 400 * 2**20),
        ("model_credentials_present", True),
    ],
)
def test_resources_require_actual_capacity_and_no_model_credentials(field, value):
    old, new = receipts()
    new["resource_transition"]["verification"][field] = value
    with pytest.raises(ValueError, match="capacity"):
        resources.validate_receipt(old, new)


def test_ambiguous_template_start_cannot_silently_repeat(tmp_path, monkeypatch):
    from e2b import Template

    old, _ = receipts()
    original, build = tmp_path / "original.json", tmp_path / "build.json"
    original.write_text(json.dumps(old))
    calls = []

    def start(*args, **kwargs):
        calls.append(True)
        raise ConnectionError("lost acknowledgement")

    monkeypatch.setattr(Template, "build_in_background", start)
    with pytest.raises(ConnectionError):
        resources.start(original, build, name="test-runtime:v1")
    assert json.loads(build.read_text())["status"] == "starting"
    with pytest.raises(FileExistsError):
        resources.start(original, build, name="test-runtime:v1")
    assert len(calls) == 1


def test_poll_existing_build_without_starting_another(tmp_path, monkeypatch):
    from e2b import Template
    from e2b.template.types import BuildInfo

    old, _ = receipts()
    build = tmp_path / "build.json"
    info = BuildInfo(template_id="template", build_id="build", name="test:v1", alias="test:v1")
    build.write_text(json.dumps({"status": "building", "original": old, "build": dataclasses.asdict(info)}))
    monkeypatch.setattr(
        Template, "get_build_status", lambda _: SimpleNamespace(status=SimpleNamespace(value="building"))
    )
    before = copy.deepcopy(json.loads(build.read_text()))
    assert resources.finish(build, tmp_path / "receipt.json") == {"status": "building", "build_id": "build"}
    assert json.loads(build.read_text()) == before and not (tmp_path / "receipt.json").exists()


@pytest.mark.parametrize("failure", [None, "manifest", "capacity", "cleanup"])
def test_finished_build_publishes_only_after_identity_capacity_and_cleanup(tmp_path, monkeypatch, failure):
    from e2b import Sandbox, Template
    from e2b.template.types import BuildInfo

    old, new = receipts()
    raw = json.dumps(old["manifest"]).encode()
    old["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    build, output = tmp_path / "build.json", tmp_path / "receipt.json"
    info = BuildInfo(template_id="template", build_id="build", name="test:v1", alias="test:v1")
    build.write_text(json.dumps({"status": "building", "original": old, "build": dataclasses.asdict(info)}))
    events = []
    measured = new["resource_transition"]["verification"]
    if failure == "capacity":
        measured["memory_total_bytes"] = 512 * 2**20

    def run(command, **kwargs):
        events.append("verify" if "--verify" in command else "stress")
        return SimpleNamespace(stdout=json.dumps(measured))

    def snapshot(**kwargs):
        assert not output.exists()
        events.append("snapshot")
        return SimpleNamespace(snapshot_id="new:default")

    def kill(**kwargs):
        assert not output.exists()
        events.append("kill")
        if failure == "cleanup":
            raise ConnectionError("cleanup acknowledgement unavailable")
        return True

    sandbox = SimpleNamespace(
        sandbox_id="verification-parent",
        create_snapshot=snapshot,
        kill=kill,
        files=SimpleNamespace(read=lambda *args, **kwargs: b"changed" if failure == "manifest" else raw),
        commands=SimpleNamespace(run=run),
    )
    monkeypatch.setattr(Sandbox, "create", lambda *args, **kwargs: sandbox)
    monkeypatch.setattr(Template, "get_build_status", lambda _: SimpleNamespace(status=SimpleNamespace(value="ready")))
    if failure:
        with pytest.raises((ValueError, ConnectionError)):
            resources.finish(build, output)
        assert not output.exists() and events[-1] == "kill"
        assert ("snapshot" in events) == (failure == "cleanup")
        claim = json.loads(output.with_suffix(".verification.json").read_text())
        assert claim["status"] == "verification_failed"
        assert claim["cleanup_confirmed"] == (failure != "cleanup")
        # An uncertain verification cannot silently repeat external work.
        with pytest.raises(FileExistsError):
            resources.finish(build, output)
    else:
        assert resources.finish(build, output)["status"] == "verified"
        resources.validate_receipt(old, json.loads(output.read_text()))
        assert events == ["verify", "stress", "snapshot", "kill"]


def test_archive_mismatch_rejected_before_external_calls(tmp_path, monkeypatch):
    from e2b import Template

    old, _ = receipts()
    build, archive = tmp_path / "build.json", tmp_path / "input.tar.gz"
    archive.write_bytes(b"unreviewed source")
    build.write_text(json.dumps({"status": "building", "original": old}))
    monkeypatch.setattr(Template, "get_build_status", lambda _: pytest.fail("must reject before external request"))
    with pytest.raises(ValueError, match="exact archived source"):
        resources.finish(build, tmp_path / "receipt.json", source_archive=archive)


@pytest.mark.parametrize("changed", [False, True])
def test_missing_manifest_restored_only_after_all_runtime_files_match(changed):
    old, _ = receipts()
    raw = (json.dumps(old["manifest"], sort_keys=True) + "\n").encode()
    old["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    writes = []

    def run(command, **kwargs):
        if "printf present" in command:
            return SimpleNamespace(stdout="present")
        actual = {"runtime": "changed"} if changed else old["manifest"]
        return SimpleNamespace(stdout=json.dumps(actual))

    sandbox = SimpleNamespace(
        commands=SimpleNamespace(run=run),
        files=SimpleNamespace(write=lambda path, value, **kw: writes.append((path, value))),
    )
    if changed:
        with pytest.raises(ValueError, match="differs from the pinned"):
            resources.restore_runtime(sandbox, b"archive", old)
        assert not writes
    else:
        assert resources.restore_runtime(sandbox, b"archive", old) == "verified_manifest_restore"
        assert writes == [(resources.MANIFEST, raw)]
