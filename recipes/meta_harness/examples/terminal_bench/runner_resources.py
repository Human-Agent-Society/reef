"""Prepare an explicitly sized copy of the frozen runner without model calls.

The template is cloned from the credential-free runtime, never from a task or
candidate sandbox. Runtime manifest identity must survive unchanged. Resources
are measured in a fresh sandbox and tested before an immutable snapshot receipt
can be used in a separately reviewed campaign transition.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shlex
from pathlib import Path

from .campaign import atomic_json
from .e2b_executor import MANIFEST, REMOTE_PYTHON

PROTOCOL = "e2b-runner-4g-v1"
MEMORY_MB = 4096
CPU_COUNT = 2


def validate_receipt(original, resized):
    provenance = resized.get("resource_transition") or {}
    if (
        provenance.get("protocol") != PROTOCOL
        or provenance.get("original_snapshot_id") != original["snapshot_id"]
        or resized.get("snapshot_id") == original["snapshot_id"]
        or resized.get("manifest_sha256") != original["manifest_sha256"]
        or resized.get("manifest") != original["manifest"]
        or resized.get("verifier_compat") != original.get("verifier_compat")
        or resized.get("source_archive_sha256") != original.get("source_archive_sha256")
    ):
        raise ValueError("resource transition must preserve the complete pinned runner identity")
    measured = provenance.get("verification") or {}
    if (
        measured.get("cpu_count") != CPU_COUNT
        or measured.get("memory_total_bytes", 0) < MEMORY_MB * 2**20 * 0.9
        or measured.get("stress_rss_bytes", 0) < 1500 * 2**20
        or measured.get("model_credentials_present") is not False
        or provenance.get("model_calls") != 0
        or provenance.get("cleanup_confirmed") is not True
    ):
        raise ValueError("resource transition lacks verified capacity, isolation or cleanup")


def start(original_path, build_path, *, name):
    from e2b import Template

    if not re.fullmatch(r"[a-z0-9_-]+:v1", name):
        raise ValueError("use a fresh lowercase template name with an explicit v1 tag")
    original = json.loads(Path(original_path).read_text())
    build_path = Path(build_path)
    # Reserve a local build identity before requesting a billable infrastructure
    # build. An ambiguous start cannot silently create another template.
    with build_path.open("x") as handle:
        json.dump({"status": "starting", "original": original, "name": name}, handle)
    info = Template.build_in_background(
        Template().from_template(original["snapshot_id"]), name, cpu_count=CPU_COUNT, memory_mb=MEMORY_MB
    )
    value = {
        "status": "building",
        "original": original,
        "name": name,
        "build": dataclasses.asdict(info),
        "requested_cpu_count": CPU_COUNT,
        "requested_memory_mb": MEMORY_MB,
    }
    atomic_json(build_path, value)
    return {"status": "building", "build": value["build"]}


def restore_runtime(sandbox, payload, original):
    """Install the exact archived input using the original pinned setup steps."""
    present = sandbox.commands.run(
        "if test -d /opt/reef; then printf present; else printf absent; fi", user="root", timeout=30
    ).stdout
    if present == "present":
        program = (
            "import json; from recipes.meta_harness.examples.terminal_bench.e2b_runtime import create_manifest; "
            "print(json.dumps(create_manifest(), sort_keys=True))"
        )
        actual = sandbox.commands.run(
            shlex.join([REMOTE_PYTHON, "-c", program]), user="root", cwd="/opt/reef", timeout=60
        )
        if json.loads(actual.stdout) != original["manifest"]:
            raise ValueError("the existing template runtime differs from the pinned source or dependencies")
        raw = (json.dumps(original["manifest"], sort_keys=True) + "\n").encode()
        if hashlib.sha256(raw).hexdigest() != original["manifest_sha256"]:
            raise ValueError("cannot reproduce the original manifest bytes")
        sandbox.files.write(MANIFEST, raw, user="root", request_timeout=30)
        sandbox.commands.run("chmod a+r /opt/reef-runtime.json", user="root", timeout=30)
        return "verified_manifest_restore"
    sandbox.files.write("/tmp/reef-source.tar.gz", payload, user="root", request_timeout=60)
    sandbox.commands.run(
        "test ! -e /opt/reef && mkdir -p /opt/reef && tar -xzf /tmp/reef-source.tar.gz -C /opt/reef "
        "&& rm /tmp/reef-source.tar.gz && useradd -m -u 1001 -s /bin/bash reef "
        "&& python3 -m pip install --break-system-packages uv==0.12.5",
        user="root",
        timeout=180,
    )
    sandbox.commands.run(
        "uv sync --locked --python 3.12.14",
        user="root",
        cwd="/opt/reef/recipes/meta_harness/examples/terminal_bench",
        timeout=900,
        envs={"UV_PYTHON_INSTALL_DIR": "/opt/uv-python", "UV_CACHE_DIR": "/opt/uv-cache"},
    )
    sandbox.commands.run(
        f"{REMOTE_PYTHON} -m recipes.meta_harness.examples.terminal_bench.runtime "
        f"&& {REMOTE_PYTHON} -m recipes.meta_harness.examples.terminal_bench.e2b_runtime --write-manifest "
        "&& chmod -R a+rX,go-w /opt/reef && chmod a+r /opt/reef-runtime.json",
        user="root",
        cwd="/opt/reef",
        timeout=60,
    )
    return "verified_archive_restore"


def finish(build_path, output, *, source_archive=None):
    from e2b import Sandbox, Template
    from e2b.template.types import BuildInfo

    build_path, output = Path(build_path), Path(output)
    value = json.loads(build_path.read_text())
    payload = Path(source_archive).read_bytes() if source_archive is not None else None
    if payload is not None and hashlib.sha256(payload).hexdigest() != value["original"]["source_archive_sha256"]:
        raise ValueError("restore input must be the exact archived source used by the original runtime")
    if value["status"] not in ("building", "ready"):
        raise ValueError("review the existing build instead of repeating an uncertain request")
    info = BuildInfo(**value["build"])
    status = Template.get_build_status(info)
    state = status.status.value
    if state != "ready":
        if state == "error":
            atomic_json(
                build_path, {**value, "status": "error", "reason": status.reason.message if status.reason else None}
            )
        return {"status": state, "build_id": info.build_id}
    if output.exists():
        raise ValueError("preserve the existing resource receipt")
    claim = output.with_suffix(".verification.json")
    with claim.open("x") as handle:
        json.dump({"status": "starting", "build": value["build"]}, handle)
    sandbox = None
    result = None
    failure_type = None
    cleanup_confirmed = False
    try:
        sandbox = Sandbox.create(
            info.template_id + ":v1",
            timeout=1800 if payload is not None else 300,
            secure=True,
            lifecycle={"on_timeout": "kill"},
            metadata={"reef_role": "runtime-resource-verification"},
            request_timeout=60,
        )
        atomic_json(
            claim,
            {
                "status": "verifying",
                "build": value["build"],
                "sandbox_id": sandbox.sandbox_id,
                "model_calls": 0,
                "cleanup_confirmed": False,
            },
        )
        original = value["original"]
        runtime_source = "template_copy"
        if payload is not None:
            runtime_source = restore_runtime(sandbox, payload, original)
        raw = bytes(sandbox.files.read(MANIFEST, format="bytes", user="root", request_timeout=30))
        if hashlib.sha256(raw).hexdigest() != original["manifest_sha256"]:
            raise ValueError("the resized template changed the pinned runtime manifest")
        sandbox.commands.run(
            shlex.join([REMOTE_PYTHON, "-m", "recipes.meta_harness.examples.terminal_bench.e2b_runtime", "--verify"]),
            user="root",
            cwd="/opt/reef",
            timeout=60,
        )
        # Load the normal harness libraries before a 1.5 GiB allocation. Only
        # aggregate resource numbers leave the sandbox; no model client runs.
        program = """import json, os, resource
from harbor.agents.terminus_2 import Terminus2
from pathlib import Path
memory_kib = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:')))
data = bytearray(1536 * 1024**2)
for offset in range(0, len(data), 4096):
    data[offset] = 1
print(json.dumps({'cpu_count': os.cpu_count(), 'memory_total_bytes': memory_kib * 1024,
    'stress_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    'model_credentials_present': any(os.environ.get(key) for key in ('OPENAI_API_KEY','ANTHROPIC_API_KEY'))}))
"""
        measured = sandbox.commands.run(
            shlex.join([REMOTE_PYTHON, "-c", program]), user="reef", cwd="/tmp", timeout=60
        )
        verification = json.loads(measured.stdout)
        provisional = {
            **original,
            "snapshot_id": "unpublished-resource-verification",
            "resource_transition": {
                "protocol": PROTOCOL,
                "original_snapshot_id": original["snapshot_id"],
                "build": value["build"],
                "requested_cpu_count": CPU_COUNT,
                "requested_memory_mb": MEMORY_MB,
                "runtime_source": runtime_source,
                "verification": verification,
                "model_calls": 0,
                "cleanup_confirmed": True,
            },
        }
        validate_receipt(original, provisional)
        snapshot = sandbox.create_snapshot(request_timeout=120)
        result = {**provisional, "snapshot_id": snapshot.snapshot_id}
        result["resource_transition"].update(verification_sandbox_id=sandbox.sandbox_id, cleanup_confirmed=False)
    except Exception as exc:
        failure_type = type(exc).__name__
        raise
    finally:
        try:
            if sandbox is not None:
                sandbox.kill(request_timeout=30)
                cleanup_confirmed = True
                if result is not None:
                    result["resource_transition"]["cleanup_confirmed"] = True
        except Exception as exc:
            failure_type = type(exc).__name__
            raise
        finally:
            atomic_json(
                claim,
                {
                    "status": "verification_failed" if failure_type else "verified_pending_publish",
                    "failure_type": failure_type,
                    "sandbox_id": sandbox.sandbox_id if sandbox else None,
                    "cleanup_confirmed": cleanup_confirmed,
                    "model_calls": 0,
                    "build": value["build"],
                },
            )
    validate_receipt(value["original"], result)
    atomic_json(output, result)
    atomic_json(
        claim,
        {"status": "verified", "snapshot_id": result["snapshot_id"], "cleanup_confirmed": True, "model_calls": 0},
    )
    atomic_json(build_path, {**value, "status": "ready", "resource_receipt": str(output)})
    return {
        "status": "verified",
        "snapshot_id": result["snapshot_id"],
        "verification": result["resource_transition"]["verification"],
        "model_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    begin = sub.add_parser("start")
    begin.add_argument("--original", type=Path, required=True)
    begin.add_argument("--build", type=Path, required=True)
    begin.add_argument("--name", required=True)
    end = sub.add_parser("finish")
    end.add_argument("--build", type=Path, required=True)
    end.add_argument("--output", type=Path, required=True)
    end.add_argument(
        "--source-archive",
        type=Path,
        help="restore the original digest-matched archive when template copying omits snapshot files",
    )
    args = parser.parse_args()
    if not os.environ.get("E2B_API_KEY"):
        parser.error("E2B_API_KEY is required; no model credential is needed")
    result = (
        start(args.original, args.build, name=args.name)
        if args.command == "start"
        else finish(args.build, args.output, source_archive=args.source_archive)
    )
    print(json.dumps(result))
    return 0 if result["status"] != "error" else 2


if __name__ == "__main__":
    raise SystemExit(main())
