"""Prepare and verify a credential-free, pinned E2B runtime snapshot."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

from .e2b_executor import MANIFEST, REMOTE_PYTHON
from .runtime import runtime_fingerprint


def source_files(root):
    # Explicit source allowlist. No .git, .venv, artifacts, logs, or environment
    # files are uploaded. Snapshot preparation never receives model credentials.
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    for prefix in ("reef", "recipes/meta_harness"):
        names += [str(path.relative_to(root)) for path in (root / prefix).rglob("*") if path.is_file()]
    return sorted(
        {
            name
            for name in names
            if name
            and (root / name).is_file()
            and not (root / name).is_symlink()
            and not any(part.startswith(".") or part in ("__pycache__", "output", "jobs") for part in Path(name).parts)
            and (
                name in ("pyproject.toml", "README.md", "LICENSE")
                or name.startswith(("reef/", "recipes/meta_harness/"))
            )
            and Path(name).suffix != ".pyc"
        }
    )


def source_archive(root):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name in source_files(root):
            archive.add(root / name, arcname=name, recursive=False)
    return stream.getvalue()


def create_manifest():
    root = Path("/opt/reef")
    files = [
        path
        for prefix in ("reef", "recipes/meta_harness")
        for path in (root / prefix).rglob("*")
        if path.is_file()
        and not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(root).parts)
    ]
    return {
        "runtime": runtime_fingerprint(),
        "files": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)
        },
    }


def verify_manifest():
    frozen = json.loads(Path(MANIFEST).read_text())
    if frozen != create_manifest():
        raise RuntimeError("prepared E2B source or runtime differs from its frozen manifest")


def prepare(root, output, *, verifier_compat=None):
    from e2b import Sandbox

    from .runtime import check_capacity

    if output.exists():
        raise ValueError("preserve the existing snapshot receipt; use a new output path")
    if verifier_compat is not None:
        from .verifier_compat import PROTOCOL

        if verifier_compat != PROTOCOL:
            raise ValueError("unknown verifier compatibility protocol")
    check_capacity(1)
    payload = source_archive(root)
    sandbox = Sandbox.create(timeout=1800, metadata={"reef_role": "meta-harness-runtime-build"}, secure=True)
    print(json.dumps({"status": "preparing", "sandbox_id": sandbox.sandbox_id}), flush=True)
    try:
        sandbox.files.write("/tmp/reef-source.tar.gz", payload, user="root", request_timeout=60)
        sandbox.commands.run(
            "mkdir -p /opt/reef && tar -xzf /tmp/reef-source.tar.gz -C /opt/reef && rm /tmp/reef-source.tar.gz "
            "&& useradd -m -u 1001 -s /bin/bash reef "
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
        manifest = bytes(sandbox.files.read(MANIFEST, format="bytes", user="root"))
        snapshot = sandbox.create_snapshot(request_timeout=120)
        receipt = {
            "snapshot_id": snapshot.snapshot_id,
            "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "source_archive_sha256": hashlib.sha256(payload).hexdigest(),
            "manifest": json.loads(manifest),
        }
        if verifier_compat:
            receipt["verifier_compat"] = verifier_compat
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(
            json.dumps({"status": "prepared", "snapshot_id": snapshot.snapshot_id, "receipt": str(output)}), flush=True
        )
    finally:
        sandbox.kill(request_timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", type=Path)
    group.add_argument("--verify", action="store_true")
    group.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--verifier-compat", choices=["tb2-torch-gloo-cleanup-v1"])
    args = parser.parse_args()
    if args.verify:
        verify_manifest()
    elif args.write_manifest:
        Path(MANIFEST).write_text(json.dumps(create_manifest(), sort_keys=True) + "\n")
    else:
        prepare(Path(__file__).resolve().parents[4], args.prepare.resolve(), verifier_compat=args.verifier_compat)


if __name__ == "__main__":
    main()
