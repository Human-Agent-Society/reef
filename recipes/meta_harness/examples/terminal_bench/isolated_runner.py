"""Remote-only entry point. Candidate imports happen inside Harbor in E2B."""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path


def require_isolation():
    # This guard prevents accidental local invocation. The actual boundary is
    # E2B's VM, created and destroyed by E2BEpisodeExecutor.
    if os.getuid() != 1001 or not Path("/opt/reef-runtime.json").is_file():
        raise RuntimeError("the executable runner must run as reef in the prepared E2B sandbox")


def make_agent(root, tree):
    require_isolation()
    from reef.harness.terminus.runner import agent_spec

    from .isolated_adapter import PREFIX, finalize_render

    finalize_render(tree)
    spec = agent_spec(root, tree)
    modules = [path for path in tree if path.startswith(PREFIX)]
    if modules:
        from harbor.agents.terminus_2 import Terminus2

        source = Path(root) / modules[0]
        module = "reef_candidate_" + hashlib.sha256(source.read_bytes()).hexdigest()
        loader = importlib.util.spec_from_file_location(module, source)
        loaded = importlib.util.module_from_spec(loader)
        sys.modules[module] = loaded
        loader.loader.exec_module(loaded)
        agent = getattr(loaded, "Agent", None)
        if not inspect.isclass(agent) or not issubclass(agent, Terminus2):
            raise ValueError("the executable Agent must subclass Harbor Terminus2")
        spec.pop("name")
        spec["import_path"] = f"{module}:Agent"
    return spec


def self_test():
    require_isolation()
    from .runtime import runtime_fingerprint

    root = Path(os.environ["REEF_TERMINUS_DIR"])
    try:
        (root / "terminus/config.json").write_text("tampered")
    except PermissionError:
        pass
    else:
        raise RuntimeError("rendered inputs are writable")
    import subprocess

    assert subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode != 0
    result = {"self_test": True, "uid": os.getuid(), "runtime": runtime_fingerprint()}
    (root / "workspace/runner-self-test.json").write_text(json.dumps(result))
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task")
    group.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    require_isolation()
    if args.self_test:
        self_test()
        return 0
    if protocol := os.environ.get("REEF_TERMINUS_VERIFIER_COMPAT"):
        from .verifier_compat import install

        install(protocol)
    from reef.harness.terminus.runner import run

    return run(args.task, make_agent=make_agent)


if __name__ == "__main__":
    raise SystemExit(main())
