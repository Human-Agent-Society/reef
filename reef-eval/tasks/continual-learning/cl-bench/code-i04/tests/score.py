"""Codebase-adaptation evaluator, run as each task's verifier.

Replays CL-Bench's evaluation recipe (``generic_runtime.py``,
Apache-2.0) inside the task container: capture the agent's ``git diff``,
strip any sections touching test-owned paths, reset the repo to the base
commit, re-apply the official test patch, apply the sanitized model
patch, then run pytest on the exact FAIL_TO_PASS and PASS_TO_PASS node
ids in collection order. Reward is 1.0 iff pytest exits 0.

One deviation from upstream: their reward scales with how early in a
40-step budget the patch landed; Harbor agents are not step-metered, so
here solved is 1.0 and unsolved is 0.0.
"""

import json
import os
import re
import shlex
import subprocess

CONFIG_PATH = "/tests/eval_config.json"
REWARD_PATH = "/logs/verifier/reward.txt"
DETAILS_PATH = "/logs/verifier/scoring.json"

APPLY_PATCH_COMMANDS = (
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
)


def sh(cmd: str, *, cwd: str, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/bash", "-lc", cmd],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# ---- vendored verbatim from upstream generic_runtime.py (Apache-2.0) ----


def test_patch_paths(test_patch: str) -> list[str]:
    return re.findall(r"^diff --git a/(.*?) b/", test_patch, re.MULTILINE)


def strip_patch_paths(patch_text: str, blocked_paths: list[str]) -> str:
    """Remove file-level diff sections whose paths are owned by test_patch."""
    if not patch_text.strip() or not blocked_paths:
        return patch_text
    blocked = set(blocked_paths)
    kept_sections: list[str] = []
    current_lines: list[str] = []
    keep_current = True

    def _flush_current():
        if current_lines and keep_current:
            kept_sections.extend(current_lines)

    for line in patch_text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            _flush_current()
            current_lines = [line]
            parts = line.strip().split()
            candidate_paths = set()
            if len(parts) >= 4:
                for token in (parts[2], parts[3]):
                    if token.startswith("a/") or token.startswith("b/"):
                        candidate_paths.add(token[2:])
            keep_current = candidate_paths.isdisjoint(blocked)
        else:
            current_lines.append(line)
    _flush_current()
    return "".join(kept_sections)


# ------------------------------------------------------------------- eval


def apply_patch(patch_text: str, *, cwd: str, timeout: int) -> bool:
    path = "/tmp/clbench_apply.patch"
    with open(path, "w") as f:
        f.write(patch_text if patch_text.endswith("\n") else patch_text + "\n")
    for base in APPLY_PATCH_COMMANDS:
        if sh(f"{base} {path}", cwd=cwd, timeout=timeout).returncode == 0:
            return True
    return False


def ordered_targets(targets: list[str], *, cwd: str, timeout: int) -> list[str]:
    """Reorder exact node ids to pytest's collection order per file."""
    by_file: dict[str, list[str]] = {}
    for t in targets:
        by_file.setdefault(t.split("::", 1)[0], []).append(t)
    ordered: list[str] = []
    for file, wanted in by_file.items():
        result = sh(
            f"python -m pytest --collect-only -q --color=no {shlex.quote(file)}",
            cwd=cwd,
            timeout=timeout,
        )
        if result.returncode not in (0, 1):
            ordered.extend(wanted)
            continue
        wanted_set = set(wanted)
        collected = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() in wanted_set
        ]
        ordered.extend(collected)
        ordered.extend(t for t in wanted if t not in set(collected))
    return ordered


def main() -> int:
    os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    workdir = config["workdir"]
    timeout = int(config["test_timeout"])
    test_patch = config["test_patch"]

    def finish(reward: float, status: str, detail: str = "") -> int:
        with open(REWARD_PATH, "w") as f:
            f.write(str(reward))
        with open(DETAILS_PATH, "w") as f:
            json.dump({"reward": reward, "status": status, "detail": detail[-2000:]}, f)
        print(f"eval: {status} -> reward {reward}")
        return 0

    model_patch = sh("git diff", cwd=workdir, timeout=timeout).stdout
    if not model_patch.strip():
        return finish(0.0, "empty_patch")
    sanitized = strip_patch_paths(model_patch, test_patch_paths(test_patch))

    sh(
        f"git reset --hard {config['base_commit']} && git clean -fd",
        cwd=workdir,
        timeout=timeout,
    )
    if not apply_patch(test_patch, cwd=workdir, timeout=timeout):
        return finish(0.0, "error", "test_patch failed to apply on reset workspace")
    if sanitized.strip() and not apply_patch(sanitized, cwd=workdir, timeout=timeout):
        return finish(0.0, "apply_failed")

    # Re-assert canonical tests in case the model patch drifted them.
    for path in test_patch_paths(test_patch):
        quoted = shlex.quote(path)
        sh(
            f"git checkout -- {quoted} 2>/dev/null || rm -f {quoted}",
            cwd=workdir,
            timeout=timeout,
        )
    if not apply_patch(test_patch, cwd=workdir, timeout=timeout):
        return finish(0.0, "error", "test_patch re-assert failed")

    targets = ordered_targets(
        [t for t in config["fail_to_pass"] + config["pass_to_pass"] if "::" in t],
        cwd=workdir,
        timeout=timeout,
    )
    command = shlex.join([*shlex.split(config["test_command"]), *targets])
    try:
        result = sh(command, cwd=workdir, timeout=timeout)
    except subprocess.TimeoutExpired:
        return finish(0.0, "timeout")
    if result.returncode == 0:
        return finish(1.0, "completed")
    return finish(0.0, "tests_failed", result.stdout[-2000:])


if __name__ == "__main__":
    raise SystemExit(main())
