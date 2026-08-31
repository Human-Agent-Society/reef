#!/usr/bin/env python3
"""Fast agent-side validation for the RocksDB patch submission."""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path


MAX_PATCH_BYTES = 256 * 1024
MAX_CHANGED_FILES = 5
ALLOWLIST = frozenset(
    {
        "db/compaction/compaction_picker.cc",
        "db/compaction/compaction_picker.h",
        "db/compaction/compaction_picker_level.cc",
        "db/compaction/compaction_picker_level.h",
        "db/version_set.cc",
    }
)
STRUCTURAL_MARKERS = (
    "new file mode",
    "deleted file mode",
    "rename from",
    "rename to",
    "copy from",
    "copy to",
    "old mode ",
    "new mode ",
    "GIT binary patch",
    "Subproject commit",
)
FORBIDDEN_IDENTIFIERS = frozenset(
    {
        "getenv",
        "secure_getenv",
        "system",
        "popen",
        "fork",
        "execve",
        "dlopen",
        "dlsym",
        "readlink",
        "realpath",
        "getcwd",
        "clock_gettime",
        "rdtsc",
        "rdtscp",
        "thread_local",
        "argv",
        "argc",
        "GetEnv",
        "GetFileSystem",
        "NowMicros",
        "NowNanos",
        "NowSeconds",
        "SleepForMicroseconds",
        "GetCurrentTime",
        "getpid",
        "getppid",
        "syscall",
        "timespec_get",
        "gettimeofday",
        "getrusage",
        "uname",
        "sysinfo",
        "sched_getcpu",
        "gethostname",
        "getuid",
        "geteuid",
        "environ",
        "srand",
        "getrandom",
        "arc4random",
        "random_device",
    }
)
FORBIDDEN_CALL_IDENTIFIERS = frozenset({"clock"})
FORBIDDEN_FRAGMENTS = (
    "/proc",
    "/sys",
    "frontier",
    "harbor",
    "submission",
    "harness",
    "profile",
    "case-file",
    "std::chrono",
    "system_clock",
    "steady_clock",
    "high_resolution_clock",
    "std::random_device",
    "std::filesystem",
    "program_invocation_name",
    "ioptions_.clock",
    "mutable_db_options_.clock",
    "Env::",
    "FileSystem::",
    "std::ifstream",
    "std::ofstream",
)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
PREPROCESSOR_RE = re.compile(r"^\s*(?:#|%:|\?\?=)\s*[A-Za-z_][A-Za-z0-9_]*")


def fail(message: str) -> None:
    print(f"ERROR: invalid solution patch: {message}", file=sys.stderr)
    raise SystemExit(7)


def strip_prefix(path: str) -> str:
    path = path.strip().split("\t", 1)[0]
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def unsafe_path(path: str) -> bool:
    return (
        not path
        or path.startswith("/")
        or "\\" in path
        or "\x00" in path
        or ".." in Path(path).parts
    )


def diff_paths(line: str) -> tuple[str, str]:
    try:
        parts = shlex.split(line)
    except ValueError:
        fail("malformed diff header")
    if len(parts) != 4 or parts[:2] != ["diff", "--git"]:
        fail("malformed diff header")
    return strip_prefix(parts[2]), strip_prefix(parts[3])


def forbidden_added_line(line: str) -> str | None:
    if PREPROCESSOR_RE.search(line):
        return "preprocessor directive"
    lowered = line.lower()
    for fragment in FORBIDDEN_FRAGMENTS:
        if fragment.lower() in lowered:
            return fragment
    identifiers = set(TOKEN_RE.findall(line))
    for identifier in FORBIDDEN_IDENTIFIERS:
        if identifier in identifiers:
            return identifier
    for identifier in FORBIDDEN_CALL_IDENTIFIERS:
        if re.search(rf"\b{re.escape(identifier)}\s*\(", line):
            return f"{identifier}()"
    return None


def validate(text: str) -> None:
    if len(text.encode("utf-8", errors="replace")) > MAX_PATCH_BYTES:
        fail("patch exceeds 256 KiB")
    if not text.strip():
        return
    for marker in STRUCTURAL_MARKERS:
        if marker in text:
            fail(f"forbidden structural change: {marker.strip()}")

    current_path = ""
    old_path = ""
    new_path = ""
    hunks = 0
    changes = 0
    additions = 0
    removals = 0
    changed_files: list[str] = []

    def finish_file() -> None:
        if not current_path:
            return
        if old_path != current_path or new_path != current_path:
            fail(f"patch path mismatch for {current_path}")
        if hunks == 0 or changes == 0:
            fail(f"patch for {current_path} has no changes")

    for raw_line in text.splitlines():
        if raw_line.startswith("diff --git "):
            finish_file()
            diff_old, diff_new = diff_paths(raw_line)
            if diff_old != diff_new:
                fail("renames are not allowed")
            if unsafe_path(diff_new):
                fail(f"unsafe patch path: {diff_new}")
            if diff_new not in ALLOWLIST:
                fail(f"changed file is outside the editable surface: {diff_new}")
            current_path = diff_new
            old_path = new_path = ""
            hunks = changes = additions = removals = 0
            changed_files.append(current_path)
        elif raw_line.startswith("--- "):
            old_path = strip_prefix(raw_line[4:])
        elif raw_line.startswith("+++ "):
            new_path = strip_prefix(raw_line[4:])
        elif raw_line.startswith("@@ "):
            hunks += 1
        elif raw_line.startswith("+") and not raw_line.startswith("+++ "):
            changes += 1
            additions += 1
            reason = forbidden_added_line(raw_line[1:])
            if reason:
                fail(f"forbidden added code in {current_path}: {reason}")
        elif raw_line.startswith("-") and not raw_line.startswith("--- "):
            changes += 1
            removals += 1

    finish_file()
    if not changed_files:
        fail("submission is not a git patch")
    if len(changed_files) > MAX_CHANGED_FILES:
        fail("patch changes too many files")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: validate_solution_patch.py SOLUTION_PATCH", file=sys.stderr)
        return 2
    validate(Path(argv[1]).read_text(encoding="utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
