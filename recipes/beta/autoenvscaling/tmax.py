"""Convert extracted TMax tasks into seed tasks: Harbor tasks that reef writes, for ``seed_tasks_path``.

TMax (Ivison et al., 2026) releases TMax-15K as ``tasks.zip`` (``allenai/TMax-15K``). Unzip it yourself, then run::

    python -m recipes.beta.autoenvscaling.tmax --tasks <extracted dir> --out <seed dir> [--select ids.json]

Reimplemented from AutoEnvScaling's ``baselines/tmax.py`` (Apache 2.0) on reef's task writer. It reads only the
extracted directory and downloads nothing. Each task directory with a ``task.json`` becomes one Harbor task:

- ``container.def``, a Singularity definition bootstrapped from a Docker image, becomes ``environment/Dockerfile``:
  ``From:`` becomes ``FROM``, each ``%environment`` export an ``ENV`` line, each ``%files`` entry a ``COPY`` of the
  file of that name in the task directory, and ``%post`` the build script ``tmax_build.sh``, which one ``RUN`` runs.
  The Dockerfile then installs tmux, which the solver's agent needs, and pytest, which the verifier needs: neither
  has network.
- ``task.json``'s ``description`` becomes ``instruction.md``, and ``test_final_state.py`` becomes
  ``tests/test_outputs.py``. ``tests/test.sh`` is ``tmax_verifier.sh``: the reward is the fraction of the tests
  that pass.
- TMax ships no reference solution, so ``solution/solve.sh`` does nothing: its oracle trial scores the untouched
  environment.

A task with a binary fixture (a reef task holds text files only), no ``From:`` image, no description or no
final-state tests is refused. ``test_initial_state.py`` is not used. ``conversion.json`` in the output directory
lists the converted tasks with the sections of ``container.def`` they leave out, and the refused tasks with the
reason.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from reef.core.tasks import HarborTask, HarborTaskError, write_harbor_task
from reef.core.tasks.harbor import TASK_NAME_PATTERN

VERIFIER = Path(__file__).with_name("tmax_verifier.sh")
CONVERSION_FILE = "conversion.json"
FIXTURES_DIR = "fixtures"
BUILD_SCRIPT = "tmax_build.sh"
# The container.def sections the conversion reads; the others are listed as left out.
HANDLED_SECTIONS = ("files", "post", "environment")
TASK_JSON_KEYS = ("domain", "tier", "skill_type")
MAX_NAME_CHARS = 100
# Upstream's budgets: the build and the verifier stay inside the generator service's bound on one oracle trial.
TASK_CONFIG: dict[str, dict[str, object]] = {
    "agent": {"timeout_sec": 1200},
    "verifier": {"timeout_sec": 600},
    "environment": {"network_mode": "no-network", "build_timeout_sec": 900},
}
# The agent's container and the verifier have no network, so the image carries what each one runs.
TOOL_STEPS = (
    "# Reef: the solver's agent needs tmux, and the verifier needs pytest; neither has network.",
    "RUN command -v tmux >/dev/null 2>&1"
    " || (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tmux)",
    "RUN python3 -m pytest --version >/dev/null 2>&1 || python3 -m pip install -q pytest"
    " || python3 -m pip install -q --break-system-packages pytest"
    " || (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pytest)",
)
SOLVE_SCRIPT = """#!/bin/bash
# TMax ships no reference solution. This one does nothing, so its oracle trial scores the untouched environment.
exit 0
"""


@dataclass(frozen=True)
class ContainerDefinition:
    """What the conversion reads from a TMax task's ``container.def``.

    Values: ``image`` is the ``From:`` value of the header, or empty when there is none. ``files`` holds each
    ``%files`` entry as (source, destination); an entry with one path copies it to the same path. ``post`` holds
    the ``%post`` lines as written, and ``environment`` the ``%environment`` exports without ``export``.
    ``unhandled_sections`` names, in order, the sections not in :data:`HANDLED_SECTIONS`.
    Purpose: :func:`container_definition` makes it, and :func:`converted_task` writes the Dockerfile from it.
    """

    image: str
    files: tuple[tuple[str, str], ...]
    post: tuple[str, ...]
    environment: tuple[str, ...]
    unhandled_sections: tuple[str, ...]


def container_definition(text: str) -> ContainerDefinition:
    """The image, files, build lines and exports of a Singularity definition, and the sections it leaves out."""
    image = ""
    files: list[tuple[str, str]] = []
    post: list[str] = []
    environment: list[str] = []
    unhandled: list[str] = []
    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not section and stripped.lower().startswith("from:"):
            image = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("%"):
            section = stripped.split()[0][1:].lower()
            if section not in HANDLED_SECTIONS and section not in unhandled:
                unhandled.append(section)
        elif section == "files" and stripped:
            paths = stripped.split()
            files.append((paths[0], paths[1] if len(paths) > 1 else paths[0]))
        elif section == "post":
            post.append(line)
        elif section == "environment" and stripped.startswith("export "):
            environment.append(stripped.removeprefix("export ").strip())
    return ContainerDefinition(image, tuple(files), tuple(post), tuple(environment), tuple(unhandled))


def task_name(task_id: str) -> str:
    """The Harbor task name of a TMax task id: lowercase, each run of other characters one hyphen.

    Raises ``ValueError`` when nothing of the id is left.
    """
    name = re.sub(r"[^a-z0-9]+", "-", task_id.lower())[:MAX_NAME_CHARS].strip("-")
    if not TASK_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"the task id {task_id!r} gives no task name")
    return name


def text_file(path: Path) -> str:
    """The file's text as its bytes hold it. Raises ``ValueError`` for a file that is not UTF-8 text."""
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name} is a binary file; a reef task holds text files only") from exc


def converted_task(task_dir: Path, verifier: str) -> tuple[HarborTask, ContainerDefinition]:
    """The seed task of one extracted TMax task directory, with ``verifier`` as its ``tests/test.sh``, and its
    container definition. Raises ``ValueError`` for a task it refuses."""
    for required in ("container.def", "task.json", "test_final_state.py"):
        if not (task_dir / required).is_file():
            raise ValueError(f"the task has no {required}")
    definition = container_definition(text_file(task_dir / "container.def"))
    if not definition.image:
        raise ValueError("container.def has no From: image")
    task_json = json.loads(text_file(task_dir / "task.json"))
    description = task_json.get("description") if isinstance(task_json, dict) else None
    if not isinstance(description, str) or not description.strip():
        raise ValueError("task.json has no description")
    environment: dict[str, str] = {}
    copies: list[str] = []
    for source, destination in definition.files:
        # A %files source is a path on the TMax authors' machine; the file itself ships in the task directory.
        file_name = Path(source).name
        local = next((path for path in sorted(task_dir.rglob(file_name)) if path.is_file()), None)
        if local is None:
            raise ValueError(f"the fixture {source} is not in the task directory")
        environment[f"{FIXTURES_DIR}/{file_name}"] = text_file(local)
        copies.append(f"COPY {FIXTURES_DIR}/{file_name} {destination}")
    environment[BUILD_SCRIPT] = "#!/bin/bash\nset -e\n" + "".join(f"{line}\n" for line in definition.post)
    environment["Dockerfile"] = (
        "\n".join(
            (
                f"FROM {definition.image}",
                *(f"ENV {export}" for export in definition.environment),
                *copies,
                f"COPY {BUILD_SCRIPT} /tmp/{BUILD_SCRIPT}",
                f"RUN bash /tmp/{BUILD_SCRIPT} && rm -f /tmp/{BUILD_SCRIPT}",
                *TOOL_STEPS,
                "WORKDIR /app",
            )
        )
        + "\n"
    )
    metadata: dict[str, object] = {"tmax_task": task_dir.name}
    metadata.update({key: task_json[key] for key in TASK_JSON_KEYS if isinstance(task_json.get(key), str)})
    try:
        task = HarborTask(
            name=task_name(task_dir.name),
            instruction=description,
            tests={"test.sh": verifier, "test_outputs.py": text_file(task_dir / "test_final_state.py")},
            environment=environment,
            config=TASK_CONFIG,
            metadata=metadata,
            solution={"solve.sh": SOLVE_SCRIPT},
        )
    except HarborTaskError as exc:
        raise ValueError(str(exc)) from exc
    return task, definition


def convert(
    tasks_dir: Path, out_dir: Path, task_ids: Sequence[str] | None = None
) -> dict[str, list[dict[str, object]]]:
    """Write the seed task of each TMax task in ``tasks_dir`` (those ``task_ids`` names, else every directory
    with a ``task.json``) under ``out_dir``, and the conversion's result as ``out_dir/conversion.json``.

    A task that is refused, or whose name another task took, is listed with the reason and not written. A task
    written before with the same content is kept; one with different content is refused.
    """
    if task_ids is None:
        task_ids = sorted(path.name for path in tasks_dir.iterdir() if (path / "task.json").is_file())
    verifier = VERIFIER.read_text(encoding="utf-8")
    converted: list[dict[str, object]] = []
    refused: list[dict[str, object]] = []
    names: dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_id in task_ids:
        task_dir = tasks_dir / task_id
        try:
            if not task_dir.is_dir():
                raise ValueError("no task directory has this id")
            task, definition = converted_task(task_dir, verifier)
            if task.name in names:
                raise ValueError(f"its task name {task.name} is taken by the task {names[task.name]}")
            write_harbor_task(task, out_dir)
        except (ValueError, HarborTaskError) as exc:
            refused.append({"task_id": task_id, "reason": str(exc)})
            continue
        names[task.name] = task_id
        converted.append(
            {
                "task_id": task_id,
                "name": task.name,
                "image": definition.image,
                "fixtures": len(definition.files),
                "unhandled_sections": list(definition.unhandled_sections),
            }
        )
    result = {"converted": converted, "refused": refused}
    (out_dir / CONVERSION_FILE).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """The command: 0 when every task converted, 1 when one was refused."""
    parser = argparse.ArgumentParser(description="Convert extracted TMax tasks into AutoEnvScaling seed tasks.")
    parser.add_argument("--tasks", type=Path, required=True, help="the extracted TMax tasks, one directory each")
    parser.add_argument("--out", type=Path, required=True, help="the seed directory, for seed_tasks_path")
    parser.add_argument("--select", type=Path, help="a JSON list of the task ids to convert; default: every task")
    arguments = parser.parse_args(argv)
    task_ids = None
    if arguments.select is not None:
        task_ids = json.loads(arguments.select.read_text(encoding="utf-8"))
        if not isinstance(task_ids, list) or not all(isinstance(task_id, str) for task_id in task_ids):
            parser.error("--select must name a JSON list of task ids")
    result = convert(arguments.tasks, arguments.out, task_ids)
    for entry in result["refused"]:
        print(f"refused {entry['task_id']}: {entry['reason']}")
    total = len(result["converted"]) + len(result["refused"])
    print(f"converted {len(result['converted'])} of {total} tasks to {arguments.out}")
    return 1 if result["refused"] else 0


if __name__ == "__main__":
    sys.exit(main())
