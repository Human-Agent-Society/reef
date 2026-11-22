"""Transport-independent handling of Harbor task archives and of a scenario's task listing.

A client creates task items from a gzip-compressed tar archive whose top-level entries are Harbor task
directories, one for each task, named by the task (``reef.control.operations.data``). ``unpack_task_archive`` unpacks and
checks such an archive. ``list_imported_tasks`` lists the tasks that a scenario uses. No aiohttp types appear
here. ``reef.api.routes.tasks`` adapts the listing to HTTP.
"""

from __future__ import annotations

import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from reef.control.dispatcher import Dispatcher
from reef.core.tasks.harbor import TASK_NAME_PATTERN
from reef.storage.data.tasks import task_limit_errors

#: The most tasks in one upload.
MAX_TASKS_PER_UPLOAD = 1000
#: The default of ``data.max_task_archive_bytes``: the most bytes of the body of a task archive upload. The route
#: streams the body to a local file, so the limit bounds disk use, not memory.
DEFAULT_MAX_TASK_ARCHIVE_BYTES = 512 * 1024 * 1024
#: The most bytes that the files of one archive unpack to. A small gzip body can unpack to far more than its size,
#: and the route unpacks the whole archive to local disk. A larger task uses the reference form.
MAX_ARCHIVE_UNPACKED_BYTES = 8 * 1024**3


def unpack_task_archive(archive: Path, target: Path) -> tuple[Path, ...]:
    """Unpack the tar.gz file ``archive`` into ``target`` and return its top-level task directories, in name order.

    Every entry is a regular file or a directory under a top-level directory whose name is a task name
    (``TASK_NAME_PATTERN``); a top-level file, a link, a device, an absolute path, a ``..`` component or another
    top-level name is refused with ``ValueError``, as is an archive of no task or of more than
    ``MAX_TASKS_PER_UPLOAD`` tasks, of files that unpack to more than ``MAX_ARCHIVE_UNPACKED_BYTES``, and a task that
    breaks a limit of ``reef.storage.data.tasks.task_limit_errors``.
    Nothing is unpacked then.
    """
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            members = tar.getmembers()
            tasks: dict[str, list[tuple[str, int | None]]] = {}
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] in ("", "."):
                    raise ValueError(
                        f"the archive entry {member.name!r} is not a relative path under a task directory"
                    )
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"the archive entry {member.name!r} is not a file or a directory")
                if len(path.parts) == 1 and not member.isdir():
                    raise ValueError(f"the archive entry {member.name!r} is a file outside a task directory")
                entries = tasks.setdefault(path.parts[0], [])
                if len(path.parts) > 1:
                    entries.append(("/".join(path.parts[1:]), member.size if member.isfile() else None))
            if not tasks:
                raise ValueError("the archive holds no task")
            for name in sorted(tasks):
                if not TASK_NAME_PATTERN.fullmatch(name):
                    raise ValueError(f"{name!r} is not a task name: it must match {TASK_NAME_PATTERN.pattern}")
            if len(tasks) > MAX_TASKS_PER_UPLOAD:
                raise ValueError(f"an upload holds at most {MAX_TASKS_PER_UPLOAD} tasks")
            if sum(member.size for member in members if member.isfile()) > MAX_ARCHIVE_UNPACKED_BYTES:
                raise ValueError(
                    f"the archive unpacks to more than {MAX_ARCHIVE_UNPACKED_BYTES} bytes; "
                    "create larger tasks by reference to the task file store"
                )
            for name, entries in sorted(tasks.items()):
                errors = task_limit_errors(entries)
                if errors:
                    raise ValueError(f"task {name!r} breaks a limit: " + " ".join(error.message for error in errors))
            tar.extractall(target, filter="data")
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise ValueError(f"the body is not a gzip-compressed tar archive: {exc}") from exc
    return tuple(target / name for name in sorted(tasks))


def list_imported_tasks(dispatcher: Dispatcher, scenario: str) -> dict[str, Any]:
    """The scenario's uses of imported tasks in the order that they began: ``{scenario, tasks: [{name, split,
    benchmark, files_digest, trained}]}``.

    ``benchmark`` is the held benchmark that gives the use, or null for the task's own mark; a task that the scenario
    uses more than once is listed once for each use (RFC #1420). ``trained`` is whether a committed job of any
    trainer of the scenario trained the task. The listing reads no task files.
    """
    return {
        "scenario": scenario,
        "tasks": [
            {
                "name": task.sample_id,
                "split": task.split,
                "benchmark": task.held_set,
                "files_digest": task.digest,
                "trained": bool(task.train_counts),
            }
            for task in dispatcher.listed_imported_tasks(scenario)
        ],
    }
