"""Transport-independent handling of Harbor task archives and of a scenario's task listing.

A client creates task items from a gzip-compressed tar archive whose top-level entries are Harbor task
directories, one for each task, named by the task (``reef.service.data``). ``unpack_task_archive`` unpacks and
checks such an archive. ``list_imported_tasks`` lists the tasks that a scenario uses. No aiohttp types appear
here. ``reef.service.routes.tasks`` adapts the listing to HTTP.
"""

from __future__ import annotations

import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


from reef.core.tasks.harbor import TASK_NAME_PATTERN
from reef.dispatcher import Dispatcher

#: The most tasks in one upload.
MAX_TASKS_PER_UPLOAD = 1000
#: The most archive entries, and the most bytes they unpack to. The route also refuses a body larger than
#: ``MAX_UNPACKED_BYTES``; it streams the body to a file, so the limit bounds disk use, not memory.
MAX_ARCHIVE_ENTRIES = 10_000
MAX_UNPACKED_BYTES = 256 * 1024 * 1024


def unpack_task_archive(archive: Path, target: Path) -> tuple[Path, ...]:
    """Unpack the tar.gz file ``archive`` into ``target`` and return its top-level task directories, in name order.

    Every entry is a regular file or a directory under a top-level directory whose name is a task name
    (``TASK_NAME_PATTERN``); a top-level file, a link, a device, an absolute path, a ``..`` component or another
    top-level name is refused with ``ValueError``, as is an archive of no task, of more
    than ``MAX_TASKS_PER_UPLOAD`` tasks, of more than ``MAX_ARCHIVE_ENTRIES`` entries, or of more than
    ``MAX_UNPACKED_BYTES`` bytes.
    """
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            members = tar.getmembers()
            if len(members) > MAX_ARCHIVE_ENTRIES:
                raise ValueError(f"the archive holds more than {MAX_ARCHIVE_ENTRIES} entries")
            if sum(member.size for member in members) > MAX_UNPACKED_BYTES:
                raise ValueError(f"the archive unpacks to more than {MAX_UNPACKED_BYTES} bytes")
            names: set[str] = set()
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
                names.add(path.parts[0])
            if not names:
                raise ValueError("the archive holds no task")
            for name in sorted(names):
                if not TASK_NAME_PATTERN.fullmatch(name):
                    raise ValueError(f"{name!r} is not a task name: it must match {TASK_NAME_PATTERN.pattern}")
            if len(names) > MAX_TASKS_PER_UPLOAD:
                raise ValueError(f"an upload holds at most {MAX_TASKS_PER_UPLOAD} tasks")
            tar.extractall(target, filter="data")
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise ValueError(f"the body is not a gzip-compressed tar archive: {exc}") from exc
    return tuple(target / name for name in sorted(names))


def list_imported_tasks(dispatcher: Dispatcher, scenario: str) -> dict[str, Any]:
    """The scenario's imported tasks in arrival order: ``{scenario, tasks: [{name, split, files_digest, trained}]}``.

    ``trained`` is whether a committed job of any trainer of the scenario trained the task.
    """
    return {
        "scenario": scenario,
        "tasks": [
            {
                "name": task.sample_id,
                "split": task.split,
                "files_digest": task.files_digest,
                "trained": bool(task.train_counts),
            }
            for task in dispatcher.imported_tasks(scenario)
        ],
    }
