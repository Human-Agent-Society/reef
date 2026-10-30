"""The coordination store of a deployment, as one process opens it.

Every service of a deployment opens the same coordination store: the ``served_models`` table, the work queues and
the control lease. ``reef.storage.sql_coordination.open_coordination`` opens it in SQL. This module names only the
contracts, so a component that receives the store does not load a database adapter.
"""

from __future__ import annotations

import os
import socket
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

from reef.storage.control_lease import ControlLease
from reef.storage.served_models import ServedModelsWriter
from reef.storage.work_queue import WorkQueue


class CoordinationResource(ABC):
    """What holds the coordination tables and must be released: a database connection pool, or nothing.

    Purpose: ``Coordination.close`` releases it; ``reef.storage.sql_coordination.CoordinationDatabase`` is one.
    """

    @abstractmethod
    def close(self) -> None:
        """Release the resource; safe to call repeatedly."""


def instance_id(role: str) -> str:
    """A name for one process of one role, unique across hosts and restarts: ``<role>@<host>:<pid>:<random>``."""
    return f"{role}@{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


@dataclass
class Coordination:
    """The coordination store of a deployment, opened by one process.

    Values: ``database`` holds the tables; ``work_queue``, ``served_models`` and ``control_lease`` are the stores
    on it.

    Purpose: assembly opens it once per process (``open_coordination``) and gives each service the store it uses.
    """

    database: CoordinationResource
    work_queue: WorkQueue
    served_models: ServedModelsWriter
    control_lease: ControlLease

    def close(self) -> None:
        self.database.close()


__all__ = ["Coordination", "CoordinationResource", "instance_id"]
