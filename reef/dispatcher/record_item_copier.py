"""The record item copier: one thread that makes each loaded scenario's records into record items (RFC #1028).

A record commits to the scenario's record store first, and its record item follows. The copier keeps, for each
scenario ``s``, the invariant of ``ScenarioSettings.record_items_sequence``: each record of ``s`` with a sequence at
most that value has its record item, if ``reef.storage.record_items`` gives it one, in the store of the owner of
``s``. A copy page creates the items, then increases the value, so a crash between the two creates the same items
again, which changes nothing. The first copy of a scenario with old records moves them in pages; the scenario
serves meanwhile.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from threading import Condition, Thread

from reef.scenario.registry import ScenarioRegistry
from reef.storage.data import DataStore
from reef.storage.record_items import copy_records
from reef.storage.scenario import ScenarioStorage

logger = logging.getLogger(__name__)


class RecordItemCopier:
    """The thread that copies the records of the scenarios that ``wake`` names into ``data``.

    Values: ``pending`` is the set of scenario names to copy, and ``closed`` is True after ``close``.

    Purpose: the dispatcher wakes the copier when it loads a scenario or accepts records, and closes it before it
    closes the scenarios and the data store.

    Lock order: registry scenario lock -> ``condition``. The thread never holds ``condition`` while it copies.
    """

    def __init__(self, registry: ScenarioRegistry, storage: ScenarioStorage, data: DataStore) -> None:
        self.registry = registry
        self.storage = storage
        self.data = data
        self.condition = Condition()
        self.pending: set[str] = set()
        self.closed = False
        self.thread = Thread(target=self.run, name="reef-record-items", daemon=True)
        self.thread.start()

    def wake(self, scenario: str) -> None:
        """Copy the records of ``scenario`` that are not record items yet, soon."""
        with self.condition:
            self.pending.add(scenario)
            self.condition.notify()

    def run(self) -> None:
        while True:
            with self.condition:
                while not self.pending and not self.closed:
                    self.condition.wait()
                if self.closed:
                    return
                scenario = min(self.pending)
                self.pending.discard(scenario)
            try:
                self.copy_scenario(scenario)
            except Exception:
                # The records stay; the next wake of the scenario tries again from the same sequence.
                logger.exception("record items of scenario %r are not up to date", scenario)

    def copy_scenario(self, scenario: str) -> None:
        """Copy the records of ``scenario`` in pages until all are record items, or the scenario unloads.

        Each page holds the scenario's lock, so the settings that the page writes do not race other settings
        changes. The lock is free between pages.
        """
        while not self.closed:
            with self.registry.lock_for(scenario):
                current = self.registry.get_loaded(scenario)
                if current is None:
                    return
                settings = self.storage.read_settings(scenario)
                if settings.owner is None:
                    return
                sequence = copy_records(
                    current.store.records, self.data, settings.owner, scenario, settings.record_items_sequence
                )
                if sequence == settings.record_items_sequence:
                    return
                self.storage.write_settings(scenario, replace(settings, record_items_sequence=sequence))

    def close(self) -> None:
        """Stop the thread after the page that it copies; the next start copies the rest."""
        with self.condition:
            self.closed = True
            self.condition.notify()
        self.thread.join()


__all__ = ["RecordItemCopier"]
