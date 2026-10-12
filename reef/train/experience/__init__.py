"""Shared parts that the processor engines use to hold, order, and reserve training units.

Each processor makes all policy decisions. It decides which records become
units, when a unit is ready, which selection policy orders the units, and what
a consumed unit releases. This package supplies only the shared parts:

* ``ExperienceUnit``: one sample or one ready group. A processor subclasses it
  to add its own data, such as the sample or the reports of a group.
* ``ExperienceBuffer``: the units that a processor holds, and its reserved batch.
* ``EligibilityCheck``: finds held units that can no longer train.
  ``StalenessCheck`` drops units that are more than ``max_staleness``
  versions behind the newest held unit.
* ``SelectionPolicy``: the order in which units go into a batch.
  ``ArrivalOrder`` takes the oldest first; ``GroupKeyOrder`` takes ungrouped
  units, then groups by key.

These parts are internal to the processor engines; they are not a recipe
extension contract. Each engine builds its buffer with a fixed selection policy
and fixed checks.
"""

from reef.train.experience.buffer import ExperienceBuffer
from reef.train.experience.eligibility import EligibilityCheck, IneligibleUnit, StalenessCheck
from reef.train.experience.selection import ArrivalOrder, ExperienceUnit, GroupKeyOrder, SelectionPolicy

__all__ = [
    "ArrivalOrder",
    "EligibilityCheck",
    "ExperienceBuffer",
    "ExperienceUnit",
    "GroupKeyOrder",
    "IneligibleUnit",
    "SelectionPolicy",
    "StalenessCheck",
]
