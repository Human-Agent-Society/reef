"""Shared parts that data processors use to hold, order, and reserve training units.

Each processor makes all policy decisions. It decides which records become
units, when a unit is ready, which selection policy orders the units, and what
a consumed unit releases. This package supplies only the shared parts:

* ``ExperienceUnit``: one sample or one ready group. A processor subclasses it
  to add its own data, such as the sample or the reports of a group.
* ``ExperienceBuffer``: the units that a processor holds, and its reserved batch.
* ``SelectionPolicy``: the order in which units go into a batch.
  ``ArrivalOrder`` takes the oldest first; ``GroupKeyOrder`` takes ungrouped
  units, then groups by key.

To change the selection, a processor overrides ``selection_policy``.
"""

from reef.train.experience.buffer import ExperienceBuffer
from reef.train.experience.selection import ArrivalOrder, ExperienceUnit, GroupKeyOrder, SelectionPolicy

__all__ = [
    "ArrivalOrder",
    "ExperienceBuffer",
    "ExperienceUnit",
    "GroupKeyOrder",
    "SelectionPolicy",
]
