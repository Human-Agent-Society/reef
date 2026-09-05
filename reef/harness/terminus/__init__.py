"""``reef-terminus`` as installed before this move binds ``reef.harness.terminus:main``; the program is ``reef.harness.runners.terminus``.

A console script is written at install time and keeps the module path it
was given, so an environment that predates the move launches terminus
episodes through this name until it is reinstalled.
"""

from reef.harness.runners.terminus import main

__all__ = ["main"]
