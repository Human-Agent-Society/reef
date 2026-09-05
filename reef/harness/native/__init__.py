"""The old home of the native runner, kept for what is baked into installed wrappers.

The runner lives in ``reef.harness.runners.native``. Wrappers written by
earlier tutorials exec ``python -m reef.harness.native``, and configs name
``reef.harness.native.seed:SEED_NODES``, so this package forwards those two
entry points for one release and nothing else.
"""

from reef.harness.runners.native import main, run_loop

__all__ = ["main", "run_loop"]
