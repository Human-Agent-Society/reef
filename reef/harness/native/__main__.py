"""``python -m reef.harness.native``: the same entry the ``reef-native`` console script uses."""

from __future__ import annotations

import sys

from reef.harness.native import main

if __name__ == "__main__":
    sys.exit(main())
