"""``reef --version`` reports the installed distribution version."""

import subprocess
import sys

import reef


def test_version_flag_prints_package_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "reef.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == f"reef {reef.__version__}"


def test_version_is_pep440_like() -> None:
    # setuptools-scm output: a tag ("0.4.0") or a dev version past one.
    assert reef.__version__[0].isdigit()
