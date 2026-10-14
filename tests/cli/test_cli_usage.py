"""``reef`` CLI usage errors print to stderr and exit with status 2."""

import subprocess
import sys

import reef


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "reef.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_no_command_prints_usage_to_stderr_and_exits_2() -> None:
    """Running ``reef`` without a command is a usage error."""
    result = _run_cli()

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("usage: reef")


def test_help_flags_print_usage_to_stdout_and_exit_0() -> None:
    """``reef -h`` and ``reef --help`` remain successful help requests."""
    for flag in ("-h", "--help"):
        result = _run_cli(flag)

        assert result.returncode == 0
        assert result.stdout.startswith("usage: reef")
        assert result.stderr == ""


def test_unknown_command_keeps_stderr_and_exit_2() -> None:
    """An unknown command still reports on stderr and exits 2."""
    result = _run_cli("bogus")

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("reef: unknown command 'bogus'")


def test_version_flags_are_unchanged() -> None:
    """``reef -V`` and ``reef --version`` still print to stdout and exit 0."""
    for flag in ("-V", "--version"):
        result = _run_cli(flag)

        assert result.returncode == 0
        assert result.stdout.strip() == f"reef {reef.__version__}"
        assert result.stderr == ""
