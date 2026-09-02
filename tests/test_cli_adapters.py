"""``reef adapters`` lists every harness adapter this installation resolves."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

from reef.cli import main


def _capture(argv):
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit) as excinfo:
        main(argv)
    return excinfo.value.code, buf.getvalue()


def test_adapters_default_table_lists_bundled_names():
    code, out = _capture(["adapters"])
    assert code == 0
    # Header plus one line per bundled adapter, whichever ones are compiled in.
    lines = out.splitlines()
    assert lines[0].startswith("name")
    names = {line.split()[0] for line in lines[1:] if line.strip()}
    assert {"claude", "opencode", "pi"}.issubset(names)


def test_adapters_json_format_carries_install_pins():
    code, out = _capture(["adapters", "--format", "json"])
    assert code == 0
    payload = json.loads(out)
    by_name = {entry["name"]: entry for entry in payload["adapters"]}
    pi = by_name["pi"]
    assert pi["binary"] == "pi"
    assert pi["trajectory_format"] == "pi-session-jsonl"
    install = pi["install"]
    assert install["kind"] == "npm"
    assert install["package"].startswith("@")


def test_adapters_rejects_unknown_format():
    code, _ = _capture(["adapters", "--format=yaml"])
    assert code == 2
