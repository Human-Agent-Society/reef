"""``reef serve`` override-parsing tests (issue #142)."""

from __future__ import annotations

import pytest

from reef.service.deploy.orchestrator import InvalidOverrideError, _parse_overrides, main


def test_key_value_pair() -> None:
    assert _parse_overrides(["--port", "9000"]) == {"port": "9000"}


def test_key_equals_value() -> None:
    assert _parse_overrides(["--port=9000"]) == {"port": "9000"}


def test_dotted_key() -> None:
    assert _parse_overrides(["--training.checkpoint_dir", "/tmp/ckpt"]) == {
        "training.checkpoint_dir": "/tmp/ckpt"
    }


def test_bare_boolean_override() -> None:
    assert _parse_overrides(["--verbose"]) == {"verbose": "true"}


def test_bare_boolean_followed_by_another_flag() -> None:
    assert _parse_overrides(["--verbose", "--port", "9000"]) == {
        "verbose": "true",
        "port": "9000",
    }


def test_orphan_positional_argument_is_rejected() -> None:
    with pytest.raises(InvalidOverrideError):
        _parse_overrides(["typo"])


def test_unknown_short_option_is_rejected() -> None:
    with pytest.raises(InvalidOverrideError):
        _parse_overrides(["-p"])


def test_empty_override_name_is_rejected() -> None:
    with pytest.raises(InvalidOverrideError):
        _parse_overrides(["--=value"])


def test_bare_double_dash_is_rejected() -> None:
    with pytest.raises(InvalidOverrideError):
        _parse_overrides(["--"])


def test_valid_override_after_an_invalid_one_does_not_mask_the_error() -> None:
    with pytest.raises(InvalidOverrideError):
        _parse_overrides(["typo", "--port", "9000"])


def test_cli_exits_with_status_2_on_orphan_argument(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["typo"])
    assert excinfo.value.code == 2
    assert "typo" in capsys.readouterr().err


def test_cli_exits_with_status_2_on_unknown_short_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["-p"])
    assert excinfo.value.code == 2
