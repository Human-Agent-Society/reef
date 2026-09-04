"""``reef serve`` override-parsing tests (issue #142)."""

from __future__ import annotations

import pytest

from reef.service.deploy.orchestrator import InvalidOverrideError, _apply_overrides, _parse_overrides, main


def test_key_value_pair() -> None:
    assert _parse_overrides(["--port", "9000"]) == {"port": "9000"}


def test_key_equals_value() -> None:
    assert _parse_overrides(["--port=9000"]) == {"port": "9000"}


def test_dotted_key() -> None:
    assert _parse_overrides(["--training.checkpoint_dir", "/tmp/ckpt"]) == {"training.checkpoint_dir": "/tmp/ckpt"}


def test_apply_overrides_rejects_a_path_through_a_scalar() -> None:
    with pytest.raises(InvalidOverrideError, match=r"reef\.artifact_dir"):
        _apply_overrides({"reef": {"artifact_dir": "/data/artifacts"}}, {"reef.artifact_dir.foo": "1"})


def test_apply_overrides_rejects_a_path_through_a_list() -> None:
    with pytest.raises(InvalidOverrideError, match=r"reef\.adapters"):
        _apply_overrides({"reef": {"adapters": ["primary"]}}, {"reef.adapters.foo": "1"})


def test_apply_overrides_creates_and_merges_nested_sections() -> None:
    assert _apply_overrides(
        {"reef": {}, "training": {"keep": "yes"}},
        {"training.checkpoint_dir": "/tmp/ckpt", "new_section.enabled": "true"},
    ) == {
        "reef": {},
        "training": {"keep": "yes", "checkpoint_dir": "/tmp/ckpt"},
        "new_section": {"enabled": True},
    }


def test_apply_overrides_replaces_a_scalar_at_its_own_path() -> None:
    assert _apply_overrides({"reef": {"artifact_dir": "/data/artifacts"}}, {"artifact_dir": "/other/artifacts"}) == {
        "reef": {"artifact_dir": "/other/artifacts"}
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


def test_cli_exits_with_status_2_before_starting_services_for_an_invalid_path(tmp_path, capsys) -> None:
    config_path = tmp_path / "reef.yaml"
    config_path.write_text("reef:\n  artifact_dir: /data/artifacts\n")

    with pytest.raises(SystemExit) as excinfo:
        main(["--config", str(config_path), "--reef.artifact_dir.foo", "1"])

    assert excinfo.value.code == 2
    assert "reef.artifact_dir" in capsys.readouterr().err
