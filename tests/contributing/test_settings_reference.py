"""The settings reference page is the output of its generator, and lists every settings class."""

import dataclasses
import importlib
from pathlib import Path
from types import ModuleType

import pytest

from reef.core.config import Settings, config_option


@pytest.fixture
def reference(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / ".github/scripts"))
    return importlib.import_module("check_settings_reference")


@dataclasses.dataclass(frozen=True)
class ExampleSettings(Settings):
    name: str = config_option(help="Name of the pool.")
    retry_seconds: float | None = config_option(None, help="Seconds between two attempts_ of *one* call.")
    workers: int = config_option(2, help="Workers of the pool.", env="EXAMPLE_WORKERS")
    delays: tuple[float, ...] = (1.0,)
    internal: int = 0


@pytest.mark.unit
def test_the_page_is_the_generator_output(reference: ModuleType) -> None:
    assert reference.PAGE.read_text(encoding="utf-8") == reference.render(), f"run: {reference.COMMAND}"


@pytest.mark.unit
def test_every_settings_class_is_listed_or_excluded(reference: ModuleType) -> None:
    reference.check_coverage()
    assert set(reference.EXCLUDED) <= reference.declared_settings_classes(), "EXCLUDED names a class without settings"


@pytest.mark.unit
def test_a_row_gives_the_key_the_type_the_default_and_the_help(reference: ModuleType) -> None:
    table = reference.Table("example", ExampleSettings, undeclared={"delays": "Seconds before each attempt."})
    assert reference.table_rows(table) == [
        "   ``example.name`` : string : required",
        "       Name of the pool.",
        "",
        "   ``example.retry-seconds`` : number : unset",
        r"       Seconds between two attempts\_ of \*one\* call.",
        "",
        "   ``example.workers`` : integer : 2",
        "       Workers of the pool. Environment fallback: ``EXAMPLE_WORKERS``.",
        "",
        "   ``example.delays`` : list : [1.0]",
        "       Seconds before each attempt.",
        "",
    ]


@pytest.mark.unit
def test_a_table_can_omit_a_field_and_keep_underscores(reference: ModuleType) -> None:
    table = reference.Table("example", ExampleSettings, omitted={"name": "the service sets it"}, underscores=True)
    keys = [line.split("``")[1] for line in reference.table_rows(table) if line.startswith("   ``")]
    assert keys == ["example.retry_seconds", "example.workers"]


@pytest.mark.unit
def test_a_table_that_names_a_missing_field_is_refused(reference: ModuleType) -> None:
    table = reference.Table("example", ExampleSettings, omitted={"gone": "removed"})
    with pytest.raises(reference.SettingsReferenceError, match="ExampleSettings has no field gone"):
        reference.table_rows(table)


@pytest.mark.unit
def test_a_setting_without_help_is_refused(reference: ModuleType) -> None:
    with pytest.raises(reference.SettingsReferenceError, match=r"example\.workers has no help text"):
        reference.row("example.workers", "integer", 2, "")
