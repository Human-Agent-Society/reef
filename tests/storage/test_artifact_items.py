"""Artifact items: the content of the workspace artifact library's items (reef.storage.artifact_items)."""

from __future__ import annotations

import pytest

from reef.storage.artifact_items import artifact_item
from reef.storage.item_kinds import check_artifact_content

ROW = {"artifact_id": "a1", "component": "weights", "content_id": "c1"}


@pytest.mark.unit
def test_the_item_of_a_scenarios_artifact_names_where_it_came_from_and_where_its_files_are() -> None:
    item = artifact_item("coding", ROW, "model", "s3://bucket/coding/r1/")
    assert item.item_id == "coding/a1"
    assert item.content == {
        "source_uri": "s3://bucket/coding/r1/",
        "kind": "model",
        "component": "weights",
    }
    assert item.metadata == {
        "kind": "artifact",
        "origin": {"scenario": "coding", "artifact_id": "a1"},
    }
    # A repository that gives no URI, and a component without a kind, leave those null.
    assert artifact_item("coding", ROW, None, None).content["source_uri"] is None


@pytest.mark.unit
def test_an_artifact_items_content_is_checked() -> None:
    uploaded = {"source_uri": "s3://bucket/u/", "kind": None, "component": None}
    check_artifact_content(uploaded, made_by_scenario=False)
    for content, made_by_scenario, reason in (
        ({**uploaded, "extra": 1}, False, "holds exactly"),
        ({**uploaded, "source_uri": None}, False, "names its source_uri"),
        ({**uploaded, "component": "weights"}, False, "no component"),
        ({**uploaded, "kind": "dataset"}, False, "content.kind"),
        ({**uploaded, "kind": ""}, False, "non-empty string or null"),
        (uploaded, True, "names its component"),
    ):
        with pytest.raises(ValueError, match=reason):
            check_artifact_content(content, made_by_scenario=made_by_scenario)


def test_only_the_item_of_a_harness_tree_names_its_adapter() -> None:
    assert artifact_item("coding", ROW, "harness_plugin", None, "pi").metadata["adapter"] == "pi"
    assert "adapter" not in artifact_item("coding", ROW, "model", None, "pi").metadata
    assert "adapter" not in artifact_item("coding", ROW, "harness_plugin", None).metadata
