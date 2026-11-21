"""Fixtures for Slime backend tests."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest


@pytest.fixture
def make_dataset_source(monkeypatch):
    """Create a Slime data source for file prompts that need no tokenizer."""
    pytest.importorskip("slime.data.data_source", reason="requires Slime's optional dataset dependencies")
    from reef.train.slime_backend import dataset_source

    monkeypatch.setattr(dataset_source, "load_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(dataset_source, "load_processor", lambda *args, **kwargs: None)

    def make_source(root: Path, *, rollout_batch_size: int):
        return dataset_source.ReefDatasetSource(
            Namespace(
                save_hf=str(root / "hf" / "{rollout_id}"),
                prompt_data=None,
                hf_checkpoint="/models/m",
                rollout_max_prompt_len=None,
                input_key="input",
                multimodal_keys=None,
                label_key="label",
                metadata_key="metadata",
                tool_key=None,
                apply_chat_template=False,
                apply_chat_template_kwargs=None,
                rollout_seed=0,
                rollout_shuffle=False,
                rollout_batch_size=rollout_batch_size,
                n_samples_per_prompt=2,
                buffer_filter_path=None,
                dump_details=None,
            )
        )

    return make_source
