"""The data source of Slime's rollout manager for dataset jobs: the prompts of the job that the save root records.

Slime's rollout manager builds its data source once and keeps it for the
serving session, across drivers. ``ReefDatasetSource`` therefore reads the
dataset job from ``DATASET_FILE`` (``reef.train.slime_backend.dataset``) when
it serves prompts. A new driver's token loads the job's dataset with Slime's
own ``Dataset``, with the tokenizer, chat template and prompt length limit of
Slime's arguments, and skips the prompts of the rollouts that the job trained
before this driver. Only the rollout manager's process imports this module.
"""

from __future__ import annotations

from typing import Any

from slime.data.data_source import RolloutDataSourceWithBuffer
from slime.utils.data import Dataset
from slime.utils.processing_utils import load_processor, load_tokenizer

from reef.runtime.durable_files import read_json
from reef.train.slime_backend.dataset import DATASET_FILE, save_root


class ReefDatasetSource(RolloutDataSourceWithBuffer):
    """Serve the prompts of the dataset job in the save root's ``DATASET_FILE``, as Slime's own data source does."""

    def __init__(self, args: Any) -> None:
        super().__init__(args)
        self.record_path = save_root(args.save_hf) / DATASET_FILE
        # The token of the driver whose job this source serves.
        self.token: str | None = None
        self.tokenizer: Any = None
        self.processor: Any = None

    def refresh(self) -> None:
        """Load the recorded job's dataset when a new driver recorded it."""
        record = read_json(self.record_path)
        if record is None:
            raise RuntimeError(f"no dataset job is recorded in {self.record_path}")
        if record["token"] == self.token:
            return
        args = self.args
        if self.tokenizer is None:
            self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        self.dataset = Dataset(
            record["path"],
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key,
            multimodal_keys=args.multimodal_keys,
            label_key=args.label_key,
            metadata_key=args.metadata_key,
            tool_key=args.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            seed=args.rollout_seed,
        )
        if not len(self.dataset):
            raise RuntimeError(f"the dataset {record['path']} has no prompt within --rollout-max-prompt-len")
        # The previous job's aborted groups are not this job's prompts.
        self.buffer = []
        self.epoch_id, self.sample_offset = divmod(
            record["skip_rollouts"] * args.rollout_batch_size, len(self.dataset)
        )
        if args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)
        self.token = record["token"]

    def get_samples(self, num_samples: int) -> list[list[Any]]:
        self.refresh()
        return super().get_samples(num_samples)

    def __len__(self) -> int:
        self.refresh()
        return len(self.dataset)


__all__ = ["ReefDatasetSource"]
