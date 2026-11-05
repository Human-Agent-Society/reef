"""The data source of Slime's rollout manager for dataset jobs and task jobs: the prompts the save root records.

Slime's rollout manager builds its data source once and keeps it for the
serving session, across drivers. ``ReefDatasetSource`` therefore reads the job
from ``DATASET_FILE`` (``reef.train.slime_backend.dataset``) when it serves
prompts. A new driver's token loads the job's prompts and skips those of the
rollouts that the job trained before this driver. A dataset job's prompts come
from Slime's own ``Dataset``, with the tokenizer, chat template and prompt
length limit of Slime's arguments. A task job's prompts are its tasks
(``reef.train.slime_backend.harbor_rollouts``): each names its task and the
generate function that runs a Harbor rollout of it. One class serves both kinds,
since a serving session has one data source and can run both kinds of job. Only
the rollout manager's process imports this module.
"""

from __future__ import annotations

from typing import Any

from slime.data.data_source import RolloutDataSourceWithBuffer
from slime.utils.data import Dataset
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

from reef.storage.durable_files import read_json
from reef.train.slime_backend.dataset import DATASET_FILE, save_root
from reef.train.slime_backend.harbor_rollouts import GENERATE_FUNCTION_PATH, TaskJobSource, TaskPrompts


class ReefDatasetSource(RolloutDataSourceWithBuffer):
    """Serve the prompts of the job in the save root's ``DATASET_FILE``, as Slime's own data source does."""

    def __init__(self, args: Any) -> None:
        super().__init__(args)
        self.record_path = save_root(args.save_hf) / DATASET_FILE
        # The token of the driver whose job this source serves.
        self.token: str | None = None
        self.tokenizer: Any = None
        self.processor: Any = None

    def load_dataset(self, path: str) -> Dataset:
        """The prompts of a dataset job's JSONL file ``path``, as Slime's own data source reads them."""
        args = self.args
        if self.tokenizer is None:
            self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        dataset = Dataset(
            path,
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
        if not len(dataset):
            raise RuntimeError(f"the dataset {path} has no prompt within --rollout-max-prompt-len")
        return dataset

    def refresh(self) -> None:
        """Load the recorded job's prompts when a new driver recorded it."""
        record = read_json(self.record_path)
        if record is None:
            raise RuntimeError(f"no dataset job is recorded in {self.record_path}")
        if record["token"] == self.token:
            return
        args = self.args
        if "tasks" in record:
            self.dataset = TaskPrompts(
                [
                    Sample(
                        prompt=name,
                        generate_function_path=GENERATE_FUNCTION_PATH,
                        metadata={"task": name, "path": path, "token": record["token"]},
                    )
                    for name, path in TaskJobSource.from_record(record).tasks
                ],
                args.rollout_seed,
            )
        else:
            self.dataset = self.load_dataset(record["path"])
        # The previous job's aborted groups are not this job's prompts.
        self.buffer: list[Any] = []
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
