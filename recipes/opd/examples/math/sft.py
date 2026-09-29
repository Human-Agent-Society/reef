"""Prepare a full-parameter Qwen3.5 language-model SFT starting checkpoint.

Launch with torchrun on four GPUs. The unused vision encoder stays frozen;
all text decoder, embedding and output parameters train, without adapters.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForImageTextToText, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments


class TokenizePair:
    """Mask the observed prompt and preserve a truncated response without inventing EOS."""

    def __init__(self, tokenizer_path: str, max_length: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length

    def __call__(self, batch: dict) -> dict:
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for messages in batch["messages"]:
            if [message["role"] for message in messages] != ["user", "assistant"]:
                raise ValueError("The OpenThoughts3 initializer expects one user/assistant pair")
            prefix_text = self.tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            if not full_text.startswith(prefix_text):
                raise ValueError("SFT prefix differs from the inference template; check thinking-token alignment")
            # Tokenize the observed prefix separately, as during generation.
            # This prevents a newline merge across the observation/target boundary.
            prefix = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            response = self.tokenizer.encode(
                full_text[len(prefix_text) :].removesuffix("\n"), add_special_tokens=False
            )
            tokens = (prefix + response)[: self.max_length]
            if len(prefix) >= len(tokens):
                raise ValueError("SFT sequence has no supervised response within the token budget")
            result["input_ids"].append(tokens)
            result["attention_mask"].append([1] * len(tokens))
            result["labels"].append([-100] * len(prefix) + tokens[len(prefix) :])
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/work/models/Qwen3.5-9B-Base")
    parser.add_argument("--tokenizer", default="/work/models/Qwen3.5-9B")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenize-only", action="store_true")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument(
        "--save-steps", type=int, default=100, help="Optimizer checkpoint interval; 0 saves only the final model"
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", type=str)
    args = parser.parse_args()
    if min(args.steps, args.global_batch_size, args.max_length, args.workers) <= 0 or args.save_steps < 0:
        parser.error("Steps, batch size, length and workers must be positive; save-steps must be nonnegative")
    if args.tokenize_only:
        data = load_dataset("json", data_files=str(args.data), split="train")
        data = data.map(
            TokenizePair(args.tokenizer, args.max_length),
            batched=True,
            batch_size=32,
            num_proc=args.workers,
            remove_columns=data.column_names,
        )
        data.save_to_disk(str(args.tokenized))
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.global_batch_size % world_size:
        raise ValueError("Global batch size must be divisible by the GPU count")
    data = load_from_disk(str(args.tokenized))
    if len(data) < args.steps * args.global_batch_size:
        raise ValueError("The SFT schedule needs more examples; do not silently repeat a small subset")
    from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available

    if not is_fast_path_available:
        raise RuntimeError("Qwen3.5 SFT requires flash-linear-attention and causal-conv1d; refusing the slow fallback")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # FP32 master parameters and optimizer state; FSDP runs the forward in bf16.
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="flash_attention_2"
    )
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    for parameter in model.model.visual.parameters():
        parameter.requires_grad_(False)
    parameter_counts = {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            json.dumps(
                {
                    "sft_initialization": parameter_counts,
                    "steps": args.steps,
                    "global_batch_size": args.global_batch_size,
                }
            ),
            flush=True,
        )
    arguments = TrainingArguments(
        output_dir=str(args.output),
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.global_batch_size // world_size,
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        warmup_steps=0,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fsdp="full_shard auto_wrap",
        fsdp_config={"transformer_layer_cls_to_wrap": ["Qwen3_5DecoderLayer"], "use_orig_params": True},
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps,
        save_total_limit=1,
        logging_steps=1,
        report_to="none",
        seed=0,
        data_seed=0,
        train_sampling_strategy="sequential",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(args.output / "final"))
    # Keep every rank alive until the final FSDP gathers have completed.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(args.output / "final"))
        (args.output / "protocol.json").write_text(
            json.dumps(
                {
                    **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    **parameter_counts,
                },
                indent=2,
            )
            + "\n"
        )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
