"""SFT training via TRL's SFTTrainer.

Trains Qwen2.5-7B-Instruct + LoRA on GSM8K solutions so the model learns to
produce step-by-step reasoning ending in '#### <answer>'.

Usage:
    python -m src.training.sft_trainer [--config configs/sft_config.yaml]
"""

import argparse
import json
import os
from typing import Optional

import datasets
from trl import SFTConfig, SFTTrainer

from src.helpers import load_config
from src.models.sft_model import format_sft_example, get_sft_model_and_tokenizer


def prepare_gsm8k_dataset(config: dict, tokenizer, local_path: Optional[str] = None) -> datasets.Dataset:
    """Load GSM8K (local JSONL if present, else HuggingFace) and format for SFTTrainer.

    SFTTrainer expects a 'text' column containing the fully-rendered chat example.
    """
    data_cfg = config["data"]
    effective_path = local_path or data_cfg.get("local_train_path", "data/raw/gsm8k_train.jsonl")

    if os.path.exists(effective_path):
        print(f"Loading GSM8K train from local file: {effective_path}")
        with open(effective_path) as f:
            records = [json.loads(line) for line in f]
        raw_dataset = datasets.Dataset.from_list(records)
    else:
        print(f"Downloading GSM8K from HuggingFace ({data_cfg['dataset_name']})...")
        raw_dataset = datasets.load_dataset(
            data_cfg["dataset_name"],
            data_cfg.get("dataset_config", "main"),
            split=data_cfg.get("train_split", "train"),
        )

    def format_fn(example):
        """Render one raw GSM8K record as a chat-formatted 'text' field."""
        question = example.get("question") or example.get("input") or ""
        answer = example.get("answer") or example.get("output") or ""
        return {"text": format_sft_example(question, answer, tokenizer)}

    formatted = raw_dataset.map(format_fn, remove_columns=raw_dataset.column_names)
    print(f"  Dataset size: {len(formatted)} examples")
    return formatted


def build_training_args(config: dict) -> SFTConfig:
    """Build an SFTConfig from the 'training' and 'output' sections of the config."""
    t = config["training"]
    out = config["output"]
    return SFTConfig(
        output_dir=out["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        optim=t["optim"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_strategy="no",
        save_total_limit=t["save_total_limit"],
        fp16=t.get("fp16", False),
        bf16=t.get("bf16", True),
        dataloader_num_workers=t.get("dataloader_num_workers", 4),
        report_to=t.get("report_to", "none"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=config["model"]["max_seq_length"],
        packing=False,
    )


def run_sft_training(config_path: str = "configs/sft_config.yaml") -> None:
    """Full SFT pipeline: load model + data, train, save the LoRA adapter."""
    config = load_config(config_path)

    print("=" * 60)
    print("SFT Training: Qwen2.5-7B-Instruct on GSM8K")
    print(f"Config: {config_path}")
    print("=" * 60)

    model, tokenizer = get_sft_model_and_tokenizer(config)
    train_dataset = prepare_gsm8k_dataset(config, tokenizer)
    training_args = build_training_args(config)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    output_dir = config["output"]["output_dir"]
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)  # PEFT models save only the adapter (~300MB)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


def main() -> None:
    """Entry point: parse args and run SFT training."""
    parser = argparse.ArgumentParser(description="Run SFT training on GSM8K")
    parser.add_argument("--config", default="configs/sft_config.yaml", help="Path to SFT config YAML")
    args = parser.parse_args()
    run_sft_training(args.config)


if __name__ == "__main__":
    main()
