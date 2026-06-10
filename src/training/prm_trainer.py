"""Custom PRM training loop.

A custom DataLoader + loop is used (rather than TRL's SFTTrainer) because each
example has a ragged number of step-boundary positions, which needs collation
logic that SFTTrainer doesn't support natively.

Usage:
    python -m src.training.prm_trainer [--config configs/prm_config.yaml]
"""

import argparse
import json
import os
from typing import List, Optional, Tuple

import safetensors.torch
import torch
import torch.nn as nn
from peft import set_peft_model_state_dict
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.helpers import find_step_positions_char, load_config, load_tokenizer
from src.models.prm_model import ProcessRewardModel, compute_prm_loss, load_prm_model


class PRMDataset(Dataset):
    """Tokenized (problem + steps) sequences with step-boundary positions and labels."""

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 2048,
        step_separator: str = "ки",
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.step_sep = step_separator
        self.step_sep_str = f" {step_separator}\n"

        print(f"Loading PRM data from {jsonl_path}...")
        with open(jsonl_path) as f:
            raw = [json.loads(line) for line in f]

        self.examples = self._preprocess(raw)
        print(f"  Loaded {len(self.examples)} examples")

    def _preprocess(self, raw: List[dict]) -> List[dict]:
        """Tokenize each raw record and locate its step-boundary positions, dropping empty examples."""
        processed = []
        for rec in tqdm(raw, desc="Tokenising"):
            problem, steps, labels = rec["problem"], rec["steps"], rec["labels"]
            if not steps:
                continue

            solution_text = self.step_sep_str.join(steps)
            full_text = f"Problem: {problem}\nSolution: {solution_text}"

            input_ids, attention_mask, positions = find_step_positions_char(
                full_text, self.tokenizer, sep=self.step_sep, max_length=self.max_seq_length
            )

            n = min(len(positions), len(labels))
            if n == 0:
                continue

            processed.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "step_positions": positions[:n],
                "step_labels": [float(labels[i]) for i in range(n)],
            })
        return processed

    def __len__(self) -> int:
        """Return the number of processed examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        """Return the pre-processed example at idx."""
        return self.examples[idx]


class PRMDataCollator:
    """Pad sequences and ragged step labels/positions to per-batch maximums."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, batch: List[dict]) -> dict:
        """Pad input_ids, attention_mask, step_labels, and step_mask to per-batch maximums."""
        max_seq = max(item["input_ids"].shape[0] for item in batch)
        max_steps = max(len(item["step_positions"]) for item in batch)

        padded_input_ids = torch.full((len(batch), max_seq), self.pad_id, dtype=torch.long)
        padded_attention = torch.zeros(len(batch), max_seq, dtype=torch.long)
        padded_labels = torch.zeros(len(batch), max_steps, dtype=torch.float)
        step_mask = torch.zeros(len(batch), max_steps, dtype=torch.bool)
        step_positions_list: List[List[int]] = []

        for i, item in enumerate(batch):
            seq_len = item["input_ids"].shape[0]
            padded_input_ids[i, :seq_len] = item["input_ids"]
            padded_attention[i, :seq_len] = item["attention_mask"]

            n_steps = len(item["step_positions"])
            padded_labels[i, :n_steps] = torch.tensor(item["step_labels"])
            step_mask[i, :n_steps] = True
            step_positions_list.append(item["step_positions"])

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention,
            "step_positions": step_positions_list,  # ragged; consumed directly by the model
            "step_labels": padded_labels,
            "step_mask": step_mask,
        }


def find_latest_checkpoint(output_dir: str) -> Tuple[Optional[str], int]:
    """Return (checkpoint_dir, global_step) for the highest-numbered checkpoint, or (None, 0)."""
    if not os.path.exists(output_dir):
        return None, 0
    best_step, best_dir = 0, None
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[1])
                if step > best_step:
                    best_step, best_dir = step, os.path.join(output_dir, name)
            except (ValueError, IndexError):
                pass
    return best_dir, best_step


def load_prm_checkpoint(prm_model: ProcessRewardModel, checkpoint_dir: str, device: str) -> None:
    """Restore PRMHead weights and the LoRA adapter from a checkpoint directory."""
    prm_model.prm_head.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "prm_head.pt"), map_location=device)
    )

    st_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "adapter_model.bin")
    if os.path.exists(st_path):
        adapter_state = safetensors.torch.load_file(st_path, device=device)
    else:
        adapter_state = torch.load(bin_path, map_location=device)
    set_peft_model_state_dict(prm_model.base_model, adapter_state)
    print(f"Resumed from checkpoint: {checkpoint_dir}")


def save_prm_checkpoint(model: ProcessRewardModel, output_dir: str, step: int) -> None:
    """Save the PRM head and LoRA adapter to output_dir/checkpoint-<step>/."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.prm_head.state_dict(), os.path.join(ckpt_dir, "prm_head.pt"))
    model.base_model.save_pretrained(ckpt_dir)
    print(f"Checkpoint saved to {ckpt_dir}")


@torch.no_grad()
def _evaluate(prm_model: ProcessRewardModel, val_loader: DataLoader, device: str, use_pav: bool) -> Tuple[float, float]:
    """Mean validation loss and step-level binary accuracy at threshold 0.5."""
    prm_model.eval()
    losses, n_correct, n_total = [], 0, 0

    for batch in tqdm(val_loader, desc="Validation"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        step_labels = batch["step_labels"].to(device)
        step_mask = batch["step_mask"].to(device)

        with autocast("cuda", dtype=torch.bfloat16):
            step_logits, model_mask = prm_model(input_ids, attention_mask, batch["step_positions"])
            mask = model_mask & step_mask
            losses.append(compute_prm_loss(step_logits, step_labels, mask, use_pav).item())

        preds = (torch.sigmoid(step_logits.squeeze(-1)) >= 0.5)[mask]
        targets = (step_labels >= 0.5)[mask]
        n_correct += (preds == targets).sum().item()
        n_total += mask.sum().item()

    avg_loss = sum(losses) / len(losses) if losses else float("nan")
    accuracy = n_correct / n_total if n_total > 0 else float("nan")
    return avg_loss, accuracy


def run_prm_training(config_path: str = "configs/prm_config.yaml") -> None:
    """Full PRM training pipeline: data, model init/resume, train + validate per epoch."""
    config = load_config(config_path)

    print("=" * 60)
    print("PRM Training: Qwen2.5-7B-Instruct + PRMHead on Math-Shepherd")
    print(f"Config: {config_path}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_cfg = config["training"]
    out_cfg = config["output"]

    tokenizer = load_tokenizer(config["model"]["base_model"])

    train_dataset = PRMDataset(
        jsonl_path=config["data"]["train_file"],
        tokenizer=tokenizer,
        max_seq_length=config["model"]["max_seq_length"],
        step_separator=config["data"]["step_separator"],
    )
    max_train = config["data"].get("max_train_samples")
    if max_train:
        train_dataset.examples = train_dataset.examples[:max_train]
        print(f"  Truncated train set to {len(train_dataset.examples)} examples")

    val_dataset = PRMDataset(
        jsonl_path=config["data"]["val_file"],
        tokenizer=tokenizer,
        max_seq_length=config["model"]["max_seq_length"],
        step_separator=config["data"]["step_separator"],
    )

    collator = PRMDataCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=t_cfg["per_device_train_batch_size"], shuffle=True,
        collate_fn=collator, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=t_cfg["per_device_train_batch_size"], shuffle=False,
        collate_fn=collator, num_workers=4,
    )

    sft_ckpt = config["model"].get("sft_checkpoint")
    prm_model = load_prm_model(config, sft_ckpt)
    prm_model.to(device)
    prm_model.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    use_pav = config["prm_head"].get("use_pav", False)
    os.makedirs(out_cfg["output_dir"], exist_ok=True)

    resume_dir, global_step = find_latest_checkpoint(out_cfg["output_dir"])
    if resume_dir:
        load_prm_checkpoint(prm_model, resume_dir, device)
        print(f"Resuming from global_step={global_step}")
    else:
        print("Starting fresh training run")

    trainable_params = [p for p in prm_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=t_cfg["learning_rate"])

    num_steps = len(train_loader) * t_cfg["num_train_epochs"]
    warmup_steps = int(num_steps * t_cfg["warmup_ratio"])
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]  # required before constructing scheduler with last_epoch >= 0
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps, last_epoch=global_step - 1)

    grad_accum = t_cfg["gradient_accumulation_steps"]
    max_steps_per_run = t_cfg.get("max_steps_per_run")
    steps_this_run = 0

    for epoch in range(t_cfg["num_train_epochs"]):
        prm_model.train()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            step_labels = batch["step_labels"].to(device)
            step_mask = batch["step_mask"].to(device)

            with autocast("cuda", dtype=torch.bfloat16):
                step_logits, model_mask = prm_model(input_ids, attention_mask, batch["step_positions"])
                # combine with the collator mask: the model mask may differ after truncation
                mask = model_mask & step_mask
                loss = compute_prm_loss(step_logits, step_labels, mask, use_pav) / grad_accum

            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % t_cfg["logging_steps"] == 0:
                    print(f"  step={global_step}  loss={loss.item() * grad_accum:.4f}  "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")

                if global_step % t_cfg["save_steps"] == 0:
                    save_prm_checkpoint(prm_model, out_cfg["output_dir"], global_step)

                steps_this_run += 1
                if max_steps_per_run and steps_this_run >= max_steps_per_run:
                    print(f"Reached max_steps_per_run={max_steps_per_run}. Saving and exiting.")
                    save_prm_checkpoint(prm_model, out_cfg["output_dir"], global_step)
                    torch.save(prm_model.prm_head.state_dict(), os.path.join(out_cfg["output_dir"], "prm_head.pt"))
                    return

        avg_val_loss, val_acc = _evaluate(prm_model, val_loader, device, use_pav)
        print(f"Epoch {epoch + 1} - val loss: {avg_val_loss:.4f}  val step-acc: {val_acc:.4f}")

    save_prm_checkpoint(prm_model, out_cfg["output_dir"], global_step)
    torch.save(prm_model.prm_head.state_dict(), os.path.join(out_cfg["output_dir"], "prm_head.pt"))
    print(f"Training complete. Final model saved to {out_cfg['output_dir']}")


def main() -> None:
    """Entry point: parse args and run PRM training."""
    parser = argparse.ArgumentParser(description="Run PRM training")
    parser.add_argument("--config", default="configs/prm_config.yaml", help="Path to PRM config YAML")
    args = parser.parse_args()
    run_prm_training(args.config)


if __name__ == "__main__":
    main()
