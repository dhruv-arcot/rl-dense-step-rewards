"""Evaluate PRM quality on held-out Math-Shepherd step labels.

Metrics:
    AUC-ROC  — ability to rank correct steps above incorrect ones
    Accuracy — binary classification accuracy at threshold 0.5

Usage:
    python -m src.evaluation.eval_prm \
        --prm_checkpoint /vol/prm_checkpoint --val_data_path data/processed/prm_val.jsonl
"""

import argparse
import os
from typing import List, Tuple

import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoTokenizer

from src.helpers import find_step_positions_char, load_config, load_tokenizer
from src.models.prm_model import ProcessRewardModel, load_prm_model


def get_prm_step_scores(
    prm_model: ProcessRewardModel,
    tokenizer: AutoTokenizer,
    examples: List[dict],
    device: str,
    sep: str = "ки",
    max_length: int = 2048,
) -> Tuple[List[float], List[int]]:
    """Run the PRM over each example and return flattened (scores, labels) for AUC."""
    all_scores: List[float] = []
    all_labels: List[int] = []
    step_sep_str = f" {sep}\n"

    for ex in tqdm(examples, desc="Scoring steps"):
        problem, steps, labels = ex["problem"], ex["steps"], ex["labels"]
        if not steps:
            continue

        full_text = f"Problem: {problem}\nSolution: {step_sep_str.join(steps)}"
        input_ids, attention_mask, positions = find_step_positions_char(
            full_text, tokenizer, sep=sep, max_length=max_length
        )
        if not positions:
            continue

        n_steps = min(len(positions), len(labels))
        positions = positions[:n_steps]
        step_labels = labels[:n_steps]

        with torch.no_grad():
            step_logits, step_mask = prm_model(
                input_ids=input_ids.unsqueeze(0).to(device),
                attention_mask=attention_mask.unsqueeze(0).to(device),
                step_positions=[positions],
            )

        scores = torch.sigmoid(step_logits.squeeze(-1)[0])
        valid_mask = step_mask[0]
        for score, label in zip(scores[valid_mask].tolist(), step_labels):
            all_scores.append(score)
            all_labels.append(int(label))

    return all_scores, all_labels


def compute_auc_roc(scores: List[float], labels: List[int]) -> float:
    """AUC-ROC; returns 0.5 (chance) if only one class is present."""
    if len(set(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, scores)


def compute_step_accuracy(scores: List[float], labels: List[int], threshold: float = 0.5) -> float:
    if not scores:
        return 0.0
    n_correct = sum((s >= threshold) == bool(l) for s, l in zip(scores, labels))
    return n_correct / len(scores)


def load_val_data(path: str) -> List[dict]:
    import json
    with open(path) as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PRM on step-level labels")
    parser.add_argument("--prm_checkpoint", required=True, help="Dir containing prm_head.pt and adapter weights")
    parser.add_argument("--sft_checkpoint", default=None,
                        help="SFT checkpoint used to init the base model (defaults to prm_config.yaml value)")
    parser.add_argument("--val_data_path", default="data/processed/prm_val.jsonl")
    parser.add_argument("--config", default="configs/prm_config.yaml")
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config)

    sft_ckpt = args.sft_checkpoint or config["model"].get("sft_checkpoint")
    print(f"Loading PRM model (SFT base: {sft_ckpt})...")
    prm_model = load_prm_model(config, sft_ckpt)
    prm_model.prm_head.load_state_dict(
        torch.load(os.path.join(args.prm_checkpoint, "prm_head.pt"), map_location=device)
    )
    prm_model.eval()

    tokenizer = load_tokenizer(config["model"]["base_model"])

    val_data = load_val_data(args.val_data_path)
    if args.max_examples:
        val_data = val_data[:args.max_examples]

    print(f"Evaluating on {len(val_data)} examples...")
    scores, labels = get_prm_step_scores(
        prm_model, tokenizer, val_data, device, sep=config["data"]["step_separator"]
    )

    auc = compute_auc_roc(scores, labels)
    acc = compute_step_accuracy(scores, labels)

    print("\n" + "=" * 40)
    print(f"PRM Evaluation ({len(scores)} steps total)")
    print(f"AUC-ROC:  {auc:.4f}")
    print(f"Accuracy: {acc:.4f}  (threshold=0.5)")
    print("=" * 40)


if __name__ == "__main__":
    main()
