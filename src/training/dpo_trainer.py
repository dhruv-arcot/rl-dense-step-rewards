"""DPO preference-pair generation and fine-tuning, guided by the PRM.

Stage A — generate_dpo_pairs: for each problem, sample N solutions from the SFT
model, score each with the PRM, and emit a (prompt, chosen, rejected) triple:
    1. correct vs. incorrect available -> chosen = best-scored correct,
       rejected = worst-scored incorrect (strong signal)
    2. all correct or all incorrect     -> chosen/rejected = highest/lowest PRM
       score (process-supervision signal); skipped if the score gap is too small
       to be informative

Stage B — run_dpo_training: standard LoRA DPO fine-tuning of the SFT policy
against a frozen copy of itself as the reference model.

Usage (Stage A pairs are produced on Modal — see modal/modal_dpo.py):
    python -m src.training.dpo_trainer --pairs_path /vol/dpo_pairs.jsonl \
        --sft_checkpoint /vol/sft_checkpoint --output_dir /vol/rl_checkpoint
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

from tqdm import tqdm

from src.evaluation.eval_gsm8k import build_generation_prompt, generate_n_solutions, score_solution_with_prm
from src.helpers import extract_answer_from_output, load_base_model, load_tokenizer
from src.models.prm_model import ProcessRewardModel

MIN_SCORE_GAP = 0.05  # minimum PRM score gap required to emit a pair when correctness can't discriminate


def generate_dpo_pairs(
    gen_model,
    prm_model: ProcessRewardModel,
    tokenizer,
    sep_token_ids: List[int],
    train_data: List[dict],
    n_samples: int = 4,
    temperature: float = 0.7,
    device: str = "cuda",
    min_score_gap: float = MIN_SCORE_GAP,
) -> Tuple[List[dict], Dict[str, int]]:
    """Sample, score, and pair solutions for each training problem. Returns (pairs, stats)."""
    pairs: List[dict] = []
    stats = {"correct_vs_incorrect": 0, "prm_ranked": 0, "skipped_gap": 0}

    for ex in tqdm(train_data, desc="Generating DPO pairs"):
        question, gold_answer = ex["question"], ex["numeric_answer"]

        candidates = generate_n_solutions(gen_model, tokenizer, question, n_samples, temperature, device)
        scored = []
        for sol in candidates:
            score = score_solution_with_prm(prm_model, tokenizer, question, sol, sep_token_ids, device)
            is_correct = extract_answer_from_output(sol) == gold_answer
            scored.append((score, is_correct, sol))

        correct = [(s, sol) for s, ok, sol in scored if ok]
        incorrect = [(s, sol) for s, ok, sol in scored if not ok]

        if correct and incorrect:
            chosen = max(correct, key=lambda x: x[0])[1]
            rejected = min(incorrect, key=lambda x: x[0])[1]
            stats["correct_vs_incorrect"] += 1
        else:
            ranked = sorted(scored, key=lambda x: x[0], reverse=True)
            if ranked[0][0] - ranked[-1][0] < min_score_gap:
                stats["skipped_gap"] += 1
                continue
            chosen, rejected = ranked[0][2], ranked[-1][2]
            stats["prm_ranked"] += 1

        pairs.append({
            "prompt": build_generation_prompt(question, tokenizer),
            "chosen": chosen,
            "rejected": rejected,
        })

    return pairs, stats


def run_dpo_training(
    pairs_path: str,
    sft_checkpoint: str,
    output_dir: str,
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
) -> None:
    """LoRA DPO fine-tuning of the SFT policy against a frozen reference copy of itself."""
    from datasets import Dataset
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer

    print(f"Loading DPO pairs from {pairs_path}...")
    with open(pairs_path) as f:
        pairs = [json.loads(line) for line in f]
    print(f"  {len(pairs)} pairs loaded")

    dataset = Dataset.from_dict({
        "prompt": [p["prompt"] for p in pairs],
        "chosen": [p["chosen"] for p in pairs],
        "rejected": [p["rejected"] for p in pairs],
    })

    tokenizer = load_tokenizer(base_model_name)

    print("Loading policy model (SFT, trainable)...")
    policy = PeftModel.from_pretrained(load_base_model(base_model_name), sft_checkpoint, is_trainable=True)

    print("Loading reference model (SFT, frozen)...")
    ref_model = PeftModel.from_pretrained(load_base_model(base_model_name), sft_checkpoint, is_trainable=False)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    os.makedirs(output_dir, exist_ok=True)

    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # effective batch = 8
        num_train_epochs=3,
        learning_rate=5e-5,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none",
        beta=0.1,                        # KL regularization coefficient
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"DPO training complete. Model saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DPO fine-tuning on pre-generated preference pairs")
    parser.add_argument("--pairs_path", required=True, help="JSONL of {prompt, chosen, rejected} triples")
    parser.add_argument("--sft_checkpoint", required=True, help="Path to the SFT LoRA adapter")
    parser.add_argument("--output_dir", required=True, help="Where to save the DPO-tuned adapter")
    args = parser.parse_args()
    run_dpo_training(args.pairs_path, args.sft_checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
