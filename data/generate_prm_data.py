"""Generate PAV-style continuous step rewards from a trained SFT model.

PAV (Progress as a Verifier, Setlur et al. 2024):
    PAV_t = P(correct | s_0..t) - P(correct | s_0..t-1)

P(correct | s_0..t) is estimated by sampling K completions from the prefix
ending at step t and measuring the fraction that reach the correct answer.
This gives a continuous, dense alternative to Math-Shepherd's binary labels -
set `prm_head.use_pav: true` in configs/prm_config.yaml to train on it.

Usage:
    python data/generate_prm_data.py \
        --sft_checkpoint /vol/sft_checkpoint \
        --gsm8k_path data/raw/gsm8k_train.jsonl \
        --output_path data/processed/prm_train_pav.jsonl \
        --k_samples 8
"""

import argparse
import json
import os
import re
from typing import List, Optional

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
STEP_SEPARATOR = " ки\n"


def extract_answer_from_output(text: str) -> Optional[int]:
    """Extract the numeric answer following '####' (GSM8K answer format)."""
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match is None:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


def generate_completions(
    model, tokenizer: AutoTokenizer, prefix: str, k: int, temperature: float, max_new_tokens: int, device: str,
) -> List[str]:
    """Sample K continuations of `prefix`."""
    inputs = tokenizer(prefix, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
            num_return_sequences=k, pad_token_id=tokenizer.eos_token_id,
        )
    return [tokenizer.decode(seq[input_len:], skip_special_tokens=True) for seq in outputs]


def compute_pav_rewards(
    model, tokenizer: AutoTokenizer, problem: str, steps: List[str], gold_answer: int,
    k_samples: int = 8, temperature: float = 0.7, max_new_tokens: int = 512, device: str = "cuda",
) -> List[float]:
    """Estimate PAV_t = P(correct | prefix_t) - P(correct | prefix_{t-1}) for each step."""
    pav_rewards = []
    prev_prob = 0.0  # P(correct | problem alone) is taken as 0

    for t in range(len(steps)):
        prefix_steps = STEP_SEPARATOR.join(steps[:t + 1])
        prefix = f"Problem: {problem}\nSolution: {prefix_steps}{STEP_SEPARATOR}"

        completions = generate_completions(model, tokenizer, prefix, k_samples, temperature, max_new_tokens, device)
        prob_correct = sum(extract_answer_from_output(c) == gold_answer for c in completions) / k_samples

        pav_rewards.append(prob_correct - prev_prob)
        prev_prob = prob_correct

    return pav_rewards


def main(
    sft_checkpoint: str,
    gsm8k_path: str,
    output_path: str,
    k_samples: int = 8,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    max_examples: Optional[int] = None,
) -> None:
    """Generate PAV reward data for each GSM8K training example and write it to output_path."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SFT model from {sft_checkpoint} on {device}...")

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, sft_checkpoint).eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(gsm8k_path) as f:
        examples = [json.loads(line) for line in f]
    if max_examples:
        examples = examples[:max_examples]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as out_f:
        for ex in tqdm(examples, desc="Generating PAV rewards"):
            # Greedily generate one solution to obtain a step trajectory
            solution = generate_completions(
                model, tokenizer, f"Problem: {ex['question']}\nSolution:",
                k=1, temperature=0.0, max_new_tokens=max_new_tokens, device=device,
            )[0]
            steps = [s.strip() for s in solution.split("ки") if s.strip()]
            if not steps:
                continue

            pav_rewards = compute_pav_rewards(
                model, tokenizer, problem=ex["question"], steps=steps, gold_answer=ex["numeric_answer"],
                k_samples=k_samples, temperature=temperature, max_new_tokens=max_new_tokens, device=device,
            )

            out_f.write(json.dumps({
                "problem": ex["question"],
                "steps": steps,
                "labels": pav_rewards,
                "label_type": "pav",
            }) + "\n")

    print(f"PAV data saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PAV reward data from an SFT model")
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--gsm8k_path", default="data/raw/gsm8k_train.jsonl")
    parser.add_argument("--output_path", default="data/processed/prm_train_pav.jsonl")
    parser.add_argument("--k_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    main(
        sft_checkpoint=args.sft_checkpoint,
        gsm8k_path=args.gsm8k_path,
        output_path=args.output_path,
        k_samples=args.k_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_examples=args.max_examples,
    )
