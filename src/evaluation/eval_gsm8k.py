"""Evaluate a model on the GSM8K test set.

Two modes:
    greedy      - generate one solution per problem, check exact-match accuracy
    best_of_n   - sample N solutions, rerank with the PRM, keep the top-scoring one

Usage:
    python -m src.evaluation.eval_gsm8k --sft_checkpoint /vol/sft_checkpoint --eval_mode greedy

    python -m src.evaluation.eval_gsm8k \
        --sft_checkpoint /vol/sft_checkpoint --prm_checkpoint /vol/prm_checkpoint \
        --eval_mode best_of_n --n_samples 8
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.helpers import extract_answer_from_output, load_config, load_tokenizer
from src.models.prm_model import ProcessRewardModel, find_step_positions, get_sep_token_ids, load_prm_model
from src.models.sft_model import SYSTEM_PROMPT


def build_generation_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    """Render a question as a generation prompt (no assistant turn appended)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_solution(model, tokenizer: AutoTokenizer, question: str, device: str, max_new_tokens: int = 512) -> str:
    """Greedy-decode a single solution."""
    prompt = build_generation_prompt(question, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=1.0, pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_n_solutions(
    model, tokenizer: AutoTokenizer, question: str, n: int, temperature: float, device: str,
    max_new_tokens: int = 512,
) -> List[str]:
    """Sample N candidate solutions for best-of-N reranking."""
    prompt = build_generation_prompt(question, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
            num_return_sequences=n, pad_token_id=tokenizer.eos_token_id,
        )
    return [tokenizer.decode(seq[input_len:], skip_special_tokens=True) for seq in output_ids]


def score_solution_with_prm(
    prm_model: ProcessRewardModel,
    tokenizer: AutoTokenizer,
    question: str,
    solution: str,
    sep_token_ids: List[int],
    device: str,
) -> float:
    """Score a solution as the mean sigmoid(step logit) across its steps.

    Falls back to scoring the final token as a single "step" if no separator
    is found (e.g. the model didn't emit "ки" during free generation).
    """
    full_text = f"Problem: {question}\nSolution: {solution}"
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).to(device)

    positions = find_step_positions(inputs["input_ids"][0], sep_token_ids)
    if not positions:
        positions = [inputs["input_ids"].shape[1] - 1]

    with torch.no_grad():
        step_logits, step_mask = prm_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            step_positions=[positions],
        )

    scores = torch.sigmoid(step_logits.squeeze(-1))
    return scores[step_mask].mean().item()


def evaluate_greedy(model, tokenizer: AutoTokenizer, test_data: List[dict], device: str, max_new_tokens: int = 512) -> Dict:
    """Greedy-decode accuracy: exact match on the final numeric answer."""
    n_correct = 0
    for ex in tqdm(test_data, desc="Greedy eval"):
        solution = generate_solution(model, tokenizer, ex["question"], device, max_new_tokens)
        predicted = extract_answer_from_output(solution)
        if predicted is not None and predicted == ex["numeric_answer"]:
            n_correct += 1

    return {
        "accuracy": n_correct / len(test_data) if test_data else 0.0,
        "n_correct": n_correct, "n_total": len(test_data), "mode": "greedy",
    }


def evaluate_best_of_n(
    gen_model, prm_model: ProcessRewardModel, tokenizer: AutoTokenizer, test_data: List[dict],
    n: int, temperature: float, sep_token_ids: List[int], device: str, max_new_tokens: int = 512,
) -> Dict:
    """Sample N solutions per problem, keep the highest-PRM-scored one, check correctness."""
    n_correct = 0
    for ex in tqdm(test_data, desc=f"Best-of-{n} eval"):
        candidates = generate_n_solutions(gen_model, tokenizer, ex["question"], n, temperature, device, max_new_tokens)
        scores = [
            score_solution_with_prm(prm_model, tokenizer, ex["question"], sol, sep_token_ids, device)
            for sol in candidates
        ]
        best_solution = candidates[scores.index(max(scores))]
        predicted = extract_answer_from_output(best_solution)
        if predicted is not None and predicted == ex["numeric_answer"]:
            n_correct += 1

    return {
        "accuracy": n_correct / len(test_data) if test_data else 0.0,
        "n_correct": n_correct, "n_total": len(test_data), "mode": f"best_of_{n}",
    }


def load_test_data(path: str = "data/raw/gsm8k_test.jsonl") -> List[dict]:
    """Load GSM8K examples from a JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def main() -> None:
    """Entry point: parse args and run GSM8K evaluation (greedy or best-of-N)."""
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K")
    parser.add_argument("--sft_checkpoint", required=True, help="Path to the SFT LoRA adapter")
    parser.add_argument("--prm_checkpoint", default=None, help="Path to the PRM checkpoint (required for best_of_n)")
    parser.add_argument("--test_data_path", default="data/raw/gsm8k_test.jsonl")
    parser.add_argument("--eval_mode", choices=["greedy", "best_of_n"], default="greedy")
    parser.add_argument("--n_samples", type=int, default=8, help="Candidates per problem for best_of_n")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_examples", type=int, default=None, help="Limit test set size for quick runs")
    parser.add_argument("--prm_config", default="configs/prm_config.yaml")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"

    print(f"Loading SFT model from {args.sft_checkpoint}...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    gen_model = PeftModel.from_pretrained(base, args.sft_checkpoint)
    gen_model.eval()
    tokenizer = load_tokenizer(base_model_name)

    test_data = load_test_data(args.test_data_path)
    if args.max_examples:
        test_data = test_data[:args.max_examples]

    if args.eval_mode == "greedy":
        results = evaluate_greedy(gen_model, tokenizer, test_data, device, args.max_new_tokens)
    else:
        assert args.prm_checkpoint, "--prm_checkpoint is required for best_of_n mode"
        prm_config = load_config(args.prm_config)
        prm_model = load_prm_model(prm_config, args.sft_checkpoint)
        prm_model.prm_head.load_state_dict(
            torch.load(os.path.join(args.prm_checkpoint, "prm_head.pt"), map_location=device)
        )
        prm_model.eval()

        sep_token_ids = get_sep_token_ids(tokenizer, sep="ки")
        results = evaluate_best_of_n(
            gen_model, prm_model, tokenizer, test_data, n=args.n_samples, temperature=args.temperature,
            sep_token_ids=sep_token_ids, device=device, max_new_tokens=args.max_new_tokens,
        )

    print("\n" + "=" * 40)
    print(f"Mode: {results['mode']}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['n_correct']}/{results['n_total']})")
    print("=" * 40)


if __name__ == "__main__":
    main()
