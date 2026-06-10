"""Parse the Math-Shepherd dataset into step-level PRM training data.

Math-Shepherd (peiyi9979/math-shepherd) marks step boundaries with the Cyrillic
token "ки", and labels each step "ки\\n+" (correct) or "ки\\n-" (incorrect).

Output format per line:
    {"problem": str, "steps": List[str], "labels": List[int]}

Usage:
    python data/prepare_prm_data.py [--output_dir data/processed]
"""

import argparse
import json
import os
import random
import re
from typing import List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm


def parse_math_shepherd_example(row: dict) -> Optional[dict]:
    """Parse one Math-Shepherd row into {problem, steps, labels}.

    Raw format:
      input: "<problem> Step 1: <text> ки\\nStep 2: <text> ки\\n..."
      label: "<problem> Step 1: <text> +\\nStep 2: <text> -\\n..."

    Steps are recovered by splitting `input` on " ки\\n" (stripping the problem
    prefix off the first chunk at "Step 1:"); labels come from the trailing
    " +"/" -" on each line of `label`. Returns None on any parse failure.
    """
    input_text = row.get("input", "").strip()
    label_text = row.get("label", "").strip()
    if not input_text or not label_text:
        return None

    raw_steps = re.split(r" ки\n", input_text)
    if not raw_steps:
        return None
    if raw_steps[-1].endswith(" ки"):
        raw_steps[-1] = raw_steps[-1][:-3]

    step1_marker = " Step 1:"
    if step1_marker in raw_steps[0]:
        split_idx = raw_steps[0].index(step1_marker)
        problem = raw_steps[0][:split_idx].strip()
        raw_steps[0] = raw_steps[0][split_idx + 1:].strip()
    else:
        problem = ""
        raw_steps[0] = raw_steps[0].strip()

    steps = [s.strip() for s in raw_steps if s.strip()]

    labels: List[int] = []
    for line in (l.strip() for l in label_text.split("\n") if l.strip()):
        if line.endswith(" +") or line.endswith("+"):
            labels.append(1)
        elif line.endswith(" -") or line.endswith("-"):
            labels.append(0)
        # lines without a trailing +/- are skipped

    if not steps or not labels:
        return None

    # Truncate to the shorter side; truncation upstream can cause a length mismatch
    n = min(len(steps), len(labels))
    return {"problem": problem, "steps": steps[:n], "labels": labels[:n]}


def load_and_parse_dataset() -> List[dict]:
    """Download Math-Shepherd from HuggingFace and parse every row into {problem, steps, labels}."""
    print("Loading Math-Shepherd from HuggingFace...")
    dataset = load_dataset("peiyi9979/math-shepherd", split="train")

    parsed, n_failed = [], 0
    for row in tqdm(dataset, desc="Parsing Math-Shepherd"):
        result = parse_math_shepherd_example(row)
        if result is None:
            n_failed += 1
        else:
            parsed.append(result)

    print(f"  Parsed {len(parsed)} examples, skipped {n_failed}")
    return parsed


def split_train_val(data: List[dict], val_fraction: float = 0.1, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Shuffle data deterministically and split off val_fraction as the validation set."""
    random.seed(seed)
    shuffled = data[:]
    random.shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]


def save_jsonl(data: List[dict], path: str) -> None:
    """Write records to a JSONL file, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")


def main(output_dir: str = "data/processed") -> None:
    """Parse Math-Shepherd into step-level PRM data and save train/val JSONL splits."""
    data = load_and_parse_dataset()
    train_data, val_data = split_train_val(data, val_fraction=0.1)

    train_path = os.path.join(output_dir, "prm_train.jsonl")
    val_path = os.path.join(output_dir, "prm_val.jsonl")
    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)

    print(f"Saved {len(train_data)} train examples to {train_path}")
    print(f"Saved {len(val_data)} val examples to {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Math-Shepherd PRM data")
    parser.add_argument("--output_dir", default="data/processed", help="Directory to save processed JSONL files")
    args = parser.parse_args()
    main(args.output_dir)
