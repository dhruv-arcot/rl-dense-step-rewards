"""Download GSM8K from HuggingFace and save train/test splits as JSONL.

Usage:
    python data/download_gsm8k.py [--output_dir data/raw]
"""

import argparse
import json
import os
import re
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm


def extract_numeric_answer(answer_text: str) -> Optional[int]:
    """Pull the final number out of a GSM8K solution's '#### <number>' tail."""
    match = re.search(r"####\s*(-?[\d,]+)", answer_text)
    if match is None:
        return None
    return int(match.group(1).replace(",", ""))


def save_jsonl(records: list, path: str) -> None:
    """Write records to a JSONL file, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def download_and_save(output_dir: str = "data/raw") -> None:
    """Download GSM8K from HuggingFace and write train/test JSONL files to output_dir."""
    print("Loading GSM8K from HuggingFace...")
    dataset = load_dataset("openai/gsm8k", "main")

    for split_name in ("train", "test"):
        records, n_failed = [], 0
        for example in tqdm(dataset[split_name], desc=f"Processing {split_name}"):
            numeric = extract_numeric_answer(example["answer"])
            if numeric is None:
                n_failed += 1
                continue
            records.append({
                "question": example["question"],
                "answer": example["answer"],
                "numeric_answer": numeric,
            })

        out_path = os.path.join(output_dir, f"gsm8k_{split_name}.jsonl")
        save_jsonl(records, out_path)
        print(f"  {split_name}: {len(records)} examples saved to {out_path} ({n_failed} skipped)")


def main() -> None:
    """Entry point: parse args and download GSM8K."""
    parser = argparse.ArgumentParser(description="Download GSM8K dataset")
    parser.add_argument("--output_dir", default="data/raw", help="Directory to save JSONL files")
    args = parser.parse_args()
    download_and_save(args.output_dir)


if __name__ == "__main__":
    main()
