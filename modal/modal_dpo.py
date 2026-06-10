"""Modal app: PRM-guided DPO preference-pair generation and fine-tuning.

Two-stage pipeline (run after PRM training is complete):

    Stage A - generate preference pairs from the GSM8K train set:
        modal run --detach modal/modal_dpo.py::generate_pairs [--max-train-examples 500] [--n-samples 4]

    Stage B - DPO fine-tune the SFT policy on those pairs:
        modal run --detach modal/modal_dpo.py::train

Checkpoints land in the 'gsm8k-training-vol' volume at /vol/rl_checkpoint.
Eval the result with:
    modal run modal/modal_eval.py::rl_eval
"""

from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent

vol = modal.Volume.from_name("gsm8k-training-vol", create_if_missing=False)
hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(_ROOT / "requirements.txt"))
    .add_local_dir(str(_ROOT / "src"), remote_path="/app/src")
    .add_local_dir(str(_ROOT / "configs"), remote_path="/app/configs")
    .add_local_dir(str(_ROOT / "data" / "raw"), remote_path="/app/data/raw")
)

app = modal.App("gsm8k-dpo")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SFT_CKPT = "/vol/sft_checkpoint"
PRM_CKPT_DIR = "/vol/prm_checkpoint"
PAIRS_PATH = "/vol/dpo_pairs.jsonl"
RL_CKPT = "/vol/rl_checkpoint"
TRAIN_DATA = "/app/data/raw/gsm8k_train.jsonl"


@app.function(
    image=image,
    gpu="a100-40gb",
    timeout=8 * 3600,
    volumes={"/vol": vol},
    secrets=[hf_secret],
    retries=0,
)
def run_generate_pairs(max_train_examples: int = 500, n_samples: int = 4) -> dict:
    """Stage A: sample candidates with the SFT model, score with the PRM, emit pairs."""
    import json
    import os
    import sys

    sys.path.insert(0, "/app")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    import yaml
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from src.evaluation.eval_gsm8k import load_test_data
    from src.helpers import load_tokenizer
    from src.models.prm_model import get_sep_token_ids, load_prm_model
    from src.training.dpo_trainer import generate_dpo_pairs
    from src.training.prm_trainer import find_latest_checkpoint, load_prm_checkpoint

    device = "cuda"

    print("Loading SFT generator...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    gen_model = PeftModel.from_pretrained(base, SFT_CKPT)
    gen_model.eval()
    tokenizer = load_tokenizer(BASE_MODEL)

    print("Loading PRM scorer...")
    with open("/app/configs/prm_config.yaml") as f:
        prm_config = yaml.safe_load(f)
    prm_config["model"]["sft_checkpoint"] = SFT_CKPT
    prm_model = load_prm_model(prm_config, SFT_CKPT).to(device)

    ckpt_dir, step = find_latest_checkpoint(PRM_CKPT_DIR)
    assert ckpt_dir, f"No PRM checkpoint found in {PRM_CKPT_DIR}"
    load_prm_checkpoint(prm_model, ckpt_dir, device)
    prm_model.eval()
    print(f"PRM loaded from checkpoint-{step}")

    sep_token_ids = get_sep_token_ids(tokenizer, sep="ки")

    train_data = load_test_data(TRAIN_DATA)
    if max_train_examples:
        train_data = train_data[:max_train_examples]

    pairs, stats = generate_dpo_pairs(
        gen_model, prm_model, tokenizer, sep_token_ids, train_data, n_samples=n_samples, device=device,
    )

    print(f"\nGenerated {len(pairs)} pairs")
    print(f"  correct vs incorrect : {stats['correct_vs_incorrect']}")
    print(f"  PRM-ranked           : {stats['prm_ranked']}")
    print(f"  skipped (score gap)  : {stats['skipped_gap']}")

    os.makedirs(os.path.dirname(PAIRS_PATH), exist_ok=True)
    with open(PAIRS_PATH, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    vol.commit()
    print(f"Pairs saved to {PAIRS_PATH}")
    return {"n_pairs": len(pairs), **stats}


@app.function(
    image=image,
    gpu="a100-40gb",
    timeout=8 * 3600,
    volumes={"/vol": vol},
    secrets=[hf_secret],
    retries=0,
)
def run_dpo_training() -> None:
    """Stage B: DPO fine-tune the SFT policy on the generated preference pairs."""
    import os
    import sys

    sys.path.insert(0, "/app")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from src.training.dpo_trainer import run_dpo_training as train_dpo
    train_dpo(pairs_path=PAIRS_PATH, sft_checkpoint=SFT_CKPT, output_dir=RL_CKPT, base_model_name=BASE_MODEL)

    vol.commit()
    print(f"DPO training complete. Checkpoint committed to {RL_CKPT}")


@app.local_entrypoint()
def generate_pairs(max_train_examples: int = 500, n_samples: int = 4):
    """Stage A: generate DPO pairs. Run after PRM training is complete."""
    print(f"Generating DPO pairs for {max_train_examples} problems (N={n_samples} each)...")
    result = run_generate_pairs.remote(max_train_examples=max_train_examples, n_samples=n_samples)
    print(f"Done. {result['n_pairs']} pairs - "
          f"correct/incorrect: {result['correct_vs_incorrect']}, "
          f"PRM-ranked: {result['prm_ranked']}, skipped: {result['skipped_gap']}")


@app.local_entrypoint()
def train():
    """Stage B: DPO fine-tuning. Run after generate_pairs finishes."""
    print("Starting DPO fine-tuning...")
    run_dpo_training.remote()
    print(f"DPO training complete. Checkpoint at {RL_CKPT}")
