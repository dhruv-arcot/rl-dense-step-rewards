"""Modal app: all evaluation modes for the GSM8K / PRM pipeline.

Commands:
    Zero-shot base model (no LoRA):
        modal run modal/modal_eval.py::zero_shot [--max-examples 500]

    SFT model, greedy decoding:
        modal run modal/modal_eval.py::sft_eval [--max-examples 0]

    SFT + PRM best-of-N reranking:
        modal run --detach modal/modal_eval.py::best_of_n [--n-samples 8]

    DPO/RL fine-tuned model, greedy decoding:
        modal run modal/modal_eval.py::rl_eval [--max-examples 0]

    PRM step-level quality (AUC-ROC + accuracy):
        modal run modal/modal_eval.py::prm_eval [--max-examples 2000]
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
    .add_local_dir(str(_ROOT / "data" / "processed"), remote_path="/app/data/processed")
)

app = modal.App("gsm8k-eval")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SFT_CKPT = "/vol/sft_checkpoint"
PRM_CKPT_DIR = "/vol/prm_checkpoint"
RL_CKPT = "/vol/rl_checkpoint"
TEST_DATA = "/app/data/raw/gsm8k_test.jsonl"

GPU_FN_KWARGS = dict(image=image, gpu="a100-40gb", timeout=4 * 3600, volumes={"/vol": vol},
                     secrets=[hf_secret], retries=0)


def _setup():
    """Common container setup shared by every remote function below."""
    import os
    import sys
    sys.path.insert(0, "/app")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _print_accuracy(label: str, results: dict) -> None:
    print(f"\n{'=' * 40}")
    print(f"{label}: {results['accuracy']:.4f}  ({results['n_correct']}/{results['n_total']})")
    print("=" * 40)


@app.function(**GPU_FN_KWARGS)
def run_zero_shot(max_examples: int = None) -> dict:
    """Greedy-eval the bare base model (no LoRA) — establishes the pre-finetuning baseline."""
    _setup()
    import torch
    from transformers import AutoModelForCausalLM

    from src.evaluation.eval_gsm8k import evaluate_greedy, load_test_data
    from src.helpers import load_tokenizer

    print("Loading base model (zero-shot, no LoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    tokenizer = load_tokenizer(BASE_MODEL)

    test_data = load_test_data(TEST_DATA)
    if max_examples:
        test_data = test_data[:max_examples]

    print(f"Running zero-shot greedy eval on {len(test_data)} examples...")
    results = evaluate_greedy(model, tokenizer, test_data, "cuda")
    _print_accuracy("Zero-shot accuracy", results)
    return results


@app.function(**GPU_FN_KWARGS)
def run_sft_eval(max_examples: int = None) -> dict:
    """Greedy-eval the SFT LoRA checkpoint."""
    _setup()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from src.evaluation.eval_gsm8k import evaluate_greedy, load_test_data
    from src.helpers import load_tokenizer

    print(f"Loading SFT model from {SFT_CKPT}...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, SFT_CKPT).eval()
    tokenizer = load_tokenizer(BASE_MODEL)

    test_data = load_test_data(TEST_DATA)
    if max_examples:
        test_data = test_data[:max_examples]

    print(f"Running SFT greedy eval on {len(test_data)} examples...")
    results = evaluate_greedy(model, tokenizer, test_data, "cuda")
    _print_accuracy("SFT accuracy", results)
    return results


@app.function(image=image, gpu="a100-40gb", timeout=8 * 3600, volumes={"/vol": vol}, secrets=[hf_secret], retries=0)
def run_best_of_n(n_samples: int = 8, max_examples: int = None) -> dict:
    """Best-of-N: sample N solutions per problem from the SFT model, rerank with the PRM."""
    _setup()
    import torch
    import yaml
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from src.evaluation.eval_gsm8k import evaluate_best_of_n, load_test_data
    from src.helpers import load_tokenizer
    from src.models.prm_model import get_sep_token_ids, load_prm_model
    from src.training.prm_trainer import find_latest_checkpoint, load_prm_checkpoint

    device = "cuda"

    print("Loading SFT generator...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    gen_model = PeftModel.from_pretrained(base, SFT_CKPT).eval()
    tokenizer = load_tokenizer(BASE_MODEL)

    print("Loading PRM scorer...")
    with open("/app/configs/prm_config.yaml") as f:
        prm_config = yaml.safe_load(f)
    prm_config["model"]["sft_checkpoint"] = SFT_CKPT
    prm_model = load_prm_model(prm_config, SFT_CKPT).to(device)

    ckpt_dir, step = find_latest_checkpoint(PRM_CKPT_DIR)
    assert ckpt_dir, f"No checkpoint found in {PRM_CKPT_DIR}"
    load_prm_checkpoint(prm_model, ckpt_dir, device)
    prm_model.eval()
    print(f"PRM loaded from checkpoint-{step}")

    sep_token_ids = get_sep_token_ids(tokenizer, sep="ки")
    test_data = load_test_data(TEST_DATA)
    if max_examples:
        test_data = test_data[:max_examples]

    print(f"Running Best-of-{n_samples} on {len(test_data)} examples...")
    results = evaluate_best_of_n(
        gen_model, prm_model, tokenizer, test_data, n=n_samples, temperature=0.7,
        sep_token_ids=sep_token_ids, device=device,
    )
    _print_accuracy(f"Best-of-{n_samples} accuracy", results)
    return results


@app.function(**GPU_FN_KWARGS)
def run_rl_eval(max_examples: int = None) -> dict:
    """Greedy-eval the DPO (RL) fine-tuned checkpoint."""
    _setup()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from src.evaluation.eval_gsm8k import evaluate_greedy, load_test_data
    from src.helpers import load_tokenizer

    print(f"Loading RL (DPO) model from {RL_CKPT}...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, RL_CKPT).eval()
    tokenizer = load_tokenizer(BASE_MODEL)

    test_data = load_test_data(TEST_DATA)
    if max_examples:
        test_data = test_data[:max_examples]

    print(f"Running RL greedy eval on {len(test_data)} examples...")
    results = evaluate_greedy(model, tokenizer, test_data, "cuda")
    _print_accuracy("RL (DPO) accuracy", results)
    return results


@app.function(**GPU_FN_KWARGS)
def run_prm_eval(max_examples: int = 2000) -> dict:
    """Step-level PRM quality: AUC-ROC and binary accuracy on held-out Math-Shepherd labels."""
    _setup()
    import yaml

    from src.evaluation.eval_prm import compute_auc_roc, compute_step_accuracy, get_prm_step_scores, load_val_data
    from src.helpers import load_tokenizer
    from src.models.prm_model import load_prm_model
    from src.training.prm_trainer import find_latest_checkpoint, load_prm_checkpoint

    device = "cuda"

    with open("/app/configs/prm_config.yaml") as f:
        config = yaml.safe_load(f)
    config["model"]["sft_checkpoint"] = SFT_CKPT

    print("Loading PRM model...")
    prm_model = load_prm_model(config, SFT_CKPT).to(device)

    ckpt_dir, step = find_latest_checkpoint(PRM_CKPT_DIR)
    assert ckpt_dir, f"No checkpoint found in {PRM_CKPT_DIR}"
    load_prm_checkpoint(prm_model, ckpt_dir, device)
    prm_model.eval()
    print(f"PRM loaded from checkpoint-{step}")

    tokenizer = load_tokenizer(config["model"]["base_model"])

    val_data = load_val_data("/vol/data/prm_val.jsonl")
    if max_examples:
        val_data = val_data[:max_examples]

    print(f"Scoring {len(val_data)} validation examples...")
    scores, labels = get_prm_step_scores(prm_model, tokenizer, val_data, device, sep=config["data"]["step_separator"])

    auc = compute_auc_roc(scores, labels)
    acc = compute_step_accuracy(scores, labels)

    print(f"\n{'=' * 40}")
    print(f"PRM Evaluation ({len(scores)} steps, checkpoint-{step})")
    print(f"AUC-ROC:  {auc:.4f}")
    print(f"Accuracy: {acc:.4f}  (threshold=0.5)")
    print("=" * 40)
    return {"auc_roc": auc, "accuracy": acc, "n_steps": len(scores), "checkpoint_step": step}


# ---------------------------------------------------------------------------
# Local entrypoints  (call by name: modal run modal/modal_eval.py::<name>)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def zero_shot(max_examples: int = 500):
    print(f"Submitting zero-shot eval ({max_examples} examples)...")
    results = run_zero_shot.remote(max_examples=max_examples or None)
    print(f"\nZero-shot: {results['accuracy']:.4f} ({results['n_correct']}/{results['n_total']})")


@app.local_entrypoint()
def sft_eval(max_examples: int = 0):
    print("Submitting SFT eval...")
    results = run_sft_eval.remote(max_examples=max_examples or None)
    print(f"\nSFT: {results['accuracy']:.4f} ({results['n_correct']}/{results['n_total']})")


@app.local_entrypoint()
def best_of_n(n_samples: int = 8, max_examples: int = 0):
    print(f"Submitting best-of-{n_samples} eval...")
    results = run_best_of_n.remote(n_samples=n_samples, max_examples=max_examples or None)
    print(f"\nBest-of-{n_samples}: {results['accuracy']:.4f} ({results['n_correct']}/{results['n_total']})")


@app.local_entrypoint()
def rl_eval(max_examples: int = 0):
    print("Submitting RL eval...")
    results = run_rl_eval.remote(max_examples=max_examples or None)
    print(f"\nRL: {results['accuracy']:.4f} ({results['n_correct']}/{results['n_total']})")


@app.local_entrypoint()
def prm_eval(max_examples: int = 2000):
    print(f"Submitting PRM eval ({max_examples} examples)...")
    results = run_prm_eval.remote(max_examples=max_examples or None)
    print(f"\nAUC-ROC: {results['auc_roc']:.4f}  |  Step Acc: {results['accuracy']:.4f}  "
          f"({results['n_steps']} steps, checkpoint-{results['checkpoint_step']})")
