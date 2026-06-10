"""Modal app: SFT training of Qwen2.5-7B-Instruct + LoRA on GSM8K.

Setup (one-time):
    pip install modal && modal setup
    modal secret create huggingface-secret HF_TOKEN=<your_token>
    python data/download_gsm8k.py

Run:
    modal run modal/modal_sft.py

The LoRA adapter is written to the 'gsm8k-training-vol' Modal Volume at
/vol/sft_checkpoint. Retrieve it locally with:
    modal volume get gsm8k-training-vol sft_checkpoint ./local_sft_checkpoint
"""

from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent

# Shared persistent volume for checkpoints across all training/eval apps
vol = modal.Volume.from_name("gsm8k-training-vol", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(_ROOT / "requirements.txt"))
    .add_local_dir(str(_ROOT / "src"), remote_path="/app/src")
    .add_local_dir(str(_ROOT / "configs"), remote_path="/app/configs")
    .add_local_dir(str(_ROOT / "data" / "raw"), remote_path="/app/data/raw")
)

app = modal.App("gsm8k-sft-training")


@app.function(
    image=image,
    gpu="a100-40gb",
    timeout=6 * 3600,
    volumes={"/vol": vol},
    secrets=[hf_secret],
    retries=0,
)
def run_sft() -> None:
    """Train inside the container, writing the LoRA adapter to the persistent volume."""
    import os
    import sys
    import tempfile

    import yaml

    sys.path.insert(0, "/app")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    with open("/app/configs/sft_config.yaml") as f:
        config = yaml.safe_load(f)

    config["output"]["output_dir"] = "/vol/sft_checkpoint"
    config["output"]["logging_dir"] = "/vol/sft_logs"
    config["data"]["local_train_path"] = "/app/data/raw/gsm8k_train.jsonl"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config, tmp)
        tmp_config_path = tmp.name

    from src.training.sft_trainer import run_sft_training
    run_sft_training(tmp_config_path)

    vol.commit()  # persist writes so they survive after the container exits
    print("SFT training complete. Checkpoint committed to volume.")


@app.local_entrypoint()
def main() -> None:
    """Submit the SFT training job to Modal."""
    print("Submitting SFT training job to Modal...")
    run_sft.remote()
    print("Job submitted. Monitor progress in the Modal dashboard.")
