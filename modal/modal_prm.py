"""Modal app: PRM training on Math-Shepherd, initialized from the SFT checkpoint.

Prerequisites:
    1. SFT training complete (sft_checkpoint present in the volume)
    2. Math-Shepherd data prepared locally:  python data/prepare_prm_data.py
    3. Upload it to the volume (once):       modal run modal/modal_prm.py::upload_data

Run training:
    modal run modal/modal_prm.py

Checkpoints land in the 'gsm8k-training-vol' volume at /vol/prm_checkpoint.
Retrieve locally with:
    modal volume get gsm8k-training-vol prm_checkpoint ./local_prm_checkpoint
"""

from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent

vol = modal.Volume.from_name("gsm8k-training-vol", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(_ROOT / "requirements.txt"))
    .add_local_dir(str(_ROOT / "src"), remote_path="/app/src")
    .add_local_dir(str(_ROOT / "configs"), remote_path="/app/configs")
    .add_local_dir(str(_ROOT / "data" / "processed"), remote_path="/app/data/processed")
)

app = modal.App("gsm8k-prm-training")


@app.function(image=image, volumes={"/vol": vol}, timeout=600)
def upload_data() -> None:
    """Copy the locally-prepared PRM data (baked into the image) onto the volume."""
    import os
    import shutil

    os.makedirs("/vol/data", exist_ok=True)
    for name in ("prm_train.jsonl", "prm_val.jsonl"):
        shutil.copy(f"/app/data/processed/{name}", f"/vol/data/{name}")
        print(f"Copied /app/data/processed/{name} -> /vol/data/{name}")
    vol.commit()


@app.function(
    image=image,
    gpu="a100-40gb",
    timeout=8 * 3600,
    volumes={"/vol": vol},
    secrets=[hf_secret],
    retries=0,
)
def run_prm() -> None:
    """Train inside the container; reads the SFT checkpoint and PRM data from the volume."""
    import os
    import sys
    import tempfile

    import yaml

    sys.path.insert(0, "/app")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    with open("/app/configs/prm_config.yaml") as f:
        config = yaml.safe_load(f)

    config["model"]["sft_checkpoint"] = "/vol/sft_checkpoint"
    config["data"]["train_file"] = "/vol/data/prm_train.jsonl"
    config["data"]["val_file"] = "/vol/data/prm_val.jsonl"
    config["output"]["output_dir"] = "/vol/prm_checkpoint"
    config["output"]["logging_dir"] = "/vol/prm_logs"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config, tmp)
        tmp_config_path = tmp.name

    from src.training.prm_trainer import run_prm_training
    run_prm_training(tmp_config_path)

    vol.commit()
    print("PRM training complete. Checkpoint committed to volume.")


@app.local_entrypoint()
def main() -> None:
    print("Submitting PRM training job to Modal...")
    print("(Make sure you've run `modal run modal/modal_prm.py::upload_data` first.)")
    run_prm.remote()
    print("Job submitted. Monitor progress in the Modal dashboard.")
