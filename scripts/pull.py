import os
import tarfile
from pathlib import Path

import modal
from tqdm import tqdm

app = modal.App("llm-pull")

volume = modal.Volume.from_name("llm-from-scratch", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install("tqdm")

CHECKPOINT_DIR = "/vol/runs"


@app.function(image=image, volumes={"/vol": volume})
def pull_tokenizers():
    tokenizer_dir = Path("/vol/tokenizers")
    if not tokenizer_dir.exists():
        raise FileNotFoundError("No tokenizers found in Modal Volume")

    tar_path = "/tmp/tokenizers.tar.gz"

    with tarfile.open(tar_path, "w:gz", compresslevel=1) as tar:
        tar.add(tokenizer_dir, arcname="tokenizers")

    with open(tar_path, "rb") as f:
        return f.read()


@app.function(image=image, volumes={"/vol": volume})
def pull_best_model(run_name: str):
    best_model_path = Path(f"{CHECKPOINT_DIR}/{run_name}/best_model.ckpt")
    if not best_model_path.exists():
        raise FileNotFoundError(f"best_model.ckpt not found for run '{run_name}'")

    with open(best_model_path, "rb") as f:
        return f.read()


@app.function(image=image, volumes={"/vol": volume})
def pull_checkpoint_by_filename(run_name: str, filename: str):
    checkpoint_path = Path(f"{CHECKPOINT_DIR}/{run_name}/{filename}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{filename} not found for run '{run_name}'")

    with open(checkpoint_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(run_name: str = None, best_model: bool = True, filename: str = None):
    tokenizers_data = pull_tokenizers.remote()

    tokenizers_path = "checkpoints/tokenizers.tar.gz"
    os.makedirs("checkpoints", exist_ok=True)

    with open(tokenizers_path, "wb") as f:
        f.write(tokenizers_data)

    with tarfile.open(tokenizers_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting tokenizers", unit="files"):
            tar.extract(member, "checkpoints")
    os.remove(tokenizers_path)

    if run_name:
        model_dir = Path(f"checkpoints/runs/{run_name}")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Pull specific filename if provided
        if filename:
            file_data = pull_checkpoint_by_filename.remote(run_name, filename)
            file_path = model_dir / filename
            with open(file_path, "wb") as f:
                with tqdm(
                    total=len(file_data),
                    desc=f"Saving {filename}",
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    f.write(file_data)
                    pbar.update(len(file_data))

        # Pull best model if requested
        if best_model and not filename:
            model_data = pull_best_model.remote(run_name)
            model_path = model_dir / "best_model.ckpt"
            with open(model_path, "wb") as f:
                with tqdm(
                    total=len(model_data),
                    desc="Saving best_model.ckpt",
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    f.write(model_data)
                    pbar.update(len(model_data))
