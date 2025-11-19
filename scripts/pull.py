import os

import modal

app = modal.App("llm-pull")

data_volume = modal.Volume.from_name("llm-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("llm-checkpoints", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install("tqdm")


@app.function(image=image, volumes={"/data": data_volume})
def pull_tokenizers():
    import tarfile
    from pathlib import Path

    tokenizer_dir = Path("/data/tokenizers")
    if not tokenizer_dir.exists():
        raise FileNotFoundError("No tokenizers found in Modal Volume")

    tar_path = "/tmp/tokenizers.tar.gz"

    with tarfile.open(tar_path, "w:gz", compresslevel=1) as tar:
        tar.add(tokenizer_dir, arcname="tokenizers")

    with open(tar_path, "rb") as f:
        return f.read()


@app.function(image=image, volumes={"/checkpoints": checkpoint_volume})
def pull_best_model(run_name: str):
    from pathlib import Path

    best_model_path = Path(f"/checkpoints/runs/{run_name}/best_model.pt")
    if not best_model_path.exists():
        raise FileNotFoundError(f"best_model.pt not found for run '{run_name}'")

    with open(best_model_path, "rb") as f:
        return f.read()


@app.function(image=image, volumes={"/checkpoints": checkpoint_volume})
def list_runs():
    from pathlib import Path

    runs_dir = Path("/checkpoints/runs")
    if not runs_dir.exists():
        return []

    runs = []
    for run_dir in sorted(
        runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True
    ):
        if run_dir.is_dir():
            has_best = (run_dir / "best_model.pt").exists()
            runs.append({"name": run_dir.name, "has_best": has_best})
    return runs


@app.local_entrypoint()
def main(run_name: str = None, list_only: bool = False, skip_tokenizers: bool = False):
    if list_only:
        runs = list_runs.remote()
        if not runs:
            print("No training runs found")
            return

        print("\nAvailable runs:")
        for run in runs:
            status = "✓" if run["has_best"] else "✗"
            print(f"  [{status}] {run['name']}")
        return

    import tarfile
    from pathlib import Path

    from tqdm import tqdm

    tokenizers_exist = Path("checkpoints/tokenizers").exists()

    if not skip_tokenizers and not tokenizers_exist:
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
        model_data = pull_best_model.remote(run_name)

        model_dir = Path(f"checkpoints/runs/{run_name}")
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "best_model.pt"
        with open(model_path, "wb") as f:
            with tqdm(
                total=len(model_data),
                desc="Saving best_model.pt",
                unit="B",
                unit_scale=True,
            ) as pbar:
                f.write(model_data)
                pbar.update(len(model_data))
