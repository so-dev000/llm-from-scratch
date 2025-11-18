import os

import modal

app = modal.App("llm-pull")

data_volume = modal.Volume.from_name("llm-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("llm-checkpoints", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={"/data": data_volume})
def pull_tokenizers():
    import tarfile
    from pathlib import Path

    tokenizer_dir = Path("/data/tokenizers")
    if not tokenizer_dir.exists():
        raise FileNotFoundError("No tokenizers found in Modal Volume")

    tar_path = "/tmp/tokenizers.tar.gz"

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(tokenizer_dir, arcname="tokenizers")

    with open(tar_path, "rb") as f:
        return f.read()


@app.function(image=image, volumes={"/checkpoints": checkpoint_volume})
def pull_checkpoints(run_name: str):
    import tarfile
    from pathlib import Path

    checkpoint_dir = Path(f"/checkpoints/runs/{run_name}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Run '{run_name}' not found in Modal Volume")

    tar_path = "/tmp/checkpoints.tar.gz"

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(checkpoint_dir, arcname=f"runs/{run_name}")

    with open(tar_path, "rb") as f:
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
            checkpoints = list(run_dir.glob("*.pt"))
            runs.append(
                {
                    "name": run_dir.name,
                    "checkpoints": len(checkpoints),
                    "has_best": (run_dir / "best_model.pt").exists(),
                }
            )
    return runs


@app.local_entrypoint()
def main(run_name: str = None, list_only: bool = False):
    if list_only:
        runs = list_runs.remote()
        if not runs:
            print("No training runs found")
            return

        print("\nAvailable runs:")
        for run in runs:
            status = " " if run["has_best"] else " "
            print(f"  [{status}] {run['name']} ({run['checkpoints']} checkpoints)")
        return

    print("Pulling tokenizers...")
    tokenizers_data = pull_tokenizers.remote()
    tokenizers_path = "checkpoints/tokenizers.tar.gz"
    os.makedirs("checkpoints", exist_ok=True)
    with open(tokenizers_path, "wb") as f:
        f.write(tokenizers_data)

    import tarfile

    with tarfile.open(tokenizers_path, "r:gz") as tar:
        tar.extractall("checkpoints")
    os.remove(tokenizers_path)
    print("Tokenizers saved to checkpoints/tokenizers/")

    if run_name:
        print(f"\nPulling checkpoints for run: {run_name}")
        checkpoints_data = pull_checkpoints.remote(run_name)
        checkpoints_path = "checkpoints/checkpoints.tar.gz"
        with open(checkpoints_path, "wb") as f:
            f.write(checkpoints_data)

        with tarfile.open(checkpoints_path, "r:gz") as tar:
            tar.extractall("checkpoints")
        os.remove(checkpoints_path)
        print(f"Checkpoints saved to checkpoints/runs/{run_name}/")
    else:
        print("\nNo run_name specified. Use --run-name to pull checkpoints.")
        print("Available runs:")
        runs = list_runs.remote()
        for run in runs[:5]:
            print(f"  - {run['name']}")
