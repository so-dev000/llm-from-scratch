import modal

from utils.masking import create_causal_mask

app = modal.App("modal-demo")

volume = modal.Volume.from_name("demo-volume", create_if_missing=True)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    .add_local_dir("utils", remote_path="/root/utils")
)


@app.function(image=image, timeout=300)
def cpu_task() -> dict:
    import numpy as np

    x = 5
    result = np.array([x, x**2, x**3])
    return {"input": x, "powers": result.tolist()}


@app.function(image=image, gpu="L4", timeout=300)
def gpu_task() -> dict:
    import torch

    matrix_size = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    matrix_a = torch.randn(matrix_size, matrix_size, device=device)
    matrix_b = torch.randn(matrix_size, matrix_size, device=device)
    result = torch.matmul(matrix_a, matrix_b)
    mask = create_causal_mask(10, device)
    result = result != mask

    return {
        "device": str(device),
        "matrix_size": matrix_size,
        "result": result,
    }


@app.function(image=image, volumes={"/data": volume})
def write_to_volume():
    filepath = "/data/demo.txt"

    with open(filepath, "w") as f:
        f.write("Hello Modal")

    volume.commit()
    return filepath


@app.function(image=image, volumes={"/data": volume})
def read_from_volume() -> str:
    filepath = "/data/demo.txt"

    with open(filepath, "r") as f:
        return f.read()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("DEMO_KEY")],
)
def use_secret() -> str:
    import os

    return os.getenv("DEMO_KEY", "not_found")


@app.local_entrypoint()
def main():
    cpu_result = cpu_task.remote()
    print(f"CPU: {cpu_result}")

    gpu_result = gpu_task.remote()
    print(f"GPU: {gpu_result}")

    filepath = write_to_volume.remote()
    print(f"Written: {filepath}")

    content = read_from_volume.remote()
    print(f"Read: {content}")

    secret = use_secret.remote()
    print(f"Secret: {secret}")
