import modal

app = modal.App("modal-demo")

volume = modal.Volume.from_name("demo-volume", create_if_missing=True)

cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    .add_local_dir("utils", remote_path="/root/utils")
)


@app.function(image=cpu_image)
def cpu_task(x: int) -> dict:
    import numpy as np

    result = np.array([x, x**2, x**3])
    return {"input": x, "powers": result.tolist()}


@app.function(image=gpu_image, gpu="L4", timeout=300)
def gpu_task(matrix_size: int) -> dict:
    import torch

    from utils.masking import create_causal_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    matrix_a = torch.randn(matrix_size, matrix_size, device=device)
    matrix_b = torch.randn(matrix_size, matrix_size, device=device)
    result = torch.matmul(matrix_a, matrix_b)

    causal_mask = create_causal_mask(10, device=device)

    return {
        "device": str(device),
        "matrix_size": matrix_size,
        "result_sum": float(result.sum()),
        "causal_mask_shape": list(causal_mask.shape),
    }


@app.function(image=cpu_image, volumes={"/data": volume})
def write_to_volume(filename: str, content: str):
    filepath = f"/data/{filename}"

    with open(filepath, "w") as f:
        f.write(content)

    volume.commit()
    return filepath


@app.function(image=cpu_image, volumes={"/data": volume})
def read_from_volume(filename: str) -> str:
    filepath = f"/data/{filename}"

    with open(filepath, "r") as f:
        return f.read()


@app.function(
    image=cpu_image,
    secrets=[modal.Secret.from_dict({"DEMO_KEY": "demo_value_12345"})],
)
def use_secret() -> str:
    import os

    return os.getenv("DEMO_KEY", "not_found")


@app.local_entrypoint()
def main():
    cpu_result = cpu_task.remote(5)
    print(f"CPU: {cpu_result}")

    gpu_result = gpu_task.remote(1000)
    print(f"GPU: {gpu_result}")

    filepath = write_to_volume.remote("demo.txt", "Hello Modal")
    print(f"Written: {filepath}")

    content = read_from_volume.remote("demo.txt")
    print(f"Read: {content}")

    secret = use_secret.remote()
    print(f"Secret: {secret}")
