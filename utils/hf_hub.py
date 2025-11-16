import os
import shutil

from huggingface_hub import HfApi, create_repo, hf_hub_download


def ensure_repo(repo_id, token, private):
    try:
        create_repo(repo_id, token=token, private=private, exist_ok=True)
    except Exception as e:
        print(f"Repo creation: {e}")


def upload_checkpoints(checkpoints_dir, repo_id, token=None, private=False):
    ensure_repo(repo_id, token, private)

    if not os.path.exists(checkpoints_dir):
        print(f"Error: {checkpoints_dir} not found")
        return False

    print(f"Uploading {checkpoints_dir} to {repo_id}")
    HfApi(token=token).upload_folder(
        folder_path=checkpoints_dir,
        repo_id=repo_id,
        path_in_repo="checkpoints",
    )
    print("Upload complete")
    return True


def confirm_overwrite(path):
    if not os.path.exists(path):
        return True
    response = input(f"{path} exists. Overwrite? [y/N]: ").strip().lower()
    return response in ("y", "yes")


def download_checkpoints(repo_id, token=None):
    files = [
        "checkpoints/tokenizers/bsd_ja_en/en_bpe.pkl",
        "checkpoints/tokenizers/bsd_ja_en/ja_bpe.pkl",
        "checkpoints/models/best_model.pt",
    ]

    print(f"Downloading from {repo_id}")
    for file in files:
        try:
            if not confirm_overwrite(file):
                print(f"Skipped {file}")
                continue

            path = hf_hub_download(repo_id, file, token=token)
            local_path = file
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            shutil.copy(path, local_path)
            print(f"Downloaded {file}")
        except Exception as e:
            print(f"Failed to download {file}: {e}")

    print("Download complete")
