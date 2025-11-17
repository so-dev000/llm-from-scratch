import os
import shutil

from huggingface_hub import HfApi, create_repo, hf_hub_download


def _ensure_repo(repo_id, token=None, private=False, repo_type="model"):
    try:
        create_repo(
            repo_id, token=token, private=private, exist_ok=True, repo_type=repo_type
        )
    except Exception as e:
        print(f"Repo creation: {e}")


def _confirm_overwrite(file_path):
    if not os.path.exists(file_path):
        return True
    response = input(f"{file_path} exists. Overwrite? [y/N]: ").strip().lower()
    return response in ("y", "yes")


def upload(
    local_dir, repo_id, token=None, private=False, repo_type="model", path_in_repo=None
):
    _ensure_repo(repo_id, token, private, repo_type)

    if not os.path.exists(local_dir):
        print(f"Error: {local_dir} not found")
        return False

    if path_in_repo is None:
        path_in_repo = os.path.basename(local_dir)

    print(f"Uploading {local_dir} to {repo_id}")
    HfApi(token=token).upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
    )
    print("Upload complete")
    return True


def download(
    repo_id,
    token=None,
    repo_type="model",
    local_dir=".",
    allow_patterns=None,
    confirm=True,
):
    print(f"Downloading checkpoints from {repo_id}")

    try:
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

        if allow_patterns:
            import fnmatch

            files = [
                f
                for f in files
                if any(fnmatch.fnmatch(f, pattern) for pattern in allow_patterns)
            ]

        downloaded = 0
        skipped = 0

        for file in files:
            local_path = os.path.join(local_dir, file)

            if confirm and not _confirm_overwrite(local_path):
                print(f"Skipped {file}")
                skipped += 1
                continue

            try:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                cached_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    token=token,
                    repo_type=repo_type,
                )
                shutil.copy(cached_path, local_path)
                print(f"Downloaded {file}")
                downloaded += 1
            except Exception as e:
                print(f"Failed to download {file}: {e}")

        print(f"\nDownload complete: {downloaded} files downloaded, {skipped} skipped")
    except Exception as e:
        print(f"Failed to download: {e}")
        raise
