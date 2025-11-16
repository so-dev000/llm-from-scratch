import argparse
import os
import sys

from dotenv import load_dotenv

from utils.hf_hub import download_checkpoints, upload_checkpoints

load_dotenv()


def cmd_push():
    repo_id = os.getenv("HF_REPO_ID")
    token = os.getenv("HF_TOKEN")

    if not repo_id or not token:
        print("Error: set HF_REPO_ID and HF_TOKEN in .env")
        return 1

    if not os.path.exists("checkpoints"):
        print("Error: checkpoints not found")
        print("Train model first: python -m scripts.train")
        return 1

    success = upload_checkpoints("checkpoints", repo_id, token, private=True)
    if success:
        print(f"\nView at: https://huggingface.co/{repo_id}")
        return 0
    return 1


def cmd_pull():
    repo_id = os.getenv("HF_REPO_ID")
    token = os.getenv("HF_TOKEN")

    if not repo_id or not token:
        print("Error: set HF_REPO_ID and HF_TOKEN in .env")
        return 1

    try:
        download_checkpoints(repo_id, token)
        return 0
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nPlease check:")
        print(f"1. Repository: https://huggingface.co/{repo_id}")
        print("2. Upload first: python -m scripts.hub push")
        print("3. HF_TOKEN is valid")
        return 1


def main():
    parser = argparse.ArgumentParser(description="HF Hub CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("push")
    sub.add_parser("pull")

    args = parser.parse_args()

    if args.command == "push":
        return cmd_push()
    elif args.command == "pull":
        return cmd_pull()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
