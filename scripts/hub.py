import argparse
import os
import sys

from dotenv import load_dotenv

from utils.hf_hub import download_checkpoints, upload_checkpoints

load_dotenv()


def cmd_push(args):
    repo_id = args.repo_id or os.getenv("HF_REPO_ID")
    token = args.token or os.getenv("HF_TOKEN")
    private = args.private or os.getenv("PRIVATE_REPO", "false").lower() == "true"

    if not repo_id:
        print("Error: set HF_REPO_ID or use --repo-id")
        return 1

    if not os.path.exists("checkpoints"):
        print("Error: checkpoints not found")
        print("Train model first: python -m scripts.train")
        return 1

    success = upload_checkpoints("checkpoints", repo_id, token, private)
    if success:
        print(f"\nView at: https://huggingface.co/{repo_id}")
        return 0
    return 1


def cmd_pull(args):
    repo_id = args.repo_id or os.getenv("HF_REPO_ID")
    token = args.token or os.getenv("HF_TOKEN")

    if not repo_id:
        print("Error: set HF_REPO_ID or use --repo-id")
        return 1

    try:
        download_checkpoints(repo_id, token)
        return 0
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nPlease check:")
        print(f"1. Repository: https://huggingface.co/{repo_id}")
        print("2. Upload first: python -m scripts.hub push")
        print("3. HF_TOKEN is valid (if private)")
        return 1


def main():
    parser = argparse.ArgumentParser(description="HF Hub CLI")
    sub = parser.add_subparsers(dest="command")

    push = sub.add_parser("push")
    push.add_argument("--repo-id")
    push.add_argument("--token")
    push.add_argument("--private", action="store_true")

    pull = sub.add_parser("pull")
    pull.add_argument("--repo-id")
    pull.add_argument("--token")

    args = parser.parse_args()

    if args.command == "push":
        return cmd_push(args)
    elif args.command == "pull":
        return cmd_pull(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
