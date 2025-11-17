import os
import pickle

import modal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

import wandb

app = modal.App("llm-training")

PROJECT_DIR = "/Users/nsota/llm-from-scratch"

# Only include necessary Python package directories
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchinfo",
        "tqdm",
        "wandb",
        "datasets",
        "transformers",
        "huggingface-hub",
    )
    .add_local_dir(f"{PROJECT_DIR}/model", remote_path="/root/llm-from-scratch/model")
    .add_local_dir(f"{PROJECT_DIR}/data", remote_path="/root/llm-from-scratch/data")
    .add_local_dir(f"{PROJECT_DIR}/utils", remote_path="/root/llm-from-scratch/utils")
    .add_local_dir(f"{PROJECT_DIR}/block", remote_path="/root/llm-from-scratch/block")
    .add_local_dir(f"{PROJECT_DIR}/layer", remote_path="/root/llm-from-scratch/layer")
    .add_local_dir(
        f"{PROJECT_DIR}/component", remote_path="/root/llm-from-scratch/component"
    )
    .add_local_dir(
        f"{PROJECT_DIR}/tokenizer", remote_path="/root/llm-from-scratch/tokenizer"
    )
)

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
MAX_LENGTH = 128
MODEL_DIM = 512
ENCODER_LAYERS = 6
DECODER_LAYERS = 6
PAD_IDX = 0
CLIP_GRAD = 1.0


def train_epoch(
    model, loader, optimizer, criterion, device, create_causal_mask, combine_masks
):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_mask = batch["src_mask"].to(device)
        tgt_mask = batch["tgt_mask"].to(device)

        # Expand src_mask for encoder self-attention:
        # (batch, src_len) -> (batch, src_len, src_len)
        src_mask_expanded = src_mask.unsqueeze(1) & src_mask.unsqueeze(2)

        # Create causal mask for target
        tgt_len = tgt.size(1)
        causal_mask = create_causal_mask(tgt_len, device=device)
        tgt_combined_mask = combine_masks(tgt_mask, causal_mask)

        # shift target sequence
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Adjust mask size to match tgt_input
        tgt_input_mask = tgt_combined_mask[:, :-1, :-1]

        # Forward pass
        output = model(
            src,
            tgt_input,
            encoder_src_mask=src_mask_expanded,
            decoder_src_mask=src_mask,
            tgt_mask=tgt_input_mask,
        )

        # Calculate loss
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()

        total_loss += loss.item()
        wandb.log({"batch_train_loss": loss.item()})

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, create_causal_mask, combine_masks):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)

            src_mask_expanded = src_mask.unsqueeze(1) & src_mask.unsqueeze(2)

            tgt_len = tgt.size(1)
            causal_mask = create_causal_mask(tgt_len, device=device)
            tgt_combined_mask = combine_masks(tgt_mask, causal_mask)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_input_mask = tgt_combined_mask[:, :-1, :-1]

            output = model(
                src,
                tgt_input,
                encoder_src_mask=src_mask_expanded,
                decoder_src_mask=src_mask,
                tgt_mask=tgt_input_mask,
            )

            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            total_loss += loss.item()

    return total_loss / len(loader)


@app.function(
    image=image,
    gpu="L4",
    timeout=10800,  # 3hour
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train():
    import sys

    sys.path.insert(0, "/root/llm-from-scratch")

    # Import local modules
    from data.translation_dataset import TranslationDataset
    from datasets import load_dataset
    from model.transformer import Transformer
    from utils.collate import collate
    from utils.hf_hub import download, upload
    from utils.masking import combine_masks, create_causal_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Modal with device: {device}")

    # Initialize wandb
    wandb.init(
        project="llm-from-scratch",
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "model_dim": MODEL_DIM,
            "encoder_layers": ENCODER_LAYERS,
            "decoder_layers": DECODER_LAYERS,
        },
    )

    # Load tokenizers from Hugging Face Hub
    print("Loading tokenizers from Hugging Face Hub...")

    hf_repo_id = os.environ.get("HF_REPO_ID")
    if not hf_repo_id:
        raise ValueError("HF_REPO_ID environment variable must be set")

    print(f"Downloading tokenizers from {hf_repo_id}...")

    en_tokenizer_path = "checkpoints/tokenizers/bsd_ja_en/en_bpe.pkl"
    ja_tokenizer_path = "checkpoints/tokenizers/bsd_ja_en/ja_bpe.pkl"

    download(
        repo_id=hf_repo_id,
        allow_patterns=[en_tokenizer_path, ja_tokenizer_path],
        token=os.environ.get("HF_TOKEN"),
        repo_type="model",
        local_dir=".",
        confirm=False,
    )

    print("Tokenizers downloaded successfully!")

    with open(en_tokenizer_path, "rb") as f:
        en_tokenizer = pickle.load(f)

    with open(ja_tokenizer_path, "rb") as f:
        ja_tokenizer = pickle.load(f)

    # Use max vocab size
    vocab_size = max(len(en_tokenizer.vocab), len(ja_tokenizer.vocab))
    print(f"Vocab size: {vocab_size}")

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("ryo0634/bsd_ja_en")

    # Create datasets
    train_dataset = TranslationDataset(
        data=dataset["train"],
        en_tokenizer=en_tokenizer,
        ja_tokenizer=ja_tokenizer,
        max_length=MAX_LENGTH,
    )

    val_dataset = TranslationDataset(
        data=dataset["validation"],
        en_tokenizer=en_tokenizer,
        ja_tokenizer=ja_tokenizer,
        max_length=MAX_LENGTH,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Initialize model
    print(f"Initializing model on {device}...")
    model = Transformer(
        vocab_size=vocab_size,
        model_dim=MODEL_DIM,
        encoder_num=ENCODER_LAYERS,
        decoder_num=DECODER_LAYERS,
    ).to(device)

    summary(model)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Training loop
    best_val_loss = float("inf")

    print(f"Using Hugging Face repo: {hf_repo_id}")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 50)

        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            create_causal_mask,
            combine_masks,
        )
        print(f"Training Loss: {train_loss:.4f}")

        # Evaluate
        val_loss = evaluate(
            model, val_loader, criterion, device, create_causal_mask, combine_masks
        )
        print(f"Validation Loss: {val_loss:.4f}")

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        # Save and upload best model to Hugging Face Hub
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save model locally first
            checkpoint_path = "/tmp/checkpoints/models/best_model.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "vocab_size": vocab_size,
                    "model_dim": MODEL_DIM,
                    "encoder_layers": ENCODER_LAYERS,
                    "decoder_layers": DECODER_LAYERS,
                },
                checkpoint_path,
            )
            print(f"Best model saved (val_loss: {val_loss:.4f})")

            # Upload to Hugging Face Hub
            try:
                upload(
                    local_dir="/tmp/checkpoints/models",
                    repo_id=hf_repo_id,
                    token=os.environ.get("HF_TOKEN"),
                    private=True,
                    repo_type="model",
                    path_in_repo="checkpoints/models",
                )
                print(f"Model uploaded to Hugging Face Hub: {hf_repo_id}")
            except Exception as e:
                print(f"Error uploading to Hugging Face Hub: {e}")

    wandb.finish()


@app.local_entrypoint()
def main():
    train.remote()
