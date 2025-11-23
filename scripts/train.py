import os
from datetime import datetime

import modal
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

app = modal.App("llm-training")

PROJECT_DIR = "/Users/nsota/llm-from-scratch"

volume = modal.Volume.from_name("llm-from-scratch", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchinfo", "tqdm", "wandb", "datasets", "regex")
    .add_local_dir(f"{PROJECT_DIR}/model", remote_path="/root/llm-from-scratch/model")
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

DATASET_NAME = "ryo0634/bsd_ja_en"

BATCH_SIZE = 256
LEARNING_RATE = 3e-4
NUM_EPOCHS = 50
MAX_LENGTH = 64
MODEL_DIM = 256
ENCODER_LAYERS = 4
DECODER_LAYERS = 4
PAD_IDX = 0
CLIP_GRAD = 1.0
NUM_WORKERS = 8
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 5
WARMUP_EPOCHS = 2


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
    volumes={"/vol": volume},
    timeout=3600 * 12,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(run_name: str = None):
    import sys

    sys.path.insert(0, "/root/llm-from-scratch")

    from datasets import load_dataset

    from model.transformer import Transformer
    from tokenizer.bpe import BPE
    from utils.collate import collate
    from utils.masking import combine_masks, create_causal_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_float32_matmul_precision("high")

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    wandb.init(
        project="llm-from-scratch",
        name=run_name,
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "model_dim": MODEL_DIM,
            "encoder_layers": ENCODER_LAYERS,
            "decoder_layers": DECODER_LAYERS,
            "dataset": DATASET_NAME,
        },
    )

    tokenizer_dir = "/vol/tokenizers/bsd_en_ja"
    en_tokenizer_path = f"{tokenizer_dir}/en_bpe.pkl"
    ja_tokenizer_path = f"{tokenizer_dir}/ja_bpe.pkl"

    if not os.path.exists(en_tokenizer_path) or not os.path.exists(ja_tokenizer_path):
        raise FileNotFoundError("Tokenizers not found. Run scripts/prepare.py first")

    en_tokenizer = BPE.load(en_tokenizer_path)
    ja_tokenizer = BPE.load(ja_tokenizer_path)

    vocab_size = max(en_tokenizer.get_vocab_size(), ja_tokenizer.get_vocab_size())

    dataset = load_dataset(DATASET_NAME, split="train")
    train_test = dataset.train_test_split(test_size=0.05, seed=42)

    def preprocess_batch(batch):
        src_ids = []
        for text in batch["en_sentence"]:
            ids = en_tokenizer.encode(text, add_special_tokens=True)
            if len(ids) > MAX_LENGTH:
                ids = ids[: MAX_LENGTH - 1] + [en_tokenizer.special_tokens["<EOS>"]]
            src_ids.append(ids)

        tgt_ids = []
        for text in batch["ja_sentence"]:
            ids = ja_tokenizer.encode(text, add_special_tokens=True)
            if len(ids) > MAX_LENGTH:
                ids = ids[: MAX_LENGTH - 1] + [ja_tokenizer.special_tokens["<EOS>"]]
            tgt_ids.append(ids)

        return {"src": src_ids, "tgt": tgt_ids}

    tokenized_datasets = train_test.map(
        preprocess_batch,
        batched=True,
        batch_size=1000,
        remove_columns=train_test["train"].column_names,
        desc="Tokenizing dataset",
    )

    tokenized_datasets.set_format(type="torch", columns=["src", "tgt"])

    train_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        tokenized_datasets["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    model = Transformer(
        vocab_size=vocab_size,
        model_dim=MODEL_DIM,
        encoder_num=ENCODER_LAYERS,
        decoder_num=DECODER_LAYERS,
    ).to(device)

    summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX, label_smoothing=LABEL_SMOOTHING
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    run_checkpoint_dir = f"/vol/runs/{run_name}"
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        # Warmup learning rate
        if epoch < WARMUP_EPOCHS:
            warmup_lr = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            create_causal_mask,
            combine_masks,
        )

        val_loss = evaluate(
            model, val_loader, criterion, device, create_causal_mask, combine_masks
        )

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            }
        )

        # Only use scheduler after warmup
        if epoch >= WARMUP_EPOCHS:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "vocab_size": vocab_size,
                    "model_dim": MODEL_DIM,
                    "encoder_layers": ENCODER_LAYERS,
                    "decoder_layers": DECODER_LAYERS,
                },
                f"{run_checkpoint_dir}/best_model.pt",
            )
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

        volume.commit()

    wandb.finish()


@app.local_entrypoint()
def main(run_name: str = None):
    train.remote(run_name=run_name)
