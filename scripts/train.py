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


volume = modal.Volume.from_name("llm-from-scratch", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchinfo", "tqdm", "wandb", "datasets", "regex", "model")
    .add_local_dir("model", remote_path="/root/llm-from-scratch/model")
    .add_local_dir("utils", remote_path="/root/llm-from-scratch/utils")
    .add_local_dir("block", remote_path="/root/llm-from-scratch/block")
    .add_local_dir("layer", remote_path="/root/llm-from-scratch/layer")
    .add_local_dir("component", remote_path="/root/llm-from-scratch/component")
    .add_local_dir("tokenizer", remote_path="/root/llm-from-scratch/tokenizer")
)

DATASET_NAME = "ryo0634/bsd_ja_en"

BATCH_SIZE = 256
NUM_EPOCHS = 50
MAX_LENGTH = 64
MODEL_DIM = 256
ENCODER_LAYERS = 4
DECODER_LAYERS = 4
PAD_IDX = 0
NUM_WORKERS = 8
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 5
# Optimizer and learning rate parameters from "Attention is All You Need" paper
WARMUP_STEPS = 4000
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.98
ADAM_EPSILON = 1e-9


def get_lr(step, model_dim, warmup_steps):
    # Learning rate schedule from "Attention is All You Need" paper.
    # lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

    step = max(step, 1)  # Avoid division by zero
    lr = (model_dim**-0.5) * min(step**-0.5, step * warmup_steps**-1.5)
    return lr


def train_epoch(
    model, loader, optimizer, criterion, device, create_causal_mask, combine_masks, step
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

        # Update learning rate
        step += 1
        lr = get_lr(step, MODEL_DIM, WARMUP_STEPS)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

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
        optimizer.step()

        total_loss += loss.item()
        wandb.log({"batch_train_loss": loss.item(), "learning_rate": lr, "step": step})

    return total_loss / len(loader), step


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
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "model_dim": MODEL_DIM,
            "encoder_layers": ENCODER_LAYERS,
            "decoder_layers": DECODER_LAYERS,
            "dataset": DATASET_NAME,
            "warmup_steps": WARMUP_STEPS,
            "adam_beta1": ADAM_BETA1,
            "adam_beta2": ADAM_BETA2,
            "adam_epsilon": ADAM_EPSILON,
            "label_smoothing": LABEL_SMOOTHING,
        },
    )

    tokenizer_dir = "/vol/tokenizers/bsd_en_ja"
    en_tokenizer_path = f"{tokenizer_dir}/en_bpe.pkl"
    ja_tokenizer_path = f"{tokenizer_dir}/ja_bpe.pkl"

    if not os.path.exists(en_tokenizer_path) or not os.path.exists(ja_tokenizer_path):
        raise FileNotFoundError("Tokenizers not found. Run scripts/prepare.py first")

    en_tokenizer = BPE.load(en_tokenizer_path)
    ja_tokenizer = BPE.load(ja_tokenizer_path)

    src_vocab_size = en_tokenizer.get_vocab_size()
    tgt_vocab_size = ja_tokenizer.get_vocab_size()

    from datasets import DatasetDict

    dataset = load_dataset(DATASET_NAME, split="train")
    # train 85%, val 10%, held-out 5% reserved for test CSV
    train_rest = dataset.train_test_split(test_size=0.15, seed=42)
    val_test = train_rest["test"].train_test_split(test_size=0.33, seed=42)
    train_val = DatasetDict({"train": train_rest["train"], "val": val_test["train"]})

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

    tokenized_datasets = train_val.map(
        preprocess_batch,
        batched=True,
        batch_size=1000,
        remove_columns=train_val["train"].column_names,
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
        tokenized_datasets["val"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        model_dim=MODEL_DIM,
        encoder_num=ENCODER_LAYERS,
        decoder_num=DECODER_LAYERS,
        padding_idx=PAD_IDX,
    ).to(device)

    summary(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPSILON,
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX, label_smoothing=LABEL_SMOOTHING
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    run_checkpoint_dir = f"/vol/runs/{run_name}"
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    step = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, step = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            create_causal_mask,
            combine_masks,
            step,
        )

        val_loss = evaluate(
            model, val_loader, criterion, device, create_causal_mask, combine_masks
        )

        current_lr = optimizer.param_groups[0]["lr"]
        loss_ratio = val_loss / train_loss if train_loss > 0 else 0
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Ratio: {loss_ratio:.4f}, LR: {current_lr:.2e}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_train_loss_ratio": loss_ratio,
                "learning_rate": current_lr,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "src_vocab_size": src_vocab_size,
                    "tgt_vocab_size": tgt_vocab_size,
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
