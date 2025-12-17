from datetime import datetime

import modal
import pytorch_lightning as L
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model.gpt import GPT
from model.transformer import Transformer
from scripts.config import Config
from utils.training_pipeline import TransformerLRScheduler, get_data_module

app = modal.App("llm-training")

volume = modal.Volume.from_name("llm-from-scratch", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchinfo",
        "pytorch-lightning",
        "wandb",
        "datasets",
        "regex",
    )
    .add_local_dir("model", remote_path="/root/llm-from-scratch/model")
    .add_local_dir("utils", remote_path="/root/llm-from-scratch/utils")
    .add_local_dir("block", remote_path="/root/llm-from-scratch/block")
    .add_local_dir("layer", remote_path="/root/llm-from-scratch/layer")
    .add_local_dir("component", remote_path="/root/llm-from-scratch/component")
    .add_local_dir("tokenizer", remote_path="/root/llm-from-scratch/tokenizer")
    .add_local_dir("scripts", remote_path="/root/llm-from-scratch/scripts")
)


class TransformerLightningModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config.to_dict())
        self.config = config
        self.model = Transformer(config.model)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config.data.pad_idx,
            label_smoothing=config.training.label_smoothing,
        )

    def forward(
        self,
        source_tokens,
        target_tokens,
        encoder_src_mask=None,
        decoder_src_mask=None,
        tgt_mask=None,
    ):
        return self.model(
            source_tokens, target_tokens, encoder_src_mask, decoder_src_mask, tgt_mask
        )

    def _shared_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        src_mask = batch["src_mask"]
        tgt_mask = batch["tgt_mask"]

        src_mask_expanded = src_mask.unsqueeze(1) & src_mask.unsqueeze(2)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_input_mask = tgt_mask[:, :-1, :-1]

        output = self(
            src,
            tgt_input,
            encoder_src_mask=src_mask_expanded,
            decoder_src_mask=src_mask,
            tgt_mask=tgt_input_mask,
        )

        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        loss = self.criterion(output, tgt_output)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.optimizer.initial_lr,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            eps=self.config.optimizer.adam_epsilon,
        )
        return optimizer


class GPTLightningModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config.to_dict())
        self.config = config
        self.model = GPT(config.model)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config.data.pad_idx,
            label_smoothing=config.training.label_smoothing,
        )

    def forward(self, tokens, mask=None):
        return self.model(tokens, mask)

    def _shared_step(self, batch, batch_idx):
        tokens = batch["tokens"]
        mask = batch.get("mask", None)

        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]

        if mask is not None:
            input_mask = mask[:, :-1, :-1]
        else:
            input_mask = None

        output = self(input_tokens, input_mask)
        output = output.reshape(-1, output.size(-1))
        target_tokens = target_tokens.reshape(-1)
        loss = self.criterion(output, target_tokens)
        perplexity = torch.exp(loss)

        return loss, perplexity

    def training_step(self, batch, batch_idx):
        loss, perplexity = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_perplexity", perplexity, on_step=True, on_epoch=True, prog_bar=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, perplexity = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_perplexity",
            perplexity,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.optimizer.initial_lr,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            eps=self.config.optimizer.adam_epsilon,
        )
        return optimizer


@app.function(
    image=image,
    gpu="L4",
    volumes={"/vol": volume},
    timeout=3600 * 12,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(config: Config):
    torch.set_float32_matmul_precision("high")

    if config.model.model_type == "transformer":
        pl_module = TransformerLightningModule(config)
    elif config.model.model_type == "gpt":
        pl_module = GPTLightningModule(config)
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")

    data_module = get_data_module(config)

    callbacks = [
        TransformerLRScheduler(config.model.model_dim, config.optimizer.warmup_steps),
        EarlyStopping(
            monitor="val_loss",
            patience=config.training.early_stopping_patience,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=f"{config.checkpoint_dir}/{config.run_name}",
            filename="best_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True,
        ),
    ]

    if config.wandb.enabled:
        logger = WandbLogger(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.run_name,
            log_model=config.wandb.log_model,
        )
    else:
        logger = None

    trainer = L.Trainer(
        max_epochs=config.training.num_epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.training.gradient_clip_val,
        precision=config.training.precision,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        val_check_interval=config.training.val_check_interval,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(pl_module, datamodule=data_module)

    volume.commit()

    checkpoint_path = f"{config.checkpoint_dir}/{config.run_name}"
    print(f"Training complete. Best model saved to {checkpoint_path}")


@app.local_entrypoint()
def main(model_type: str = "transformer", run_name: str = None):
    if model_type == "transformer":
        config = Config.for_transformer()
    elif model_type == "gpt":
        config = Config.for_gpt()
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Must be 'transformer' or 'gpt'"
        )

    if run_name is None:
        run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config.run_name = run_name

    config.validate()

    print(f"Starting training run: {run_name}")
    print(f"Model type: {model_type}")
    print(f"Config: {config.to_dict()}")

    train.remote(config)
