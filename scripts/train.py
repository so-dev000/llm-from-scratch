from datetime import datetime

import modal
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from scripts.config import Config
from scripts.lightning_module import GPTLightningModule, TransformerLightningModule
from utils.training_pipeline import get_data_module

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
    .add_local_dir("model", remote_path="/root/model")
    .add_local_dir("utils", remote_path="/root/utils")
    .add_local_dir("block", remote_path="/root/block")
    .add_local_dir("layer", remote_path="/root/layer")
    .add_local_dir("component", remote_path="/root/component")
    .add_local_dir("tokenizer", remote_path="/root/tokenizer")
    .add_local_dir("scripts", remote_path="/root/scripts")
)


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/vol": volume},
    timeout=3600 * 12,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(config: Config):
    torch.set_float32_matmul_precision("high")

    data_module = get_data_module(config)
    data_module.prepare_data()
    data_module.setup(stage="fit")

    if config.model.model_type == "transformer":
        pl_module = TransformerLightningModule(config)
    elif config.model.model_type == "gpt":
        pl_module = GPTLightningModule(config)
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")

    callbacks = [
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
