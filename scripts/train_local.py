import os
from datetime import datetime
from pathlib import Path

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from scripts.config import Config
from scripts.lightning_module import (
    GPTLightningModule,
    LlamaLightningModule,
    TransformerLightningModule,
)
from utils.training_pipeline import get_data_module


def train(config: Config):
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    data_module = get_data_module(config)
    data_module.setup(stage="fit")

    if config.model.model_type == "transformer":
        pl_module = TransformerLightningModule(config)
    elif config.model.model_type == "gpt":
        pl_module = GPTLightningModule(config)
    elif config.model.model_type == "llama":
        pl_module = LlamaLightningModule(config)
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
        logger.experiment.config.update(config.to_dict())
    else:
        logger = None

    trainer = L.Trainer(
        max_epochs=config.training.num_epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.training.gradient_clip_val,
        precision=config.training.precision,
        val_check_interval=config.training.val_check_interval,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(pl_module, datamodule=data_module)


def main(
    model_type: str = "transformer",
    run_name: str = None,
    checkpoint_dir: str = "./checkpoints",
    tokenizer_dir: str = "checkpoints/tokenizers",
):
    if model_type == "transformer":
        config = Config.for_transformer()
    elif model_type == "gpt":
        config = Config.for_gpt()
    elif model_type == "llama":
        config = Config.for_llama()
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Must be 'transformer', 'gpt', or 'llama'"
        )

    if run_name is None:
        run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config.run_name = run_name
    config.checkpoint_dir = checkpoint_dir
    config.tokenizer_dir = tokenizer_dir
    config.validate()

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(tokenizer_dir).mkdir(parents=True, exist_ok=True)

    train(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="transformer")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--tokenizer-dir", type=str, default="checkpoints/tokenizers")

    args = parser.parse_args()

    main(
        model_type=args.model_type,
        run_name=args.run_name,
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_dir=args.tokenizer_dir,
    )
