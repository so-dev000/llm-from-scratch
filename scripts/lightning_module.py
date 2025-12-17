import pytorch_lightning as L
import torch
from torch import nn

from model.gpt import GPT
from model.transformer import Transformer


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
        tgt_padding_mask = batch["tgt_mask"]

        src_mask_expanded = src_mask.unsqueeze(1) & src_mask.unsqueeze(2)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        tgt_len = tgt_input.size(1)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        tgt_padding = tgt_padding_mask[:, :-1]
        tgt_input_mask = (
            causal_mask.unsqueeze(0)
            & tgt_padding.unsqueeze(1)
            & tgt_padding.unsqueeze(2)
        )

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
