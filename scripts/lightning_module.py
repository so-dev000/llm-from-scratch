import pytorch_lightning as L
import torch
from torch import nn

from model.gpt import GPT
from model.transformer import Transformer
from utils.decoding_strategy import BeamSearch, GreedyDecoding


def get_optimizer_and_scheduler(module, config):
    optim_cls = getattr(torch.optim, config.optimizer.optimizer_type)
    optimizer = optim_cls(
        module.parameters(),
        lr=config.optimizer.initial_lr,
        betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
        eps=config.optimizer.adam_epsilon,
    )

    scheduler = None
    if config.optimizer.scheduler_type == "inverse_sqrt":

        def lr_lambda(step):
            step = max(step, 1)
            model_dim = config.model.model_dim
            warmup_steps = config.optimizer.warmup_steps
            return (model_dim**-0.5) * min(step**-0.5, step * warmup_steps**-1.5)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if scheduler:
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    return optimizer


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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        src_tokens = batch["src"]
        src_mask = batch["src_mask"]

        if self.config.inference.beam_size > 1:
            decoder = BeamSearch(self.config.inference)
        else:
            decoder = GreedyDecoding(self.config.inference)

        return decoder.decode(
            self.model, src_tokens, src_mask, self.config.inference.max_gen_len
        )

    def configure_optimizers(self):
        return get_optimizer_and_scheduler(self, self.config)


class GPTLightningModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config.to_dict())
        self.config = config
        self.model = GPT(config.model)

    def forward(self, tokens, mask=None):
        raise NotImplementedError("GPTLightningModule not yet implemented")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("GPTLightningModule not yet implemented")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("GPTLightningModule not yet implemented")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError("GPTLightningModule not yet implemented")

    def configure_optimizers(self):
        raise NotImplementedError("GPTLightningModule not yet implemented")
