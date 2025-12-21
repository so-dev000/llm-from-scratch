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
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_validation_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss_epoch")
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if train_loss is not None and val_loss is not None and val_loss != 0:
            loss_ratio = train_loss / val_loss
            self.log("train_val_loss_ratio", loss_ratio, prog_bar=False, logger=True)

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
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config.data.pad_idx,
            label_smoothing=config.training.label_smoothing,
        )

    def forward(self, tokens, mask=None):
        return self.model(tokens, mask)

    def _shared_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        mask = batch["mask"]
        input_tokens = input_ids[:, :-1]
        target_tokens = input_ids[:, 1:]
        input_mask = mask[:, :-1]
        batch_size, seq_len = input_tokens.shape
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=input_tokens.device)
        ).bool()
        combined_mask = (
            causal_mask.unsqueeze(0) & input_mask.unsqueeze(1) & input_mask.unsqueeze(2)
        )
        logits = self(input_tokens, combined_mask)
        logits_flat = logits.reshape(-1, logits.size(-1))
        target_flat = target_tokens.reshape(-1)
        loss = self.criterion(logits_flat, target_flat)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_validation_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss_epoch")
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if train_loss is not None and val_loss is not None and val_loss != 0:
            loss_ratio = train_loss / val_loss
            self.log("train_val_loss_ratio", loss_ratio, prog_bar=False, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids = batch["input_ids"]
        max_gen_len = self.config.inference.max_gen_len
        generated = input_ids[:, :1]
        for _ in range(max_gen_len - 1):
            seq_len = generated.size(1)
            mask = torch.tril(
                torch.ones(seq_len, seq_len, device=generated.device)
            ).bool()
            mask = mask.unsqueeze(0).expand(generated.size(0), -1, -1)
            logits = self(generated, mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == self.config.inference.eos_idx).all():
                break
        return generated

    def configure_optimizers(self):
        # Separate parameters for weight decay
        decay = set()
        no_decay = set()

        for name, param in self.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ["bias", "LayerNorm", "norm"]):
                    no_decay.add(name)
                else:
                    decay.add(name)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": 0.01},
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.optimizer.initial_lr,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            eps=self.config.optimizer.adam_epsilon,
        )

        # GPT-2: Cosine annealing scheduler
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.config.optimizer.warmup_steps

        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265359)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
