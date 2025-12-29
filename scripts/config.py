from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DataConfig:
    dataset_name: str
    batch_size: int
    max_length: int
    num_workers: int
    pad_idx: int
    vocab_size: int = 8000
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    src_lang: str = "en"
    tgt_lang: str = "ja"
    src_column: str = "en_sentence"
    tgt_column: str = "ja_sentence"
    text_column: str = "text"
    dataset_config: str = None
    tokenizer_train_samples: int = None
    val_split_size: float = 0.05
    preprocess_batch_size: int = 10000
    preprocess_num_proc: int = 2


@dataclass
class OptimizerConfig:
    warmup_steps: int
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    initial_lr: float = 1.0
    optimizer_type: str = "Adam"
    scheduler_type: str = "inverse_sqrt"


@dataclass
class TransformerModelConfig:
    model_type: Literal["transformer"] = "transformer"
    src_vocab_size: int = None
    tgt_vocab_size: int = None
    model_dim: int = 256
    encoder_layers: int = 4
    decoder_layers: int = 4
    num_heads: int = 8
    feedforward_dim: int = None
    dropout: float = 0.1
    activation: str = "relu"
    max_seq_len: int = 5000
    padding_idx: int = 0


@dataclass
class GPTModelConfig:
    model_type: Literal["gpt"] = "gpt"
    vocab_size: int = None
    model_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    feedforward_dim: int = None
    max_seq_len: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    padding_idx: int = 0


@dataclass
class LlamaModelConfig:
    model_type: Literal["llama"] = "llama"
    vocab_size: int = None
    model_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    feedforward_dim: int = None
    max_seq_len: int = 2048
    dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    padding_idx: int = 0
    use_gradient_checkpointing: bool = False


@dataclass
class TrainingConfig:
    num_epochs: int
    label_smoothing: float
    early_stopping_patience: int
    gradient_clip_val: float = 1.0
    precision: str = "32-true"
    val_check_interval: float = 1.0
    accumulate_grad_batches: int = 1


@dataclass
class InferenceConfig:
    pad_idx: int = 0
    unk_idx: int = 1
    bos_idx: int = 2
    eos_idx: int = 3

    beam_size: int = 5
    length_penalty: float = 0.6
    max_output_offset: int = 10

    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.2

    max_gen_len: int = 100


@dataclass
class ModalConfig:
    timeout_hours: int = 12
    volume_name: str = "llm-from-scratch"
    secret_name: str = "wandb-secret"


@dataclass
class WandbConfig:
    project: str = "llm-from-scratch"
    entity: str = None
    enabled: bool = True
    log_model: bool = False


@dataclass
class Config:
    data: DataConfig
    optimizer: OptimizerConfig
    model: TransformerModelConfig | GPTModelConfig | LlamaModelConfig
    training: TrainingConfig
    inference: InferenceConfig

    modal: ModalConfig = field(default_factory=ModalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    run_name: str = None
    checkpoint_dir: str = "/vol/runs"
    tokenizer_dir: str = "/vol/tokenizers"

    def _apply_overrides(self, overrides: dict) -> None:
        for key, value in overrides.items():
            if "." in key:
                parts = key.split(".")
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(self, key, value)

    @classmethod
    def for_transformer(cls, **overrides) -> "Config":
        data_config = DataConfig(
            dataset_name="ryo0634/bsd_ja_en",
            batch_size=256,
            max_length=64,
            num_workers=8,
            pad_idx=0,
            val_split_size=0.15,
            preprocess_batch_size=1000,
            preprocess_num_proc=8,
        )

        optimizer_config = OptimizerConfig(
            warmup_steps=4000,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
        )

        model_config = TransformerModelConfig(
            model_dim=256,
            encoder_layers=4,
            decoder_layers=4,
            num_heads=8,
            feedforward_dim=1024,
            dropout=0.1,
            activation="relu",
            max_seq_len=5000,
        )

        training_config = TrainingConfig(
            num_epochs=50,
            label_smoothing=0.1,
            early_stopping_patience=5,
            gradient_clip_val=1.0,
            precision="32-true",
        )

        inference_config = InferenceConfig(
            pad_idx=0,
            unk_idx=1,
            bos_idx=2,
            eos_idx=3,
            beam_size=5,
            length_penalty=0.6,
            max_output_offset=10,
        )

        modal_config = ModalConfig()
        wandb_config = WandbConfig()

        config = cls(
            data=data_config,
            optimizer=optimizer_config,
            model=model_config,
            training=training_config,
            inference=inference_config,
            modal=modal_config,
            wandb=wandb_config,
        )

        config._apply_overrides(overrides)
        return config

    @classmethod
    def for_gpt(cls, **overrides) -> "Config":
        data_config = DataConfig(
            dataset_name="HuggingFaceFW/fineweb-edu",
            batch_size=72,
            max_length=512,
            num_workers=8,
            pad_idx=0,
            vocab_size=50257,
            text_column="text",
            dataset_config="sample-10BT",
            prefetch_factor=4,
            tokenizer_train_samples=100_000,
        )

        optimizer_config = OptimizerConfig(
            warmup_steps=2000,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            optimizer_type="AdamW",
            initial_lr=2.5e-4,
        )

        model_config = GPTModelConfig(
            model_dim=768,
            num_layers=12,
            num_heads=12,
            feedforward_dim=3072,
            max_seq_len=1024,
            dropout=0.1,
            activation="gelu",
        )

        training_config = TrainingConfig(
            num_epochs=3,
            label_smoothing=0.0,
            early_stopping_patience=2,
            gradient_clip_val=1.0,
            precision="bf16-true",
        )

        inference_config = InferenceConfig(
            pad_idx=0,
            unk_idx=1,
            bos_idx=2,
            eos_idx=3,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            max_gen_len=100,
        )

        modal_config = ModalConfig()
        wandb_config = WandbConfig()

        config = cls(
            data=data_config,
            optimizer=optimizer_config,
            model=model_config,
            training=training_config,
            inference=inference_config,
            modal=modal_config,
            wandb=wandb_config,
        )

        config._apply_overrides(overrides)
        return config

    @classmethod
    def for_llama(cls, **overrides) -> "Config":
        data_config = DataConfig(
            dataset_name="HuggingFaceFW/fineweb-edu",
            batch_size=16,
            max_length=512,
            num_workers=4,
            pad_idx=0,
            vocab_size=32000,
            text_column="text",
            dataset_config="sample-10BT",
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
            tokenizer_train_samples=2_500_000,
        )

        optimizer_config = OptimizerConfig(
            warmup_steps=2000,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8,
            optimizer_type="AdamW",
            initial_lr=3e-4,
        )

        model_config = LlamaModelConfig(
            model_dim=512,
            num_layers=8,
            num_heads=8,
            num_kv_heads=4,
            feedforward_dim=2048,
            max_seq_len=512,
            dropout=0.1,
            norm_eps=1e-5,
            rope_theta=10000.0,
            use_gradient_checkpointing=False,
        )

        training_config = TrainingConfig(
            num_epochs=3,
            label_smoothing=0.0,
            early_stopping_patience=2,
            gradient_clip_val=1.0,
            precision="bf16-mixed",
            accumulate_grad_batches=2,
        )

        inference_config = InferenceConfig(
            pad_idx=0,
            unk_idx=1,
            bos_idx=2,
            eos_idx=3,
            temperature=0.8,
            top_k=20,
            top_p=0.85,
            repetition_penalty=1.5,
            max_gen_len=100,
        )

        modal_config = ModalConfig()
        wandb_config = WandbConfig()

        config = cls(
            data=data_config,
            optimizer=optimizer_config,
            model=model_config,
            training=training_config,
            inference=inference_config,
            modal=modal_config,
            wandb=wandb_config,
        )

        config._apply_overrides(overrides)
        return config

    def validate(self) -> None:
        if self.model.vocab_size is None:
            self.model.vocab_size = self.data.vocab_size

        if self.model.num_heads > 0:
            if self.model.model_dim % self.model.num_heads != 0:
                raise ValueError(
                    f"model_dim ({self.model.model_dim}) must be divisible by "
                    f"num_heads ({self.model.num_heads})"
                )

        if isinstance(self.model, TransformerModelConfig):
            if self.model.feedforward_dim is None:
                self.model.feedforward_dim = 4 * self.model.model_dim
        elif isinstance(self.model, GPTModelConfig):
            if self.model.feedforward_dim is None:
                self.model.feedforward_dim = 4 * self.model.model_dim
        elif isinstance(self.model, LlamaModelConfig):
            if self.model.feedforward_dim is None:
                self.model.feedforward_dim = int(8 * self.model.model_dim / 3)
                self.model.feedforward_dim = (
                    (self.model.feedforward_dim + 255) // 256
                ) * 256

        if self.data.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.data.batch_size}")

        if self.optimizer.warmup_steps <= 0:
            raise ValueError(
                f"warmup_steps must be positive, got {self.optimizer.warmup_steps}"
            )

    def to_dict(self) -> dict:
        result = {}

        for attr_name in ["data", "optimizer", "model", "training", "inference"]:
            attr = getattr(self, attr_name)
            for field_name in attr.__dataclass_fields__:
                value = getattr(attr, field_name)
                result[f"{attr_name}.{field_name}"] = value

        result["run_name"] = self.run_name
        result["checkpoint_dir"] = self.checkpoint_dir

        return result
