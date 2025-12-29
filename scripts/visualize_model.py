import sys
from pathlib import Path

import torch
from torchinfo import summary

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gpt import GPT
from model.llama import Llama
from model.transformer import Transformer
from scripts.config import Config


def visualize_model(model_type: str):
    if model_type == "llama":
        config = Config.for_llama()
        config.validate()
    elif model_type == "gpt":
        config = Config.for_gpt()
        config.validate()
    elif model_type == "transformer":
        config = Config.for_transformer()
        config.model.src_vocab_size = config.data.vocab_size
        config.model.tgt_vocab_size = config.data.vocab_size
        if config.model.feedforward_dim is None:
            config.model.feedforward_dim = 4 * config.model.model_dim
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type == "llama":
        model = Llama(config.model)
    elif model_type == "gpt":
        model = GPT(config.model)
    else:
        model = Transformer(config.model)

    if model_type == "transformer":
        input_data = (
            torch.randint(0, config.model.src_vocab_size, (2, 32), dtype=torch.long),
            torch.randint(0, config.model.tgt_vocab_size, (2, 32), dtype=torch.long),
        )
    else:
        input_data = (
            torch.randint(0, config.model.vocab_size, (2, 32), dtype=torch.long),
        )

    print("\n" + "=" * 80)
    print(f"  {model_type.upper()} MODEL ARCHITECTURE")
    print("=" * 80 + "\n")

    summary(
        model,
        input_data=input_data,
        depth=5,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "params_percent",
            "trainable",
        ],
        row_settings=["var_names"],
    )

    print("\n" + "=" * 80)
    print("  MODEL CONFIGURATION")
    print("=" * 80 + "\n")

    print(f"Model Type: {model_type.upper()}\n")

    if model_type == "transformer":
        print(f"  Model Dimension:      {config.model.model_dim}")
        print(f"  Encoder Layers:       {config.model.encoder_layers}")
        print(f"  Decoder Layers:       {config.model.decoder_layers}")
        print(f"  Attention Heads:      {config.model.num_heads}")
        print(f"  Feedforward Dim:      {config.model.feedforward_dim}")
        print(f"  Source Vocab Size:    {config.model.src_vocab_size:,}")
        print(f"  Target Vocab Size:    {config.model.tgt_vocab_size:,}")
    else:
        print(f"  Model Dimension:      {config.model.model_dim}")
        print(f"  Layers:               {config.model.num_layers}")
        print(f"  Attention Heads:      {config.model.num_heads}")
        print(f"  Feedforward Dim:      {config.model.feedforward_dim}")
        print(f"  Vocabulary Size:      {config.model.vocab_size:,}")
        print(f"  Max Sequence Length:  {config.model.max_seq_len:,}")
        if model_type == "llama":
            print(f"  KV Heads (GQA):       {config.model.num_kv_heads}")
            print(f"  RoPE Theta:           {config.model.rope_theta:,}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/visualize_model.py [llama|gpt|transformer]")
        sys.exit(1)

    model_type = sys.argv[1].lower()
    if model_type not in ["llama", "gpt", "transformer"]:
        print(f"Error: Invalid model type '{model_type}'")
        print("Available models: llama, gpt, transformer")
        sys.exit(1)

    visualize_model(model_type)
