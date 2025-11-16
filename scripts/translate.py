import pickle

import torch

from model.transformer import Transformer
from utils.masking import combine_masks, create_causal_mask

# Hyperparameters (must match training)
MAX_LENGTH = 128
MODEL_DIM = 512
ENCODER_LAYERS = 6
DECODER_LAYERS = 6
PAD_IDX = 0
BOS_IDX = 2
EOS_IDX = 3

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def translate_sentence(
    model,
    sentence,
    en_tokenizer,
    ja_tokenizer,
    max_length=MAX_LENGTH,
    device=device,
):
    model.eval()

    # Tokenize and prepare input
    src_ids = en_tokenizer.encode(sentence, add_special_tokens=True)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(
        0
    )  # Add batch dimension

    # Create src_masks
    src_padding_mask = src_tensor != PAD_IDX
    encoder_src_mask = src_padding_mask.unsqueeze(1) & src_padding_mask.unsqueeze(2)

    # Initialize target sequence with BOS token
    tgt_tensor = torch.tensor([[BOS_IDX]], dtype=torch.long, device=device)

    for _ in range(max_length):
        # Create tgt_mask (causal mask + padding mask)
        tgt_len = tgt_tensor.size(1)
        causal_mask = create_causal_mask(tgt_len, device=device)
        tgt_padding_mask = tgt_tensor != PAD_IDX
        tgt_combined_mask = combine_masks(tgt_padding_mask, causal_mask)
        tgt_input_mask = tgt_combined_mask

        # Forward pass
        with torch.no_grad():
            output = model(
                src_tensor,
                tgt_tensor,
                encoder_src_mask=encoder_src_mask,
                decoder_src_mask=src_padding_mask,
                tgt_mask=tgt_input_mask,
            )

        # Get the next token prediction
        pred_token_id = output.argmax(dim=-1)[:, -1].item()

        # Append to target sequence
        tgt_tensor = torch.cat(
            [
                tgt_tensor,
                torch.tensor([[pred_token_id]], dtype=torch.long, device=device),
            ],
            dim=1,
        )

        # If EOS token is predicted, stop
        if pred_token_id == EOS_IDX:
            break

    # Decode the generated sequence
    translated_text = ja_tokenizer.decode(
        tgt_tensor.squeeze(0).tolist(), skip_special_tokens=True
    )
    return translated_text


def main():
    # Load tokenizers
    print("Loading tokenizers...")
    with open("checkpoints/tokenizers/bsd_ja_en/en_bpe.pkl", "rb") as f:
        en_tokenizer = pickle.load(f)

    with open("checkpoints/tokenizers/bsd_ja_en/ja_bpe.pkl", "rb") as f:
        ja_tokenizer = pickle.load(f)

    # Use max vocab size
    vocab_size = max(len(en_tokenizer.vocab), len(ja_tokenizer.vocab))

    # Initialize model
    print(f"Initializing model on {device}...")
    model = Transformer(
        vocab_size=vocab_size,
        model_dim=MODEL_DIM,
        encoder_num=ENCODER_LAYERS,
        decoder_num=DECODER_LAYERS,
    ).to(device)

    # Load trained model weights
    print("Loading model weights...")
    model_path = "checkpoints/models/best_model.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        print("Please train the model first using 'python -m scripts.train'")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Model loaded. Enter English sentences to translate (type 'exit' to quit):")

    while True:
        english_sentence = input("English: ")
        if english_sentence.lower() == "exit":
            break

        japanese_translation = translate_sentence(
            model, english_sentence, en_tokenizer, ja_tokenizer, device=device
        )
        print(f"Japanese: {japanese_translation}")


if __name__ == "__main__":
    import os

    main()
