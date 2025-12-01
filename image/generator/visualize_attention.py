import matplotlib.pyplot as plt
import torch

from model.transformer import Transformer
from tokenizer.bpe import BPE
from utils.inference import PAD_IDX, get_device, translate_sentence_beam
from utils.masking import combine_masks, create_causal_mask

plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "Yu Gothic", "Arial Unicode MS"]
SENTENCE = input("EN: ").strip()

device = get_device()

# Load model
checkpoint = torch.load(
    "checkpoints/runs/20251123_153359/best_model.pt",
    map_location=device,
    weights_only=False,
)
model = Transformer(
    checkpoint["src_vocab_size"],
    checkpoint["tgt_vocab_size"],
    checkpoint["model_dim"],
    checkpoint["encoder_layers"],
    checkpoint["decoder_layers"],
    PAD_IDX,
).to(device)

state_dict = checkpoint["model_state_dict"]
if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
    state_dict = {
        key.replace("_orig_mod.", ""): value for key, value in state_dict.items()
    }
model.load_state_dict(state_dict)
model.eval()

# Load tokenizers and translate
en_tokenizer = BPE.load("data/tokenizers/bsd_en_ja/en_bpe.pkl")
ja_tokenizer = BPE.load("data/tokenizers/bsd_en_ja/ja_bpe.pkl")

translation = translate_sentence_beam(model, SENTENCE, en_tokenizer, ja_tokenizer)

print(f"JA: {translation}")

# Encode and get encoder output
src_ids = en_tokenizer.encode(SENTENCE, add_special_tokens=True)
tgt_ids = ja_tokenizer.encode(translation, add_special_tokens=True)
src = torch.tensor(src_ids, device=device).unsqueeze(0)
src_mask = src != PAD_IDX

with torch.no_grad():
    enc_out = model.encoder(
        model.positional_encoding(model.src_embedding(src)),
        src_mask.unsqueeze(1) & src_mask.unsqueeze(2),
    )

# Force-decode and capture attention
attention_list = []
for i in range(1, len(tgt_ids)):
    tgt = torch.tensor([tgt_ids[:i]], device=device).long()
    tgt_emb = model.positional_encoding(model.tgt_embedding(tgt))
    tgt_mask = combine_masks(tgt != PAD_IDX, create_causal_mask(i, device=device))

    dec_in, layer_attns = tgt_emb, []
    with torch.no_grad():
        for layer in model.decoder.decoders:
            norm1 = layer.normalizer_1(
                dec_in + layer.masked_attention(dec_in, mask=tgt_mask)
            )
            cross_attn_out = layer.attention(norm1, enc_out, mask=src_mask)
            layer_attns.append(layer.attention.last_attention_weights[0].mean(0)[-1])
            norm2 = layer.normalizer_2(norm1 + cross_attn_out)
            dec_in = layer.normalizer_3(norm2 + layer.feed_forward(norm2))
    attention_list.append(torch.stack(layer_attns).mean(0).cpu())


# Visualize
def ids2tok(tok, ids):
    return [tok.vocab[i].decode("utf-8", errors="replace") for i in ids]


src_tokens, tgt_tokens = (
    ids2tok(en_tokenizer, src_ids),
    ids2tok(ja_tokenizer, tgt_ids[1:]),
)

plt.figure(figsize=(12, 8))
plt.imshow(
    torch.stack(attention_list).numpy(),
    cmap="RdBu_r",
    aspect="auto",
    interpolation="nearest",
)
plt.colorbar(label="Attention")
plt.xlabel("English")
plt.ylabel("Japanese")
plt.title("Cross-Attention")
plt.xticks(range(len(src_tokens)), src_tokens, rotation=45, ha="right")
plt.yticks(range(len(tgt_tokens)), tgt_tokens)
plt.tight_layout()
plt.savefig("image/attention_en_ja.png", dpi=150)
plt.close()
