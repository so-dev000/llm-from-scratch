import matplotlib.pyplot as plt

from component.positional_encoding import PositionalEncoding

pe_layer = PositionalEncoding(model_dim=512, max_len=64)
pe = pe_layer.pe.squeeze(0).numpy()

plt.figure(figsize=(10, 6))
plt.imshow(pe, cmap="RdBu_r", aspect="auto", interpolation="bicubic")
plt.xlabel("Dimension")
plt.ylabel("Position")
plt.colorbar()
plt.tight_layout()
plt.savefig("image/positional_encoding.png", dpi=150, bbox_inches="tight")
plt.close()
