import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from component.token_embedding import TokenEmbedding

plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "Yu Gothic", "Arial Unicode MS"]

vocab_size, dim = 5, 7
words = ["cat", "dog", "apple", "book", "car"]
sel = 1

emb = TokenEmbedding(vocab_size, dim)
emb_matrix = emb.embedding.weight.data
sel_vec = emb_matrix[sel].unsqueeze(1)
scaled_vec = sel_vec * (dim**0.5)

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(14, 5), gridspec_kw={"width_ratios": [6, 1, 1]}
)


def show_matrix(ax, data, vmin, vmax):
    ax.imshow(data, cmap="coolwarm", aspect="auto", vmin=vmin, vmax=vmax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.axis("off")


# Matrix
show_matrix(ax1, emb_matrix, -2, 2)
ax1.add_patch(
    Rectangle((-0.5, sel - 0.5), dim, 1, fill=False, edgecolor="red", linewidth=2)
)
for i, w in enumerate(words):
    ax1.text(
        -0.8,
        i,
        w,
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="red" if i == sel else "black",
    )
ax1.set_xlim(-1.5, dim - 0.5)
ax1.set_ylim(vocab_size - 0.5, -1.2)

# Selected
show_matrix(ax2, sel_vec, -2, 2)
ax2.text(0, -0.6, words[sel], ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_xlim(-0.8, 0.5)
ax2.set_ylim(dim - 0.5, -1.2)

# Scaled
show_matrix(ax3, scaled_vec, -2 * (dim**0.5), 2 * (dim**0.5))
ax3.text(0, -0.6, words[sel], ha="center", va="bottom", fontsize=11, fontweight="bold")
ax3.set_xlim(-0.8, 0.5)
ax3.set_ylim(dim - 0.5, -1.2)

plt.subplots_adjust(wspace=0.3)

# Get subplot positions
pos1 = ax1.get_position()
pos2 = ax2.get_position()
pos3 = ax3.get_position()

# Add arrows and labels in gaps between subplots
# Arrow 1: between ax1 and ax2
arrow1_start_x = pos1.x1 + 0.02
arrow1_end_x = pos2.x0 - 0.005
arrow1_y = pos1.y0 + (pos1.y1 - pos1.y0) * 0.35
arrow1_mid_x = (arrow1_start_x + arrow1_end_x) / 2

arrow1 = FancyArrowPatch(
    (arrow1_start_x, arrow1_y),
    (arrow1_end_x, arrow1_y),
    transform=fig.transFigure,
    arrowstyle="->",
    mutation_scale=20,
    lw=2.5,
    color="black",
)
fig.patches.append(arrow1)
fig.text(
    arrow1_mid_x,
    arrow1_y + 0.03,
    "選択",
    ha="center",
    va="bottom",
    fontsize=11,
    fontweight="bold",
)

# Arrow 2: between ax2 and ax3
arrow2_start_x = pos2.x1 + 0.02
arrow2_end_x = pos3.x0 - 0.005
arrow2_y = pos2.y0 + (pos2.y1 - pos2.y0) * 0.35
arrow2_mid_x = (arrow2_start_x + arrow2_end_x) / 2

arrow2 = FancyArrowPatch(
    (arrow2_start_x, arrow2_y),
    (arrow2_end_x, arrow2_y),
    transform=fig.transFigure,
    arrowstyle="->",
    mutation_scale=20,
    lw=2.5,
    color="black",
)
fig.patches.append(arrow2)
fig.text(
    arrow2_mid_x,
    arrow2_y + 0.03,
    "スケーリング",
    ha="center",
    va="bottom",
    fontsize=11,
    fontweight="bold",
)

plt.savefig("image/token_embedding.png", dpi=300, bbox_inches="tight")
plt.close()
