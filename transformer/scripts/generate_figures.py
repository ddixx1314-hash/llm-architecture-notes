"""Generate visualization figures for the Transformer learning notes.

Outputs (written to ../images/):
  - rnn-vs-transformer.png:   Section 0 — RNN vs Transformer dependency graph
  - positional-encoding-heatmap.png: Section 3 — sin/cos PE heatmap

Run from anywhere; output paths are resolved relative to this file.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "images"
OUT.mkdir(exist_ok=True)


def figure_rnn_vs_transformer():
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.5))
    tokens = ["x1", "x2", "x3", "x4", "x5"]
    n = len(tokens)
    xs = np.arange(n)

    # ---------- Left: RNN (serial) ----------
    ax_l.set_title("RNN: serial along sequence", fontsize=13)
    for i, t in enumerate(tokens):
        ax_l.scatter(i, 0, s=900, c="#bcd0f5", edgecolors="#2b5fb8", zorder=3)
        ax_l.text(i, 0, t, ha="center", va="center", fontsize=10, zorder=4)
        ax_l.scatter(i, 1, s=900, c="#f5d6bc", edgecolors="#b8762b", zorder=3)
        ax_l.text(i, 1, f"h{i+1}", ha="center", va="center", fontsize=10, zorder=4)
        # vertical: x_t -> h_t
        ax_l.annotate("", xy=(i, 0.85), xytext=(i, 0.15),
                      arrowprops=dict(arrowstyle="->", color="#666"))
    # horizontal: h_t -> h_{t+1}
    for i in range(n - 1):
        ax_l.annotate("", xy=(i + 0.85, 1), xytext=(i + 0.15, 1),
                      arrowprops=dict(arrowstyle="->", color="#b8762b", lw=2))
    ax_l.text(n / 2 - 0.5, 1.55, "must compute h1 → h2 → ... → h5 in order",
              ha="center", fontsize=10, color="#b8762b", style="italic")
    ax_l.set_xlim(-0.6, n - 0.4)
    ax_l.set_ylim(-0.6, 1.8)
    ax_l.set_xticks([])
    ax_l.set_yticks([0, 1])
    ax_l.set_yticklabels(["input x", "hidden h"], fontsize=10)
    ax_l.spines[:].set_visible(False)

    # ---------- Right: Transformer (parallel, all-to-all) ----------
    ax_r.set_title("Transformer: parallel, every position sees every other", fontsize=13)
    for i, t in enumerate(tokens):
        ax_r.scatter(i, 0, s=900, c="#bcd0f5", edgecolors="#2b5fb8", zorder=3)
        ax_r.text(i, 0, t, ha="center", va="center", fontsize=10, zorder=4)
        ax_r.scatter(i, 1, s=900, c="#c5e8c5", edgecolors="#2a7a2a", zorder=3)
        ax_r.text(i, 1, f"y{i+1}", ha="center", va="center", fontsize=10, zorder=4)
    # all-to-all edges from x_j to y_i
    for i in range(n):
        for j in range(n):
            ax_r.plot([j, i], [0.1, 0.9], color="#2a7a2a", alpha=0.18, lw=1, zorder=1)
    ax_r.text(n / 2 - 0.5, 1.55, "all y_i computed in parallel (one big matmul)",
              ha="center", fontsize=10, color="#2a7a2a", style="italic")
    ax_r.set_xlim(-0.6, n - 0.4)
    ax_r.set_ylim(-0.6, 1.8)
    ax_r.set_xticks([])
    ax_r.set_yticks([0, 1])
    ax_r.set_yticklabels(["input x", "output y"], fontsize=10)
    ax_r.spines[:].set_visible(False)

    fig.tight_layout()
    out_path = OUT / "rnn-vs-transformer.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def figure_positional_encoding_heatmap():
    d_model = 128
    max_len = 100

    pos = np.arange(max_len)[:, None]              # (max_len, 1)
    i = np.arange(d_model)[None, :]                # (1, d_model)
    div_term = np.exp(-(np.log(10000.0) * (i // 2 * 2) / d_model))
    angles = pos * div_term                        # (max_len, d_model)

    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 2]}
    )

    # Top: full heatmap
    im = ax_top.imshow(pe.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax_top.set_xlabel("position (pos)", fontsize=11)
    ax_top.set_ylabel("dimension index (i)", fontsize=11)
    ax_top.set_title("sin/cos positional encoding (d_model=128, max_len=100)",
                     fontsize=13)
    fig.colorbar(im, ax=ax_top, label="PE value")

    # Bottom: a few selected dimensions to show frequency
    for dim, color in [(0, "#d62728"), (4, "#ff7f0e"), (20, "#2ca02c"),
                       (60, "#1f77b4"), (120, "#9467bd")]:
        ax_bot.plot(pe[:, dim], label=f"dim {dim}", color=color, lw=1.6)
    ax_bot.set_xlabel("position (pos)", fontsize=11)
    ax_bot.set_ylabel("PE value", fontsize=11)
    ax_bot.set_title("Low dims oscillate fast; high dims oscillate slow",
                     fontsize=12)
    ax_bot.legend(loc="upper right", fontsize=9, ncol=5)
    ax_bot.grid(alpha=0.3)
    ax_bot.set_xlim(0, max_len - 1)

    fig.tight_layout()
    out_path = OUT / "positional-encoding-heatmap.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    figure_rnn_vs_transformer()
    figure_positional_encoding_heatmap()
