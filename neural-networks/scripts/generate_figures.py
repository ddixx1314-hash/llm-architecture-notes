"""Generate visualization figures for the foundational neural-network notes.

Outputs are written to ../images/.

Run:
    python neural-networks/scripts/generate_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "images"
OUT.mkdir(exist_ok=True)


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


def box(ax, xy, w, h, text, fc="#e8f0fe", ec="#3454d1", fs=10):
    rect = patches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)
    return rect


def arrow(ax, start, end, color="#555", lw=1.7, style="->"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=4, shrinkB=4),
    )


def figure_training_loop():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    clean(ax)

    items = [
        ((0.4, 1.8), "batch\n(x, y)", "#d8e8ff"),
        ((2.2, 1.8), "forward\nlogits = f(x)", "#dff3df"),
        ((4.2, 1.8), "loss\nL(logits, y)", "#fff1c7"),
        ((6.2, 1.8), "backward\ncompute grads", "#ffe0d6"),
        ((8.3, 1.8), "optimizer.step\nupdate params", "#eadcff"),
    ]
    for xy, text, fc in items:
        box(ax, xy, 1.45, 0.85, text, fc=fc, ec="#555")

    centers = [(x + 1.45, y + 0.42) for (x, y), _, _ in items]
    for i in range(len(centers) - 1):
        arrow(ax, centers[i], (items[i + 1][0][0], items[i + 1][0][1] + 0.42))

    arrow(ax, (9.0, 1.75), (2.8, 1.65), color="#8a4fd3", lw=1.5, style="-|>")
    ax.text(5.9, 1.35, "next mini-batch repeats the same loop", ha="center", fontsize=10, color="#6a38ad")
    ax.text(5.2, 3.0, "A training step is a closed feedback loop", ha="center", fontsize=14, weight="bold")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(1.0, 3.3)

    out = OUT / "training-loop.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_cnn_convolution():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.1, 0.9, 1.1]})
    ax0, ax1, ax2 = axes
    for ax in axes:
        clean(ax)

    img = np.zeros((7, 7))
    img[1:6, 3] = 1
    img[2:5, 2:5] += 0.25
    ax0.imshow(img, cmap="Blues", vmin=0, vmax=1.25)
    ax0.set_title("input image", fontsize=12)
    rect = patches.Rectangle((2 - 0.5, 2 - 0.5), 3, 3, fill=False, edgecolor="#d62728", linewidth=2.5)
    ax0.add_patch(rect)
    for i in range(8):
        ax0.axhline(i - 0.5, color="white", lw=0.8)
        ax0.axvline(i - 0.5, color="white", lw=0.8)

    kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    ax1.imshow(kernel, cmap="Oranges", vmin=0, vmax=1)
    ax1.set_title("3x3 kernel", fontsize=12)
    for i in range(4):
        ax1.axhline(i - 0.5, color="white", lw=0.8)
        ax1.axvline(i - 0.5, color="white", lw=0.8)

    feature = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            feature[i, j] = (img[i : i + 3, j : j + 3] * kernel).sum()
    ax2.imshow(feature, cmap="Greens")
    ax2.set_title("feature map", fontsize=12)
    for i in range(6):
        ax2.axhline(i - 0.5, color="white", lw=0.8)
        ax2.axvline(i - 0.5, color="white", lw=0.8)

    fig.text(0.5, 0.02, "The same small kernel slides across the image: local connection + weight sharing", ha="center", fontsize=11)
    out = OUT / "cnn-convolution-overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_rnn_unroll():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    clean(ax)
    n = 5
    for i in range(n):
        x = 1 + i * 1.8
        box(ax, (x, 0.4), 0.9, 0.55, f"x{i+1}", fc="#d8e8ff", ec="#2b5fb8")
        box(ax, (x, 1.55), 0.9, 0.65, f"h{i+1}", fc="#ffe4c8", ec="#b8762b")
        arrow(ax, (x + 0.45, 0.95), (x + 0.45, 1.55), color="#666")
        if i > 0:
            arrow(ax, (x - 0.9, 1.88), (x, 1.88), color="#b8762b", lw=2.2)
    ax.text(4.6, 2.75, "RNN unrolled through time", ha="center", fontsize=14, weight="bold")
    ax.text(4.6, 2.45, "h_t depends on h_{t-1}, so the sequence dimension is serial", ha="center", fontsize=10, color="#8a4d1f")
    ax.set_xlim(0.4, 9.6)
    ax.set_ylim(0.1, 3.1)
    out = OUT / "rnn-unroll.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_lstm_gates():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    clean(ax)
    box(ax, (0.4, 2.0), 1.0, 0.55, "c_{t-1}", fc="#fff1c7", ec="#b88700")
    box(ax, (10.4, 2.0), 1.0, 0.55, "c_t", fc="#fff1c7", ec="#b88700")
    arrow(ax, (1.4, 2.28), (10.4, 2.28), color="#b88700", lw=2.5)
    ax.text(5.9, 2.65, "cell state: additive memory highway", ha="center", fontsize=11, color="#8c6500")

    gate_info = [
        (2.0, "forget gate\nf_t", "#ffe0d6"),
        (4.1, "input gate\ni_t", "#dff3df"),
        (6.2, "candidate\n~c_t", "#d8e8ff"),
        (8.3, "output gate\no_t", "#eadcff"),
    ]
    for x, text, fc in gate_info:
        box(ax, (x, 0.8), 1.35, 0.75, text, fc=fc, ec="#555")
        arrow(ax, (x + 0.68, 1.55), (x + 0.68, 2.03), color="#555")

    box(ax, (4.7, 0.05), 1.9, 0.45, "[x_t ; h_{t-1}]", fc="#f0f0f0", ec="#777")
    for x, _, _ in gate_info:
        arrow(ax, (5.65, 0.5), (x + 0.68, 0.8), color="#777", lw=1.2)
    box(ax, (10.4, 0.8), 1.0, 0.55, "h_t", fc="#cfe5cf", ec="#2a7a2a")
    arrow(ax, (9.0, 1.55), (10.35, 1.08), color="#6a38ad", lw=1.7)
    arrow(ax, (10.9, 2.0), (10.9, 1.35), color="#2a7a2a", lw=1.7)
    ax.text(6.0, 3.35, "LSTM gates control what to forget, write, and expose", ha="center", fontsize=14, weight="bold")
    ax.set_xlim(0.1, 11.8)
    ax.set_ylim(-0.1, 3.7)
    out = OUT / "lstm-gates.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_attention_bridge():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    clean(ax)
    enc_x = [1.2, 2.4, 3.6, 4.8]
    for i, x in enumerate(enc_x, 1):
        box(ax, (x, 0.55), 0.75, 0.5, f"x{i}", fc="#d8e8ff", ec="#2b5fb8")
        box(ax, (x, 1.55), 0.75, 0.58, f"h{i}", fc="#dff3df", ec="#2a7a2a")
        arrow(ax, (x + 0.38, 1.05), (x + 0.38, 1.55), color="#555")
    box(ax, (8.4, 1.55), 1.0, 0.58, "s_t\nQuery", fc="#ffe4c8", ec="#b8762b")
    box(ax, (6.5, 2.7), 1.55, 0.65, "softmax\nscores", fc="#fff1c7", ec="#b88700")
    box(ax, (6.5, 0.35), 1.55, 0.65, "context c_t\nweighted sum", fc="#eadcff", ec="#6a38ad")
    for x in enc_x:
        arrow(ax, (x + 0.38, 2.13), (6.5, 3.03), color="#2a7a2a", lw=1.2)
        arrow(ax, (7.25, 0.35), (x + 0.38, 1.55), color="#6a38ad", lw=1.0, style="<-")
    arrow(ax, (8.4, 1.84), (8.05, 3.03), color="#b8762b", lw=1.5)
    arrow(ax, (7.25, 2.7), (7.25, 1.0), color="#b88700", lw=1.7)
    arrow(ax, (8.05, 0.68), (8.4, 1.65), color="#6a38ad", lw=1.4)
    ax.text(5.6, 3.85, "Seq2Seq attention: decoder query reads encoder keys/values", ha="center", fontsize=14, weight="bold")
    ax.text(5.6, 3.55, "This is the conceptual bridge to Transformer cross-attention and self-attention", ha="center", fontsize=10, color="#444")
    ax.set_xlim(0.6, 10.0)
    ax.set_ylim(0.0, 4.1)
    out = OUT / "seq2seq-attention-bridge.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_gnn_message_passing():
    fig, ax = plt.subplots(figsize=(8.5, 6))
    clean(ax)
    pos = {
        "v": (0.0, 0.0),
        "a": (-1.7, 1.1),
        "b": (1.7, 1.1),
        "c": (-1.6, -1.2),
        "d": (1.5, -1.25),
        "e": (0.0, 2.0),
    }
    edges = [("a", "v"), ("b", "v"), ("c", "v"), ("d", "v"), ("e", "a"), ("e", "b")]
    for u, v in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#aaa", lw=1.5, zorder=1)
    for name, (x, y) in pos.items():
        color = "#ffe4c8" if name == "v" else "#d8e8ff"
        edge = "#b8762b" if name == "v" else "#2b5fb8"
        circ = patches.Circle((x, y), 0.33, facecolor=color, edgecolor=edge, linewidth=1.4, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, name, ha="center", va="center", fontsize=12, zorder=4)
    for name in ["a", "b", "c", "d"]:
        x0, y0 = pos[name]
        x1, y1 = pos["v"]
        arrow(ax, (x0, y0), (x1, y1), color="#2a7a2a", lw=1.7)
    box(ax, (-1.05, -2.25), 2.1, 0.55, "aggregate neighbor messages", fc="#dff3df", ec="#2a7a2a")
    arrow(ax, (0, -1.85), (0, -0.35), color="#2a7a2a", lw=1.8)
    ax.text(0, 2.75, "GNN message passing", ha="center", fontsize=14, weight="bold")
    ax.text(0, 2.45, "node v updates its representation from local neighbors", ha="center", fontsize=10, color="#444")
    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-2.65, 3.0)
    out = OUT / "gnn-message-passing.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_modern_cnn_blocks():
    fig, ax = plt.subplots(figsize=(12, 4.4))
    clean(ax)
    # ResNet block
    box(ax, (0.5, 1.8), 0.8, 0.5, "x", fc="#f0f0f0", ec="#777")
    box(ax, (2.0, 1.8), 1.1, 0.5, "3x3 conv", fc="#d8e8ff", ec="#2b5fb8")
    box(ax, (3.7, 1.8), 1.1, 0.5, "3x3 conv", fc="#d8e8ff", ec="#2b5fb8")
    box(ax, (5.4, 1.8), 0.8, 0.5, "+", fc="#fff1c7", ec="#b88700")
    box(ax, (6.9, 1.8), 1.0, 0.5, "ReLU", fc="#dff3df", ec="#2a7a2a")
    for s, e in [((1.3, 2.05), (2.0, 2.05)), ((3.1, 2.05), (3.7, 2.05)), ((4.8, 2.05), (5.4, 2.05)), ((6.2, 2.05), (6.9, 2.05))]:
        arrow(ax, s, e)
    ax.plot([1.3, 1.3, 5.4], [1.7, 1.0, 1.0], color="#b88700", lw=1.7)
    arrow(ax, (5.4, 1.0), (5.75, 1.75), color="#b88700", lw=1.7)
    ax.text(3.9, 2.75, "ResNet basic block: y = x + F(x)", ha="center", fontsize=12, weight="bold")

    # Depthwise separable
    box(ax, (1.2, 0.05), 1.45, 0.5, "depthwise 3x3\nspatial per channel", fc="#ffe0d6", ec="#d62728", fs=9)
    box(ax, (3.5, 0.05), 1.45, 0.5, "pointwise 1x1\nchannel mixing", fc="#eadcff", ec="#6a38ad", fs=9)
    arrow(ax, (2.65, 0.3), (3.5, 0.3), color="#555")
    ax.text(3.1, -0.45, "Depthwise separable conv splits spatial filtering and channel mixing", ha="center", fontsize=10, color="#444")
    ax.set_xlim(0.2, 8.5)
    ax.set_ylim(-0.8, 3.1)
    out = OUT / "modern-cnn-blocks.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    figure_training_loop()
    figure_cnn_convolution()
    figure_rnn_unroll()
    figure_lstm_gates()
    figure_attention_bridge()
    figure_gnn_message_passing()
    figure_modern_cnn_blocks()


if __name__ == "__main__":
    main()
