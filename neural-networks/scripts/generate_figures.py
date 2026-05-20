"""Generate visualization figures for the foundational neural-network notes.

The figures are original redrawn teaching diagrams. They intentionally do not
copy figures from papers; see ../paper-figures.md for pointers to the original
paper figures worth reading.

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

COLORS = {
    "blue": "#d8e8ff",
    "blue_edge": "#2b5fb8",
    "green": "#dff3df",
    "green_edge": "#2a7a2a",
    "orange": "#ffe4c8",
    "orange_edge": "#b8762b",
    "yellow": "#fff1c7",
    "yellow_edge": "#b88700",
    "purple": "#eadcff",
    "purple_edge": "#6a38ad",
    "red": "#ffe0d6",
    "red_edge": "#d62728",
    "gray": "#f4f4f4",
    "gray_edge": "#666666",
}


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


def box(ax, xy, w, h, text, fc=None, ec=None, fs=10, weight="normal"):
    fc = fc or COLORS["blue"]
    ec = ec or COLORS["blue_edge"]
    rect = patches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.045",
        linewidth=1.35,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        weight=weight,
    )
    return rect


def circle(ax, xy, r, text, fc, ec, fs=10, weight="normal"):
    circ = patches.Circle(xy, r, facecolor=fc, edgecolor=ec, linewidth=1.35, zorder=3)
    ax.add_patch(circ)
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs, weight=weight, zorder=4)
    return circ


def arrow(ax, start, end, color="#555", lw=1.65, style="->", rad=0.0):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            shrinkA=4,
            shrinkB=4,
            connectionstyle=f"arc3,rad={rad}",
        ),
    )


def save(fig, name):
    out = OUT / name
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def draw_tensor_stack(ax, x, y, w, h, depth, color, edge, label, dx=0.08, dy=0.08):
    for k in range(depth - 1, -1, -1):
        rect = patches.Rectangle(
            (x + k * dx, y + k * dy),
            w,
            h,
            facecolor=color,
            edgecolor=edge,
            linewidth=1.2,
            alpha=0.95,
        )
        ax.add_patch(rect)
    ax.text(x + w / 2 + (depth - 1) * dx / 2, y - 0.25, label, ha="center", fontsize=10)


def figure_training_loop():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    clean(ax)

    items = [
        ((0.4, 2.0), "mini-batch\nx, y", COLORS["blue"], COLORS["blue_edge"]),
        ((2.25, 2.0), "forward\nlogits = f_theta(x)", COLORS["green"], COLORS["green_edge"]),
        ((4.45, 2.0), "loss\nL(logits, y)", COLORS["yellow"], COLORS["yellow_edge"]),
        ((6.4, 2.0), "backward\ngrad = dL/dtheta", COLORS["red"], COLORS["red_edge"]),
        ((8.75, 2.0), "optimizer\nAdamW / SGD", COLORS["purple"], COLORS["purple_edge"]),
    ]
    for xy, text, fc, ec in items:
        box(ax, xy, 1.45, 0.85, text, fc=fc, ec=ec, fs=9.5)
    for i in range(len(items) - 1):
        x0, y0 = items[i][0]
        x1, y1 = items[i + 1][0]
        arrow(ax, (x0 + 1.45, y0 + 0.43), (x1, y1 + 0.43), lw=1.8)

    box(ax, (8.7, 0.65), 1.55, 0.55, "theta <- theta - step", fc=COLORS["gray"], ec=COLORS["gray_edge"], fs=9)
    arrow(ax, (9.5, 2.0), (9.5, 1.2), color=COLORS["purple_edge"], lw=1.6)
    arrow(ax, (8.7, 0.93), (2.7, 1.95), color=COLORS["purple_edge"], lw=1.5, rad=0.25)
    ax.text(5.5, 3.35, "Training loop: prediction, error signal, gradient, parameter update", ha="center", fontsize=14, weight="bold")
    ax.text(5.5, 3.05, "The model improves because the loss sends feedback to every parameter through backpropagation", ha="center", fontsize=10, color="#444")
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0.35, 3.7)
    save(fig, "training-loop.png")


def figure_cnn_convolution():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    clean(ax)
    ax.text(5.8, 4.55, "CNN convolution: local receptive fields + shared kernels + channel mixing", ha="center", fontsize=14, weight="bold")

    draw_tensor_stack(ax, 0.7, 1.65, 1.75, 1.75, 3, COLORS["blue"], COLORS["blue_edge"], "input\nC_in x H x W")
    rect = patches.Rectangle((1.1, 2.05), 0.75, 0.75, facecolor="none", edgecolor=COLORS["red_edge"], linewidth=2.2)
    ax.add_patch(rect)
    ax.text(1.5, 3.67, "same 3x3 window\nslides everywhere", ha="center", fontsize=9, color=COLORS["red_edge"])
    arrow(ax, (1.5, 3.42), (1.5, 2.85), color=COLORS["red_edge"], lw=1.3)

    for i, y in enumerate([2.85, 2.0, 1.15]):
        draw_tensor_stack(ax, 3.35, y, 0.82, 0.52, 3, COLORS["orange"], COLORS["orange_edge"], "" if i else "kernel bank\nC_out filters", dx=0.045, dy=0.045)
    ax.text(3.85, 0.7, "each filter spans all input channels", ha="center", fontsize=9, color="#555")

    arrow(ax, (2.55, 2.52), (3.25, 2.52), color="#555", lw=1.8)
    ax.text(2.9, 2.78, "dot product", ha="center", fontsize=9)

    draw_tensor_stack(ax, 6.1, 1.55, 1.7, 1.7, 4, COLORS["green"], COLORS["green_edge"], "feature maps\nC_out x H_out x W_out")
    arrow(ax, (4.4, 2.52), (6.0, 2.52), color="#555", lw=1.8)
    ax.text(5.18, 2.84, "one map per filter", ha="center", fontsize=9)

    box(ax, (8.7, 2.7), 2.3, 0.55, "parameter count", fc=COLORS["gray"], ec=COLORS["gray_edge"], fs=10, weight="bold")
    box(ax, (8.7, 1.9), 2.3, 0.55, "C_out x C_in x k x k", fc="#ffffff", ec=COLORS["gray_edge"], fs=10)
    box(ax, (8.7, 1.1), 2.3, 0.55, "independent of image H,W", fc=COLORS["yellow"], ec=COLORS["yellow_edge"], fs=10)
    ax.set_xlim(0.1, 11.4)
    ax.set_ylim(0.35, 4.9)
    save(fig, "cnn-convolution-overview.png")


def figure_rnn_unroll():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    clean(ax)
    n = 5
    ax.text(5.7, 4.05, "RNN unrolled through time: same cell, serial hidden state", ha="center", fontsize=14, weight="bold")
    ax.text(5.7, 3.75, "Every h_t waits for h_{t-1}; BPTT sends gradients back through the same chain", ha="center", fontsize=10, color="#444")

    for i in range(n):
        x = 1.0 + i * 2.0
        box(ax, (x, 0.65), 0.75, 0.45, f"$x_{i+1}$", fc=COLORS["blue"], ec=COLORS["blue_edge"], fs=12)
        box(ax, (x - 0.08, 1.75), 0.92, 0.65, "RNN\ncell", fc=COLORS["orange"], ec=COLORS["orange_edge"], fs=9.5)
        box(ax, (x, 2.9), 0.75, 0.45, f"$h_{i+1}$", fc=COLORS["green"], ec=COLORS["green_edge"], fs=12)
        arrow(ax, (x + 0.37, 1.1), (x + 0.37, 1.75), color="#555")
        arrow(ax, (x + 0.37, 2.4), (x + 0.37, 2.9), color=COLORS["green_edge"])
        if i > 0:
            arrow(ax, (x - 1.08, 2.08), (x - 0.08, 2.08), color=COLORS["orange_edge"], lw=2.1)
        ax.text(x + 0.37, 1.52, "$W_x$", ha="center", fontsize=8, color="#555")
    ax.text(4.98, 2.48, "shared $W_x, W_h$ at every step", ha="center", fontsize=10, color=COLORS["orange_edge"])
    arrow(ax, (9.35, 3.55), (1.45, 3.55), color=COLORS["red_edge"], lw=1.6, style="-|>")
    ax.text(5.4, 3.35, "gradient path for early tokens can become very long", ha="center", fontsize=9, color=COLORS["red_edge"])
    ax.set_xlim(0.25, 10.6)
    ax.set_ylim(0.3, 4.35)
    save(fig, "rnn-unroll.png")


def figure_lstm_gates():
    fig, ax = plt.subplots(figsize=(13, 5.8))
    clean(ax)
    ax.text(6.4, 5.25, "LSTM cell: gates regulate long-term memory flow", ha="center", fontsize=14, weight="bold")

    # Cell-state highway
    box(ax, (0.55, 3.25), 0.95, 0.5, "$c_{t-1}$", fc=COLORS["yellow"], ec=COLORS["yellow_edge"], fs=12)
    circle(ax, (3.1, 3.5), 0.22, "$\\times$", COLORS["gray"], COLORS["gray_edge"], fs=11)
    circle(ax, (7.0, 3.5), 0.25, "$+$", COLORS["yellow"], COLORS["yellow_edge"], fs=13, weight="bold")
    box(ax, (11.1, 3.25), 0.85, 0.5, "$c_t$", fc=COLORS["yellow"], ec=COLORS["yellow_edge"], fs=12)
    arrow(ax, (1.5, 3.5), (2.88, 3.5), color=COLORS["yellow_edge"], lw=2.2)
    arrow(ax, (3.32, 3.5), (6.75, 3.5), color=COLORS["yellow_edge"], lw=2.2)
    arrow(ax, (7.25, 3.5), (11.1, 3.5), color=COLORS["yellow_edge"], lw=2.2)
    ax.text(6.3, 3.9, "additive cell-state path helps preserve long-range information", ha="center", fontsize=10, color="#805d00")

    # Input source
    box(ax, (5.15, 0.35), 2.2, 0.48, "$u_t = [x_t ; h_{t-1}]$", fc=COLORS["gray"], ec=COLORS["gray_edge"], fs=11)

    gates = [
        ((2.15, 1.45), "$f_t=\\sigma(W_fu_t)$\nforget", COLORS["red"], COLORS["red_edge"], (3.1, 3.28)),
        ((4.55, 1.45), "$i_t=\\sigma(W_iu_t)$\nwrite amount", COLORS["green"], COLORS["green_edge"], (5.35, 2.45)),
        ((6.95, 1.45), "$\\tilde{c}_t=\\tanh(W_cu_t)$\ncandidate", COLORS["blue"], COLORS["blue_edge"], (6.65, 2.45)),
        ((9.35, 1.45), "$o_t=\\sigma(W_ou_t)$\nexpose", COLORS["purple"], COLORS["purple_edge"], (10.2, 2.45)),
    ]
    for xy, text, fc, ec, target in gates:
        box(ax, xy, 1.65, 0.75, text, fc=fc, ec=ec, fs=9.3)
        arrow(ax, (6.25, 0.83), (xy[0] + 0.82, xy[1]), color="#777", lw=1.15)
        arrow(ax, (xy[0] + 0.82, xy[1] + 0.75), target, color=ec, lw=1.45)

    circle(ax, (5.35, 2.7), 0.22, "$\\times$", COLORS["gray"], COLORS["gray_edge"], fs=11)
    circle(ax, (6.65, 2.7), 0.22, "$\\times$", COLORS["gray"], COLORS["gray_edge"], fs=11)
    arrow(ax, (5.55, 2.7), (6.43, 2.7), color=COLORS["green_edge"], lw=1.5)
    arrow(ax, (6.65, 2.92), (6.9, 3.28), color=COLORS["green_edge"], lw=1.5)

    circle(ax, (10.2, 2.7), 0.22, "$\\times$", COLORS["gray"], COLORS["gray_edge"], fs=11)
    box(ax, (11.1, 1.6), 0.85, 0.5, "$h_t$", fc=COLORS["green"], ec=COLORS["green_edge"], fs=12)
    arrow(ax, (11.5, 3.25), (10.35, 2.9), color=COLORS["yellow_edge"], lw=1.4, rad=0.12)
    arrow(ax, (10.42, 2.7), (11.1, 1.85), color=COLORS["green_edge"], lw=1.4)

    ax.text(5.4, 4.55, "$c_t=f_t\\odot c_{t-1}+i_t\\odot\\tilde{c}_t$", ha="center", fontsize=12, color="#333")
    ax.text(10.4, 4.55, "$h_t=o_t\\odot\\tanh(c_t)$", ha="center", fontsize=12, color="#333")
    ax.set_xlim(0.2, 12.4)
    ax.set_ylim(0.1, 5.55)
    save(fig, "lstm-gates.png")


def figure_attention_bridge():
    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    hm = fig.add_subplot(gs[0, 1])
    clean(ax)

    ax.text(4.55, 4.5, "Seq2Seq attention: decoder query reads encoder annotations", ha="center", fontsize=13, weight="bold")
    enc_x = [0.75, 1.75, 2.75, 3.75]
    for i, x in enumerate(enc_x, 1):
        box(ax, (x, 0.5), 0.58, 0.42, f"$x_{i}$", fc=COLORS["blue"], ec=COLORS["blue_edge"], fs=10)
        box(ax, (x, 1.45), 0.58, 0.5, f"$h_{i}$", fc=COLORS["green"], ec=COLORS["green_edge"], fs=10)
        arrow(ax, (x + 0.29, 0.92), (x + 0.29, 1.45), color="#555")
    box(ax, (6.9, 1.45), 0.95, 0.5, "$s_t$\nquery", fc=COLORS["orange"], ec=COLORS["orange_edge"], fs=10)
    box(ax, (5.15, 3.15), 1.55, 0.55, "score + softmax\n$\\alpha_{t,j}$", fc=COLORS["yellow"], ec=COLORS["yellow_edge"], fs=10)
    box(ax, (5.15, 0.45), 1.55, 0.55, "context $c_t$\nweighted sum", fc=COLORS["purple"], ec=COLORS["purple_edge"], fs=10)
    for x in enc_x:
        arrow(ax, (x + 0.29, 1.95), (5.15, 3.43), color=COLORS["green_edge"], lw=1.05)
        arrow(ax, (5.85, 1.0), (x + 0.29, 1.45), color=COLORS["purple_edge"], lw=1.0, style="<-")
    arrow(ax, (6.9, 1.75), (6.65, 3.43), color=COLORS["orange_edge"], lw=1.4)
    arrow(ax, (5.92, 3.15), (5.92, 1.0), color=COLORS["yellow_edge"], lw=1.5)
    arrow(ax, (6.7, 0.72), (6.9, 1.55), color=COLORS["purple_edge"], lw=1.2)
    ax.text(4.4, 4.2, "Bahdanau-style soft alignment removes the fixed-vector bottleneck", ha="center", fontsize=9.5, color="#444")
    ax.set_xlim(0.25, 8.2)
    ax.set_ylim(0.15, 4.8)

    src = ["the", "cat", "sat", "."]
    tgt = ["le", "chat", "assis", "."]
    weights = np.array(
        [
            [0.72, 0.15, 0.08, 0.05],
            [0.07, 0.78, 0.11, 0.04],
            [0.05, 0.18, 0.70, 0.07],
            [0.05, 0.06, 0.10, 0.79],
        ]
    )
    im = hm.imshow(weights, cmap="YlGnBu", vmin=0, vmax=1)
    hm.set_title("alignment weights $\\alpha_{t,j}$", fontsize=12)
    hm.set_xticks(np.arange(len(src)), labels=src, fontsize=9)
    hm.set_yticks(np.arange(len(tgt)), labels=tgt, fontsize=9)
    hm.set_xlabel("source positions", fontsize=10)
    hm.set_ylabel("target step", fontsize=10)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            hm.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#222")
    fig.colorbar(im, ax=hm, fraction=0.046, pad=0.04)
    save(fig, "seq2seq-attention-bridge.png")


def figure_gnn_message_passing():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    layers = ["input features\n$H^{(0)}$", "1-hop mixed\n$H^{(1)}$", "2-hop mixed\n$H^{(2)}$"]
    pos = {
        0: (0.0, 0.0),
        1: (-1.1, 0.75),
        2: (1.1, 0.75),
        3: (-1.05, -0.9),
        4: (1.05, -0.9),
        5: (0.0, 1.65),
    }
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 5)]
    for layer, ax in enumerate(axes):
        clean(ax)
        ax.set_title(layers[layer], fontsize=12, weight="bold")
        for u, v in edges:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#b7b7b7", lw=1.4, zorder=1)
        for node, (x, y) in pos.items():
            highlight = node == 0 or (layer >= 1 and node in [1, 2, 3, 4]) or (layer >= 2 and node == 5)
            fc = COLORS["orange"] if node == 0 else (COLORS["green"] if highlight else COLORS["blue"])
            ec = COLORS["orange_edge"] if node == 0 else (COLORS["green_edge"] if highlight else COLORS["blue_edge"])
            circle(ax, (x, y), 0.23, str(node), fc, ec, fs=9)
        if layer == 1:
            for n in [1, 2, 3, 4]:
                arrow(ax, pos[n], pos[0], color=COLORS["green_edge"], lw=1.5)
        if layer == 2:
            arrow(ax, pos[5], pos[1], color=COLORS["green_edge"], lw=1.2)
            arrow(ax, pos[5], pos[2], color=COLORS["green_edge"], lw=1.2)
            for n in [1, 2, 3, 4]:
                arrow(ax, pos[n], pos[0], color=COLORS["green_edge"], lw=1.2)
        ax.set_xlim(-1.75, 1.75)
        ax.set_ylim(-1.35, 2.0)
    fig.suptitle("GCN intuition: each layer expands the receptive field by one graph hop", fontsize=14, weight="bold", y=1.02)
    save(fig, "gnn-message-passing.png")


def figure_modern_cnn_blocks():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax, ax2 = axes
    clean(ax)
    clean(ax2)
    ax.text(3.9, 3.95, "ResNet block: learn a residual correction", ha="center", fontsize=13, weight="bold")
    box(ax, (0.35, 2.0), 0.65, 0.45, "$x$", fc=COLORS["gray"], ec=COLORS["gray_edge"], fs=11)
    box(ax, (1.55, 2.0), 1.0, 0.45, "3x3\nconv", fc=COLORS["blue"], ec=COLORS["blue_edge"], fs=9)
    box(ax, (3.0, 2.0), 1.0, 0.45, "BN\nReLU", fc=COLORS["green"], ec=COLORS["green_edge"], fs=9)
    box(ax, (4.45, 2.0), 1.0, 0.45, "3x3\nconv", fc=COLORS["blue"], ec=COLORS["blue_edge"], fs=9)
    circle(ax, (6.2, 2.23), 0.24, "$+$", COLORS["yellow"], COLORS["yellow_edge"], fs=13, weight="bold")
    box(ax, (7.0, 2.0), 0.8, 0.45, "ReLU", fc=COLORS["green"], ec=COLORS["green_edge"], fs=9)
    for s, e in [((1.0, 2.23), (1.55, 2.23)), ((2.55, 2.23), (3.0, 2.23)), ((4.0, 2.23), (4.45, 2.23)), ((5.45, 2.23), (5.96, 2.23)), ((6.44, 2.23), (7.0, 2.23))]:
        arrow(ax, s, e)
    ax.plot([1.0, 1.0, 6.0], [1.9, 1.15, 1.15], color=COLORS["yellow_edge"], lw=1.8)
    arrow(ax, (6.0, 1.15), (6.12, 2.0), color=COLORS["yellow_edge"], lw=1.8)
    ax.text(3.7, 0.62, "$y = F(x) + x$", ha="center", fontsize=12)
    ax.set_xlim(0.1, 8.1)
    ax.set_ylim(0.25, 4.25)

    ax2.text(3.0, 3.95, "Depthwise separable convolution", ha="center", fontsize=13, weight="bold")
    draw_tensor_stack(ax2, 0.25, 1.8, 0.9, 1.0, 4, COLORS["blue"], COLORS["blue_edge"], "input")
    box(ax2, (1.85, 2.1), 1.25, 0.55, "depthwise\nk x k", fc=COLORS["red"], ec=COLORS["red_edge"], fs=9)
    box(ax2, (3.8, 2.1), 1.25, 0.55, "pointwise\n1x1", fc=COLORS["purple"], ec=COLORS["purple_edge"], fs=9)
    draw_tensor_stack(ax2, 5.75, 1.8, 0.9, 1.0, 5, COLORS["green"], COLORS["green_edge"], "output")
    arrow(ax2, (1.2, 2.35), (1.85, 2.35))
    arrow(ax2, (3.1, 2.35), (3.8, 2.35))
    arrow(ax2, (5.05, 2.35), (5.75, 2.35))
    ax2.text(2.5, 1.25, "spatial filtering\nper channel", ha="center", fontsize=9, color=COLORS["red_edge"])
    ax2.text(4.45, 1.25, "channel mixing", ha="center", fontsize=9, color=COLORS["purple_edge"])
    box(ax2, (1.0, 0.4), 5.1, 0.48, "params: C_in*k*k + C_in*C_out instead of C_in*C_out*k*k", fc=COLORS["yellow"], ec=COLORS["yellow_edge"], fs=9)
    ax2.set_xlim(0.0, 7.1)
    ax2.set_ylim(0.15, 4.25)
    save(fig, "modern-cnn-blocks.png")


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
