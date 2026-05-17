"""Generate visualization figures for the Jamba learning notes.

Outputs (written to ../images/):
  - ssm-state-evolution.png:      Section 1 — x(t) = e^{At} x(0) for different A eigenvalues
  - delta-controls-memory.png:    Section 2 / 4 — how Δ_t controls effective memory length
  - prefix-scan-tree.png:         Section 5 — serial RNN scan vs Hillis-Steele parallel scan
  - cache-growth.png:             Section 8 — KV cache vs Mamba state size growth with L

Run from anywhere; output paths are resolved relative to this file.

    python3 scripts/generate_figures.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "images"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1 (Section 1.5): SSM state evolution under different A
# ---------------------------------------------------------------------------
def figure_ssm_state_evolution():
    """Show x(t) = e^{a*t} * x(0) for three qualitatively different scalar a.

    a < 0  -> stable decay (good for memory)
    a = ib -> oscillation (anti-symmetric A in higher dim)
    a > 0  -> blow-up (unstable)
    """
    t = np.linspace(0, 5, 400)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    cases = [
        ("a = -0.8  (stable decay)", -0.8, "#2a7a2a",
         "good for long-range memory:\nold state fades smoothly"),
        ("a = 0.0 ± 2j  (oscillation)", 2.0j, "#1f77b4",
         "anti-symmetric A in 2D:\nperiodic state, never decays"),
        ("a = +0.4  (blow-up)", 0.4, "#d62728",
         "unstable: state explodes\n— bad init for SSM"),
    ]

    for ax, (title, a, color, note) in zip(axes, cases):
        if isinstance(a, complex):
            # 2D oscillation: x(t) = [cos(b*t), sin(b*t)] from x(0)=[1,0]
            x = np.cos(a.imag * t)
            x2 = np.sin(a.imag * t)
            ax.plot(t, x, color=color, lw=2, label="x_1(t)")
            ax.plot(t, x2, color=color, lw=1.5, ls="--", alpha=0.6, label="x_2(t)")
            ax.legend(loc="lower right", fontsize=9)
        else:
            x = np.exp(a * t)
            ax.plot(t, x, color=color, lw=2)
        ax.axhline(0, color="#888", lw=0.5)
        ax.set_xlabel("time t", fontsize=10)
        ax.set_ylabel("state x(t)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.text(0.03, 0.96, note, transform=ax.transAxes,
                fontsize=9, color="#333", va="top",
                bbox=dict(boxstyle="round,pad=0.35",
                          fc="#f7f7f7", ec="#bbb", lw=0.6))
        ax.grid(alpha=0.3)

    axes[2].set_ylim(0.5, 8)
    fig.suptitle(
        "SSM state evolution  x(t) = e^{At} x(0)   — A's eigenvalues decide behavior",
        fontsize=12, y=1.02
    )
    fig.tight_layout()
    out_path = OUT / "ssm-state-evolution.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 (Section 2.4 / 4.3): Δ_t controls effective memory length
# ---------------------------------------------------------------------------
def figure_delta_controls_memory():
    """Mamba lets Δ_t depend on input. Show how a single discrete SSM step
    A_bar = exp(a * Δ) behaves under different Δ values, given fixed a < 0.

    Small Δ -> A_bar ~ 1 -> state mostly preserved (long memory).
    Large Δ -> A_bar ~ 0 -> state heavily forgotten (write new input).
    """
    a = -1.0  # fixed continuous-time decay rate
    deltas = np.linspace(0.01, 4.0, 200)
    A_bar = np.exp(a * deltas)  # state retention per step

    # Right panel: state trajectory under three different Δ
    L = 30
    t_steps = np.arange(L)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.2))

    # ---- Left: retention vs Δ ----
    ax_l.plot(deltas, A_bar, lw=2.5, color="#1f77b4")
    ax_l.fill_between(deltas, 0, A_bar, alpha=0.15, color="#1f77b4")
    for d, label, color in [(0.2, "small Δ\n(remember)", "#2a7a2a"),
                             (1.0, "medium Δ", "#ff7f0e"),
                             (3.0, "large Δ\n(forget)", "#d62728")]:
        y = np.exp(a * d)
        ax_l.plot([d, d], [0, y], color=color, lw=1.2, ls="--")
        ax_l.scatter([d], [y], color=color, s=80, zorder=5)
        ax_l.annotate(label, xy=(d, y), xytext=(d + 0.15, y + 0.08),
                      fontsize=9, color=color)
    ax_l.set_xlabel("step size  Δ", fontsize=11)
    ax_l.set_ylabel("state retention  Ā = exp(aΔ)", fontsize=11)
    ax_l.set_title("Fixed a = -1: Δ controls how much old state survives",
                   fontsize=11)
    ax_l.set_xlim(0, 4)
    ax_l.set_ylim(0, 1.05)
    ax_l.grid(alpha=0.3)

    # ---- Right: state trajectory under three Δ (single impulse at t=0) ----
    for d, label, color in [(0.2, "Δ = 0.2 (long memory)", "#2a7a2a"),
                             (1.0, "Δ = 1.0 (medium)", "#ff7f0e"),
                             (3.0, "Δ = 3.0 (short memory)", "#d62728")]:
        x = np.zeros(L)
        x[0] = 1.0  # impulse input writes state to 1
        A_b = np.exp(a * d)
        for t in range(1, L):
            x[t] = A_b * x[t - 1]  # no new input, just decay
        ax_r.plot(t_steps, x, lw=2, color=color, label=label, marker="o", ms=4)

    ax_r.set_xlabel("step t (after a single impulse at t=0)", fontsize=11)
    ax_r.set_ylabel("state x_t", fontsize=11)
    ax_r.set_title("How long can the impulse 'survive' in the state?",
                   fontsize=11)
    ax_r.set_xlim(0, L - 1)
    ax_r.set_ylim(-0.02, 1.05)
    ax_r.legend(loc="upper right", fontsize=9)
    ax_r.grid(alpha=0.3)

    fig.suptitle(
        "Selective SSM intuition: Δ_t is the per-token 'memory dial'",
        fontsize=12, y=1.02
    )
    fig.tight_layout()
    out_path = OUT / "delta-controls-memory.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 (Section 5.4): prefix scan — serial vs parallel
# ---------------------------------------------------------------------------
def figure_prefix_scan_tree():
    """Two side-by-side dependency graphs:
    Left:  serial scan, depth = L
    Right: Hillis-Steele parallel scan, depth = log2(L)
    """
    L = 8
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- Left: serial ----
    ax_l.set_title(f"Serial scan: depth O(L) = {L} sequential steps",
                   fontsize=12)
    for i in range(L):
        ax_l.scatter(i, 0, s=600, c="#bcd0f5", edgecolors="#2b5fb8", zorder=3)
        ax_l.text(i, 0, f"z{i+1}", ha="center", va="center", fontsize=9)
        ax_l.scatter(i, 1, s=600, c="#f5d6bc", edgecolors="#b8762b", zorder=3)
        ax_l.text(i, 1, f"y{i+1}", ha="center", va="center", fontsize=9)
        # z_i -> y_i
        ax_l.annotate("", xy=(i, 0.78), xytext=(i, 0.22),
                      arrowprops=dict(arrowstyle="->", color="#888", lw=1))
    for i in range(L - 1):
        # y_i -> y_{i+1}
        ax_l.annotate("", xy=(i + 0.78, 1), xytext=(i + 0.22, 1),
                      arrowprops=dict(arrowstyle="->", color="#b8762b", lw=2.2))
    ax_l.text(L / 2 - 0.5, 1.7, "critical path: y1 → y2 → ... → y8 (8 steps)",
              ha="center", fontsize=10, style="italic", color="#b8762b")
    ax_l.set_xlim(-0.6, L - 0.4)
    ax_l.set_ylim(-0.6, 2.1)
    ax_l.set_xticks([])
    ax_l.set_yticks([0, 1])
    ax_l.set_yticklabels(["input  z", "prefix y"], fontsize=10)
    ax_l.spines[:].set_visible(False)

    # ---- Right: Hillis-Steele parallel scan ----
    # At step k (1-indexed), each position i with i >= 2^(k-1) gets
    # combined with position i - 2^(k-1).
    depth = int(np.log2(L))  # = 3 for L=8
    ax_r.set_title(f"Hillis-Steele parallel scan: depth O(log L) = {depth} rounds",
                   fontsize=12)
    # Draw L columns, depth+1 rows (row 0 = inputs, rows 1..depth = after each round)
    palette = ["#bcd0f5", "#cfe5cf", "#ffe6b3", "#f5c0c0"]
    for row in range(depth + 1):
        for i in range(L):
            ax_r.scatter(i, -row, s=550,
                         c=palette[row % len(palette)],
                         edgecolors="#444", zorder=3)
        # arrows from previous row
        if row == 0:
            continue
        stride = 2 ** (row - 1)
        for i in range(L):
            # identity edge from same column
            ax_r.annotate("", xy=(i, -row + 0.22), xytext=(i, -row + 0.78),
                          arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8))
            # merge edge from column i - stride
            if i - stride >= 0:
                ax_r.annotate("", xy=(i, -row + 0.22),
                              xytext=(i - stride, -row + 0.78),
                              arrowprops=dict(arrowstyle="->",
                                              color="#2b5fb8", lw=1.6))

    # Row labels
    ax_r.text(-0.9, 0, "z", fontsize=11, va="center")
    for row in range(1, depth + 1):
        ax_r.text(-0.9, -row, f"round {row}", fontsize=9, va="center", color="#444")
    ax_r.text(L / 2 - 0.5, -(depth + 0.9),
              f"critical path: only {depth} parallel rounds (each fires all L elements at once)",
              ha="center", fontsize=10, style="italic", color="#2b5fb8")

    ax_r.set_xlim(-1.2, L - 0.4)
    ax_r.set_ylim(-(depth + 1.4), 0.7)
    ax_r.set_xticks([])
    ax_r.set_yticks([])
    ax_r.spines[:].set_visible(False)

    fig.suptitle(
        "Prefix scan: same total work, but serial waits L steps while parallel waits log L",
        fontsize=12, y=1.02
    )
    fig.tight_layout()
    out_path = OUT / "prefix-scan-tree.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 (Section 8.5): cache size vs context length L
# ---------------------------------------------------------------------------
def figure_cache_growth():
    """Compare per-sample inference cache size (FP16) for four configs as
    context length grows from 4K to 256K:

        full MHA          :  attn cache = 2 * L * D * n_layers * 2 bytes
        full GQA(8x)      :  attn cache / 8       (kv head sharing)
        Jamba 1:7 + GQA   :  attn cache / 8 / 8   (1/8 layers + GQA per attn)
        pure Mamba        :  only state, ≈ const

    Numbers use D=4096, n_layers=32, FP16 (2 bytes).
    Jamba's real config uses GQA on its few attention layers, hence /8 twice.
    """
    L = np.linspace(4_000, 256_000, 200)
    D = 4096
    n_layers = 32
    fp_bytes = 2

    # Bytes / sample
    full_mha = 2 * L * D * n_layers * fp_bytes
    full_gqa = full_mha / 8
    n_attn_layers_jamba = 4  # 1:7 → 4 of 32
    jamba_attn = 2 * L * D * n_attn_layers_jamba * fp_bytes / 8  # GQA(8) on each
    mamba_state_per_layer = D * 16 * fp_bytes  # N=16
    pure_mamba = np.full_like(L, mamba_state_per_layer * n_layers)
    jamba_total = jamba_attn + mamba_state_per_layer * (n_layers - n_attn_layers_jamba)

    fig, ax = plt.subplots(figsize=(10, 5))
    gb = 1024 ** 3
    ax.plot(L / 1000, full_mha / gb, lw=2.5, color="#d62728",
            label=f"full MHA  (D={D}, {n_layers} layers)")
    ax.plot(L / 1000, full_gqa / gb, lw=2.5, color="#ff7f0e",
            label="full GQA  (group=8)")
    ax.plot(L / 1000, jamba_total / gb, lw=2.5, color="#2b5fb8",
            label="Jamba  (1:7 layers × GQA-8 + Mamba states)")
    ax.plot(L / 1000, pure_mamba / gb, lw=2.5, color="#2a7a2a",
            label="pure Mamba  (state only)")

    ax.set_yscale("log")
    ax.set_xlabel("context length  L  (×1000 tokens)", fontsize=11)
    ax.set_ylabel("inference cache per sample  (GB, log scale)", fontsize=11)
    ax.set_title("Long-context cache cost: why Jamba combines fewer attn layers + GQA",
                 fontsize=12)
    ax.axhline(80, color="#888", lw=1, ls="--")
    ax.text(8, 95, "single A100 80 GB", fontsize=9, color="#888")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.set_xlim(4, 256)

    # Annotate the 256K endpoint values
    for series, color in [
        (full_mha, "#d62728"),
        (full_gqa, "#ff7f0e"),
        (jamba_total, "#2b5fb8"),
        (pure_mamba, "#2a7a2a"),
    ]:
        val_gb = series[-1] / gb
        if val_gb >= 1:
            txt = f"{val_gb:.1f} GB"
        elif val_gb >= 1 / 1024:
            txt = f"{val_gb * 1024:.0f} MB"
        else:
            txt = f"{val_gb * 1024 * 1024:.1f} KB"
        ax.annotate(txt, xy=(256, val_gb), xytext=(258, val_gb),
                    fontsize=9, color=color, va="center")

    fig.tight_layout()
    out_path = OUT / "cache-growth.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    figure_ssm_state_evolution()
    figure_delta_controls_memory()
    figure_prefix_scan_tree()
    figure_cache_growth()
