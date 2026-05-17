"""Mini-Mamba: a from-scratch Selective SSM language model that ties together
every concept from the Jamba learning notes (sections 1–5).

Each component is annotated with the section that introduces it.

Run directly:
    python3 scripts/mini_mamba.py

What this demo does:
    1. Builds a tiny Mamba LM (~30K params, character level).
    2. Trains it on Shakespeare's Sonnet 18 (same corpus as transformer/mini_gpt.py).
    3. Watches loss drop from ~3.4 (random) to <1.5.
    4. Generates text via recurrent step-by-step inference.
    5. Sanity-checks: parallel scan-mode forward must equal step-by-step
       recurrent forward (analogous to mini_gpt's KV-cache sanity check).

This is a *teaching* implementation: naive Python scan, no Triton kernel,
no FFT, no parallel scan tree. Performance is irrelevant; clarity is everything.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Section 4.2.1: Selective SSM — input-dependent Δ, B, C; A is fixed diagonal
# ---------------------------------------------------------------------------
class SelectiveSSM(nn.Module):
    """One layer of selective state-space model.

    Per-channel SSM over D channels, each with state size N.
    A is a learned diagonal (D, N) with negative real part (forced via -exp).
    Δ_t, B_t, C_t come from the input via a small Linear projection.

    Discretization: simplified ZOH from Mamba paper (§2.13 in notes):
        Ā_t = exp(Δ_t * A)         (D, N)
        B̄_t ≈ Δ_t * B_t            (one-step Euler / first-order approx)

    Forward modes:
        - forward(u):           parallel-friendly scan over the full L
        - step(u_t, state):     recurrent single-token update for decode
    Both must agree (verified in main()).
    """

    def __init__(self, d_model: int, d_state: int = 8, dt_rank: int = 4):
        super().__init__()
        self.D = d_model
        self.N = d_state

        # A is parameterized as A = -exp(A_log) so its real part is always < 0
        # (Mamba paper §3.5 trick — see notes §4.13).
        # Initialize like S4D-Real: A_real = 1..N per channel.
        A = torch.arange(1, d_state + 1, dtype=torch.float).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))           # (D, N)
        self.D_skip = nn.Parameter(torch.ones(d_model))   # (D,) skip connection

        # x_proj: from u_t produce raw Δ, B, C parameters
        # Δ has its own rank-dt_rank low-rank parameterization (Mamba trick).
        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)
        # dt_proj: dt_rank -> D (broadcast across channels then softplus)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        self.dt_rank = dt_rank

        # Bias init so initial Δ ≈ 1 (softplus(0.5) ≈ 1)
        with torch.no_grad():
            self.dt_proj.bias.fill_(0.5)

    def _compute_params(self, u: torch.Tensor):
        """u: (B, L, D) -> per-position (delta, B, C)."""
        params = self.x_proj(u)                                  # (B, L, dt_rank + 2N)
        dt_raw, Bx, Cx = torch.split(
            params, [self.dt_rank, self.N, self.N], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt_raw))                 # (B, L, D), > 0
        return delta, Bx, Cx                                     # B,C: (B, L, N)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Naive scan over the L dimension. (B, L, D) -> (B, L, D).

        Notes §5: real Mamba uses parallel prefix scan in CUDA/Triton.
        We use a plain Python for-loop for clarity — quadratically slow but
        gives mathematically identical outputs to the recurrent step.
        """
        B, L, D = u.shape
        N = self.N
        A = -torch.exp(self.A_log)                               # (D, N), real < 0

        delta, B_t, C_t = self._compute_params(u)                # see above
        # Discretize per (b, l, d, n):
        #   A_bar = exp(delta * A)   -> (B, L, D, N)
        #   B_bar ≈ delta * B_t      -> (B, L, D, N)  (broadcast B_t across D)
        A_bar = torch.exp(delta.unsqueeze(-1) * A)               # (B, L, D, N)
        B_bar = delta.unsqueeze(-1) * B_t.unsqueeze(2)           # (B, L, D, N)

        # Scan: x_l = A_bar_l ⊙ x_{l-1} + B_bar_l * u_l   (elementwise over D,N)
        x = u.new_zeros(B, D, N)
        ys = []
        for l in range(L):
            x = A_bar[:, l] * x + B_bar[:, l] * u[:, l].unsqueeze(-1)  # (B, D, N)
            # Read out: y_l = C_l · x_l   (sum over N), per channel
            y = (C_t[:, l].unsqueeze(1) * x).sum(dim=-1)         # (B, D)
            ys.append(y)
        y_seq = torch.stack(ys, dim=1)                           # (B, L, D)

        # Skip connection (Mamba `D` parameter, see notes §3.13)
        return y_seq + self.D_skip * u

    @torch.no_grad()
    def step(self, u_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Recurrent single-token update. (B, D), (B, D, N) -> (B, D), (B, D, N)."""
        A = -torch.exp(self.A_log)                               # (D, N)
        delta, B_t, C_t = self._compute_params(u_t.unsqueeze(1)) # add fake L=1
        delta = delta[:, 0]                                      # (B, D)
        B_t = B_t[:, 0]                                          # (B, N)
        C_t = C_t[:, 0]                                          # (B, N)

        A_bar = torch.exp(delta.unsqueeze(-1) * A)               # (B, D, N)
        B_bar = delta.unsqueeze(-1) * B_t.unsqueeze(1)           # (B, D, N)
        state = A_bar * state + B_bar * u_t.unsqueeze(-1)        # (B, D, N)
        y = (C_t.unsqueeze(1) * state).sum(dim=-1)               # (B, D)
        return y + self.D_skip * u_t, state


# ---------------------------------------------------------------------------
# Section 4.5: Mamba block — Linear → depthwise conv → SSM → SiLU gate → Linear
# ---------------------------------------------------------------------------
class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.d_inner = d_inner
        self.d_conv = d_conv

        # Two parallel projections: main path x, gate path z
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Causal depthwise conv (notes §4.6) — kernel of 4 sees ~4 tokens
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            groups=d_inner, padding=d_conv - 1, bias=True,
        )
        self.ssm = SelectiveSSM(d_inner, d_state=d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, L, D)
        B, L, _ = u.shape
        x, z = self.in_proj(u).chunk(2, dim=-1)                  # both (B, L, d_inner)

        # Causal conv: (B, L, d_inner) -> (B, d_inner, L) -> conv -> back
        x = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x = F.silu(x)

        # Selective SSM
        x = self.ssm(x)                                          # (B, L, d_inner)

        # SiLU gate (notes §4.7)
        x = x * F.silu(z)
        return self.out_proj(x)

    @torch.no_grad()
    def step(self, u_t: torch.Tensor, ssm_state: torch.Tensor,
             conv_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-token recurrent update.

        Args:
            u_t:        (B, D)        current token's residual stream
            ssm_state:  (B, d_inner, N)
            conv_state: (B, d_inner, d_conv)  ring buffer of last d_conv x-values

        Returns: (out (B, D), new_ssm_state, new_conv_state)
        """
        x, z = self.in_proj(u_t).chunk(2, dim=-1)                # (B, d_inner) ×2

        # Slide conv buffer: drop oldest, append newest
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = x

        # Apply the (causal) depthwise conv as a dot product with the buffer
        weight = self.conv1d.weight.squeeze(1)                   # (d_inner, d_conv)
        x = (conv_state * weight.unsqueeze(0)).sum(dim=-1) + self.conv1d.bias
        x = F.silu(x)

        x, ssm_state = self.ssm.step(x, ssm_state)
        x = x * F.silu(z)
        return self.out_proj(x), ssm_state, conv_state


# ---------------------------------------------------------------------------
# The model itself — stack of Mamba blocks + LM head
# ---------------------------------------------------------------------------
class MiniMamba(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        d_state: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # Standard small init so initial logits are reasonable
        # (PyTorch default N(0,1) gives huge logits via tied lm_head — bad).
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)

        # Pre-norm in each block; one final norm before lm_head
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state=d_state) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight              # weight tying

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, L) -> logits (B, L, vocab)
        x = self.token_emb(idx)
        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))
        x = self.final_norm(x)
        return self.lm_head(x)

    # Cache layout for recurrent decode:
    #   ssm_states:  list[(B, d_inner, N)]      length = n_layers
    #   conv_states: list[(B, d_inner, d_conv)] length = n_layers
    def empty_cache(self, batch_size: int, device: torch.device):
        ssm_states, conv_states = [], []
        for blk in self.blocks:
            ssm_states.append(
                torch.zeros(batch_size, blk.d_inner, self.d_state, device=device)
            )
            conv_states.append(
                torch.zeros(batch_size, blk.d_inner, blk.d_conv, device=device)
            )
        return ssm_states, conv_states

    @torch.no_grad()
    def step(self, idx_t: torch.Tensor, ssm_states, conv_states):
        """Single-token recurrent forward. idx_t: (B,) -> logits (B, vocab)."""
        x = self.token_emb(idx_t)                                # (B, D)
        new_ssm, new_conv = [], []
        for i, (norm, block) in enumerate(zip(self.norms, self.blocks)):
            h, s, c = block.step(norm(x), ssm_states[i], conv_states[i])
            x = x + h
            new_ssm.append(s)
            new_conv.append(c)
        x = self.final_norm(x)
        return self.lm_head(x), new_ssm, new_conv

    @torch.no_grad()
    def prefill(self, idx: torch.Tensor):
        """Run the parallel forward and warm the caches from the resulting states."""
        B, L = idx.shape
        device = idx.device
        ssm_states, conv_states = self.empty_cache(B, device)
        # Re-run step-by-step to populate caches (slow but simplest).
        # In real Mamba you'd extract the final state from a fused scan kernel.
        logits = None
        for t in range(L):
            logits, ssm_states, conv_states = self.step(
                idx[:, t], ssm_states, conv_states
            )
        return logits, ssm_states, conv_states


# ---------------------------------------------------------------------------
# Section 8.4: autoregressive generation via recurrent step
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(
    model: MiniMamba,
    start_ids: torch.Tensor,         # (1, prompt_len)
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    model.eval()
    logits, ssm_states, conv_states = model.prefill(start_ids)
    out = start_ids
    for _ in range(max_new_tokens):
        last = logits / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(last, top_k)
            last[last < v[:, [-1]]] = float("-inf")
        probs = F.softmax(last, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)        # (B, 1)
        out = torch.cat([out, next_id], dim=1)
        logits, ssm_states, conv_states = model.step(
            next_id.squeeze(-1), ssm_states, conv_states
        )
    return out


# ---------------------------------------------------------------------------
# Training (next-token prediction, teacher forcing) — same recipe as mini_gpt
# ---------------------------------------------------------------------------
def get_batch(data: torch.Tensor, batch_size: int, seq_len: int):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
    return x, y


def train(model, data, steps=800, batch_size=16, seq_len=32, lr=3e-3):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    losses: list[float] = []
    for step in range(1, steps + 1):
        x, y = get_batch(data, batch_size, seq_len)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 100 == 0 or step == 1:
            recent = sum(losses[-100:]) / len(losses[-100:])
            print(f"step {step:4d} | loss {loss.item():.4f} | "
                  f"avg100 {recent:.4f} | elapsed {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Demo on Shakespeare's Sonnet 18 — same corpus as transformer/mini_gpt.py
# ---------------------------------------------------------------------------
CORPUS = """Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
But thy eternal summer shall not fade
Nor lose possession of that fair thou ow'st;
Nor shall Death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe or eyes can see,
So long lives this and this gives life to thee.
"""


def main() -> None:
    torch.manual_seed(0)

    chars = sorted(set(CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab_size = len(chars)
    data = torch.tensor([stoi[c] for c in CORPUS], dtype=torch.long)

    print(f"corpus: {len(CORPUS)} chars, {vocab_size} unique tokens")

    model = MiniMamba(vocab_size=vocab_size, d_model=64, n_layers=2, d_state=8)
    n_params = sum(p.numel() for p in model.parameters())
    n_params -= model.lm_head.weight.numel()                     # tied
    print(f"model: {n_params:,} parameters")

    # ---- Train ----
    print("\n--- training ---")
    train(model, data, steps=800, batch_size=16, seq_len=32, lr=3e-3)

    # ---- Generate ----
    print("\n--- generation (recurrent step inference) ---")
    prompt = "Shall I "
    start_ids = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long)
    out = generate(model, start_ids, max_new_tokens=200, temperature=0.8, top_k=10)
    print("".join(itos[i.item()] for i in out[0]))

    # ---- Sanity: parallel scan vs step-by-step must agree ----
    print("\n--- sanity check: scan forward vs recurrent step ---")
    model.eval()
    with torch.no_grad():
        idx = out[:, :30]                                        # any short slice
        logits_parallel = model(idx)                             # (1, 30, V)

        ssm_s, conv_s = model.empty_cache(1, idx.device)
        step_logits = []
        for t in range(30):
            lt, ssm_s, conv_s = model.step(idx[:, t], ssm_s, conv_s)
            step_logits.append(lt)
        logits_step = torch.stack(step_logits, dim=1)            # (1, 30, V)

        diff = (logits_step - logits_parallel).abs().max().item()
        print(f"max |logits_step - logits_parallel|: {diff:.2e}")
        print("✓ scan == recurrent step (Mamba state is consistent)"
              if diff < 1e-4
              else "✗ MISMATCH — scan and step disagree!")


if __name__ == "__main__":
    main()
