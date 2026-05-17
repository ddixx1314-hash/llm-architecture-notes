"""Mini-Jamba: a from-scratch hybrid Attention + Mamba + MoE language model
that ties together every concept from the Jamba learning notes (sections 6–8).

Each component is annotated with the section that introduces it.

Run directly:
    python3 scripts/mini_jamba.py

What this demo does:
    1. Reuses MambaBlock from mini_mamba.py.
    2. Adds a causal MultiHeadAttention layer with KV cache.
    3. Adds a Top-2-of-4 MoE FFN.
    4. Stacks them in a Jamba-style hybrid:
         layer 0: Mamba + FFN
         layer 1: Mamba + MoE
         layer 2: Attention + FFN   ← the one attention layer (a:m = 1:3 here)
         layer 3: Mamba + MoE
       (Real Jamba uses 1:7 + e=2; for a small demo 1:3 keeps it tractable.)
    5. Trains on Sonnet 18.
    6. Generates text with mixed Attention KV cache + Mamba state.
    7. Sanity check: parallel forward must equal step-by-step recurrent forward
       on every layer type.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse MambaBlock from mini_mamba.py in the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mini_mamba import MambaBlock  # noqa: E402


# ---------------------------------------------------------------------------
# Section 7.5: Causal multi-head attention (with KV cache) — same as mini_gpt
# ---------------------------------------------------------------------------
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor,
                past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, n, _ = x.shape
        q, k, v = self.W_qkv(x).chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B, n, self.h, self.d_k).transpose(1, 2)
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v)

        L_q, L_k = q.size(2), k.size(2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if L_q > 1:
            mask = torch.ones(L_q, L_k, dtype=torch.bool, device=x.device).triu(
                diagonal=L_k - L_q + 1
            )
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        return self.W_o(out), new_kv


# ---------------------------------------------------------------------------
# Section 6.4: dense FFN (used in non-MoE layers)
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Section 6.8: Top-K MoE — token×slot expanded routing (the recommended form)
# ---------------------------------------------------------------------------
class TopKMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.E = num_experts
        self.K = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForward(d_model, d_ff) for _ in range(num_experts)]
        )

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)                        # (T, D), T = B*L
        T = x_flat.size(0)

        logits = self.router(x_flat)                     # (T, E)
        top_logits, top_idx = logits.topk(self.K, dim=-1)  # (T, K)
        top_w = F.softmax(top_logits, dim=-1)            # (T, K), already normalized

        # Expand to T*K (token_id, expert_id, weight) triples — notes §6.8 version B
        flat_token = (torch.arange(T, device=x.device)
                      .repeat_interleave(self.K))        # (T*K,)
        flat_expert = top_idx.reshape(-1)                # (T*K,)
        flat_w = top_w.reshape(-1)                       # (T*K,)

        out = torch.zeros_like(x_flat)
        for e in range(self.E):
            mask = (flat_expert == e)
            if not mask.any():
                continue
            sel = flat_token[mask]
            sel_w = flat_w[mask].unsqueeze(-1)
            out.index_add_(0, sel, self.experts[e](x_flat[sel]) * sel_w)

        return out.reshape(B, L, D)


# ---------------------------------------------------------------------------
# Section 7.5: Jamba block — either Attention or Mamba for sequence mixing,
# followed by either FFN or MoE for channel mixing. Pre-LN, residuals.
# ---------------------------------------------------------------------------
class JambaBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 mixer: str = "mamba", channel: str = "ffn"):
        super().__init__()
        assert mixer in ("mamba", "attention")
        assert channel in ("ffn", "moe")
        self.mixer_type = mixer
        self.channel_type = channel
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if mixer == "mamba":
            self.mixer = MambaBlock(d_model)
        else:
            self.mixer = CausalMultiHeadAttention(d_model, num_heads)
        if channel == "ffn":
            self.channel = FeedForward(d_model, d_ff)
        else:
            self.channel = TopKMoE(d_model, d_ff)

    def forward(self, x, past_kv=None):
        """Parallel forward over a full sequence.

        For mamba: past_kv is unused, returns (out, None)
        For attention: returns (out, (K, V) cache)
        """
        if self.mixer_type == "attention":
            h, new_kv = self.mixer(self.norm1(x), past_kv=past_kv)
            x = x + h
        else:
            x = x + self.mixer(self.norm1(x))
            new_kv = None
        x = x + self.channel(self.norm2(x))
        return x, new_kv

    @torch.no_grad()
    def step(self, x_t, ssm_state=None, conv_state=None, past_kv=None):
        """Single-token recurrent update.

        Args:
            x_t: (B, D)  — residual stream at this token
            ssm_state, conv_state: only used for Mamba layers
            past_kv: only used for Attention layers
        Returns: (out_t (B, D), new_ssm, new_conv, new_kv)
        """
        if self.mixer_type == "attention":
            h, new_kv = self.mixer(self.norm1(x_t).unsqueeze(1), past_kv=past_kv)
            x_t = x_t + h.squeeze(1)
            new_ssm, new_conv = ssm_state, conv_state
        else:
            h, new_ssm, new_conv = self.mixer.step(
                self.norm1(x_t), ssm_state, conv_state
            )
            x_t = x_t + h
            new_kv = past_kv
        # channel mixing applies elementwise per token, so the parallel form
        # already works on a (B, 1, D) sequence — reuse it
        ch_out = self.channel(self.norm2(x_t).unsqueeze(1)).squeeze(1)
        x_t = x_t + ch_out
        return x_t, new_ssm, new_conv, new_kv


# ---------------------------------------------------------------------------
# The model: Mamba-dominant stack with one attention layer + MoE every 2 layers
# ---------------------------------------------------------------------------
class MiniJamba(nn.Module):
    """Tiny Jamba-style model.

    Layer schedule (real Jamba is 1:7 + e=2; we shrink to 1:3 + e=2 for demo):
        layer 0: Mamba + FFN
        layer 1: Mamba + MoE
        layer 2: Attention + FFN
        layer 3: Mamba + MoE
    """

    DEFAULT_SCHEDULE = [
        ("mamba", "ffn"),
        ("mamba", "moe"),
        ("attention", "ffn"),
        ("mamba", "moe"),
    ]

    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=128,
                 schedule=None, max_len=512):
        super().__init__()
        self.schedule = schedule or list(self.DEFAULT_SCHEDULE)
        self.d_model = d_model
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)

        # We need positional info only for the (rare) attention layers; use
        # a simple sin/cos PE applied at input (Mamba layers ignore it anyway).
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

        self.blocks = nn.ModuleList(
            [JambaBlock(d_model, num_heads, d_ff, mixer=m, channel=c)
             for m, c in self.schedule]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def _embed(self, idx: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        x = self.token_emb(idx)
        L = idx.size(1)
        return x + self.pe[start_pos:start_pos + L]

    def forward(self, idx: torch.Tensor):
        """Parallel forward over full sequence. Returns logits (B, L, V)."""
        x = self._embed(idx, start_pos=0)
        for block in self.blocks:
            x, _ = block(x)
        return self.lm_head(self.final_norm(x))

    # Mixed cache: per-layer (ssm_state, conv_state, kv) — None where unused.
    def empty_cache(self, batch_size, device):
        caches = []
        for i, block in enumerate(self.blocks):
            if block.mixer_type == "mamba":
                m = block.mixer
                caches.append((
                    torch.zeros(batch_size, m.d_inner, m.ssm.N, device=device),
                    torch.zeros(batch_size, m.d_inner, m.d_conv, device=device),
                    None,
                ))
            else:
                caches.append((None, None, None))  # KV cache starts empty
        return caches

    @torch.no_grad()
    def step(self, idx_t: torch.Tensor, caches, position: int):
        """Decode one token. idx_t: (B,) -> logits (B, V)."""
        x = self.token_emb(idx_t) + self.pe[position]
        new_caches = []
        for i, block in enumerate(self.blocks):
            ssm_s, conv_s, kv = caches[i]
            x, new_ssm, new_conv, new_kv = block.step(x, ssm_s, conv_s, kv)
            new_caches.append((new_ssm, new_conv, new_kv))
        x = self.final_norm(x)
        return self.lm_head(x), new_caches

    @torch.no_grad()
    def prefill(self, idx: torch.Tensor):
        """Step-by-step prefill (slow but simple and exactly consistent)."""
        B, L = idx.shape
        caches = self.empty_cache(B, idx.device)
        logits = None
        for t in range(L):
            logits, caches = self.step(idx[:, t], caches, position=t)
        return logits, caches


@torch.no_grad()
def generate(model, start_ids, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    logits, caches = model.prefill(start_ids)
    out = start_ids
    pos = start_ids.size(1)
    for _ in range(max_new_tokens):
        last = logits / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(last, top_k)
            last[last < v[:, [-1]]] = float("-inf")
        probs = F.softmax(last, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_id], dim=1)
        logits, caches = model.step(next_id.squeeze(-1), caches, position=pos)
        pos += 1
    return out


def get_batch(data, batch_size, seq_len):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x, y


def train(model, data, steps=600, batch_size=16, seq_len=32, lr=3e-3):
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


def main():
    torch.manual_seed(0)

    chars = sorted(set(CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab_size = len(chars)
    data = torch.tensor([stoi[c] for c in CORPUS], dtype=torch.long)

    print(f"corpus: {len(CORPUS)} chars, {vocab_size} unique tokens")
    model = MiniJamba(
        vocab_size=vocab_size,
        d_model=64, num_heads=4, d_ff=128,
    )

    # Parameter accounting: total vs active (per-token)
    total = sum(p.numel() for p in model.parameters())
    total -= model.lm_head.weight.numel()  # tied
    # Active = total - (E-K) * (per-expert size) per MoE layer
    active = total
    for block in model.blocks:
        if block.channel_type == "moe":
            moe = block.channel
            per_expert = sum(p.numel() for p in moe.experts[0].parameters())
            # K of E experts used per token
            active -= (moe.E - moe.K) * per_expert
    print(f"model: total={total:,} params, "
          f"active per token={active:,} ({100*active/total:.0f}%)")
    print(f"layer schedule: {model.schedule}")

    # ---- Train ----
    print("\n--- training ---")
    train(model, data, steps=600, batch_size=16, seq_len=32, lr=3e-3)

    # ---- Generate ----
    print("\n--- generation (mixed KV cache + SSM state) ---")
    prompt = "Shall I "
    start_ids = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long)
    out = generate(model, start_ids, max_new_tokens=200, temperature=0.8, top_k=10)
    print("".join(itos[i.item()] for i in out[0]))

    # ---- Sanity check: parallel forward vs recurrent step ----
    print("\n--- sanity check: parallel forward vs step-by-step ---")
    model.eval()
    with torch.no_grad():
        idx = out[:, :30]
        logits_parallel = model(idx)

        caches = model.empty_cache(1, idx.device)
        step_logits = []
        for t in range(30):
            lt, caches = model.step(idx[:, t], caches, position=t)
            step_logits.append(lt)
        logits_step = torch.stack(step_logits, dim=1)

        diff = (logits_step - logits_parallel).abs().max().item()
        print(f"max |logits_step - logits_parallel|: {diff:.2e}")
        print("✓ all 4 layer types agree (Mamba scan == step, Attention parallel == cached)"
              if diff < 1e-4
              else "✗ MISMATCH — investigate!")


if __name__ == "__main__":
    main()
