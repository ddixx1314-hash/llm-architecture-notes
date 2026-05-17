"""Mini-GPT: a from-scratch decoder-only Transformer that ties together
every concept from the learning notes (sections 1–7).

Each component is annotated with the section that introduces it.

Run directly:
    python3 scripts/mini_gpt.py

What this demo does:
    1. Builds a tiny decoder-only Transformer (~50K params).
    2. Trains it on Shakespeare's Sonnet 18 at the character level.
    3. Watches train loss drop from ~3.4 (random) to <1.5.
    4. Generates text autoregressively, with KV cache.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Section 3: Positional Encoding (sin/cos, non-learned)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        # 1 / 10000^(2i/d_model), implemented as exp(-2i * log(10000) / d)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        # x: (B, n, d_model). When generating with KV cache, start_pos
        # is the cached length so PE keeps advancing.
        n = x.size(1)
        if start_pos + n > self.pe.size(0):
            raise ValueError(
                f"position {start_pos + n} exceeds PE max_len {self.pe.size(0)}; "
                "increase `max_len` when constructing the model"
            )
        return x + self.pe[start_pos : start_pos + n]


# ---------------------------------------------------------------------------
# Sections 1–2 + 5: Causal Multi-Head Self-Attention (with optional KV cache)
# ---------------------------------------------------------------------------
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model

        # Fused Q/K/V projection — equivalent to three separate Linears
        # (Section 2.6: one big matmul + reshape, no per-head loop).
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, n, _ = x.shape
        qkv = self.W_qkv(x)  # (B, n, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, n, d_model) -> (B, h, n, d_k)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, n, self.h, self.d_k).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # Section 6.8–6.9: prepend cached K/V, then write back the
        # concatenated tensor as the new cache.
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v)

        L_q, L_k = q.size(2), k.size(2)

        # Section 1.5–1.6: scaled dot product + softmax
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Section 5.4: causal mask (upper triangular -> -inf).
        # During decode with cache, L_q == 1 — the single new query can
        # attend to all cached keys, so no mask needed.
        if L_q > 1:
            mask = torch.ones(L_q, L_k, dtype=torch.bool, device=x.device).triu(
                diagonal=L_k - L_q + 1
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, h, L_q, d_k)

        # Merge heads back: (B, h, L_q, d_k) -> (B, L_q, d_model)
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        return self.W_o(out), new_kv


# ---------------------------------------------------------------------------
# Section 4.6: Position-wise Feed-Forward Network
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Sections 4–5: Pre-LN Decoder Block (causal self-attention + FFN)
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Pre-LN: residual path stays clean (Section 4.5)
        attn_out, new_kv = self.attn(self.norm1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


# ---------------------------------------------------------------------------
# The model itself
# ---------------------------------------------------------------------------
class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: Optional[int] = None,
        max_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, num_heads, d_ff or 4 * d_model)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: lm_head shares weights with token_emb
        # (common trick, slightly outside the notes but standard in nanoGPT)
        self.lm_head.weight = self.token_emb.weight

    def forward(
        self,
        idx: torch.Tensor,
        past_kvs: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        # Section 3.12: scale embedding by sqrt(d_model)
        x = self.token_emb(idx) * math.sqrt(self.d_model)
        start_pos = past_kvs[0][0].size(2) if past_kvs is not None else 0
        x = self.pos_enc(x, start_pos=start_pos)

        new_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            past = past_kvs[i] if past_kvs is not None else None
            x, kv = block(x, past_kv=past)
            new_kvs.append(kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_kvs


# ---------------------------------------------------------------------------
# Section 6: Autoregressive generation with KV cache
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(
    model: MiniGPT,
    start_ids: torch.Tensor,  # (1, prompt_len)
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    model.eval()
    # Prefill: process the whole prompt once, get initial cache
    logits, kvs = model(start_ids)
    out = start_ids
    for _ in range(max_new_tokens):
        # Decode: feed only the last predicted token, reuse cache
        last_logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(last_logits, top_k)
            last_logits[last_logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(last_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_id], dim=1)
        logits, kvs = model(next_id, past_kvs=kvs)
    return out


# ---------------------------------------------------------------------------
# Section 6: Training loop (next-token prediction + teacher forcing)
# ---------------------------------------------------------------------------
def get_batch(data: torch.Tensor, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Sample random windows of length seq_len+1, split into input/label
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])  # right-shifted (Section 6.4)
    return x, y


def train(
    model: MiniGPT,
    data: torch.Tensor,
    steps: int = 800,
    batch_size: int = 32,
    seq_len: int = 32,
    lr: float = 3e-3,
) -> None:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    losses: list[float] = []
    for step in range(1, steps + 1):
        x, y = get_batch(data, batch_size, seq_len)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
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
# Demo on Shakespeare's Sonnet 18 (public domain, ~600 chars, ~30 unique)
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

    # Character-level tokenizer (Section 6: each char is a "token")
    chars = sorted(set(CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab_size = len(chars)
    data = torch.tensor([stoi[c] for c in CORPUS], dtype=torch.long)

    print(f"corpus: {len(CORPUS)} chars, {vocab_size} unique tokens")
    print(f"sample tokens: {list(chars[:20])}")

    # Small model — small enough to train in seconds on CPU
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_len=512,  # must exceed prompt_len + max_new_tokens
    )
    n_params = sum(p.numel() for p in model.parameters())
    # Subtract tied lm_head weight to avoid double counting
    n_params -= model.lm_head.weight.numel() if model.lm_head.weight is model.token_emb.weight else 0
    print(f"model: {n_params:,} parameters")

    # ---- Train ----
    print("\n--- training ---")
    train(model, data, steps=800, batch_size=32, seq_len=32, lr=3e-3)

    # ---- Generate ----
    print("\n--- generation (with KV cache) ---")
    prompt = "Shall I "
    start_ids = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long)
    out = generate(model, start_ids, max_new_tokens=200, temperature=0.8, top_k=10)
    text = "".join(itos[i.item()] for i in out[0])
    print(text)

    # ---- Verify KV cache matches no-cache forward ----
    print("\n--- sanity check: KV cache vs full forward ---")
    model.eval()
    with torch.no_grad():
        full_input = out[:, :30]
        logits_full, _ = model(full_input)

        # Step-by-step with cache
        logits_step, kvs = model(full_input[:, :1])
        for t in range(1, 30):
            logits_step, kvs = model(full_input[:, t : t + 1], past_kvs=kvs)

        diff = (logits_step[:, -1] - logits_full[:, -1]).abs().max().item()
        print(f"max |logits_step - logits_full| at last position: {diff:.2e}")
        print("✓ cache implementation is correct" if diff < 1e-4 else "✗ cache mismatch")


if __name__ == "__main__":
    main()
