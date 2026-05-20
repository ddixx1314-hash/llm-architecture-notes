"""Runnable sequence-model examples.

Covers:
    - Section 2: RNN
    - Section 3: LSTM / GRU
    - Section 9: Embedding + weight tying
    - Section 10: additive attention as a bridge to Transformer attention

Run:
    python neural-networks/scripts/sequence_models.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 1) -> None:
    torch.manual_seed(seed)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


def make_token_sequences(
    batch_size: int = 256,
    seq_len: int = 12,
    vocab_size: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Label is whether the final token is in the upper half of the vocab.

    This is intentionally easy so all recurrent cells converge quickly in a
    tiny CPU demo. The point is shape and data flow, not benchmark accuracy.
    """

    x = torch.randint(1, vocab_size, (batch_size, seq_len))
    y = (x[:, -1] >= vocab_size // 2).long()
    return x, y


# ---------------------------------------------------------------------------
# Sections 2-3: RNN / LSTM / GRU
# ---------------------------------------------------------------------------
class SequenceClassifier(nn.Module):
    def __init__(
        self,
        cell: str,
        vocab_size: int = 30,
        emb_dim: int = 16,
        hidden_dim: int = 32,
        num_classes: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(emb_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        outputs, _ = self.rnn(x)
        return self.head(outputs[:, -1, :])


def train_sequence_classifier(cell: str) -> None:
    x, y = make_token_sequences()
    model = SequenceClassifier(cell)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    model.train()
    for step in range(80):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x)
        print(f"[{cell.upper()}] loss={F.cross_entropy(logits, y).item():.4f}, acc={accuracy(logits, y):.3f}")


# ---------------------------------------------------------------------------
# Section 9: Embedding + weight tying
# ---------------------------------------------------------------------------
class TinyEmbeddingLM(nn.Module):
    def __init__(self, vocab_size: int = 30, d_model: int = 24, max_len: int = 16):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        return self.lm_head(self.norm(x))


def check_embedding_lm() -> None:
    model = TinyEmbeddingLM()
    input_ids = torch.randint(1, 30, (4, 8))
    labels = torch.randint(0, 30, (4, 8))
    logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    same_storage = model.lm_head.weight.data_ptr() == model.token_emb.weight.data_ptr()
    print(f"[Embedding LM] logits={tuple(logits.shape)}, loss={loss.item():.4f}, tied={same_storage}")


# ---------------------------------------------------------------------------
# Section 10: additive attention
# ---------------------------------------------------------------------------
class AdditiveAttention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        self.w_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.w_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        enc_outputs: torch.Tensor,
        dec_state: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.v(
            torch.tanh(self.w_enc(enc_outputs) + self.w_dec(dec_state)[:, None, :])
        ).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        context = weights[:, None, :] @ enc_outputs
        return context.squeeze(1), weights


def check_additive_attention() -> None:
    enc_outputs = torch.randn(3, 5, 32)
    dec_state = torch.randn(3, 24)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ]
    )
    attn = AdditiveAttention(enc_dim=32, dec_dim=24, attn_dim=16)
    context, weights = attn(enc_outputs, dec_state, mask)
    print(
        "[Additive Attention] "
        f"context={tuple(context.shape)}, weights={tuple(weights.shape)}, "
        f"row_sums={weights.sum(dim=-1).round(decimals=4).tolist()}"
    )


def main() -> None:
    set_seed(1)
    for cell in ["rnn", "lstm", "gru"]:
        train_sequence_classifier(cell)
    check_embedding_lm()
    check_additive_attention()


if __name__ == "__main__":
    main()
