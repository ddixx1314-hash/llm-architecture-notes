"""Runnable representation-learning examples.

Covers:
    - Section 4: Autoencoder
    - Section 5: GCN-style graph neural network

Run:
    python neural-networks/scripts/representation_models.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 2) -> None:
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Section 4: Autoencoder
# ---------------------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 20, latent_dim: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def make_low_rank_data(n: int = 256, input_dim: int = 20, latent_dim: int = 3) -> torch.Tensor:
    z = torch.randn(n, latent_dim)
    basis = torch.randn(latent_dim, input_dim)
    x = z @ basis + 0.05 * torch.randn(n, input_dim)
    return x


def train_autoencoder() -> None:
    x = make_low_rank_data()
    model = Autoencoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    model.train()
    for step in range(200):
        x_hat, z = model(x)
        loss = F.mse_loss(x_hat, x)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        x_hat, z = model(x)
        print(f"[Autoencoder] recon_mse={F.mse_loss(x_hat, x).item():.4f}, latent={tuple(z.shape)}")


# ---------------------------------------------------------------------------
# Section 5: GCN
# ---------------------------------------------------------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        n = adj.size(0)
        adj_hat = adj + torch.eye(n, device=adj.device)
        degree = adj_hat.sum(dim=1)
        degree_inv_sqrt = degree.pow(-0.5)
        norm = degree_inv_sqrt[:, None] * adj_hat * degree_inv_sqrt[None, :]
        return norm @ self.linear(x)


class TinyGCN(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 16, num_classes: int = 2):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gcn1(x, adj))
        return self.gcn2(h, adj)


def make_two_cluster_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_per_cluster = 10
    n = 2 * n_per_cluster
    x0 = torch.randn(n_per_cluster, 2) * 0.2 + torch.tensor([-1.0, 0.0])
    x1 = torch.randn(n_per_cluster, 2) * 0.2 + torch.tensor([1.0, 0.0])
    x = torch.cat([x0, x1], dim=0)
    y = torch.cat(
        [
            torch.zeros(n_per_cluster, dtype=torch.long),
            torch.ones(n_per_cluster, dtype=torch.long),
        ]
    )

    adj = torch.zeros(n, n)
    for start in [0, n_per_cluster]:
        end = start + n_per_cluster
        adj[start:end, start:end] = 1.0
    adj.fill_diagonal_(0.0)
    return x, adj, y


def train_gcn() -> None:
    x, adj, y = make_two_cluster_graph()
    model = TinyGCN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    model.train()
    for step in range(120):
        logits = model(x, adj)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x, adj)
        acc = (logits.argmax(dim=-1) == y).float().mean().item()
        print(f"[GCN] loss={F.cross_entropy(logits, y).item():.4f}, acc={acc:.3f}")


def main() -> None:
    set_seed(2)
    train_autoencoder()
    train_gcn()


if __name__ == "__main__":
    main()
