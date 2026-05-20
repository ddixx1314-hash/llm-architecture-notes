"""Runnable examples for the foundational neural-network notes.

Covers:
    - Section 0: MLP + training loop
    - Section 1: CNN on synthetic images
    - Section 7: AdamW, train/eval, clipping
    - Section 8: initialization
    - Section 11: ResNet-style block + depthwise separable conv

Run:
    python neural-networks/scripts/basic_models.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Section 0: MLP
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_blobs(n: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    half = n // 2
    x0 = torch.randn(half, 2) * 0.45 + torch.tensor([-1.0, -1.0])
    x1 = torch.randn(n - half, 2) * 0.45 + torch.tensor([1.0, 1.0])
    x = torch.cat([x0, x1], dim=0)
    y = torch.cat(
        [
            torch.zeros(half, dtype=torch.long),
            torch.ones(n - half, dtype=torch.long),
        ]
    )
    perm = torch.randperm(n)
    return x[perm], y[perm]


def train_mlp() -> None:
    x, y = make_blobs()
    model = MLP()
    model.apply(init_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

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
        print(f"[MLP] loss={F.cross_entropy(logits, y).item():.4f}, acc={accuracy(logits, y):.3f}")


# ---------------------------------------------------------------------------
# Section 1: CNN
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x.flatten(1))


def make_bar_images(n: int = 256, size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(n, 1, size, size) * 0.1
    y = torch.randint(0, 2, (n,))
    for i, label in enumerate(y.tolist()):
        if label == 0:
            col = torch.randint(3, size - 3, ()).item()
            x[i, 0, :, col - 1 : col + 2] += 1.0
        else:
            row = torch.randint(3, size - 3, ()).item()
            x[i, 0, row - 1 : row + 2, :] += 1.0
    return x, y


def train_cnn() -> None:
    x, y = make_bar_images()
    model = SmallCNN()
    model.apply(init_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    model.train()
    for step in range(60):
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x)
        print(f"[CNN] loss={F.cross_entropy(logits, y).item():.4f}, acc={accuracy(logits, y):.3f}")


# ---------------------------------------------------------------------------
# Section 11: modern CNN blocks
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + residual)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


def check_modern_cnn_blocks() -> None:
    x = torch.randn(4, 8, 16, 16)
    block = BasicBlock(8, 16, stride=2)
    depthwise = DepthwiseSeparableConv(16, 24)
    with torch.no_grad():
        y = block(x)
        z = depthwise(y)
        pooled = z.mean(dim=(2, 3))
    print(f"[Modern CNN] block={tuple(y.shape)}, depthwise={tuple(z.shape)}, gap={tuple(pooled.shape)}")


def main() -> None:
    set_seed(0)
    train_mlp()
    train_cnn()
    check_modern_cnn_blocks()


if __name__ == "__main__":
    main()
