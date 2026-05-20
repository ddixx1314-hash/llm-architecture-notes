# 第 6 节:归一化与正则化

> 本节目标:理解 BatchNorm、LayerNorm、Dropout、权重衰减和残差连接分别解决什么训练问题,以及它们为什么在现代网络中几乎无处不在。

---

## 6.1 为什么深层网络难训练?

网络变深后,常见问题包括:

- 激活分布不断漂移,后面的层难以适应
- 梯度在多层链式相乘中消失或爆炸
- 参数很多,容易过拟合
- 优化路径崎岖,训练不稳定

归一化、正则化和残差连接就是为了让训练更稳、更深、更不容易过拟合。

---

## 6.2 BatchNorm

BatchNorm 对一个 mini-batch 的激活做标准化。

对某个特征维度:

$$
\mu_B = \frac{1}{B}\sum_{i=1}^{B} x_i
$$

$$
\sigma_B^2 = \frac{1}{B}\sum_{i=1}^{B}(x_i-\mu_B)^2
$$

$$
\hat{x}_i = \frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

再学习缩放和平移:

$$
y_i = \gamma \hat{x}_i + \beta
$$

BatchNorm 常用于 CNN,因为图像训练通常 batch 较大,统计量比较稳定。

---

## 6.3 LayerNorm

LayerNorm 对单个样本的特征维度做归一化:

$$
\mu = \frac{1}{D}\sum_{j=1}^{D} x_j
$$

$$
\sigma^2 = \frac{1}{D}\sum_{j=1}^{D}(x_j-\mu)^2
$$

$$
\text{LayerNorm}(x)_j =
\gamma_j \frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta_j
$$

它不依赖 batch 统计,所以特别适合:

- NLP
- Transformer
- batch size 很小或序列长度变化大的任务

Transformer 里的 Add & Norm 基本就是残差连接 + LayerNorm。

---

## 6.4 Dropout

Dropout 训练时随机把一部分激活置 0:

$$
\tilde{h} = m \odot h,\quad m_i \sim \text{Bernoulli}(1-p)
$$

直觉:

> 不让网络过度依赖某几个神经元,迫使表示更分散、更鲁棒。

训练时启用 dropout,推理时关闭 dropout。

PyTorch 中 `model.train()` 和 `model.eval()` 会影响 Dropout/BatchNorm 行为。

---

## 6.5 Weight Decay

权重衰减在 loss 中加入参数惩罚:

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{task}}
+
\lambda \|\theta\|_2^2
$$

它鼓励参数不要变得过大,可以降低过拟合风险。

AdamW 把 weight decay 和 Adam 的梯度更新解耦,是现代深度学习里非常常见的优化器。

---

## 6.6 残差连接

残差连接写成:

$$
y = x + F(x)
$$

它的好处是提供一条直接路径。即使 $F(x)$ 一开始学不好,网络也至少可以近似恒等映射。

反向传播时:

$$
\frac{\partial y}{\partial x}
=
I + \frac{\partial F(x)}{\partial x}
$$

这个 $I$ 提供了更直接的梯度通道,让深层网络更容易训练。

---

## 6.7 常见组合

| 场景 | 常见组合 |
|------|----------|
| CNN | Conv + BatchNorm + ReLU |
| Transformer | Residual + LayerNorm + Attention/FFN |
| 小数据分类 | Dropout + Weight Decay |
| 很深的网络 | Residual + Normalization |

---

## 6.8 PyTorch 代码骨架

```python
import torch
import torch.nn as nn

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        return x + self.ffn(self.norm(x))

block = ResidualMLPBlock(dim=256)
x = torch.randn(32, 256)
y = block(x)
```

---

## 6.9 本节核心要点

1. BatchNorm 使用 batch 统计,常见于 CNN。
2. LayerNorm 使用单样本特征统计,常见于 Transformer/NLP。
3. Dropout 通过随机失活降低过拟合。
4. Weight decay 鼓励参数更小,提升泛化。
5. 残差连接提供直接信息路径和梯度路径。
6. 深层现代网络通常把归一化、残差、正则化组合使用。

## 思考题

<details>
<summary>为什么 Transformer 更常用 LayerNorm 而不是 BatchNorm?</summary>

Transformer 常处理变长序列,并且训练/推理场景中的 batch 统计可能不稳定。LayerNorm 对每个 token 的特征维度单独归一化,不依赖 batch size,因此更适合序列模型。

</details>
