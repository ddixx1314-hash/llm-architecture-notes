# 第 4 节:Autoencoder 自编码器

> 本节目标:理解自编码器如何通过"压缩再重构"学习表示,以及 bottleneck、denoising、VAE 的基本直觉。

---

## 4.1 自编码器在学什么?

Autoencoder 的目标不是直接预测人工标签,而是重构输入本身:

$$
\hat{x} = g_\phi(f_\theta(x))
$$

其中:

- Encoder $f_\theta$:把输入压缩成 latent representation
- Decoder $g_\phi$:从 latent representation 重构输入

训练目标:

$$
\min_{\theta,\phi} \mathcal{L}(\hat{x}, x)
$$

如果输入是连续值,常用 MSE:

$$
\mathcal{L} = \|x - \hat{x}\|_2^2
$$

---

## 4.2 Bottleneck 为什么重要?

如果 latent 维度很大,模型可能只是学会复制输入。

所以自编码器常设置一个 bottleneck:

$$
x \in \mathbb{R}^{D}
\rightarrow
z \in \mathbb{R}^{d}
\rightarrow
\hat{x} \in \mathbb{R}^{D}
$$

其中 $d < D$。

这迫使模型把输入压缩成更有用的表示 $z$。

直觉:

| 输入 | bottleneck 可能保留 |
|------|-------------------|
| 手写数字图片 | 数字形状、笔画方向 |
| 人脸图片 | 姿态、表情、光照 |
| 用户行为 | 兴趣、偏好、活跃度 |

---

## 4.3 Linear Autoencoder 和 PCA 的关系

如果:

- encoder/decoder 都是线性的
- loss 是 MSE
- latent 维度小于输入维度

那么 linear autoencoder 学到的子空间和 PCA 有密切关系。

这说明自编码器可以看成是"可非线性化、可深度化、可任务定制的表示压缩"。

---

## 4.4 Denoising Autoencoder

Denoising autoencoder 不直接输入干净样本,而是输入加噪版本:

$$
\tilde{x} = x + \epsilon
$$

训练目标仍然是重构干净 $x$:

$$
\hat{x} = g_\phi(f_\theta(\tilde{x}))
$$

$$
\mathcal{L} = \|x - \hat{x}\|_2^2
$$

这会迫使模型学习稳定结构,而不是记住像素级噪声。

---

## 4.5 VAE 的一句话直觉

VAE(Variational Autoencoder)也是 encoder-decoder,但它不把输入编码成一个固定向量,而是编码成一个分布:

$$
q_\theta(z \mid x) = \mathcal{N}(\mu_\theta(x), \sigma_\theta^2(x))
$$

然后从这个分布采样 $z$,再解码。

VAE 的 loss 通常包括:

$$
\mathcal{L} =
\mathcal{L}_{\text{recon}}
+
\beta \, D_{\text{KL}}(q_\theta(z \mid x) \| p(z))
$$

直觉:

- 重构项:生成结果要像原输入
- KL 项:latent 空间要规整,方便采样生成

---

## 4.6 PyTorch 代码骨架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

model = Autoencoder()
x = torch.rand(32, 784)
x_hat, z = model(x)
loss = F.mse_loss(x_hat, x)
```

---

## 4.7 Autoencoder 能做什么?

| 用途 | 做法 |
|------|------|
| 降维可视化 | 取 latent $z$ 再画图 |
| 异常检测 | 重构误差大可能是异常 |
| 去噪 | 输入噪声样本,输出干净样本 |
| 预训练 | 先学表示,再接下游任务 |
| 生成模型基础 | VAE 是经典生成模型之一 |

---

## 4.8 本节核心要点

1. Autoencoder 通过重构输入学习表示。
2. Encoder 压缩输入,decoder 从 latent 表示恢复输入。
3. Bottleneck 迫使模型保留最有用的信息。
4. Denoising autoencoder 学习从噪声中恢复稳定结构。
5. VAE 把 latent 表示变成分布,使采样生成更自然。
6. 自编码器是无监督/自监督表示学习的重要基础。

## 思考题

<details>
<summary>如果 latent 维度比输入维度还大,自编码器一定没用吗?</summary>

不一定,但更容易学成简单复制。可以通过加噪、稀疏约束、dropout、正则化等方式限制模型,让它仍然必须学习稳定结构。

</details>
