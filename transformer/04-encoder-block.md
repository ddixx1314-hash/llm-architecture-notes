# 第 4 节:完整 Encoder Block

> 本节目标:把前面学过的 Embedding、Positional Encoding、Multi-Head Attention 组装成 Transformer Encoder 的基本层,理解残差连接、LayerNorm 和 FFN 的作用。

---

## 4.1 从零件到模块

到目前为止,我们已经有了三个核心零件:

1. Token Embedding:把 token id 变成向量
2. Positional Encoding:把位置信息加入向量
3. Multi-Head Attention:让每个位置和所有位置交互

但 Transformer 的 Encoder Block 不是只有 Attention。原论文中的每一层 Encoder 包含两个子层:

$$
\text{Multi-Head Self-Attention}
$$

$$
\text{Feed-Forward Network}
$$

并且每个子层外面都包了一层:

$$
\text{Residual Connection} + \text{LayerNorm}
$$

---

## 4.2 Encoder Block 的整体结构

论文中的 Post-LN 写法是:

$$
z = \text{LayerNorm}(x + \text{MultiHeadAttention}(x))
$$

$$
y = \text{LayerNorm}(z + \text{FFN}(z))
$$

也就是:

```
x
│
├── Multi-Head Self-Attention
│
├── Add & Norm
│
├── Feed-Forward Network
│
└── Add & Norm
    ↓
output
```

输出形状和输入形状完全一样:

$$
x, y \in \mathbb{R}^{n \times d_{\text{model}}}
$$

这保证了 Encoder Block 可以堆叠很多层:

$$
X^{(0)} \rightarrow X^{(1)} \rightarrow \cdots \rightarrow X^{(N)}
$$

<div align="center"><img src="images/transformer-architecture.png" width="70%"></div>

图:原始 Transformer 的 Encoder-Decoder 总体结构。左侧是 Encoder stack,右侧是 Decoder stack。来源:Vaswani et al., 2017, *Attention Is All You Need*, Figure 1。

---

## 4.3 为什么需要残差连接?

如果没有残差,每一层都只能输出:

$$
y = F(x)
$$

多层堆叠后:

$$
y = F_N(F_{N-1}(\cdots F_1(x)))
$$

这会带来两个问题:

1. **信息容易被覆盖**:如果某一层暂时学得不好,原始输入信息可能直接丢掉
2. **梯度路径太长**:反向传播要穿过每一个非线性模块,深层网络很难训练

残差连接改成:

$$
y = x + F(x)
$$

这相当于给信息和梯度开了一条"直通路径":

$$
\frac{\partial y}{\partial x} = I + \frac{\partial F(x)}{\partial x}
$$

即使 $F$ 的梯度很小,还有一个恒等项 $I$ 可以让梯度继续往前传。

直觉上,每一层不需要从零生成新表示,只需要学习一个**增量修正**:

$$
\text{新表示} = \text{旧表示} + \text{这一层学到的变化}
$$

---

## 4.4 为什么需要 LayerNorm?

深层网络训练时,每一层输出的数值分布会不断变化。如果某层输出突然变得很大或很小,下一层就会难以适应。

LayerNorm 的作用是:对每个 token 的特征维度做归一化。

设一个 token 的隐藏向量为:

$$
x \in \mathbb{R}^{d_{\text{model}}}
$$

先计算均值和方差:

$$
\mu = \frac{1}{d_{\text{model}}}\sum_{i=1}^{d_{\text{model}}} x_i
$$

$$
\sigma^2 = \frac{1}{d_{\text{model}}}\sum_{i=1}^{d_{\text{model}}}(x_i - \mu)^2
$$

再归一化:

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

最后加上可学习的缩放和平移:

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

其中 $\gamma,\beta \in \mathbb{R}^{d_{\text{model}}}$ 是可学习参数。

⚠️ 注意:LayerNorm 是**对每个 token 自己的特征维度**做归一化,不是在 batch 维度上归一化。这一点对可变长度序列和自回归生成非常重要。

---

## 4.5 Post-LN 和 Pre-LN

原始 Transformer 使用的是 Post-LN:

$$
y = \text{LayerNorm}(x + F(x))
$$

现代大模型更常用 Pre-LN:

$$
y = x + F(\text{LayerNorm}(x))
$$

### Post-LN

```
x → F → Add → LayerNorm
```

优点:和原论文一致,输出分布更规整。

缺点:很深时训练不稳定,因为残差路径上仍然会经过 LayerNorm。

### Pre-LN

```
x → LayerNorm → F → Add
```

优点:残差路径更"干净",梯度可以沿着 $x$ 直接穿过很多层,深层模型更容易训练。

缺点:最后输出通常还需要再接一个最终 LayerNorm。

📌 这就是为什么 GPT、LLaMA 等现代 Decoder-only 模型通常采用 Pre-LN 结构。

---

## 4.6 FFN:每个位置独立的非线性变换

Encoder Block 的第二个子层是 Feed-Forward Network:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

其中:

$$
W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}
$$

$$
W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}
$$

论文中:

$$
d_{\text{model}} = 512,\quad d_{\text{ff}} = 2048
$$

也就是先把维度放大 4 倍,经过 ReLU,再压回原维度。

---

## 4.7 Attention 和 FFN 分工不同

Attention 负责**跨位置混合信息**:

$$
y_i = \sum_j \alpha_{ij} v_j
$$

也就是说,第 $i$ 个位置的新表示来自所有位置的加权组合。

FFN 负责**在每个位置内部做特征变换**:

$$
y_i = \text{FFN}(x_i)
$$

它不看其他 token,但提供非线性和更大的中间维度,让每个 token 的表示可以被重新编码。

可以粗略理解为:

| 模块 | 作用 |
|------|------|
| Attention | token 之间交流 |
| FFN | token 内部思考 |
| Residual | 保留旧信息,稳定梯度 |
| LayerNorm | 稳定数值分布 |

---

## 4.8 Dropout 放在哪里?

原论文在多个位置使用 Dropout:

1. Attention 权重上
2. 子层输出加回残差之前
3. Embedding + Positional Encoding 之后

Encoder Block 中常见写法:

$$
x = \text{LayerNorm}(x + \text{Dropout}(\text{MHA}(x)))
$$

$$
x = \text{LayerNorm}(x + \text{Dropout}(\text{FFN}(x)))
$$

Dropout 的目的不是改变模型结构,而是训练时正则化,减少过拟合。推理时 Dropout 会关闭。

---

## 4.9 PyTorch 实现:Post-LN 版本

```python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlockPostLN(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, n, d_model)
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x
```

这和原论文结构一致。

---

## 4.10 PyTorch 实现:Pre-LN 版本

现代实现更常见:

```python
class EncoderBlockPreLN(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, n, d_model)
        x = x + self.dropout(self.self_attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
```

如果堆叠很多 Pre-LN block,通常在所有层结束后再加一个:

```python
self.final_norm = nn.LayerNorm(d_model)
```

---

## 4.11 完整 Encoder

把 Embedding 和多个 EncoderBlock 接起来:

```python
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        max_len=5000,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderBlockPostLN(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, token_ids):
        # token_ids: (batch, n)
        x = self.embedding(token_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        return x  # (batch, n, d_model)
```

这就是 Transformer Encoder 的主体。

---

## 4.12 论文符号对照

《Attention Is All You Need》3.1 节描述 Encoder:

- Encoder 由 $N = 6$ 个相同层堆叠而成
- 每层有两个子层:Multi-Head Self-Attention 和 Position-wise Feed-Forward Network
- 每个子层使用 residual connection,再接 layer normalization
- 所有子层和 embedding 的输出维度都是 $d_{\text{model}} = 512$

其中 "position-wise" 的意思是:FFN 对每个位置使用**同一组参数**,但不同位置之间互不混合。

---

## 4.13 本节核心要点

1. Encoder Block = Self-Attention + FFN,每个子层外面包 Add & Norm
2. 残差连接让信息和梯度有直通路径,深层网络更容易训练
3. LayerNorm 对每个 token 的特征维归一化,稳定数值分布
4. FFN 是逐位置的非线性变换,通常把维度放大 4 倍再压回
5. 原论文是 Post-LN,现代大模型更常用 Pre-LN
6. 每个 Block 输入输出形状一致,所以可以堆叠多层

---

## 4.14 思考题

1. 如果去掉残差连接,Encoder 堆叠 24 层会遇到什么训练问题?
2. 为什么 FFN 要先升维到 $d_{\text{ff}}$,而不是直接做 $d_{\text{model}} \rightarrow d_{\text{model}}$?
3. LayerNorm 为什么适合序列模型?如果用 BatchNorm 会有什么问题?
4. Pre-LN 的残差路径为什么比 Post-LN 更利于深层训练?

---

## 4.15 下一节预告

Encoder 已经能把一整段输入编码成上下文化表示。下一节进入 Decoder:

- 为什么生成任务不能看到未来 token?
- causal mask 如何用上三角矩阵实现?
- Decoder 的 self-attention 和 cross-attention 有什么区别?
- Encoder-Decoder 架构如何用于机器翻译?

→ [第 5 节:Decoder + Masked Attention](05-decoder-masked-attention.md)
