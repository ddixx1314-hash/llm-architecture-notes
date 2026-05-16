# 第 7 节:现代变种(GPT/LLaMA)

> 本节目标:理解从原始 Transformer 到现代 LLM 的几个关键变化:RMSNorm、SwiGLU、RoPE、MQA/GQA,以及它们分别解决什么问题。

---

## 7.1 原始 Transformer 不是终点

《Attention Is All You Need》给出了核心框架:

```
Embedding + PE
↓
Attention
↓
FFN
↓
Add & Norm
```

但 GPT、LLaMA、Qwen、Mistral 等现代 LLM 在工程上做了很多调整:

| 原始 Transformer | 现代 LLM 常见做法 |
|------------------|------------------|
| Encoder-Decoder | Decoder-only |
| Post-LN | Pre-LN |
| LayerNorm | RMSNorm |
| ReLU FFN | SwiGLU / GeGLU |
| sin/cos 加法位置编码 | RoPE |
| Multi-Head Attention | MQA / GQA |

这些变化不是推翻 Transformer,而是在同一个骨架上做稳定性、效率和长上下文能力的优化。

---

## 7.2 Decoder-only:为什么只保留 Decoder?

对通用语言模型来说,训练目标是:

$$
P(x_t \mid x_{<t})
$$

这天然只需要 causal self-attention。输入 prompt 和输出回答本质上都可以看成同一条 token 序列:

```
用户: 解释 Transformer
助手: Transformer 是 ...
```

所以 GPT 类模型不需要单独的 Encoder 和 Cross-Attention。

结构变成:

```
token embedding
↓
causal transformer block × N
↓
lm head
```

这让模型形式更统一:翻译、问答、代码补全、摘要,都变成"给定前文,预测后文"。

---

## 7.3 RMSNorm:只保留均方根缩放

LayerNorm 做两件事:

1. 减去均值
2. 除以标准差

公式:

$$
\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

RMSNorm 更简单:不减均值,只除以均方根:

$$
\text{RMSNorm}(x) = \gamma \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}}
$$

它去掉了 $\mu$ 和 $\beta$,计算更少,实现更简单。

直觉:深层 Transformer 最需要的是控制向量的**尺度**,不一定非要把均值移到 0。

---

## 7.4 RMSNorm 代码

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # x: (..., d_model)
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return self.weight * x
```

LLaMA 系列就使用 RMSNorm。

---

## 7.5 FFN 从 ReLU 到 SwiGLU

原始 FFN:

$$
\text{FFN}(x) = \text{ReLU}(xW_1)W_2
$$

现代 LLM 常用 SwiGLU:

$$
\text{SwiGLU}(x) = (\text{SiLU}(xW_1) \odot xW_3)W_2
$$

其中:

$$
\text{SiLU}(z) = z \cdot \sigma(z)
$$

$\odot$ 是逐元素乘法。

直觉上,SwiGLU 多了一条"门控分支":

```
x ── W1 ── SiLU ──┐
                  × ── W2
x ── W3 ──────────┘
```

门控分支让模型可以动态决定哪些特征通过,比 ReLU 的硬截断更灵活。

---

## 7.6 SwiGLU 代码

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
```

因为多了一条投影分支,实际 hidden_dim 通常不是简单的 $4d$,而会取接近 $\frac{8}{3}d$,让参数量和原始 FFN 大致接近。

---

## 7.7 RoPE:把位置编码变成旋转

第 3 节我们看到 sin/cos 编码有一个关键性质:

$$
\text{PE}_{pos+k}
$$

可以由 $\text{PE}_{pos}$ 经过只依赖 $k$ 的线性变换得到。

RoPE(Rotary Position Embedding)直接把这个"旋转"用在 Q 和 K 上:

$$
\tilde{q}_m = R_m q_m
$$

$$
\tilde{k}_n = R_n k_n
$$

然后 attention 分数变成:

$$
\tilde{q}_m^T \tilde{k}_n = q_m^T R_m^T R_n k_n = q_m^T R_{n-m} k_n
$$

注意最后只依赖相对距离:

$$
n - m
$$

这就是 RoPE 的核心:通过旋转 Q/K,让点积分数天然包含相对位置信息。

---

## 7.8 二维旋转形式

对每一对维度 $(2i,2i+1)$,RoPE 做:

$$
\begin{bmatrix}
\tilde{x}_{2i} \\
\tilde{x}_{2i+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix}
\begin{bmatrix}
x_{2i} \\
x_{2i+1}
\end{bmatrix}
$$

其中:

$$
\theta = pos \cdot \omega_i
$$

$$
\omega_i = 10000^{-2i/d}
$$

这和第 3 节 sin/cos 的频率设计是一脉相承的。

---

## 7.9 RoPE 代码骨架

```python
def rotate_half(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rope(x, cos, sin):
    # x:   (batch, heads, seq_len, head_dim)
    # cos: (1, 1, seq_len, head_dim)
    # sin: (1, 1, seq_len, head_dim)
    return x * cos + rotate_half(x) * sin
```

在 attention 中:

```python
Q = apply_rope(Q, cos, sin)
K = apply_rope(K, cos, sin)
scores = Q @ K.transpose(-2, -1)
```

注意:RoPE 通常只作用在 Q 和 K 上,不作用在 V 上。因为位置信息是用来影响"谁关注谁"的,也就是 attention score。

---

## 7.10 MQA 和 GQA:减少 KV Cache

标准 Multi-Head Attention 中,每个 query head 都有自己的 K/V:

```
Q heads: h
K heads: h
V heads: h
```

KV Cache 大小和 K/V head 数量成正比。

### MQA:Multi-Query Attention

所有 query head 共享同一组 K/V:

```
Q heads: h
K heads: 1
V heads: 1
```

优点:KV Cache 大幅减少。

缺点:表达能力可能下降。

### GQA:Grouped-Query Attention

折中:把 query heads 分组,每组共享一组 K/V:

```
Q heads: 32
KV heads: 8
每 4 个 Q head 共享 1 个 KV head
```

LLaMA 2 70B、LLaMA 3 等模型使用 GQA 来兼顾效果和推理效率。

---

## 7.11 GQA 的形状

设:

$$
h_q = 32,\quad h_{kv} = 8
$$

则:

$$
Q \in \mathbb{R}^{B \times h_q \times n \times d_k}
$$

$$
K,V \in \mathbb{R}^{B \times h_{kv} \times n \times d_k}
$$

计算 attention 前,把 K/V repeat 到 query head 数量:

```python
def repeat_kv(x, n_rep):
    # x: (batch, kv_heads, seq_len, head_dim)
    batch, kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, kv_heads * n_rep, seq_len, head_dim)
```

这样:

```python
K = repeat_kv(K, h_q // h_kv)
V = repeat_kv(V, h_q // h_kv)
```

逻辑上仍然是 $h_q$ 个 attention head,但实际缓存只存 $h_{kv}$ 份 K/V。

---

## 7.12 一个现代 Decoder Block

把这些变化放在一起,现代 LLM 的 block 常长这样:

```python
class ModernDecoderBlock(nn.Module):
    def __init__(self, d_model, attn, ffn):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.attn = attn      # causal attention + RoPE + GQA
        self.ffn = ffn        # SwiGLU

    def forward(self, x, mask=None, cache=None):
        x = x + self.attn(self.attn_norm(x), mask=mask, cache=cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

一句话总结:

```
Pre-RMSNorm + RoPE Attention(GQA) + SwiGLU FFN
```

这就是很多现代 Decoder-only LLM 的骨架。

---

## 7.13 本节核心要点

1. 现代 LLM 多数是 decoder-only,统一成 next-token prediction
2. Pre-LN/Pre-RMSNorm 比 Post-LN 更适合深层训练
3. RMSNorm 只控制向量尺度,比 LayerNorm 更轻
4. SwiGLU 用门控 FFN 替代 ReLU,表达能力更强
5. RoPE 把位置编码变成 Q/K 的旋转,让 attention 分数显式依赖相对距离
6. MQA/GQA 减少 KV head 数量,显著降低 KV Cache 显存

---

## 7.14 思考题

1. RoPE 为什么只作用在 Q/K 上,而不是 Q/K/V 都作用?
2. MQA 为什么可能损失效果?GQA 为什么是折中?
3. RMSNorm 不减均值,为什么仍然能稳定训练?
4. 如果没有 KV Cache 压力,GQA 还有必要吗?

---

## 7.15 完结与下一步

到这里,Transformer 主线已经串起来了:

```
为什么需要 Transformer
→ Self-Attention
→ Multi-Head Attention
→ Positional Encoding
→ Encoder/Decoder Block
→ 训练与推理
→ 现代 LLM 变种
```

下一条学习线可以进入更长上下文、更低复杂度的架构:Jamba / Mamba / SSM。

→ [Jamba 学习笔记](../jamba/README.md)
