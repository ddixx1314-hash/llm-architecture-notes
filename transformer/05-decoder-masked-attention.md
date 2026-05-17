# 第 5 节:Decoder + Masked Attention

> 本节目标:理解 Transformer Decoder 为什么需要 mask,推导 causal self-attention,并区分 self-attention 与 cross-attention。

---

## 5.1 Encoder 和 Decoder 的任务不同

Encoder 的任务是**理解整段输入**。例如机器翻译中,Encoder 可以一次性看到完整英文句子:

> I love deep learning.

所以 Encoder 的每个位置都可以关注任意位置。

Decoder 的任务是**一步一步生成输出**。例如生成中文翻译:

```
我 → 喜欢 → 深度 → 学习
```

在生成 "喜欢" 的时候,模型只能看到已经生成的 "我",不能偷看未来的 "深度 学习"。

这就是 Decoder 和 Encoder 最大的区别:

| 模块 | 能否看未来位置? | 目的 |
|------|----------------|------|
| Encoder Self-Attention | 可以 | 编码完整输入 |
| Decoder Self-Attention | 不可以 | 自回归生成 |
| Decoder Cross-Attention | 可以看完整 Encoder 输出 | 对齐源序列信息 |

---

## 5.2 自回归生成

语言模型把序列概率分解成:

$$
P(y_1, y_2, \dots, y_n) = \prod_{t=1}^{n} P(y_t \mid y_{<t})
$$

也就是说,第 $t$ 个 token 只能依赖它前面的 token:

$$
P(y_t \mid y_1, \dots, y_{t-1})
$$

不能依赖:

$$
y_{t+1}, y_{t+2}, \dots
$$

这叫 **causal** 或 **autoregressive**。Decoder 的 mask 就是为了在训练时强制满足这个约束。

---

## 5.3 问题:训练时 Decoder 会一次性看到完整目标序列

训练机器翻译时,我们通常把目标句子整体喂给 Decoder:

```
输入给 Decoder: <bos> 我 喜欢 深度 学习
预测目标:       我   喜欢 深度 学习 <eos>
```

如果不加 mask,第 2 个位置 "喜欢" 的 query 可以直接关注第 3、4 个位置 "深度 学习"。这等于考试时偷看答案。

所以训练虽然并行喂入整段序列,但注意力矩阵必须被限制成:

$$
\alpha_{ij} = 0 \quad \text{if } j > i
$$

也就是位置 $i$ 只能看自己和之前的位置。

---

## 5.4 Causal Mask 长什么样?

设序列长度 $n=5$,允许关注的位置记为 0,禁止关注的位置记为 $-\infty$:

$$
M =
\begin{bmatrix}
0 & -\infty & -\infty & -\infty & -\infty \\
0 & 0       & -\infty & -\infty & -\infty \\
0 & 0       & 0       & -\infty & -\infty \\
0 & 0       & 0       & 0       & -\infty \\
0 & 0       & 0       & 0       & 0
\end{bmatrix}
$$

然后在 softmax 之前加到注意力分数上:

$$
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)
$$

为什么是 $-\infty$?

因为:

$$
e^{-\infty} = 0
$$

所以被 mask 的位置经过 softmax 后概率就是 0。

---

## 5.5 Masked Self-Attention 的矩阵形状

普通 self-attention:

$$
Q,K,V \in \mathbb{R}^{n \times d_k}
$$

$$
S = QK^T \in \mathbb{R}^{n \times n}
$$

Masked self-attention:

$$
S_{\text{masked}} = \frac{QK^T}{\sqrt{d_k}} + M
$$

$$
\text{MaskedAttention}(Q,K,V) = \text{softmax}(S_{\text{masked}})V
$$

唯一变化就是在 softmax 之前加 mask。其他计算完全一样。

---

## 5.6 PyTorch 实现 Causal Mask

```python
import torch

def causal_mask(n, device=None):
    # True 表示需要被 mask 的位置
    return torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=device),
        diagonal=1,
    )
```

示例:

```python
mask = causal_mask(5)
print(mask)
```

输出:

```text
tensor([[False,  True,  True,  True,  True],
        [False, False,  True,  True,  True],
        [False, False, False,  True,  True],
        [False, False, False, False,  True],
        [False, False, False, False, False]])
```

用在 attention 分数上:

```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
scores = scores.masked_fill(mask, float("-inf"))
attn = torch.softmax(scores, dim=-1)
out = torch.matmul(attn, V)
```

实际多头场景中:

```python
# scores: (batch, h, n, n)
# mask:   (n, n)
scores = scores.masked_fill(mask, float("-inf"))
```

PyTorch 会把 mask 广播到 batch 和 head 维度。

---

## 5.7 Decoder Block 的三个子层

原始 Transformer Decoder 每层有三个子层:

1. Masked Multi-Head Self-Attention
2. Encoder-Decoder Cross-Attention
3. Feed-Forward Network

结构:

```
decoder x
│
├── Masked Self-Attention
├── Add & Norm
│
├── Cross-Attention  ← K,V 来自 Encoder 输出
├── Add & Norm
│
├── FFN
└── Add & Norm
```

---

## 5.8 Cross-Attention:Q 来自 Decoder,K/V 来自 Encoder

Self-attention 中:

$$
Q = XW^Q,\quad K = XW^K,\quad V = XW^V
$$

三者都来自同一个序列。

Cross-attention 中有两个输入:

- Decoder 当前隐藏状态 $Y \in \mathbb{R}^{m \times d_{\text{model}}}$
- Encoder 输出 $H \in \mathbb{R}^{n \times d_{\text{model}}}$

计算:

$$
Q = YW^Q
$$

$$
K = HW^K,\quad V = HW^V
$$

注意力分数:

$$
S = QK^T \in \mathbb{R}^{m \times n}
$$

含义:Decoder 的每个目标位置,去 Encoder 的所有源位置里查找相关信息。

---

## 5.9 Cross-Attention 不需要 Causal Mask

Cross-attention 的 key/value 来自源序列,不是未来目标 token。

在机器翻译中,Decoder 生成每一个中文 token 时,都可以看完整英文句子:

```
源序列: I love deep learning
目标当前: 我 喜欢 ...
```

所以 cross-attention 通常不加 causal mask。它可能只需要 padding mask,用来忽略源序列里的 `<pad>`。

---

## 5.10 Padding Mask 和 Causal Mask 的区别

实际训练中常见两种 mask:

| Mask | 目的 | 形状直觉 |
|------|------|----------|
| Padding mask | 忽略 `<pad>` token | 每个样本不同 |
| Causal mask | 禁止看未来 token | 所有样本相同的上三角 |

Padding mask 例子:

```
真实 token: [I, love, you, <pad>, <pad>]
mask:       [0, 0,    0,   1,     1]
```

Causal mask 例子:

```
位置 i 不能看 j > i
```

在 Decoder self-attention 中,两者通常会合并使用:

$$
M = M_{\text{causal}} + M_{\text{padding}}
$$

---

## 5.11 PyTorch:带 mask 的 Multi-Head Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionWithMask(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key, value, mask=None):
        # query: (batch, q_len, d_model)
        # key/value: (batch, kv_len, d_model)
        batch_size, q_len, _ = query.shape
        kv_len = key.size(1)

        Q = self.W_q(query).view(batch_size, q_len, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, kv_len, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, kv_len, self.h, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.W_o(out)
```

这个版本既可以做 self-attention:

```python
self_attn(x, x, x, mask=causal_mask)
```

也可以做 cross-attention:

```python
cross_attn(decoder_x, encoder_out, encoder_out, mask=src_padding_mask)
```

---

## 5.12 Decoder Block 代码骨架

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithMask(d_model, num_heads)
        self.cross_attn = MultiHeadAttentionWithMask(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, tgt_mask=None, src_mask=None):
        # 1. masked self-attention
        self_out = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_out))

        # 2. cross-attention
        cross_out = self.cross_attn(x, encoder_out, encoder_out, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_out))

        # 3. feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x
```

---

## 5.13 Decoder-only 模型

GPT/LLaMA 这类语言模型没有 Encoder,只有 Decoder 的 masked self-attention:

```
token embedding
↓
masked self-attention block × N
↓
linear vocab head
↓
预测下一个 token
```

因为没有源序列,所以也没有 cross-attention。

这时 "Decoder Block" 通常简化为:

1. Masked Self-Attention
2. FFN

也就是第 4 节 Encoder Block 的结构,但 Self-Attention 必须加 causal mask。

---

## 5.14 论文符号对照

《Attention Is All You Need》3.1 节描述 Decoder:

- Decoder 也由 $N = 6$ 个相同层堆叠而成
- 每层除了 Encoder 的两个子层,中间还插入一个 Encoder-Decoder Attention
- Decoder self-attention 需要 mask,防止当前位置关注后续位置
- Cross-attention 中 queries 来自上一层 Decoder,keys 和 values 来自 Encoder 输出

---

## 5.15 本节核心要点

1. Decoder 用于自回归生成,第 $t$ 个位置不能看未来 token
2. Causal mask 是上三角矩阵,在 softmax 前把未来位置设为 $-\infty$
3. Masked self-attention 和普通 attention 公式相同,只多了 mask
4. Cross-attention 中 $Q$ 来自 Decoder,$K,V$ 来自 Encoder
5. Cross-attention 可以看完整源序列,通常不需要 causal mask
6. GPT/LLaMA 是 decoder-only,保留 masked self-attention,去掉 cross-attention

---

## 5.16 思考题

1. 如果训练 Decoder 时不加 causal mask,loss 会怎样?推理时会出现什么问题?
2. 为什么 causal mask 要加在 softmax 之前,而不是 softmax 之后直接把概率置 0?
3. Cross-attention 的注意力矩阵形状为什么是 $m \times n$,而不是 $n \times n$?
4. Decoder-only 模型为什么也叫语言模型?它和 Encoder-Decoder 翻译模型的训练目标有什么不同?

<details>
<summary><b>参考思路</b></summary>

**1.** 训练 loss 会**异常低**(模型可以"作弊":预测位置 $t$ 时直接看位置 $t$ 的真实答案)。推理时模型没有未来 token 可看,**输出垃圾** — 训练和推理分布完全不一致。这是个典型的"数据泄漏"陷阱。

**2.** 两个原因:
- **数值正确性**:softmax 之后置 0 会**破坏归一化**($\sum_j \alpha_{ij}$ 不再等于 1),剩余权重之和小于 1;要再次归一化才能用。
- **数学等价**:softmax 之前加 $-\infty$,经过 $\exp(-\infty)=0$,自然得到 0 概率,**剩余位置的概率自动归一**(分母只包含未 mask 位置)。一步完成,且数值稳定。

**3.** Query 来自 Decoder(长度 $m$),Key/Value 来自 Encoder(长度 $n$)。$QK^T$ 形状是 (m, d) × (d, n) = (m, n)。**含义**:Decoder 每个目标位置(m 个)去 Encoder 的所有源位置(n 个)里查找信息。$m$ 和 $n$ 可以不同(英译中:英文 5 词 → 中文 6 词)。

**4.** 因为它的训练目标 $P(x_t | x_{<t})$ 就是**语言模型的标准定义** — 预测下一个词。Encoder-Decoder 模型的目标是 $P(y | x)$(条件生成,如英→中翻译),需要两个不同序列。Decoder-only 把两者统一成 $P(\text{whole sequence})$,把 prompt 和回答都放在同一条序列里,用 causal mask 自然实现条件生成。这种统一性是 GPT 范式胜出的关键。

</details>

---

## 5.17 下一节预告

我们已经有了完整的 Transformer 结构。下一节关注训练和推理:

- 语言模型如何用 next-token prediction 训练?
- teacher forcing 是什么?
- 为什么推理时不能像训练一样完全并行?
- KV Cache 如何把自回归生成从重复计算中救出来?

→ [第 6 节:训练目标与推理](06-training-inference.md)
