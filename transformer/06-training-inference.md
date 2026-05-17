# 第 6 节:训练目标与推理

> 本节目标:理解 Transformer 语言模型如何训练、为什么训练可以并行但推理必须逐 token 生成,以及 KV Cache 如何降低自回归推理成本。

---

## 6.1 训练到底在学什么?

对语言模型来说,最常见的训练目标是:

> 给定前面的 token,预测下一个 token。

假设一句话被 tokenizer 切成:

```
[I, love, deep, learning, <eos>]
```

训练样本会被错开一位:

```
输入 x: [I,    love, deep,     learning]
目标 y: [love, deep, learning, <eos>]
```

模型在每个位置输出一个词表分布:

$$
p_t = P(y_t \mid x_{\leq t})
$$

然后希望真实下一个 token 的概率越大越好。

---

## 6.2 Next-Token Prediction 的概率分解

完整序列概率可以写成:

$$
P(x_1, x_2, \dots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_{<t})
$$

取 log:

$$
\log P(x_1, \dots, x_n) = \sum_{t=1}^{n} \log P(x_t \mid x_{<t})
$$

训练时最大化这个 log-likelihood,等价于最小化交叉熵损失:

$$
\mathcal{L} = -\sum_{t=1}^{n} \log P(x_t \mid x_{<t})
$$

这就是语言模型最核心的训练目标。

---

## 6.3 从隐藏状态到词表概率

Transformer 最后一层输出:

$$
H \in \mathbb{R}^{B \times n \times d_{\text{model}}}
$$

其中 $B$ 是 batch size。

接一个线性层映射到词表大小:

$$
\text{logits} = H W_{\text{vocab}}
$$

$$
W_{\text{vocab}} \in \mathbb{R}^{d_{\text{model}} \times |\mathcal{V}|}
$$

所以:

$$
\text{logits} \in \mathbb{R}^{B \times n \times |\mathcal{V}|}
$$

再沿词表维度做 softmax:

$$
P(x_{t+1} \mid x_{\leq t}) = \text{softmax}(\text{logits}_t)
$$

---

## 6.4 Teacher Forcing

训练时有一个重要技巧:Decoder 的输入不是模型自己上一步生成的 token,而是**真实答案右移一位**。

例如目标句子:

```
我 喜欢 深度 学习 <eos>
```

Decoder 输入:

```
<bos> 我 喜欢 深度 学习
```

预测目标:

```
我   喜欢 深度 学习 <eos>
```

这叫 **teacher forcing**。

好处:训练时可以一次性并行计算所有位置的 loss,不用真的一个 token 一个 token 地生成。

⚠️ 但注意:虽然训练输入一次性喂入完整序列,causal mask 仍然保证第 $t$ 个位置看不到未来答案。

---

## 6.5 PyTorch 训练骨架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers):
        super().__init__()
        self.transformer = DecoderOnlyTransformer(vocab_size, d_model, num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        # input_ids: (batch, n)
        hidden = self.transformer(input_ids)      # (batch, n, d_model)
        logits = self.lm_head(hidden)             # (batch, n, vocab_size)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss
```

准备输入和标签:

```python
tokens = torch.tensor([[1, 20, 305, 42, 2]])

input_ids = tokens[:, :-1]  # [1, 20, 305, 42]
labels = tokens[:, 1:]      # [20, 305, 42, 2]

logits, loss = model(input_ids, labels)
```

---

## 6.6 为什么训练能并行,推理不能?

训练时我们知道完整答案,所以可以用 teacher forcing:

```
输入: <bos> 我 喜欢 深度
目标: 我   喜欢 深度 学习
```

加上 causal mask 后,所有位置可以一次大矩阵乘并行算出来。

推理时不一样。模型一开始只有 prompt:

```
<bos>
```

要先生成第一个 token:

```
<bos> 我
```

再把 "我" 放回输入,生成下一个:

```
<bos> 我 喜欢
```

因为后一个 token 依赖前一个 token 的采样结果,所以推理天然是串行的:

$$
x_1 \rightarrow x_2 \rightarrow x_3 \rightarrow \cdots
$$

---

## 6.7 最朴素的生成循环

```python
@torch.no_grad()
def generate(model, input_ids, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)

        # 只取最后一个位置的 logits
        next_logits = logits[:, -1, :]
        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_id], dim=1)

    return input_ids
```

这个版本能工作,但非常浪费。

假设已经生成了 1000 个 token,下一步只想预测第 1001 个 token。朴素实现会把前 1000 个 token 全部重新跑一遍 Transformer。

---

## 6.8 重复计算在哪里?

看一层 self-attention:

$$
Q = XW^Q,\quad K = XW^K,\quad V = XW^V
$$

当我们在第 $t$ 步生成新 token 时:

- 老 token 的 $K,V$ 已经算过了
- 新 token 只需要和过去所有 $K,V$ 做 attention
- 老 token 的输出不需要重新计算,因为 causal mask 下它们看不到新 token

所以真正需要缓存的是每一层的:

$$
K_{\leq t}, V_{\leq t}
$$

这就是 **KV Cache**。

---

## 6.9 KV Cache 的核心思想

没有 cache 时,第 $t$ 步要对长度 $t$ 的序列重新计算:

$$
K_{1:t}, V_{1:t}
$$

有 cache 时:

1. 只对新 token 计算 $q_t,k_t,v_t$
2. 把 $k_t,v_t$ 追加到 cache
3. 用 $q_t$ 去 attend 所有缓存的 $K_{1:t},V_{1:t}$

公式:

$$
K_{\text{cache}} \leftarrow [K_{\text{cache}}; k_t]
$$

$$
V_{\text{cache}} \leftarrow [V_{\text{cache}}; v_t]
$$

$$
o_t = \text{softmax}\left(\frac{q_t K_{\text{cache}}^T}{\sqrt{d_k}}\right)V_{\text{cache}}
$$

---

## 6.10 KV Cache 降低了什么复杂度?

⚠️ **下面的复杂度只算 attention 部分**(因为 KV Cache 优化的就是 attention 重复计算)。FFN 部分两种情况差不多 — 每生成一个新 token 都要过一次 FFN,这是无法 cache 的。

设已经生成到长度 $L$。

### 没有 KV Cache

每一步都重算整段序列,第 $t$ 步 attention 约为:

$$
O(t^2 d)
$$

总生成 $L$ 个 token:

$$
\sum_{t=1}^{L} O(t^2 d) = O(L^3 d)
$$

### 有 KV Cache

第 $t$ 步只算新 token 的 query 对历史 key:

$$
O(t d)
$$

总生成 $L$ 个 token:

$$
\sum_{t=1}^{L} O(t d) = O(L^2 d)
$$

所以 KV Cache 不会让自回归生成变成并行,但能避免大量重复计算。

### 把 FFN 算进来

每个 token 过一次 FFN 是 $O(d \cdot d_{\text{ff}}) \approx O(d^2)$。生成 $L$ 个 token 总 FFN 计算是 $O(L d^2)$,**两种情况都一样**(因为 FFN 是 position-wise,无 cache 可言)。

合计:

| 部分 | 无 cache | 有 cache |
|------|---------|----------|
| Attention | $O(L^3 d)$ | $O(L^2 d)$ |
| FFN | $O(L d^2)$ | $O(L d^2)$ |

**关键观察**:
- 当 $L \gg d$(长上下文),attention 主导,KV Cache 至关重要(否则 $L^3$ 爆炸)
- 当 $L \ll d$(短上下文、大模型),FFN 主导,KV Cache 收益相对小
- 实际推理两者都重要,但 KV Cache 是"必备",FFN 优化(如 MoE)是另一条线

---

## 6.11 KV Cache 的代价:显存

每层都要缓存 K 和 V(以下假设是标准 MHA,每个 head 都缓存自己的 K/V):

$$
\text{cache size} \approx 2 \cdot L \cdot d_{\text{model}} \cdot \text{num\_layers}
$$

再乘上 batch size 和数据类型字节数。

例如:

- $L = 32768$
- $d_{\text{model}} = 4096$
- layers = 32
- fp16 每个数 2 bytes

缓存大小约为:

$$
2 \times 32768 \times 4096 \times 32 \times 2 \approx 17.2\text{GB}
$$

这就是长上下文推理很吃显存的根源之一。

⚠️ **GQA / MQA 会按比例减少**:若 KV head 数为 $h_{kv}$,query head 数为 $h$,则 cache 大小变为:

$$
2 \cdot L \cdot d_{\text{model}} \cdot \frac{h_{kv}}{h} \cdot \text{num\_layers}
$$

LLaMA 3 70B 用 $h = 64, h_{kv} = 8$,KV cache 直接降到 MHA 的 $1/8$。这就是为什么现代大模型几乎都采用 GQA(详见 [第 7 节](07-modern-variants.md))。

---

## 6.12 采样策略

最后一层 logits 只是一个分布,真正生成 token 时还要选择采样策略。

### Greedy

每次选概率最大的 token:

$$
x_{t+1} = \arg\max_i p_i
$$

稳定,但容易重复和无聊。

### Temperature

调整 logits 的尖锐程度:

$$
p = \text{softmax}(\text{logits}/T)
$$

- $T < 1$:更保守
- $T > 1$:更随机

### Top-k

只在概率最高的 $k$ 个 token 中采样。

### Top-p / nucleus sampling

选择累计概率达到 $p$ 的最小 token 集合,再从里面采样。

---

## 6.13 训练和推理的差异总结

| 阶段 | 输入 | 是否并行 | 是否用 causal mask | 是否用 KV Cache |
|------|------|----------|-------------------|----------------|
| 训练 | 完整序列右移 | ✅ | ✅ | 通常不用 |
| Prefill | prompt 全量输入 | ✅ | ✅ | 写入 cache |
| Decode | 每次一个新 token | ❌ | 不需要完整矩阵 mask | ✅ |

Prefill 指的是推理开始时把 prompt 一次性跑完,把每层 K/V 写入 cache。Decode 指的是后续逐 token 生成。

---

## 6.14 本节核心要点

1. 语言模型训练目标是 next-token prediction
2. 最大化序列概率等价于最小化交叉熵 loss
3. Teacher forcing 让训练可以并行计算所有位置
4. 推理必须自回归,因为下一个 token 依赖上一个采样结果
5. KV Cache 缓存每层历史 K/V,避免重复计算
6. KV Cache 用显存换速度,长上下文时显存压力很大

---

## 6.15 思考题

1. 为什么训练时通常不用 KV Cache,但推理时必须用?
2. causal mask 已经防止偷看未来,为什么 labels 还要右移一位?
3. 如果把 temperature 设得非常大,生成结果会发生什么?
4. KV Cache 的大小为什么和序列长度线性相关,而不是平方相关?

<details>
<summary><b>参考思路</b></summary>

**1.** 训练时**整段序列一次性并行计算**,所有位置的 K/V 同时算出来,没有"过去 vs 新增"的区分,cache 没有意义。推理时每步只新增 1 个 token,前面 token 的 K/V 在结构上已经算过(且因为 causal mask,它们的值不会因为新 token 出现而改变),所以缓存能节省 $O(L)$ 倍计算。

**2.** Causal mask 保证位置 $t$ 的输出 $h_t$ 只依赖输入位置 $\leq t$。但 $h_t$ 应该预测什么?我们希望 $h_t$ 预测**下一个 token**,即 $x_{t+1}$。如果 labels 不右移,$h_t$ 就被强制预测 $x_t$(它自己,经过 mask 后是个恒等映射,trivial)。右移让 $h_t$ 真正预测"未来",才形成 next-token prediction 任务。

**3.** $T \to \infty$ 时 $\text{logits}/T \to 0$,softmax 趋于**均匀分布**,采样退化为词表均匀随机选择。生成结果是**完全随机的乱码**,丢失所有语言结构。反过来 $T \to 0$ 等价于 greedy。

**4.** Cache 存的是 K 和 V — 每个 token 的 K/V 在算出来之后**不再改变**(因为 causal mask),也不需要重算。所以每生成一个新 token,只在 cache 末尾追加一个固定大小的条目,总大小线性增长。注意力**计算**复杂度仍是 $O(L^2 d)$(每个新 query 要和 $L$ 个 cached key 做点积),但**存储**只需 $O(L)$。这是 cache 经常和 attention 复杂度被混淆的地方。

</details>

---

## 6.16 下一节预告

到这里,原始 Transformer 的主线已经完整了。下一节我们看现代大模型常见改造:

- RMSNorm 如何简化 LayerNorm?
- SwiGLU 为什么替代 ReLU FFN?
- RoPE 如何把位置编码变成旋转?
- MQA/GQA 如何减少 KV Cache?

→ [第 7 节:现代变种](07-modern-variants.md)
