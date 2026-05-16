# 第 6 节:Mixture of Experts(MoE)

> 本节目标:理解 MoE 如何用稀疏路由扩大模型总参数量,同时控制每个 token 的实际计算量。

---

## 6.1 Dense FFN 的成本

Transformer block 里 FFN 通常占大量参数:

$$
\text{FFN}(x)=\sigma(xW_1)W_2
$$

如果:

$$
d_{\text{model}}=4096,\quad d_{\text{ff}}=16384
$$

那么一个 FFN 的参数大约是:

$$
4096 \times 16384 \times 2 \approx 134M
$$

每个 token 都要走同一套 FFN。

Dense 模型想变大,就要让每个 token 付出更多计算。

---

## 6.2 MoE 的想法

MoE 把一个大 FFN 换成多个专家:

$$
E_1,E_2,\dots,E_N
$$

每个专家通常都是一个 FFN。

对每个 token,路由器只选择少数几个专家:

$$
\text{TopK}(x) \subset \{1,\dots,E\}
$$

例如 $E=16,K=2$:

```
总共有 16 个专家
每个 token 只激活 2 个专家
```

这就带来一个重要区分:

- **总参数**:所有专家参数加起来
- **激活参数**:当前 token 实际用到的专家参数

---

## 6.3 Router / Gate

路由器通常是一个线性层:

$$
g = xW_g
$$

其中:

$$
g \in \mathbb{R}^{E}
$$

对 $g$ 做 softmax:

$$
p = \text{softmax}(g)
$$

然后选概率最高的 Top-K 个专家。

输出:

$$
y=\sum_{i \in \text{TopK}(p)} p_i E_i(x)
$$

注意这里 $p_i$ 是专家输出的加权系数。

---

## 6.4 Top-2 MoE 示例

假设有 4 个专家,某个 token 的 router 分数是:

```text
expert 0: 0.10
expert 1: 0.65
expert 2: 0.20
expert 3: 0.05
```

Top-2 选择 expert 1 和 expert 2。

输出:

$$
y=0.65E_1(x)+0.20E_2(x)
$$

实际实现中通常会对 Top-K 内部的概率重新归一化:

$$
\tilde{p}_i=\frac{p_i}{\sum_{j\in \text{TopK}}p_j}
$$

---

## 6.5 为什么需要负载均衡?

如果不加约束,router 可能会把大多数 token 都送给同一个专家。

这会导致两个问题:

1. 热门专家过载,并行效率差
2. 冷门专家训练不到,参数浪费

所以 MoE 通常会加一个 load balance loss,鼓励 token 分配更均匀。

直觉目标:

$$
\text{每个专家收到的 token 数差不多}
$$

但也不能强制完全平均,否则会破坏专家分工。

---

## 6.6 Capacity Factor

训练时每个专家通常有容量上限:

$$
\text{capacity} = \left\lceil \frac{K \cdot B \cdot L}{E} \cdot \text{capacity factor} \right\rceil
$$

含义:

- $B \cdot L$:当前 batch 的 token 数
- $K$:每个 token 选几个专家
- $E$:专家数量

如果某个专家收到太多 token,超过容量的 token 可能被丢弃或走备用路径。

这听起来粗暴,但能让分布式训练的张量形状更稳定。

---

## 6.7 MoE 的通信成本

MoE 不只是数学结构,还是分布式系统问题。

因为不同专家可能放在不同 GPU 上,token 要被发送到对应专家:

```
token batch
↓ router
按专家重排 token
↓ all-to-all communication
每个专家处理自己的 token
↓ all-to-all communication
还原 token 顺序
```

所以 MoE 的瓶颈常常不是矩阵乘本身,而是跨设备通信和负载均衡。

---

## 6.8 MoE 代码骨架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(d_model, d_ff)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (batch, L, d_model)
        gate_logits = self.router(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        top_probs, top_idx = torch.topk(gate_probs, self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            mask = top_idx == expert_id
            if not mask.any():
                continue

            token_mask = mask.any(dim=-1)
            expert_input = x[token_mask]
            expert_output = expert(expert_input)

            weights = torch.where(mask[token_mask], top_probs[token_mask], 0.0).sum(dim=-1)
            out[token_mask] += expert_output * weights.unsqueeze(-1)

        return out
```

这个版本方便理解,但真实 MoE 实现会用更高效的 token dispatch 和 expert parallel。

---

## 6.9 MoE 放在 Transformer 哪?

多数 MoE 模型把 FFN 替换成 MoE FFN:

```
Attention
↓
MoE FFN
```

也就是:

$$
x = x + \text{Attention}(\text{Norm}(x))
$$

$$
x = x + \text{MoE}(\text{Norm}(x))
$$

Jamba 也是用 MoE 增大模型容量,但不会让每个 token 激活所有专家。

---

## 6.10 本节核心要点

1. MoE 用多个专家替代单个 dense FFN
2. Router 为每个 token 选择 Top-K 专家
3. 总参数可以很大,但每个 token 只激活少数专家
4. Load balance loss 防止专家使用极度不均
5. MoE 的工程难点在 token dispatch、通信和容量控制

---

## 6.11 下一节预告

下一节把组件装成 Jamba:

- Jamba 如何混合 Mamba 层和 Attention 层?
- 1:7 比例是什么意思?
- MoE 放在哪些层?
- 为什么 Jamba 可以同时有长上下文和较大总参数?

→ [第 7 节:Jamba 整体架构](07-jamba-architecture.md)
