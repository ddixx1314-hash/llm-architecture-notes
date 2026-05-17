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

含义:**对每个 token,把它的 $D$ 维表示映射成 $E$ 个分数,一个分数对应一个专家**。

⚠️ **softmax 在 $E$ 维上做,不是 token 维**。这点很容易写错——我们要的是"一个 token 在所有专家间的概率分布",所以 softmax 应该作用在 expert 维上。如果在 token 维上做,会变成"每个专家选择哪些 token"——意思完全相反。

对 $g$ 做 softmax:

$$
p = \text{softmax}(g),\quad p \in \mathbb{R}^E
$$

然后选概率最高的 Top-K 个专家。

输出:

$$
y=\sum_{i \in \text{TopK}(p)} p_i E_i(x)
$$

注意这里 $p_i$ 是专家输出的加权系数。

📌 **两种 softmax 顺序的微妙区别**:
- **先 softmax 再 TopK**(常见教程版):上面的写法。Top-K 之外的概率会被丢弃,所以选中的 K 个概率和 $< 1$,通常需要重新归一化
- **先 TopK 再 softmax**(Mixtral / Jamba 实际用的):先选 Top-K logits,只在这 K 个上做 softmax,天然归一化为 1,**且未选中的 logits 完全不进 softmax 分母**,数值更稳定

两种数学上不严格等价,但实践中差别不大。第 6.8 节代码用的是第二种。

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

下面给两个版本:**版本 A** 容易理解但低效(逐专家循环),**版本 B** 是工业实现里常见的"按 (token, slot) 展开"风格,逻辑更清楚也避免聚合错误。

### 版本 A:逐专家循环 (易读)

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
            FeedForward(d_model, d_ff) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (B, L, D) → 把 token 维展平到 (T, D),T = B*L
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)                          # (T, D)
        T = x_flat.size(0)

        # 1. 路由:对每个 token 选 Top-K 专家
        logits = self.router(x_flat)                       # (T, E)
        top_logits, top_idx = logits.topk(self.top_k, dim=-1)  # (T, K)
        top_weights = F.softmax(top_logits, dim=-1)        # (T, K),只在 K 上归一化

        # 2. 逐专家收集 token、跑 FFN、按权重写回
        out = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            # 哪些 (token, slot) 对选中了专家 e
            hit = (top_idx == e)                           # (T, K) bool
            if not hit.any():
                continue

            token_ids, slot_ids = hit.nonzero(as_tuple=True)  # 每个命中的 (t, k)
            expert_in = x_flat[token_ids]                  # (M, D)
            expert_out = self.experts[e](expert_in)        # (M, D)

            # 取出每个命中位置对应的权重(只取该 slot,不要 sum 别的 slot)
            w = top_weights[token_ids, slot_ids].unsqueeze(-1)  # (M, 1)
            out.index_add_(0, token_ids, expert_out * w)

        return out.reshape(B, L, D)
```

### 版本 B:按 (token, slot) 展开 (推荐)

完全避开"一个 token 选了多个专家时如何聚合权重"的坑——每个 (token, slot) 对就是一个独立的工作单元。

```python
def forward(self, x):
    B, L, D = x.shape
    x_flat = x.reshape(-1, D)                              # (T, D)
    T, K, E = x_flat.size(0), self.top_k, self.num_experts

    logits = self.router(x_flat)                           # (T, E)
    top_logits, top_idx = logits.topk(K, dim=-1)           # (T, K)
    top_weights = F.softmax(top_logits, dim=-1)            # (T, K)

    # 把 (T, K) 展开成 T*K 个 (token_id, expert_id, weight) 三元组
    flat_token = torch.arange(T, device=x.device).repeat_interleave(K)  # (T*K,)
    flat_expert = top_idx.reshape(-1)                      # (T*K,)
    flat_weight = top_weights.reshape(-1)                  # (T*K,)

    out = torch.zeros_like(x_flat)
    for e in range(E):
        mask = (flat_expert == e)
        if not mask.any():
            continue
        sel_tokens = flat_token[mask]                      # 命中专家 e 的 token id
        sel_w = flat_weight[mask].unsqueeze(-1)            # 对应权重
        expert_out = self.experts[e](x_flat[sel_tokens])   # (M, D)
        out.index_add_(0, sel_tokens, expert_out * sel_w)

    return out.reshape(B, L, D)
```

> ⚠️ **常见 bug 提醒**:不要写 `weights = torch.where(mask, top_probs, 0.0).sum(dim=-1)` 来取权重。当一个 token 在 Top-K 里**同时**命中两个专家、且当前遍历到其中一个专家时,`.sum(dim=-1)` 会把**两个 slot 的权重都累加**,但实际只应该用当前专家对应的那个 slot。上面两版都通过取 `(token_id, slot_id)` 二维索引避开了这个陷阱。

> 真实 MoE 实现(megablocks、tutel、scattermoe)还会进一步:用 grouped GEMM 替代逐专家循环、用 token sort 提高内存连续性、用 capacity factor 把不规则形状变成静态形状。这些都是为了贴合 GPU。

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

---

## 6.12 思考题(可选)

1. MoE 通过稀疏激活把"总参数"和"激活参数"解耦——但 GPU 上**总参数仍然要全部装载到显存**。那 MoE 相比同总参数的 dense 模型,真正的优势在哪?
2. 如果 Top-K 设置成 K=1 (Switch Transformer 风格) vs K=2 (Mixtral/Jamba 风格),分别有什么取舍?
3. Load balance loss 鼓励均匀分配,但这和"专家分工"的初衷不冲突吗?直觉上不同 token 类型应该走不同专家才对。

<details>
<summary><b>参考思路</b>(先自己想 3-5 分钟再展开)</summary>

**1.** 训练时:总参数全部要算 backward,但**每个 token 的 forward / backward 算力只和激活参数成正比**。在 token 数远多于专家数的情况下,batch 级的总 FLOPs 接近"激活参数 × token 数",dense 模型则是"总参数 × token 数",MoE 节约的算力很可观。推理时:显存装载确实是 dense 的代价,但 decode 的瓶颈是**每个 token 的 FLOPs 和延迟**,MoE 在这里赢——所以适合云端推理,但不太适合手机端(那里显存是硬约束)。

**2.** **K=1** (Switch):路由简单(每 token 单专家),通信少,但路由错误成本高(一个 token 全押到某个专家)、训练不稳定,需要更激进的 capacity / aux loss。**K=2** (Mixtral/Jamba):路由错误时还有 backup,梯度信号更平滑,工程上更稳;代价是计算量翻倍(虽然 token 多到能摊销)。实际经验:K=2 更鲁棒,K=1 更便宜但调参难。

**3.** **目标是"软均匀"**——load balance loss 不是强制完全均匀(完全均匀就 = 随机路由,失去 MoE 意义),而是阻止**极端不平衡**(99% token 走同一个专家)。它鼓励"在统计意义上各专家被选概率接近,但任意单个 token 仍然可以有强烈倾向"。可以想成正则化:类似 dropout,牺牲一点"专家分工的极致性"换取**训练稳定性 + 不浪费参数**。

</details>

---

## 6.13 论文/源码对照

| 概念 | 论文符号 / 章节 | 源码位置 |
|---|---|---|
| Top-K Sparse Gating | Shazeer 2017 (arxiv 1701.06538) §2.1 | HuggingFace `transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock` |
| Switch Transformer (K=1) | Fedus 2021 (arxiv 2101.03961) | `t5x` 仓库 |
| Load balance loss $\mathcal{L}_{\text{aux}} = E \cdot \sum_e f_e \cdot P_e$ | Fedus 2021 Eq.(4) | Mixtral 实现中的 `load_balancing_loss_func` |
| Capacity factor | GShard / Switch Transformer | `megablocks` 仓库 |
| Token dispatch (all-to-all) | GShard paper Figure 2 | `tutel`、`megablocks::ops` |
| Jamba MoE 配置 | Jamba paper §3.1 (E=16, K=2, 每 2 层一次) | HuggingFace `transformers.JambaSparseMoeBlock` |
