# 第 0 节:为什么需要混合架构?

> 本节目标:理解纯 Transformer 和纯 Mamba/SSM 各自的长短板,从而看懂 Jamba 为什么选择 Transformer + Mamba + MoE 的混合路线。

---

## 0.1 Transformer 的强项:精确检索

Transformer Attention 的核心公式是:

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 $QK^T$ 会显式计算每一对 token 的相似度。

这带来一个巨大优点:

> 任意两个位置都能直接交互。

例如在 100K token 的长文档中,最后一句话需要引用第 300 个 token 的一个名字。Attention 理论上可以直接给那个位置很高权重。

这就是 Transformer 很擅长:

- 精确召回
- 复制某个片段
- 多处信息对齐
- 代码中的变量引用
- RAG 场景里从长上下文抽取证据

---

## 0.2 Transformer 的代价:$O(L^2)$

问题也很直接:注意力矩阵是:

$$
QK^T \in \mathbb{R}^{L \times L}
$$

含义:**每一对 token 都要算一次相似度**,共 $L \times L$ 个数。

所以复杂度是:

$$
O(L^2 \cdot D)
$$

显存也要处理一个接近 $L^2$ 的注意力结构。

当上下文长度从 4K 增长到 256K:

$$
\left(\frac{256K}{4K}\right)^2 = 64^2 = 4096
$$

注意力的二次项会爆炸。

KV Cache 也有压力:

$$
\text{KV Cache} \approx 2 \cdot L \cdot D \cdot \text{layers}
$$

⚠️ **注意**: 因子 $2$ 来自要分别缓存 $K$ 和 $V$。这里假设的是 **MHA**(每层每个头都缓存)。**GQA / MQA**(LLaMA 3、Mixtral 用的)通过让多个 query head 共享同一组 K/V,把这个 $2 \cdot L \cdot D$ 进一步缩小到 $2 \cdot L \cdot D \cdot \frac{n_{\text{kv}}}{n_h}$。但即便如此,KV Cache 仍然**线性**于 $L$,长上下文下依然吃紧。

所以长上下文推理时,Transformer 的瓶颈不只在算力,也在显存和带宽。

---

## 0.3 RNN/SSM 的强项:线性扫描

RNN 类模型使用递推:

$$
h_t = f(h_{t-1}, x_t)
$$

每一步只维护一个状态 $h_t$,不会显式保存所有 token 两两关系。

状态空间模型(SSM)也是类似思想:

$$
x_t = \bar{A}x_{t-1} + \bar{B}u_t
$$

$$
y_t = Cx_t
$$

这里 $x_t$ 是状态,它把历史压缩在一个固定大小的向量里。

复杂度:

$$
O(L)
$$

这对长序列非常诱人。

---

## 0.4 纯线性状态模型的痛点

如果所有历史都被压缩进固定大小的状态 $x_t$,就会遇到一个问题:

> 状态容量有限,很难像 Attention 那样精确取回任意一个历史 token。

直觉上,SSM 像是在读书时不断做摘要:

```
读第 1 页 → 更新笔记
读第 2 页 → 更新笔记
...
读第 1000 页 → 笔记里保留重要信息
```

这很高效,但如果你突然问:

> 第 17 页第三段的那个变量名是什么?

摘要式状态可能不如 Attention 直接查原文可靠。

---

## 0.5 Mamba 做了什么改进?

Mamba 的关键是 **Selective SSM**。

传统 SSM 的参数大致固定:

$$
A,B,C,\Delta
$$

Mamba 让其中一些参数依赖当前输入:

$$
B_t = B(x_t),\quad C_t = C(x_t),\quad \Delta_t = \Delta(x_t)
$$

这意味着模型可以根据当前 token 动态决定:

- 什么信息写入状态
- 什么信息从状态读出
- 当前步的时间尺度多大

这比固定 SSM 灵活得多。

---

## 0.6 但 Mamba 仍然不是 Attention

Mamba 的选择机制让状态更新更聪明,但它仍然不是显式的 $L \times L$ 两两比较。

所以 Mamba 更像:

> 用一个聪明的动态记忆系统流式读序列。

Attention 更像:

> 每次需要时,直接在完整上下文里做内容寻址。

两者不是谁完全替代谁,而是擅长不同事情。

---

## 0.7 Jamba 的直觉:不要二选一

Jamba 的核心选择:

> 大部分层用 Mamba 提供长上下文效率,少量层保留 Attention 提供精确检索能力。

再加上 MoE:

> 用很多专家扩大总参数量,但每个 token 只激活少数专家,控制推理成本。

所以 Jamba 同时追求三件事:

| 组件 | 解决什么 |
|------|----------|
| Mamba | 长序列线性处理,降低内存和计算压力 |
| Attention | 全局精确交互,补足检索和对齐能力 |
| MoE | 增大模型容量,但保持激活参数可控 |

---

## 0.8 复杂度对比

设序列长度为 $L$,隐藏维度为 $D$。

| 架构 | 序列混合复杂度 | 长上下文特点 |
|------|----------------|--------------|
| Transformer | $O(L^2D)$ | 精确但贵 |
| RNN/SSM | $O(LD)$ | 便宜但压缩历史 |
| Mamba | $O(LD)$ | 动态选择性状态 |
| Jamba | Mamba 为主 + 少量 Attention | 在效率和精确召回之间折中 |

---

## 0.9 本节核心要点

1. Transformer 的优势是显式两两交互,适合精确检索
2. Transformer 的长上下文瓶颈来自 $O(L^2)$ attention 和 KV Cache
3. SSM/Mamba 用固定大小状态递推,复杂度是 $O(L)$
4. Mamba 通过输入依赖参数增强了状态模型的选择能力
5. Jamba 的核心不是替代 Attention,而是把 Attention、Mamba、MoE 混合起来

---

## 0.10 下一节预告

下一节进入 SSM 基础:

- 连续时间状态空间模型是什么?
- 为什么它本质上是一个线性 ODE?
- $A,B,C$ 三个矩阵分别控制什么?
- 矩阵指数 $e^{At}$ 为什么会出现?

→ [第 1 节:状态空间模型基础](01-ssm-basics.md)

---

## 0.11 思考题(可选)

1. Transformer 长上下文推理慢,主要是 **算力瓶颈** 还是 **显存带宽瓶颈**?为什么 KV Cache 比注意力矩阵本身更早成为问题?
2. 如果一个任务只需要"看见最近的 $W$ 个 token"(比如 sliding window attention),它的复杂度回到了 $O(L \cdot W)$。这种受限注意力相比 Mamba 各有什么取舍?
3. Mamba 把历史压进固定大小状态——状态维度 $N$ 应该选多大?太小会怎样?太大又会怎样?

<details>
<summary><b>参考思路</b>(先自己想 3-5 分钟再展开)</summary>

**1.** **显存带宽**更先成为瓶颈。Decode 阶段每生成一个 token,理论 FLOPs 不大(只算一个 query 的 attention),但要**完整读一遍 KV Cache**。例如 70B 模型在 32K 上下文下,每解码一个 token 要从 HBM 读几十 GB 的 KV——这远超 GPU 的带宽承受。所以 long-context decode 是 memory-bound,不是 compute-bound,这也解释了为什么 PagedAttention/FlashDecoding/MQA 这些优化都瞄准 KV Cache 而不是 FLOPs。

**2.** Sliding window 把复杂度降到 $O(L \cdot W)$,但**显式丢掉**了超出窗口的 token——模型对"窗口外的内容"完全没有访问能力。Mamba 通过状态把历史**压缩**保留下来:不能精确召回,但能传递摘要。两者哲学不同:sliding window 是"砍掉",Mamba 是"压缩"。混合架构中两种思路都有用。

**3.** $N$ 太小,长程信息容量不够,状态写满后会快速覆盖旧信息(类似一个太小的 RNN 隐藏层)。$N$ 太大,每通道的状态张量 $(B, L, D, N)$ 显存占用线性增长,scan 计算量也增加,且模型容易过拟合到状态的高频维度。Mamba-1 实际取 $N=16$,Mamba-2 因为有 SSD 风格的并行可以取到 $N=64\sim256$。

</details>

---

## 0.12 论文/源码对照

| 概念 | 论文符号 / 章节 | 源码位置 |
|---|---|---|
| Attention $O(L^2)$ 复杂度 | Vaswani 2017, §3.2.2 | `torch.nn.functional.scaled_dot_product_attention` |
| KV Cache 增长 | LLaMA/GPT 实现通用 | HuggingFace `transformers` 的 `DynamicCache` / `StaticCache` |
| SSM 状态压缩 | Mamba paper §2.1, Eq.(1a)(1b) | `mamba_ssm/modules/mamba_simple.py` |
| Selective $\Delta, B, C$ | Mamba paper §3.2, Algorithm 2 | `mamba_ssm/ops/selective_scan_interface.py` |
| Jamba 1:7 混合 | Jamba paper §3, Figure 1 | AI21 模型卡 + HuggingFace `JambaConfig` |
