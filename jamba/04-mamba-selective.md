# 第 4 节:Mamba:Selective SSM

> 本节目标:理解 Mamba 的选择性状态空间模型,看懂 $\Delta_t,B_t,C_t$ 随输入变化的意义,以及 Mamba block 的基本结构。

---

## 4.1 固定 SSM 的限制

普通离散 SSM:

$$
x_t=\bar{A}x_{t-1}+\bar{B}u_t
$$

$$
y_t=Cx_t
$$

这里 $\bar{A},\bar{B},C$ 对所有位置都一样。

这意味着模型对每个 token 使用同一种更新规则。

但语言中 token 的重要性不同:

```
姓名、数字、代码变量、否定词
```

这些信息可能要长期保留。

而普通停用词、标点、重复片段可能只需要快速扫过。

---

## 4.2 Mamba 的选择性

Mamba 的核心是让**部分** SSM 参数依赖输入:

$$
\Delta_t = \Delta(u_t),\quad B_t = B(u_t),\quad C_t = C(u_t)
$$

然后每个位置使用自己的离散化参数:

$$
\bar{A}_t = \exp(\Delta_t A),\quad \bar{B}_t = \text{discretize}(A,B_t,\Delta_t)
$$

递推:

$$
x_t=\bar{A}_t x_{t-1}+\bar{B}_t u_t,\quad y_t=C_t x_t
$$

> ⚠️ **关键澄清:谁依赖输入,谁不依赖**
>
> Mamba 中 **$A$ 本身是固定参数**,**不**依赖输入。只有 $\Delta_t, B_t, C_t$ 是输入依赖的。
>
> $\bar{A}_t$ 看起来随 $t$ 变化,实际是因为 $\Delta_t$ 随输入变 ——$A$ 矩阵本身从训练开始到推理结束都是同一个参数表(每通道一个 diagonal,见 4.2.1)。
>
> 这点在源码里很直观:`mamba-ssm` 里 `A_log` 是 `nn.Parameter`,而 `B, C` 是从 `x_proj(x)` 即时算出来的。

### 4.2.1 多通道:每个 channel 一个独立 SSM

到目前为止我们一直把输入写成标量 $u_t \in \mathbb{R}$。但 Mamba 处理的是 $D$ 维 embedding 序列:

$$
u \in \mathbb{R}^{B \times L \times D}
$$

**关键设计:每个 channel $d \in \{1,\dots,D\}$ 独立运行一个 $N$ 维 SSM**,通道间不混合。所以单层内部:

| 张量 | 形状 | 含义 |
|------|------|------|
| 输入 $u$ | $(B, L, D)$ | $D$ 个标量序列 |
| 状态 $x$ | $(B, L, D, N)$ | 每个通道有 $N$ 维状态 |
| $A$ | $(D, N)$ | 每通道一个 diagonal,共 $D$ 套 |
| $\Delta_t, B_t, C_t$ | $(B, L, D), (B, L, N), (B, L, N)$ | $B, C$ 跨通道共享,$\Delta$ 每通道独立 |

通道间的混合留给 block 外的 Linear/MLP 完成。这种"channel-wise SSM + channel-mixing linear"的分工和 depthwise-separable conv 思路相同。

<div align="center"><img src="images/mamba-block-architecture.png" width="65%"></div>

图:Mamba block 把 H3 风格的 SSM 分支和 gated MLP 思路合到一个重复堆叠的模块里。来源:Gu & Dao, 2023, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, Figure 3。

---

## 4.3 三个选择性参数的直觉

### $\Delta_t$:时间步长

$\Delta_t$ 控制状态更新速度。

- 大 $\Delta_t$:状态快速变化,更容易遗忘旧信息
- 小 $\Delta_t$:状态慢慢变化,保留更多历史

### $B_t$:写入方向

$B_t$ 决定当前输入如何写入状态。

重要 token 可以写入更多维度,普通 token 可以少写。

### $C_t$:读出方式

$C_t$ 决定当前 token 从状态里读什么。

问题 token 可能需要读出很久以前保存的信息。

---

## 4.4 选择性像门控记忆

可以把 Mamba 想象成一个动态笔记本:

```
读到重要信息 → 用 B_t 写进笔记
读到普通信息 → 少写或快速覆盖
遇到问题 → 用 C_t 从笔记里读相关内容
```

Attention 是显式翻原文:

```
我现在要找谁?直接对所有 token 算相似度
```

Mamba 是流式维护笔记:

```
每一步决定写什么、忘什么、读什么
```

---

## 4.5 Mamba Block 的高层结构

一个简化的 Mamba block:

```
输入 x
│
├── Linear expand
│
├── depthwise conv  ← 提供局部上下文
│
├── 生成 Δ, B, C
│
├── selective scan  ← 长程序列混合
│
├── gate 分支
│
└── Linear project
```

它不像 Transformer 那样用 attention 矩阵混合 token,而是通过 selective scan 沿序列更新状态。

---

## 4.6 为什么还有卷积?

Mamba block 中有一个短卷积(depthwise convolution)。

作用:

> 在进入 SSM 前,先让每个位置看到附近几个 token。

SSM 擅长长程递推,但局部模式(比如短语、字符组合、局部语法)用小卷积很便宜也很有效。

Depthwise conv 表示每个通道单独卷积,参数和计算都比较少。

---

## 4.7 门控分支

Mamba 还有一条 gate 分支,常见形式类似:

$$
\text{out} = \text{SSM}(x) \odot \text{SiLU}(z)
$$

这和 SwiGLU 的思想相似:

> 一条分支产生内容,另一条分支决定哪些内容通过。

门控能提高表达能力,也让模型更容易控制状态输出。

---

## 4.8 伪代码

```python
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, d_state):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=4,
            groups=d_inner,
            padding=3,
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state)
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x):
        # x: (batch, L, d_model)
        x, gate = self.in_proj(x).chunk(2, dim=-1)

        # depthwise conv expects (batch, channels, L)
        x = self.conv(x.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)
        x = torch.nn.functional.silu(x)

        # 生成输入依赖的 Δ, B, C
        params = self.x_proj(x)
        dt, B, C = split_params(params)

        y = selective_scan(x, dt, B, C)
        y = y * torch.nn.functional.silu(gate)
        return self.out_proj(y)
```

这不是完整源码,但抓住了结构。

---

## 4.9 和 Transformer Block 对比

| Transformer | Mamba |
|-------------|-------|
| Attention 混合全局信息 | Selective scan 混合历史信息 |
| FFN 做通道变换 | Linear + gate 做通道变换 |
| KV Cache 保存历史 token 表示 | SSM state 保存压缩历史 |
| $O(L^2)$ attention | $O(L)$ scan |

关键区别:

> Transformer 保存可检索的历史表示;Mamba 保存递推压缩后的状态。

---

## 4.10 本节核心要点

1. Mamba 让 $\Delta_t,B_t,C_t$ 依赖输入,形成 Selective SSM
2. $\Delta_t$ 控制时间尺度,$B_t$ 控制写入,$C_t$ 控制读出
3. Mamba block 结合了线性投影、局部卷积、selective scan 和门控
4. Mamba 用固定大小状态替代 KV Cache,长序列更省内存
5. 它不是 Attention 的简单替代,而是一种不同的序列混合机制

---

## 4.11 下一节预告

Mamba 公式看起来是递推的,似乎训练时必须串行。但 Mamba 的重要贡献之一就是硬件感知的并行扫描:

- scan 为什么能并行?
- associative operator 是什么?
- 为什么不能把所有中间状态都写到 HBM?
- selective scan 如何贴合 GPU?

→ [第 5 节:硬件感知并行扫描](05-mamba-parallel-scan.md)

---

## 4.12 思考题(可选)

1. 为什么 Mamba 让 $\Delta, B, C$ 依赖输入,但**不**让 $A$ 依赖输入?如果 $A$ 也依赖输入,数学上和工程上会出什么问题?
2. Mamba block 里的 depthwise conv (kernel size 通常是 4) 看起来很短,它真的能贡献什么?去掉它会怎样?
3. 对比 Mamba block 和 Transformer block:Attention 对应什么、FFN 对应什么、LayerNorm 对应什么?哪些组件 Mamba 完全没有?

<details>
<summary><b>参考思路</b>(先自己想 3-5 分钟再展开)</summary>

**1.** **数学上**:$A$ 控制状态的演化矩阵,只有它**不**依赖输入,$\bar{A}_t = e^{\Delta_t A}$ 才能在 prefix scan 里被合并成简单的对角矩阵乘积(每通道一个标量)。如果 $A$ 也变,scan 仍然能做(矩阵乘有结合律),但每一步的"合并"代价从 $O(N)$ 变成 $O(N^2)$,GPU kernel 极难高效。**工程上**:Mamba 通过让 $\Delta_t$ 输入依赖,已经隐式实现了"通过 $\bar{A}_t = e^{\Delta_t A}$ 的输入依赖性",在不放弃 scan 效率的前提下拿到了等价表达力。

**2.** Depthwise conv (kernel=4) 让每个位置看到前 3 个位置——这是一个**局部混合**操作。SSM 的状态混合是全局但"逐通道独立"的,缺乏跨通道的局部上下文。短卷积补上了"几 token 内的字符/子词模式"。消融实验显示去掉它困惑度明显上升,特别是对短语级模式(中文里的"机器学习"类似 chunked 模式)。

**3.** 对应关系:
- Transformer Attention ↔ Mamba Selective SSM (序列混合)
- Transformer FFN ↔ Mamba Linear projection (通道混合)
- Transformer LayerNorm/RMSNorm ↔ Mamba 也用 RMSNorm
- Transformer 位置编码 ↔ Mamba **不需要**(SSM 本身是递推,位置信息隐含在状态中)
- 残差 + 因果 mask ↔ 都一样

Mamba 比 Transformer 多的:depthwise conv、gate 分支。这两个其实是从 H3/GSS/RetNet 这条线继承的"recurrent + gated" 设计。

</details>

---

## 4.13 论文/源码对照

| 概念 | 论文符号 / 章节 | 源码位置 |
|---|---|---|
| Selective SSM 公式 | Mamba paper Algorithm 2 | `mamba_ssm/ops/selective_scan_interface.py` |
| $\Delta_t = \text{softplus}(\Delta_{\text{bias}} + \text{Linear}(u_t))$ | Mamba paper §3.2 | `mamba_simple.py` 中 `dt_proj` + `F.softplus` |
| $A$ 参数化为 `A = -exp(A_log)` | Mamba paper Eq.(8) | `mamba_simple.py` 中 `self.A_log` |
| Multi-channel SSM (B,L,D,N) | Mamba paper §3.4 | `selective_scan_fn(u, delta, A, B, C, D, ...)` 中 `u: (B,D,L)`、状态 `(B,D,N)` |
| Depthwise conv 1D | Mamba paper §3.4 Figure 3 | `mamba_simple.py` 中 `self.conv1d` |
| SiLU gate | Mamba paper §3.4 | `mamba_simple.py` 中 `F.silu(z) * y` |
