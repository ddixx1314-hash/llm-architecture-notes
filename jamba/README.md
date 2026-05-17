# Jamba 2.0 mini 架构学习笔记

> 目标:从原理 + 数学推导出发,理解 Jamba 的混合架构(Transformer + Mamba + MoE),最终能读懂 Jamba 论文 (arxiv 2403.19887)、Mamba 论文 (arxiv 2312.00752) 和 `mamba-ssm` 源码。

## 为什么 Jamba 值得学

Jamba (AI21 Labs, 2024-2026) 是第一个**生产级**把三种范式融合的 LLM:

- **Transformer Attention**: 精确召回 / 全局依赖,$O(n^2)$ 复杂度
- **Mamba (Selective SSM)**: 线性 $O(n)$ 复杂度,长序列友好
- **MoE**: 稀疏激活,扩容不增推理算力

Jamba 2.0 mini 规格:**52B 总参数 / 12B 激活参数**, **256K 上下文**, Apache 2.0 开源。

## 学习路线

| 节 | 主题 | 关键数学 | 状态 |
|----|------|---------|------|
| [0](00-why-hybrid.md) | 为什么需要混合架构(Transformer 与 RNN 的两难) | $O(n^2)$ vs $O(n)$ 的张力 | ✅ |
| [1](01-ssm-basics.md) | 状态空间模型(SSM)基础(连续时间) | 线性 ODE、矩阵指数 | ✅ |
| [2](02-ssm-discretization.md) | SSM 离散化:从 ODE 到递推 | 零阶保持(ZOH)、双线性变换 | ✅ |
| [3](03-s4-hippo.md) | S4 与 HiPPO:长程记忆的核心 | Legendre 多项式、HiPPO-LegS | ✅ |
| [4](04-mamba-selective.md) | Mamba:Selective SSM(输入依赖参数化) | $\Delta, B, C$ 随输入变化 | ✅ |
| [5](05-mamba-parallel-scan.md) | 硬件感知并行扫描 | parallel prefix scan, SRAM 编排 | ✅ |
| [6](06-moe.md) | Mixture of Experts(稀疏路由) | Top-K gating、load balance loss | ✅ |
| [7](07-jamba-architecture.md) | Jamba 整体架构:1:7 比例 + MoE | block 设计、参数预算 | ✅ |
| [8](08-jamba-training-inference.md) | 训练 & 推理:KV cache vs 隐状态 | recurrent inference | ✅ |

## 学习方式

- **每节包含**:动机 → 数学推导 → 论文符号对照 → 代码骨架(PyTorch 伪代码)
- **符号约定**(贯穿全部笔记):
  - $L$:序列长度
  - $D$ 或 $d_{\text{model}}$:模型隐藏维度
  - $N$:SSM 的**状态维度**(state size,Mamba 中常用 $N=16$)
  - $E$:MoE 的专家数量
  - $K$:MoE 的 Top-K(Jamba 中 $K=2$)
  - $u(t) \in \mathbb{R}$:SSM 的标量输入信号(连续时间)
  - $x(t) \in \mathbb{R}^N$:SSM 的**状态向量**(注意:这里 $x$ 不是输入!和 Transformer 笔记符号不同)
  - $y(t) \in \mathbb{R}$:SSM 的标量输出
  - $A \in \mathbb{R}^{N \times N}, B \in \mathbb{R}^{N \times 1}, C \in \mathbb{R}^{1 \times N}$:SSM 的三个核心矩阵

> ⚠️ **符号警告**:控制论传统里 $x$ 是"状态",$u$ 是"输入"。这和 Transformer 笔记中 $X$ 表示输入嵌入是冲突的。本目录沿用 Mamba 论文的控制论符号,请在脑中区分。

## 前置知识(本目录假设你已掌握)

- 线性代数:矩阵指数 $e^{At}$、特征分解
- 微积分:一阶线性 ODE 的解
- 神经网络基础 + Transformer Self-Attention(并行学习中,见 [../transformer/](../transformer/))

## 参考资源

- **Jamba 论文**: Lieber et al., 2024, *Jamba: A Hybrid Transformer-Mamba Language Model* ([本地 PDF](references/jamba-hybrid-transformer-mamba-2403.19887.pdf), [arxiv 2403.19887](https://arxiv.org/abs/2403.19887))
- **Mamba 论文**: Gu & Dao, 2023, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* ([本地 PDF](references/mamba-linear-time-sequence-modeling-2312.00752.pdf), [arxiv 2312.00752](https://arxiv.org/abs/2312.00752))
- **S4 论文**: Gu et al., 2021, *Efficiently Modeling Long Sequences with Structured State Spaces* ([本地 PDF](references/s4-efficiently-modeling-long-sequences-2111.00396.pdf), [arxiv 2111.00396](https://arxiv.org/abs/2111.00396))
- **HiPPO 论文**: Gu et al., 2020, *HiPPO: Recurrent Memory with Optimal Polynomial Projections* ([本地 PDF](references/hippo-recurrent-memory-polynomial-projections-2008.07669.pdf), [arxiv 2008.07669](https://arxiv.org/abs/2008.07669))
- **官方实现**: [state-spaces/mamba](https://github.com/state-spaces/mamba)
- **AI21 模型卡**: https://docs.ai21.com/docs/jamba-foundation-models

## 配套脚本与可视化

笔记里的图和小型可运行实现都在 [scripts/](scripts/):

| 脚本 | 作用 | 涉及章节 |
|---|---|---|
| [scripts/generate_figures.py](scripts/generate_figures.py) | matplotlib 生成 4 张可视化图(SSM 状态演化、Δ 控制记忆、prefix scan 对比、cache 增长) | §1.5, §2.4, §5.4, §8.5 |
| [scripts/mini_mamba.py](scripts/mini_mamba.py) | 端到端可跑的微型 Mamba(~60K 参数,Sonnet 18 训练 ~1 min CPU,sanity check 验证 scan == recurrent step) | §1–§5 |
| [scripts/mini_jamba.py](scripts/mini_jamba.py) | 混合架构演示:4 层(3 Mamba + 1 Attention),其中 2 层用 Top-2-of-4 MoE,共存的 KV cache + SSM state,一致性验证全部 4 类层 | §6–§8 |

运行:

```bash
conda activate learning                # 使用主目录共享的 conda 环境
cd jamba/
python scripts/generate_figures.py     # 生成 images/*.png
python scripts/mini_mamba.py           # ~1 分钟 CPU
python scripts/mini_jamba.py           # ~5-10 分钟 CPU
```

依赖:`torch` + `matplotlib` + `numpy`,全部包含在 [learning/ 主目录的共享 conda 环境](../README.md#-环境设置)中。无 `mamba-ssm`、`flash-attn` 等高性能库——naive Python scan,目的是清晰而非快。
