# learning — 深度学习/大语言模型架构学习笔记

> 从原理 + 数学推导出发,系统学习现代大语言模型的核心架构。每个子项目都遵循同一套教科书化模板:**动机 → 数学推导 → 论文符号对照 → 可跑代码 → 思考题(含参考思路)**。

## 已完成的子项目

### 📙 [neural-networks/](neural-networks/) — 基础神经网络

从最基础的 MLP、反向传播开始,补齐 CNN、RNN、LSTM/GRU、Autoencoder、GNN、训练技巧、Embedding 以及 Seq2Seq Attention 等常见模块,作为阅读 Transformer/Jamba 前的神经网络底座。

| 节 | 主题 |
|----|------|
| [0](neural-networks/00-mlp-backprop.md) | MLP 与反向传播(线性层、激活、loss、梯度下降) |
| [1](neural-networks/01-cnn.md) | CNN 卷积神经网络(局部连接、权重共享、padding/stride) |
| [2](neural-networks/02-rnn.md) | RNN 循环神经网络(hidden state、BPTT、串行瓶颈) |
| [3](neural-networks/03-lstm-gru.md) | LSTM 与 GRU(门控、长期记忆、梯度消失缓解) |
| [4](neural-networks/04-autoencoder.md) | Autoencoder 自编码器(压缩、重构、去噪、VAE 直觉) |
| [5](neural-networks/05-gnn.md) | GNN 图神经网络(message passing、GCN、GAT 直觉) |
| [6](neural-networks/06-normalization-regularization.md) | 归一化与正则化(BatchNorm、LayerNorm、Dropout、残差) |
| [7](neural-networks/07-optimizers-training-loop.md) | 优化器与训练循环(SGD、Momentum、AdamW、scheduler) |
| [8](neural-networks/08-initialization-gradients.md) | 初始化与梯度稳定(Xavier、Kaiming、梯度消失/爆炸) |
| [9](neural-networks/09-embeddings-representation.md) | Embedding 与表示学习(token/id 到连续向量) |
| [10](neural-networks/10-seq2seq-attention.md) | Seq2Seq 与 Attention 过渡(RNN attention 到 Transformer) |
| [11](neural-networks/11-modern-cnn-blocks.md) | 现代 CNN Block(ResNet、1x1 卷积、深度可分离卷积) |

**配套代码**:
[basic_models.py](neural-networks/scripts/basic_models.py)、
[sequence_models.py](neural-networks/scripts/sequence_models.py)、
[representation_models.py](neural-networks/scripts/representation_models.py)、
[generate_figures.py](neural-networks/scripts/generate_figures.py)

**参考论文**:[references.md](neural-networks/references.md)
**论文原图导航**:[paper-figures.md](neural-networks/paper-figures.md)

### 📘 [transformer/](transformer/) — Transformer 架构

从 RNN 的局限出发,推导 Self-Attention 完整数学,搭建到完整 Encoder-Decoder,最后看现代 LLM(GPT/LLaMA)的关键改造。

| 节 | 主题 |
|----|------|
| [0](transformer/00-why-transformer.md) | 为什么需要 Transformer(RNN 的并行性 + 长程依赖问题) |
| [1](transformer/01-self-attention.md) | Self-Attention(Q/K/V 推导,√dₖ 缩放的统计分析) |
| [2](transformer/02-multi-head-attention.md) | Multi-Head Attention(子空间并行的"免费午餐") |
| [3](transformer/03-positional-encoding.md) | 位置编码(sin/cos 频率分解 + 相对位置的旋转矩阵) |
| [4](transformer/04-encoder-block.md) | Encoder Block(残差、LayerNorm、FFN、Pre/Post-LN) |
| [5](transformer/05-decoder-masked-attention.md) | Decoder + Masked Attention(causal mask + cross-attention) |
| [6](transformer/06-training-inference.md) | 训练 & 推理(teacher forcing + KV Cache) |
| [7](transformer/07-modern-variants.md) | 现代变种(RMSNorm、SwiGLU、RoPE、GQA) |

**配套代码**:[scripts/mini_gpt.py](transformer/scripts/mini_gpt.py)(~300 行 nanoGPT 风格实现,CPU 上 10 秒训练)

### 📗 [jamba/](jamba/) — Jamba 2.0 mini 混合架构

理解为什么需要把 Transformer + Mamba(SSM)+ MoE 融合,从 SSM 数学基础到 Jamba 完整架构。

| 节 | 主题 |
|----|------|
| [0](jamba/00-why-hybrid.md) | 为什么需要混合架构 |
| [1](jamba/01-ssm-basics.md) | 状态空间模型基础(连续 ODE) |
| [2](jamba/02-ssm-discretization.md) | SSM 离散化(ZOH、双线性) |
| [3](jamba/03-s4-hippo.md) | S4 与 HiPPO 长程记忆 |
| [4](jamba/04-mamba-selective.md) | Mamba Selective SSM(输入依赖参数) |
| [5](jamba/05-mamba-parallel-scan.md) | 硬件感知并行扫描 |
| [6](jamba/06-moe.md) | Mixture of Experts |
| [7](jamba/07-jamba-architecture.md) | Jamba 整体架构(1:7 比例 + MoE) |
| [8](jamba/08-jamba-training-inference.md) | 训练 & 推理(KV Cache vs 隐状态) |

**配套代码**:[mini_mamba.py](jamba/scripts/mini_mamba.py)、[mini_jamba.py](jamba/scripts/mini_jamba.py)

## 学习路线建议

```
neural-networks/ (基础) ──► transformer/ (必读) ──► jamba/ (建议读完 transformer 再学)
```

`transformer/` 默认你已经理解线性层、激活函数、反向传播、归一化、残差等基础概念;`jamba/` 假设你已经熟悉 Self-Attention、Multi-Head、KV Cache 等概念,会直接对比"Mamba 在哪些地方替代 Transformer、在哪些地方共存"。

## 通用符号约定

| 符号 | 含义 |
|------|------|
| $L$ 或 $n$ | 序列长度 |
| $d_{\text{model}}$ 或 $D$ | 模型隐藏维度 |
| $h$ | 注意力头数 |
| $d_k = d_{\text{model}}/h$ | 每个 attention head 的 K/Q 维度 |
| $N$(仅 jamba) | SSM 状态维度 |
| $E, K$(仅 jamba) | MoE 专家总数 / Top-K |

> ⚠️ jamba/ 沿用 Mamba 论文的**控制论符号**:$x$ 表示**状态**(不是输入),$u$ 表示输入。和 transformer/ 笔记中 $X$ 表示输入嵌入不同。详见 [jamba/README.md](jamba/README.md)。

---

## 🛠 环境设置

整个 learning 项目**共享一个 conda 环境**(`learning`),所有子项目都可直接用,不需要分别建环境。

### 第一次设置(只做一次)

```bash
# 1. 装 Miniforge(社区 conda 发行版,无 Anaconda 商业许可问题)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
~/miniforge3/bin/conda init bash
# 重启 shell 后,prompt 会出现 (base)

# 2. 从仓库根目录的 environment.yml 创建 learning 环境
cd /home/dong/learning
conda env create -f environment.yml      # 几分钟,装 Python 3.11 + PyTorch CPU + numpy + matplotlib

# 3. 激活
conda activate learning
```

### 日常使用

```bash
conda activate learning

# 跑 transformer 配套代码
python transformer/scripts/mini_gpt.py
python transformer/scripts/generate_figures.py

# 跑 neural-networks 配套代码
python neural-networks/scripts/basic_models.py
python neural-networks/scripts/sequence_models.py
python neural-networks/scripts/representation_models.py
python neural-networks/scripts/generate_figures.py

# 跑 jamba 配套代码
python jamba/scripts/mini_mamba.py
python jamba/scripts/mini_jamba.py
python jamba/scripts/generate_figures.py

conda deactivate
```

### 环境文件说明

- [environment.yml](environment.yml) — **精简版**(5 行直接依赖),conda solver 自动决定子依赖版本。日常修改环境编辑这个。
- [environment.lock.yml](environment.lock.yml) — **锁文件**(169 行,所有传递依赖锁到具体版本),用于精确复现。每次 `conda install` 新包后用 `conda env export --no-builds > environment.lock.yml` 重新生成。

### 修改环境

```bash
# 加新包后,先更新精简版 environment.yml,再:
conda env update -f environment.yml --prune

# 装完后同步 lock 文件:
conda env export --no-builds > environment.lock.yml
```

### 切换到 GPU 版(WSL2 + NVIDIA)

```bash
# 改 environment.yml:把 `cpuonly` 改成 `pytorch-cuda=12.1`,然后:
conda env update -f environment.yml --prune
```

详见 [Transformer 笔记中 §1](transformer/README.md) 和 [Jamba 笔记中 §6](jamba/README.md) 关于环境的注意事项。

---

## 项目目录结构

```
learning/
├── README.md                 ← 你在看的这个文件
├── environment.yml           ← 共享 conda 环境(精简)
├── environment.lock.yml      ← 共享 conda 环境(精确锁版本)
├── .gitignore
│
├── neural-networks/          ← 子项目 0:基础神经网络
│   ├── README.md
│   ├── references.md         经典论文清单
│   ├── paper-figures.md      论文原图导航
│   ├── references/           可公开下载的论文 PDF
│   ├── 00..11-*.md           12 节基础笔记
│   ├── images/               可视化图
│   └── scripts/              MLP/CNN/RNN/Autoencoder/GNN 等可运行示例 + 图片生成
│
├── transformer/              ← 子项目 1:Transformer
│   ├── README.md             该子项目的学习路线 + 阅读建议
│   ├── 00..07-*.md           8 节笔记
│   ├── images/               可视化图
│   ├── scripts/              配套代码 + 图片生成脚本
│   └── references/           论文 PDF
│
└── jamba/                    ← 子项目 2:Jamba 混合架构
    ├── README.md
    ├── 00..08-*.md           9 节笔记
    ├── images/
    ├── scripts/              mini_mamba.py + mini_jamba.py + 图
    └── references/           论文 PDF
```

## 写作风格(各子项目通用)

每一节笔记的结构:

1. **动机**:这一节解决什么问题?上一节留下了什么疑问?
2. **数学推导**:严谨写出关键公式 + 证明(用 `$$...$$`),关键步骤展开
3. **几何/物理直觉**:把数学翻译成可视化的图像或者比喻
4. **论文符号对照**:把笔记的符号映射到原论文 3.x 节的符号,方便对照阅读
5. **代码骨架**:PyTorch 伪代码,注释指明对应章节
6. **复杂度分析**(如有):时间/空间复杂度,与替代方案对比
7. **本节核心要点**:6 条左右,可以当复习卡片
8. **思考题 + 参考思路**:每节末尾的 `<details>` 折叠块,先想再展开

⚠️ 使用 `⚠️` 标注容易踩坑或反直觉的细节。
📌 使用 `📌` 标注会在后续章节继续使用的关键性质。

## 相关链接

- Transformer 原论文: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Jamba 论文: [Jamba: A Hybrid Transformer-Mamba LM](https://arxiv.org/abs/2403.19887)
- Mamba 论文: [Mamba: Linear-Time Sequence Modeling with Selective SSMs](https://arxiv.org/abs/2312.00752)
- Miniforge 项目: [conda-forge/miniforge](https://github.com/conda-forge/miniforge)
