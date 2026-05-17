# Transformer 架构学习笔记

> 目标:从原理 + 数学推导出发,最终能看懂 Transformer/LLM 相关论文(如《Attention Is All You Need》)和源码(如 GPT/LLaMA 实现)。

## 学习路线

| 节 | 主题 | 关键数学 | 状态 |
|----|------|---------|------|
| [0](00-why-transformer.md) | 为什么需要 Transformer(RNN 的局限) | 序列依赖、并行性 | ✅ |
| [1](01-self-attention.md) | Self-Attention 的核心:Q、K、V | 点积、softmax、缩放 | ✅ |
| [2](02-multi-head-attention.md) | Multi-Head Attention | 分头投影、拼接 | ✅ |
| [3](03-positional-encoding.md) | 位置编码 (Positional Encoding) | sin/cos 频率分解 | ✅ |
| [4](04-encoder-block.md) | 完整 Encoder Block(Add & Norm、FFN) | LayerNorm、残差 | ✅ |
| [5](05-decoder-masked-attention.md) | Decoder + Masked Attention | 上三角 mask、交叉注意力 | ✅ |
| [6](06-training-inference.md) | 训练目标与推理(KV Cache、causal LM) | 自回归 | ✅ |
| [7](07-modern-variants.md) | 现代变种(GPT/LLaMA):RMSNorm、RoPE、GQA | 旋转位置编码 | ✅ |

## 学习方式

- **每节包含**:动机 → 数学推导 → 论文符号对照 → 代码骨架(PyTorch 伪代码)
- **符号约定**(贯穿全部笔记):
  - $n$:序列长度(token 数量)
  - $d_{\text{model}}$:模型隐藏维度(论文中常用 $d_{\text{model}}=512$)
  - $h$:注意力头数(论文中 $h=8$)
  - $d_k = d_v = d_{\text{model}} / h$:每个头的 K/V 维度
  - $X \in \mathbb{R}^{n \times d_{\text{model}}}$:输入序列的嵌入矩阵

## 阅读建议

**线性阅读**(推荐第一次学习):按 0 → 1 → 2 → … → 7 顺序读。前面的概念被后面反复使用。

**按需跳读**(已有部分基础时):

| 你的情况 | 建议起点 |
|---------|---------|
| 完全没接触过 Transformer | 从 [第 0 节](00-why-transformer.md) 开始 |
| 知道 Attention 概念但不懂 Q/K/V 数学 | 从 [第 1 节](01-self-attention.md) 开始 |
| 懂 Self-Attention,想了解多头 | 从 [第 2 节](02-multi-head-attention.md) 开始 |
| 想搞懂位置编码 / RoPE 的数学动机 | [第 3 节](03-positional-encoding.md) → 跳到 [第 7.7-7.9 节](07-modern-variants.md) |
| 想看完整模型怎么搭起来 | [第 4 节](04-encoder-block.md) + [第 5 节](05-decoder-masked-attention.md) |
| 想理解 LLM 推理为什么慢、KV Cache 怎么工作 | [第 6 节](06-training-inference.md) |
| 想知道 GPT/LLaMA 在原始 Transformer 上改了什么 | [第 7 节](07-modern-variants.md) |
| 想学习长上下文/低复杂度替代架构 | 完成本系列后看 [Jamba 笔记](../jamba/README.md) |

**思考题**:每节末尾的思考题都附有"参考思路"折叠块。建议先自己想 5-10 分钟再展开看。

## 参考资源

- 原论文:Vaswani et al., 2017, *Attention Is All You Need* ([本地 PDF](references/attention-is-all-you-need-1706.03762.pdf), [arxiv 1706.03762](https://arxiv.org/abs/1706.03762))
- Attention 可视化分析:Clark et al., 2019, *What Does BERT Look At?* ([本地 PDF](references/what-does-bert-look-at-1906.04341.pdf), [arxiv 1906.04341](https://arxiv.org/abs/1906.04341))
- RMSNorm:Zhang & Sennrich, 2019, *Root Mean Square Layer Normalization* ([本地 PDF](references/root-mean-square-layer-normalization-1910.07467.pdf), [arxiv 1910.07467](https://arxiv.org/abs/1910.07467))
- 经典源码:[The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- 现代实现参考:nanoGPT、llama.cpp、HuggingFace Transformers

## 配套代码

学完所有章节后,可以跑一遍把零件串起来的完整实现:

- [scripts/mini_gpt.py](scripts/mini_gpt.py) — ~300 行从零实现的 decoder-only Transformer:
  - 涵盖 PE / 多头因果自注意力 / Pre-LN block / FFN / lm_head 权重绑定 / KV cache 推理
  - 每个组件的注释指向对应章节
  - 在 CPU 上 10 秒训练一个 ~66K 参数模型,字符级拟合莎士比亚十四行诗
  - 自带 KV cache 正确性 sanity check(对比 step-by-step decode 和 full forward 的 logits 差异)
  - 运行:`python scripts/mini_gpt.py`
- [scripts/generate_figures.py](scripts/generate_figures.py) — 生成 [images/](images/) 下的可视化图(RNN vs Transformer、PE 热图)

## 环境设置

本项目使用 [learning/ 主目录共享的 conda 环境](../README.md#-环境设置)。第一次设置完成后:

```bash
conda activate learning
python scripts/mini_gpt.py
python scripts/generate_figures.py
```

完整环境设置 / 切换 GPU / 修改依赖,见 [主 README 的环境章节](../README.md#-环境设置)。


