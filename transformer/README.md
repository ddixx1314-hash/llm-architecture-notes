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

## 参考资源

- 原论文:Vaswani et al., 2017, *Attention Is All You Need* ([本地 PDF](references/attention-is-all-you-need-1706.03762.pdf), [arxiv 1706.03762](https://arxiv.org/abs/1706.03762))
- Attention 可视化分析:Clark et al., 2019, *What Does BERT Look At?* ([本地 PDF](references/what-does-bert-look-at-1906.04341.pdf), [arxiv 1906.04341](https://arxiv.org/abs/1906.04341))
- RMSNorm:Zhang & Sennrich, 2019, *Root Mean Square Layer Normalization* ([本地 PDF](references/root-mean-square-layer-normalization-1910.07467.pdf), [arxiv 1910.07467](https://arxiv.org/abs/1910.07467))
- 经典源码:[The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- 现代实现参考:nanoGPT、llama.cpp、HuggingFace Transformers
