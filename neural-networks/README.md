# 基础神经网络学习笔记

> 目标:先把常见神经网络家族的基本结构、数学形式、适用数据类型和 PyTorch 骨架打通,再进入 Transformer、Mamba、Jamba 等现代架构时就不会只看到一堆陌生模块。

## 学习路线

| 节 | 主题 | 解决什么问题 | 状态 |
|----|------|-------------|------|
| [0](00-mlp-backprop.md) | MLP 与反向传播 | 神经网络最小闭环:线性层、激活、loss、梯度下降 | ✅ |
| [1](01-cnn.md) | CNN 卷积神经网络 | 图像/局部模式识别,为什么卷积比全连接更省参数 | ✅ |
| [2](02-rnn.md) | RNN 循环神经网络 | 序列建模,隐藏状态如何携带历史信息 | ✅ |
| [3](03-lstm-gru.md) | LSTM 与 GRU | RNN 长程依赖困难,门控如何缓解梯度消失 | ✅ |
| [4](04-autoencoder.md) | Autoencoder 自编码器 | 无监督表示学习,压缩、去噪、重构 | ✅ |
| [5](05-gnn.md) | GNN 图神经网络 | 图结构数据,节点如何从邻居聚合信息 | ✅ |
| [6](06-normalization-regularization.md) | 归一化与正则化 | 为什么 BatchNorm/LayerNorm/Dropout/残差能让网络更好训练 | ✅ |

## 阅读建议

**第一次学习**建议按 0 → 1 → 2 → 3 → 4 → 5 → 6 顺序读:

- 第 0 节是所有后续模型的共同底座。
- CNN 偏图像和局部结构,RNN/LSTM/GRU 偏序列,GNN 偏关系结构。
- 第 6 节不是一个单独模型,但它解释了很多现代网络为什么能训练得深。

**和 Transformer 系列的关系**:

| 这里的基础概念 | 后续会在哪里继续出现 |
|---------------|--------------------|
| 矩阵乘、非线性、反向传播 | Transformer/Jamba 所有模块 |
| RNN 的隐藏状态和串行瓶颈 | Transformer 第 0 节的动机 |
| 门控机制 | LSTM/GRU、SwiGLU、Mamba selective gate |
| LayerNorm、残差连接 | Transformer Encoder/Decoder block |
| 表示学习和瓶颈 | Autoencoder、embedding、latent representation |

## 通用符号

| 符号 | 含义 |
|------|------|
| $B$ | batch size |
| $D$ | 输入特征维度 |
| $H$ | hidden size |
| $C$ | 通道数或类别数,具体看上下文 |
| $T$ | 序列长度 |
| $x$ | 输入样本或输入向量 |
| $h$ | hidden state / hidden representation |
| $\hat{y}$ | 模型预测 |
| $y$ | 真实标签 |
| $\mathcal{L}$ | loss function |

## 学完之后应该能回答

1. 为什么神经网络本质上是"可学习的函数组合"?
2. CNN 的参数量为什么和图片尺寸不直接绑定?
3. RNN 为什么训练和推理都天然串行?
4. LSTM/GRU 的 gate 到底在控制什么?
5. Autoencoder 学到的 latent representation 有什么用?
6. GNN 的 message passing 和 attention 有什么相似之处?
7. BatchNorm、LayerNorm、Dropout、残差分别在解决什么训练问题?
