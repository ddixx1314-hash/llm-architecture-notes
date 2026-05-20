# 基础神经网络参考论文

> 这份清单按本子项目章节组织。第一次学习时不必逐篇精读,可以先把每篇论文当成"概念出处 + 深挖入口"。

## 0. MLP 与反向传播

| 论文 | 关键词 | 对应章节 |
|------|--------|----------|
| Rumelhart, Hinton & Williams, 1986, [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) | 反向传播、隐藏层表示 | [第 0 节](00-mlp-backprop.md) |
| Glorot & Bengio, 2010, [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html) | 深层网络训练困难、Xavier 初始化 | [第 8 节](08-initialization-gradients.md) |

## 1. CNN 与现代 CNN

| 论文 | 关键词 | 对应章节 |
|------|--------|----------|
| LeCun et al., 1998, [Gradient-based learning applied to document recognition](https://doi.org/10.1109/5.726791) | LeNet、卷积网络、手写识别 | [第 1 节](01-cnn.md) |
| He et al., 2015/2016, [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | ResNet、残差连接、深层 CNN | [第 11 节](11-modern-cnn-blocks.md) |
| Howard et al., 2017, [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) | depthwise separable convolution、轻量 CNN | [第 11 节](11-modern-cnn-blocks.md) |

## 2. RNN、LSTM、GRU 与 Seq2Seq

| 论文 | 关键词 | 对应章节 |
|------|--------|----------|
| Hochreiter & Schmidhuber, 1997, [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) | LSTM、长期记忆、门控 | [第 3 节](03-lstm-gru.md) |
| Cho et al., 2014, [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) | RNN Encoder-Decoder、GRU | [第 3 节](03-lstm-gru.md) |
| Sutskever, Vinyals & Le, 2014, [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) | Seq2Seq、encoder-decoder | [第 10 节](10-seq2seq-attention.md) |
| Bahdanau, Cho & Bengio, 2014, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) | attention、动态对齐、固定长度瓶颈 | [第 10 节](10-seq2seq-attention.md) |

## 3. 表示学习与生成模型

| 论文 | 关键词 | 对应章节 |
|------|--------|----------|
| Kingma & Welling, 2013, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) | VAE、reparameterization、latent variable model | [第 4 节](04-autoencoder.md) |

## 4. GNN

| 论文 | 关键词 | 对应章节 |
|------|--------|----------|
| Kipf & Welling, 2016/2017, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) | GCN、归一化邻接矩阵、节点分类 | [第 5 节](05-gnn.md) |
| Velickovic et al., 2017/2018, [Graph Attention Networks](https://arxiv.org/abs/1710.10903) | GAT、图上的 masked self-attention | [第 5 节](05-gnn.md) |

## 5. 归一化、正则化、优化器

| 论文 | 关键词 | 对应章节 |
|------|--------|----------|
| Ioffe & Szegedy, 2015, [Batch Normalization](https://arxiv.org/abs/1502.03167) | BatchNorm、mini-batch 统计 | [第 6 节](06-normalization-regularization.md) |
| Ba, Kiros & Hinton, 2016, [Layer Normalization](https://arxiv.org/abs/1607.06450) | LayerNorm、单样本特征归一化 | [第 6 节](06-normalization-regularization.md) |
| Srivastava et al., 2014, [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.jmlr.org/papers/v15/srivastava14a.html) | Dropout、正则化、过拟合 | [第 6 节](06-normalization-regularization.md) |
| Kingma & Ba, 2014/2015, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) | Adam、一阶/二阶矩估计 | [第 7 节](07-optimizers-training-loop.md) |
| Loshchilov & Hutter, 2017/2019, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | AdamW、decoupled weight decay | [第 7 节](07-optimizers-training-loop.md) |
| He et al., 2015, [Delving Deep into Rectifiers](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_Rectifiers_ICCV_2015_paper.html) | Kaiming 初始化、PReLU、ReLU 网络 | [第 8 节](08-initialization-gradients.md) |

## 建议读法

| 目标 | 推荐阅读顺序 |
|------|--------------|
| 先懂训练底层 | Rumelhart 1986 → Glorot 2010 → He 2015 初始化 → Adam/AdamW |
| 先懂图像网络 | LeCun 1998 → ResNet → MobileNet |
| 先懂序列到 Transformer | LSTM → GRU/Encoder-Decoder → Seq2Seq → Bahdanau Attention |
| 先懂图结构 | GCN → GAT |
