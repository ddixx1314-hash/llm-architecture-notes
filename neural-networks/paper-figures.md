# 论文原图导航

> 本项目中的示意图是重新绘制的原创教学图,便于统一风格和中文讲义引用。论文原图通常更权威,但很多论文 PDF 不等于图片可自由复制再发布,所以这里给出"去哪里看原图"的导航,而不是把原图截图进仓库。

## 推荐优先看的论文图

| 主题 | 推荐看哪篇论文 | 图/表位置 | 为什么值得看 |
|------|----------------|-----------|--------------|
| 反向传播 | Rumelhart, Hinton & Williams, 1986, [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) | Nature 页面全文 | 反向传播的历史入口,适合了解隐藏表示为什么重要。 |
| CNN / LeNet | LeCun et al., 1998, [本地 PDF](references/lecun-gradient-based-learning-document-recognition-1998.pdf) | 论文中的 LeNet-5 结构图 | 最经典的卷积、下采样、全连接分类器结构图。 |
| 初始化 | Glorot & Bengio, 2010, [本地 PDF](references/glorot-bengio-deep-feedforward-training-2010.pdf) | 激活/梯度统计相关图 | 能直观看到深层网络为什么会训练困难。 |
| LSTM | Hochreiter & Schmidhuber, 1997, [online](https://doi.org/10.1162/neco.1997.9.8.1735) | LSTM cell 相关图示 | 原始 LSTM 论文的图比较老,但有助于理解 constant error carousel 的动机。 |
| Seq2Seq Attention | Bahdanau, Cho & Bengio, 2014, [本地 PDF](references/bahdanau-attention-nmt-2014.pdf) | Figure 1, Figure 3 | Figure 1 是模型结构图;Figure 3 是 alignment heatmap,非常适合理解 attention 的可解释性。 |
| ResNet | He et al., 2015, [本地 PDF](references/he-resnet-2015.pdf) | Figure 2, Figure 3, Figure 4 | Figure 2 是残差块核心结构;Figure 4 展示 plain net 和 ResNet 的训练差异。 |
| GCN | Kipf & Welling, 2016, [本地 PDF](references/kipf-welling-gcn-2016.pdf) | Figure 1 | 多层 GCN 和节点分类设置的一张总览图。 |
| GAT | Velickovic et al., 2017, [本地 PDF](references/velickovic-gat-2017.pdf) | 模型结构/attention 权重图 | 看图结构上如何做 masked self-attention。 |
| BatchNorm | Ioffe & Szegedy, 2015, [本地 PDF](references/ioffe-szegedy-batchnorm-2015.pdf) | algorithm / network comparison | 重点看 BatchNorm transform 如何嵌入网络层。 |
| Dropout | Srivastava et al., 2014, [本地 PDF](references/srivastava-dropout-2014.pdf) | dropout network 示意图 | 一眼能看懂训练时随机删除神经元、推理时使用完整网络。 |
| MobileNet | Howard et al., 2017, [本地 PDF](references/howard-mobilenets-2017.pdf) | depthwise separable convolution 图/表 | 和普通卷积的计算量对比非常直观。 |

## 和本项目图片的对应关系

| 本项目图片 | 对应章节 | 建议搭配看的论文原图 |
|------------|----------|----------------------|
| [training-loop.png](images/training-loop.png) | [第 0 节](00-mlp-backprop.md) | Rumelhart 1986, Glorot 2010 |
| [cnn-convolution-overview.png](images/cnn-convolution-overview.png) | [第 1 节](01-cnn.md) | LeCun 1998 LeNet-5 结构图 |
| [rnn-unroll.png](images/rnn-unroll.png) | [第 2 节](02-rnn.md) | Cho et al. 2014 encoder-decoder 结构 |
| [lstm-gates.png](images/lstm-gates.png) | [第 3 节](03-lstm-gru.md) | Hochreiter & Schmidhuber 1997 LSTM cell |
| [seq2seq-attention-bridge.png](images/seq2seq-attention-bridge.png) | [第 10 节](10-seq2seq-attention.md) | Bahdanau et al. 2014 Figure 1 和 Figure 3 |
| [gnn-message-passing.png](images/gnn-message-passing.png) | [第 5 节](05-gnn.md) | Kipf & Welling 2016 Figure 1 |
| [modern-cnn-blocks.png](images/modern-cnn-blocks.png) | [第 11 节](11-modern-cnn-blocks.md) | ResNet Figure 2, MobileNet depthwise separable conv 说明 |

## 如果以后要直接使用论文图

建议先确认这三点:

1. 论文或出版社是否明确给出可再发布许可。
2. 图注中是否要求特定 attribution。
3. 仓库是否允许分发该图,而不是只允许个人阅读论文 PDF。

如果这些条件不清楚,更稳的选择是像本项目这样重画概念图,并在文字中引用论文与图号。
