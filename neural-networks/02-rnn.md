# 第 2 节:RNN 循环神经网络

> 本节目标:理解 RNN 如何通过 hidden state 建模序列,为什么它适合时间序列/文本,以及为什么长序列训练容易遇到梯度消失和串行瓶颈。

---

## 2.1 序列数据有什么不同?

很多数据不是独立样本,而是有顺序:

- 文本:前面的词影响后面的词
- 时间序列:过去的观测影响未来趋势
- 语音:前后音素连续变化
- 用户行为:点击序列有上下文

MLP/CNN 通常一次处理固定输入,而 RNN 的核心思想是:

> 每一步读入当前输入,同时保留一个 hidden state 作为历史摘要。

---

## 2.2 Vanilla RNN 公式

给定序列:

$$
x_1, x_2, \dots, x_T
$$

RNN 在每个时间步更新 hidden state:

$$
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
$$

输出可以是每一步都有:

$$
y_t = W_y h_t + c
$$

也可以只取最后一步:

$$
y = W_y h_T + c
$$

其中 $h_t$ 既看当前输入 $x_t$,也看过去摘要 $h_{t-1}$。

---

## 2.3 展开来看

RNN 名字里有"循环",但训练时常把它按时间展开:

```
x1 -> [RNN cell] -> h1
       |
x2 -> [RNN cell] -> h2
       |
x3 -> [RNN cell] -> h3
       |
      ...
```

更准确地写:

```
h0 -> h1 -> h2 -> h3 -> ... -> hT
      ^     ^     ^           ^
      x1    x2    x3          xT
```

所有时间步共享同一组参数 $W_x,W_h,b$。

---

## 2.4 RNN 可以做哪些任务?

| 任务形式 | 输入 | 输出 | 例子 |
|----------|------|------|------|
| many-to-one | 序列 | 一个标签 | 文本情感分类 |
| many-to-many | 序列 | 等长序列 | 词性标注 |
| one-to-many | 一个输入 | 序列 | 图片描述生成 |
| seq2seq | 一个序列 | 另一个序列 | 机器翻译 |

---

## 2.5 BPTT:穿过时间的反向传播

RNN 的训练叫 BPTT(Backpropagation Through Time)。

如果 loss 在最后一步:

$$
\mathcal{L} = \ell(y_T, y)
$$

那么早期 hidden state 的梯度要穿过很多个时间步:

$$
\frac{\partial \mathcal{L}}{\partial h_1}
=
\frac{\partial \mathcal{L}}{\partial h_T}
\frac{\partial h_T}{\partial h_{T-1}}
\cdots
\frac{\partial h_2}{\partial h_1}
$$

其中每一步都有类似 $W_h$ 和 $\tanh'$ 的乘法。

如果这些乘积的范数小于 1,梯度会越来越小;如果大于 1,梯度会越来越大。

这就是:

- **梯度消失**:模型学不到很早以前的信息
- **梯度爆炸**:训练不稳定,loss 乱飞

---

## 2.6 为什么 RNN 难并行?

因为:

$$
h_t = f(x_t, h_{t-1})
$$

计算 $h_t$ 前必须先知道 $h_{t-1}$。

所以时间维度上天然串行:

$$
h_1 \rightarrow h_2 \rightarrow h_3 \rightarrow \cdots \rightarrow h_T
$$

这正是 Transformer 想解决的问题之一:用 self-attention 让所有位置在训练时可以并行交互。

---

## 2.7 PyTorch RNN 骨架

```python
import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)      # (B, T, emb_dim)
        outputs, h_n = self.rnn(x)         # outputs: (B, T, hidden_dim)
        last_hidden = outputs[:, -1, :]    # (B, hidden_dim)
        return self.head(last_hidden)

model = RNNClassifier(vocab_size=10000, emb_dim=128, hidden_dim=256, num_classes=2)
input_ids = torch.randint(0, 10000, (32, 50))
logits = model(input_ids)
```

---

## 2.8 本节核心要点

1. RNN 用 hidden state 携带历史信息。
2. 所有时间步共享同一组参数。
3. BPTT 是把 RNN 沿时间展开后做反向传播。
4. 长序列会带来梯度消失/爆炸问题。
5. RNN 时间维度难并行,因为 $h_t$ 依赖 $h_{t-1}$。
6. Transformer 的并行性动机可以从 RNN 的瓶颈自然引出。

## 思考题

<details>
<summary>为什么 RNN 的推理和训练都难以在时间维度并行?</summary>

因为每个时间步的 hidden state 都依赖前一个 hidden state。即使训练时完整序列已经给定,$h_t$ 仍然必须等 $h_{t-1}$ 算完才能计算。

</details>
