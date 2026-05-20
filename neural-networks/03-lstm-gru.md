# 第 3 节:LSTM 与 GRU

> 本节目标:理解门控循环网络如何缓解 vanilla RNN 的长程依赖问题,重点掌握 LSTM/GRU 的 gate 在控制什么。

---

## 3.1 Vanilla RNN 的问题

RNN 的状态更新是:

$$
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
$$

这意味着历史信息每过一步都要被一次非线性变换重新压缩。

如果序列很长,很早的信息要经过很多次变换才到达后面,容易出现:

- 信息被覆盖
- 梯度消失
- 很难学到"几十步以前的某个关键信号"

LSTM 和 GRU 的核心思想是引入 **gate**:

> 不是什么都强行写入 hidden state,而是让网络自己学会"记住多少、忘掉多少、输出多少"。

---

## 3.2 LSTM 的两个状态

LSTM 有两个状态:

- $h_t$: hidden state,对外输出的短期表示
- $c_t$: cell state,更像一条长期记忆通道

LSTM 的关键是 $c_t$ 有一条相对直接的加法路径:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

这比 vanilla RNN 每一步都用 $\tanh(W_hh_{t-1})$ 重写状态更容易保留长期信息。

---

## 3.3 LSTM 三个门

把当前输入和旧 hidden 拼起来:

$$
u_t = [x_t; h_{t-1}]
$$

**遗忘门**决定旧记忆保留多少:

$$
f_t = \sigma(W_f u_t + b_f)
$$

**输入门**决定新候选记忆写入多少:

$$
i_t = \sigma(W_i u_t + b_i)
$$

候选记忆:

$$
\tilde{c}_t = \tanh(W_c u_t + b_c)
$$

更新 cell state:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

**输出门**决定暴露多少记忆给 hidden state:

$$
o_t = \sigma(W_o u_t + b_o)
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

其中 $\odot$ 是逐元素乘法。

---

## 3.4 GRU:更简洁的门控

GRU(Gated Recurrent Unit)把状态合并成一个 $h_t$,结构比 LSTM 更简单。

更新门:

$$
z_t = \sigma(W_z x_t + U_z h_{t-1})
$$

重置门:

$$
r_t = \sigma(W_r x_t + U_r h_{t-1})
$$

候选状态:

$$
\tilde{h}_t = \tanh(W_h x_t + U_h(r_t \odot h_{t-1}))
$$

最终状态:

$$
h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t
$$

直觉:

- $z_t$ 大:更多采用新状态
- $z_t$ 小:更多保留旧状态
- $r_t$ 控制计算候选状态时看多少旧历史

---

## 3.5 LSTM vs GRU

| 模型 | 状态 | 门数量 | 特点 |
|------|------|--------|------|
| Vanilla RNN | $h_t$ | 0 | 简单,但长程依赖弱 |
| LSTM | $h_t,c_t$ | 3 | 表达强,参数多 |
| GRU | $h_t$ | 2 | 更简洁,训练快一些 |

实践中:

- 数据少或模型小:GRU 常是不错起点
- 需要更强长程控制:LSTM 仍然稳
- 大规模语言模型:Transformer 已经大幅替代传统 RNN/LSTM

---

## 3.6 PyTorch 代码骨架

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        outputs, (h_n, c_n) = self.lstm(x)
        last_hidden = outputs[:, -1, :]
        return self.head(last_hidden)

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        outputs, h_n = self.gru(x)
        return self.head(outputs[:, -1, :])
```

---

## 3.7 和现代网络的关系

门控思想没有消失,只是换了形态:

| 模块 | 门控影子 |
|------|----------|
| LSTM/GRU | 显式 sigmoid gate |
| GLU/SwiGLU | 用一条分支控制另一条分支 |
| Mamba selective SSM | 输入决定状态空间参数 |
| Attention | query 决定从哪些 key/value 读取信息 |

📌 读现代架构时,看到"一个分支控制另一个分支"或"输入决定信息流比例",都可以联想到门控。

---

## 3.8 本节核心要点

1. LSTM/GRU 都是为缓解 RNN 长程依赖困难而设计。
2. LSTM 用 cell state 提供更稳定的长期记忆通道。
3. forget/input/output gate 分别控制忘旧、写新、输出。
4. GRU 更简洁,用 update/reset gate 控制状态更新。
5. 门控本质是让模型动态控制信息流。
6. 现代架构中仍然大量保留门控思想。

## 思考题

<details>
<summary>LSTM 为什么比 vanilla RNN 更容易保留长期信息?</summary>

因为 LSTM 的 cell state 有一条加法更新路径:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

当 forget gate 接近 1,input gate 接近 0 时,旧记忆可以近似原样传到下一步,不必每一步都被完全重写。

</details>
