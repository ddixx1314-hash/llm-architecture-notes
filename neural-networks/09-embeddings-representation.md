# 第 9 节:Embedding 与表示学习

> 本节目标:理解离散符号如何变成连续向量,为什么 embedding 是语言模型、推荐系统、图模型里的共同基础。

---

## 9.1 为什么需要 Embedding?

神经网络擅长处理连续向量,但很多输入是离散符号:

- 单词/token id
- 用户 id
- 商品 id
- 类别特征
- 图里的节点 id

不能直接把 id 数字当成大小使用。比如 token id 100 并不比 token id 10 "大 10 倍"。

Embedding 做的事是:

> 为每个离散 id 学一个向量。

---

## 9.2 Embedding 查表

设词表大小为 $V$,embedding 维度为 $D$:

$$
E \in \mathbb{R}^{V \times D}
$$

输入 token id:

$$
i \in \{0,1,\dots,V-1\}
$$

输出就是第 $i$ 行:

$$
x_i = E[i]
$$

这本质上是查表,不是普通的数值缩放。

---

## 9.3 One-hot 与矩阵乘

如果把 id $i$ 写成 one-hot 向量:

$$
e_i \in \mathbb{R}^{V}
$$

那么:

$$
x_i = e_i^T E
$$

结果就是取出 $E$ 的第 $i$ 行。

所以 embedding lookup 可以看成是 one-hot 乘矩阵的高效实现。

---

## 9.4 Embedding 学到了什么?

Embedding 的训练信号来自下游任务。

在语言模型里,如果两个 token 经常出现在相似上下文,它们的 embedding 往往会靠近。

在推荐系统里,如果两个商品被相似用户喜欢,它们的 embedding 往往会靠近。

这就是表示学习的直觉:

> 不是手工规定"相似",而是让任务 loss 推动向量空间自己形成结构。

---

## 9.5 Token Embedding + Position Embedding

Transformer 输入通常是:

$$
x_t = E_{\text{token}}[\text{id}_t] + E_{\text{pos}}[t]
$$

token embedding 表示"是什么词"。

position embedding 表示"在第几个位置"。

如果没有位置信息,self-attention 本身对输入顺序不敏感,模型很难区分:

```
猫 追 狗
狗 追 猫
```

---

## 9.6 权重绑定

语言模型输入端有 token embedding:

$$
E \in \mathbb{R}^{V \times D}
$$

输出端 lm_head 常把 hidden state 映射回词表 logits:

$$
\text{logits} = h E_{\text{out}}^T
$$

很多模型会做 weight tying:

$$
E_{\text{out}} = E_{\text{token}}
$$

直觉:输入侧"词的表示"和输出侧"预测词的分类器"共享同一套词向量空间。

---

## 9.7 PyTorch 代码骨架

```python
import torch
import torch.nn as nn

class TinyEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids):
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        logits = self.lm_head(x)
        return logits

model = TinyEmbeddingModel(vocab_size=10000, d_model=256, max_len=512)
input_ids = torch.randint(0, 10000, (4, 32))
logits = model(input_ids)
print(logits.shape)  # (4, 32, 10000)
```

---

## 9.8 Embedding 的几个坑

| 坑 | 说明 |
|----|------|
| 把 id 当数值 | id 只是编号,没有连续大小意义 |
| 词表过大 | embedding 参数量会很大 |
| 稀有 token | 训练次数少,向量质量差 |
| OOV | 词表外 token 需要 `<unk>` 或 tokenizer 处理 |
| padding | padding id 通常不参与 loss |

PyTorch 里可以设置:

```python
nn.Embedding(vocab_size, d_model, padding_idx=0)
```

这样 padding 向量不会被正常梯度更新。

---

## 9.9 本节核心要点

1. Embedding 把离散 id 映射成连续向量。
2. Embedding lookup 等价于 one-hot 乘 embedding 矩阵。
3. 向量空间结构由训练任务塑造。
4. Transformer 输入通常是 token embedding 加 position embedding。
5. 语言模型常用输入 embedding 和输出 lm_head 权重绑定。
6. padding、稀有 token、词表大小都是实际训练中的关键细节。

## 思考题

<details>
<summary>为什么不能直接把 token id 当成模型输入?</summary>

因为 token id 只是人为编号,数值大小没有语义。直接输入 id 会让模型误以为 id 之间存在连续大小关系,而 embedding 可以为每个 id 学独立向量,再由任务决定它们在向量空间中的关系。

</details>
