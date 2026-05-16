# 第 3 节:位置编码 (Positional Encoding)

> 本节目标:理解为什么 Attention 需要位置信息,推导论文中的 sin/cos 编码公式
> $$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

---

## 3.1 致命问题:Attention 是"置换等变"的

先证明一个吓人的事实:

> **如果你打乱输入序列的顺序,Self-Attention 的输出也只是被同样打乱;模型完全不知道哪个 token 在前。**

### 形式化陈述

设置换矩阵 $P \in \{0, 1\}^{n \times n}$($P$ 是单位矩阵行打乱的结果)。它满足:

- $P X$ 表示对 $X$ 的**行**(token)做置换
- $P^T P = I$($P$ 是正交矩阵)

我们证明:

$$
\text{Attention}(PX) = P \cdot \text{Attention}(X)
$$

### 证明

记 $Q = XW^Q$, $K = XW^K$, $V = XW^V$,则:

$$
Q_P = (PX)W^Q = PQ, \quad K_P = PK, \quad V_P = PV
$$

分数矩阵:

$$
S_P = Q_P K_P^T = (PQ)(PK)^T = PQ K^T P^T = P S P^T
$$

注意力分布(逐行 softmax,$P$ 是行置换):

$$
A_P = \text{softmax}\left(\frac{S_P}{\sqrt{d_k}}\right) = P A P^T
$$

输出:

$$
A_P V_P = (P A P^T)(P V) = P A (P^T P) V = P A V = P \cdot \text{Attention}(X) \quad \blacksquare
$$

**结论**:把 "猫追老鼠" 和 "老鼠追猫" 喂给 Attention,出来的只是同样的输出向量被打乱顺序 — 但每个 token 的表示**没有任何不同**。这显然不行,因为顺序对语义至关重要。

---

## 3.2 解决思路:把"位置"也变成一个向量

最自然的想法:**给每个位置 $pos$ 也分配一个 $d_{\text{model}}$ 维的向量** $\text{PE}_{pos}$,然后把它**加到** token 嵌入上:

$$
\tilde{x}_{pos} = x_{pos} + \text{PE}_{pos}
$$

这样模型看到的就是"内容 + 位置"的混合,不同位置的同一个词在输入端就已经不同了。

⚠️ 关键问题:$\text{PE}_{pos}$ 该怎么定?

---

## 3.3 候选方案 1:直接用位置序号 $1, 2, 3, \dots$

最简单的:让 $\text{PE}_{pos} = pos$(广播到所有维度)。

**问题**:
- 位置 1000 的数值会盖过原始嵌入(原始嵌入通常方差 $\approx 1$)
- 模型见过的最大位置是有限的,长序列推理时位置 100000 会"超纲"

**结论**:数值无界,不可行。

---

## 3.4 候选方案 2:归一化到 $[0, 1]$

令 $\text{PE}_{pos} = pos / n$。

**问题**:
- 不同长度的序列里,**同一个绝对位置的编码值不同**(长度 10 中的位置 3 是 0.3,长度 100 中的位置 3 是 0.03)
- 模型学不到"距离 5 个 token"这种**相对位置概念**

**结论**:相对位置无法稳定表达。

---

## 3.5 候选方案 3:可学习的位置嵌入

像词嵌入一样,直接学一个 $\text{PE} \in \mathbb{R}^{n_{\max} \times d_{\text{model}}}$。BERT、GPT-2 用的就是这种。

**优点**:简单、灵活。

**缺点**:
- 必须预先定一个最大长度 $n_{\max}$,超过就没了
- 无法外推到训练时没见过的长度

---

## 3.6 论文方案:正弦/余弦位置编码

Vaswani 等人选择**用固定的三角函数生成位置编码**,无需学习:

$$
\boxed{\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)}
$$

含义:
- $pos$ 是位置索引($0, 1, 2, \dots$)
- $i$ 是维度对的索引($i = 0, 1, \dots, d_{\text{model}}/2 - 1$)
- 偶数维度 $2i$ 用 sin,奇数维度 $2i+1$ 用 cos

这看起来很神秘,但其实有清晰的设计动机。

---

## 3.7 为什么是 sin/cos?— 三个核心性质

### 性质 1:值域有界 $[-1, 1]$

不管 $pos$ 多大,$\sin$ 和 $\cos$ 永远在 $[-1, 1]$,不会盖过 token 嵌入。

### 性质 2:不同维度有不同"频率"

观察分母 $10000^{2i/d_{\text{model}}}$:

- 当 $i = 0$:$10000^0 = 1$,**频率最高**(波长 $2\pi$)
- 当 $i = d_{\text{model}}/2 - 1$:$10000^1 = 10000$,**频率最低**(波长 $2\pi \cdot 10000 \approx 62832$)

也就是说,**低维度变化快,高维度变化慢**。这像**二进制编码**:

```
位置 0: 0000 0000  ← 低位变化快(0,1,0,1,...)
位置 1: 0000 0001
位置 2: 0000 0010
位置 3: 0000 0011  ← 高位变化慢(每 2^i 才变一次)
...
```

不同频率的正弦波组合,可以**唯一地表示一个很长范围内的每个位置**。$d_{\text{model}} = 512$ 维理论上可以无歧义表示远超训练长度的位置。

### 性质 3(最重要):相对位置可以由线性变换得到

这是 sin/cos 真正"魔法"的地方。

设两个位置 $pos$ 和 $pos + k$,它们对应同一对 $(2i, 2i+1)$ 维度的编码是 $(\sin(pos \cdot \omega_i), \cos(pos \cdot \omega_i))$,其中 $\omega_i = 1/10000^{2i/d_{\text{model}}}$。

由三角恒等式:

$$
\sin((pos+k) \omega_i) = \sin(pos \cdot \omega_i)\cos(k \omega_i) + \cos(pos \cdot \omega_i)\sin(k \omega_i)
$$
$$
\cos((pos+k) \omega_i) = \cos(pos \cdot \omega_i)\cos(k \omega_i) - \sin(pos \cdot \omega_i)\sin(k \omega_i)
$$

写成矩阵形式:

$$
\begin{bmatrix} \sin((pos+k)\omega_i) \\ \cos((pos+k)\omega_i) \end{bmatrix}
=
\underbrace{\begin{bmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{bmatrix}}_{\text{只依赖 } k\text{ 的旋转矩阵}}
\begin{bmatrix} \sin(pos \cdot \omega_i) \\ \cos(pos \cdot \omega_i) \end{bmatrix}
$$

**关键观察**:从位置 $pos$ 到位置 $pos+k$,只是**乘以一个只依赖于偏移 $k$ 的旋转矩阵** $R_k$。

这意味着:**模型可以通过学习一个线性投影,把"位置 $pos$ 的编码"映射成"位置 $pos+k$ 的编码"**。换句话说,$\text{PE}_{pos+k}$ 是 $\text{PE}_{pos}$ 的线性函数(只依赖偏移 $k$)。

这就让模型很容易**学习相对位置的概念** — 不需要记住每个绝对位置在做什么,只需要学一个"位移 $k$ 对应什么变换"。

📌 **这一性质会在第 7 节 RoPE(Rotary Position Embedding)中被显式利用 — RoPE 直接把"旋转"作用在 Q、K 上,而不是加到嵌入上。**

---

## 3.8 几何直觉:每对维度是一个"指针"

把每两个相邻维度 $(2i, 2i+1)$ 看作平面上的一个二维点 $(\sin\theta, \cos\theta)$,它就在**单位圆上的某个角度** $\theta$。

随着 $pos$ 增加,这个点**沿圆周旋转**:

```
i=0(高频):     pos=0  pos=1  pos=2  pos=3
                ↑      →      ↓      ←        ← 旋转很快

i=255(低频):    pos=0  pos=1  pos=2  pos=3
                ↑      ↑     ↑     ↑          ← 几乎不动
```

整个 $d_{\text{model}}$ 维向量就是 $d_{\text{model}}/2$ 个不同转速的"时钟指针"的快照。

位置 $pos$ ↔ 一组(快慢不一的)时钟指针的当前角度。

---

## 3.9 论文为什么用 $10000$?

底数 $10000$ 是个超参数,控制最低频率的波长。计算最低频率维度的波长:

$$
\lambda_{\max} = 2\pi \cdot 10000 \approx 62832
$$

含义:最慢的那个"指针"转一圈,要走 62832 个 token。

论文的考虑:训练时最长序列大约几千 token,$62832$ 远大于此,保证模型见到的位置范围内,每个位置都能在"最慢指针"上对应一个**独特的相位**(没有走完一整圈,不会出现混淆)。

---

## 3.10 PyTorch 实现

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 预计算所有可能位置的 PE
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)

        # div_term: 1/10000^(2i/d_model) = exp(-2i * log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维

        # 注册为 buffer(不参与梯度更新,但会跟模型一起保存/加载)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, n, d_model)
        # 把 PE 的前 n 行加到 x 上(广播)
        return x + self.pe[:x.size(1)]
```

注意 `register_buffer`:PE 不是可学习参数,但作为模型状态的一部分。

---

## 3.11 为什么 PE 是"加"到 token 嵌入上,而不是 concat?

直觉上 concat 似乎更"干净"(信息不会相互干扰)。但论文选择加法,理由:

1. **计算效率**:加法不增加维度,后续矩阵运算的形状不变
2. **维度学习**:嵌入空间是高维的($d_{\text{model}} = 512$),token 信息和位置信息可以**自动分布到不同子空间** — 模型自然学会"哪些维度更关心内容,哪些更关心位置"
3. **数学上的合理性**:由 3.7 节,位置变换是 $\text{PE}$ 上的线性映射;只要 token 嵌入和 PE 的"线性子空间"基本正交,加法不会丢失信息

实证上,加法效果不错,沿用至今(直到 RoPE 出现,见第 7 节)。

---

## 3.12 完整 Embedding 层

把词嵌入 + 位置编码组合起来:

```python
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=5000):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.d_model = d_model

    def forward(self, x):
        # x: (batch, n) token id 序列
        # 论文 3.4 节:嵌入要乘以 √d_model
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        return x  # (batch, n, d_model)
```

⚠️ 论文有一个细节:嵌入要乘以 $\sqrt{d_{\text{model}}}$,这是为了让嵌入的数值量级和位置编码的 $[-1, 1]$ 范围匹配(嵌入初始化方差通常是 $1/d_{\text{model}}$,乘了 $\sqrt{d_{\text{model}}}$ 之后方差变成 $1$)。

---

## 3.13 论文符号对照

《Attention Is All You Need》3.5 节:

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})
$$
$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})
$$

论文说:

> We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.

这就是我们在 3.7 节推导的内容。

---

## 3.14 本节核心要点

1. **Attention 是置换等变的** — 不加位置信息就丢失顺序
2. **位置编码是 $d_{\text{model}}$ 维向量,加到 token 嵌入上**
3. **sin/cos 编码** 用不同频率的三角函数,值域有界、相位独特
4. **关键性质**:$\text{PE}_{pos+k} = R_k \cdot \text{PE}_{pos}$,模型可以**线性学习**相对位置
5. **几何直觉**:每对维度是一个时钟指针,$d/2$ 个不同转速的指针共同定位
6. 嵌入要乘 $\sqrt{d_{\text{model}}}$ 来匹配 PE 的数值量级

---

## 3.15 思考题

1. 如果用底数 $2$ 而不是 $10000$,会发生什么?最长可表示的不重复位置是多少?
2. 为什么不直接用复数表示 $e^{i \cdot pos \cdot \omega}$?这样每个维度只需要一个复数,但维度是一半。论文为什么选实数 sin/cos 而不是复数?
3. 如果序列长度超过 max_len(模型训练时的最大),sin/cos 编码能否外推?会有什么问题?(提示:思考最低频维度走过的"圈数")

---

## 3.16 下一节预告

到目前为止,我们手上有了所有 Attention 的零件:Q/K/V → Scaled Dot-Product → Multi-Head → 加位置编码。

现在该把这些组装成一个**完整的 Encoder Block**了。下一节会讲:
- 残差连接(Residual Connection)为什么必要 — 信号衰减和梯度通路
- LayerNorm 的数学,以及 Pre-LN vs Post-LN 的工程差异
- FFN(前馈层)$\text{FFN}(x) = \max(0, xW_1)W_2$ 的角色 — 它其实是 Attention 之外的另一半计算
- 多层堆叠时的注意事项

→ [第 4 节:完整 Encoder Block](04-encoder-block.md)
