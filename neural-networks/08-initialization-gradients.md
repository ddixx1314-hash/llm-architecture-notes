# 第 8 节:初始化与梯度稳定

> 本节目标:理解为什么参数不能随便初始化,以及 Xavier、Kaiming、梯度消失/爆炸、残差缩放这些训练稳定性问题。

---

## 8.1 为什么初始化重要?

神经网络训练从随机参数开始。如果初始化太大:

- 激活值可能层层放大
- 梯度可能爆炸
- sigmoid/tanh 进入饱和区,梯度接近 0

如果初始化太小:

- 信号层层衰减
- 梯度也可能消失
- 网络一开始几乎什么都传不过去

好的初始化目标是:

> 前向传播时激活方差不要炸也不要消失;反向传播时梯度方差也尽量稳定。

---

## 8.2 一个方差视角

考虑线性层:

$$
y = Wx
$$

其中:

$$
y_i = \sum_{j=1}^{D} W_{ij}x_j
$$

如果 $W_{ij}$ 和 $x_j$ 独立、均值为 0,那么近似有:

$$
\text{Var}(y_i) = D \cdot \text{Var}(W_{ij}) \cdot \text{Var}(x_j)
$$

为了让:

$$
\text{Var}(y_i) \approx \text{Var}(x_j)
$$

需要:

$$
\text{Var}(W_{ij}) \approx \frac{1}{D}
$$

这就是很多初始化方法背后的直觉。

---

## 8.3 Xavier 初始化

Xavier 初始化常用于 tanh/sigmoid 这类较对称的激活:

$$
\text{Var}(W) = \frac{2}{\text{fan\_in}+\text{fan\_out}}
$$

其中:

- fan_in:输入维度
- fan_out:输出维度

PyTorch:

```python
nn.init.xavier_uniform_(linear.weight)
```

---

## 8.4 Kaiming 初始化

ReLU 会把大约一半负值置 0,所以需要稍微更大的方差:

$$
\text{Var}(W) = \frac{2}{\text{fan\_in}}
$$

这就是 Kaiming/He 初始化。

PyTorch:

```python
nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
```

CNN + ReLU 常用 Kaiming 初始化。

---

## 8.5 梯度消失与梯度爆炸

深层网络反向传播时会出现很多 Jacobian 相乘:

$$
\frac{\partial \mathcal{L}}{\partial h_0}
=
\frac{\partial \mathcal{L}}{\partial h_L}
\prod_{\ell=1}^{L}
\frac{\partial h_\ell}{\partial h_{\ell-1}}
$$

如果每一层平均把梯度缩小一点,很多层后就接近 0。

如果每一层平均把梯度放大一点,很多层后就会爆炸。

所以深层网络需要:

- 合理初始化
- 归一化
- 残差连接
- 梯度裁剪
- 合适的 learning rate

---

## 8.6 为什么残差连接稳定?

普通层:

$$
y = F(x)
$$

残差层:

$$
y = x + F(x)
$$

反向传播:

$$
\frac{\partial y}{\partial x}
=
I + \frac{\partial F(x)}{\partial x}
$$

这个 $I$ 像一条直通梯度路径,让深层网络不必把所有梯度都挤过复杂的 $F$。

---

## 8.7 Transformer 中的小初始化技巧

现代 Transformer 常对某些残差分支做更小初始化。

直觉:如果一开始每个 block 都往残差流里加很大的 $F(x)$,很多层叠起来会让激活尺度变乱。

一种常见思想是让残差分支初始更接近 0:

$$
y = x + \epsilon F(x)
$$

其中 $\epsilon$ 比较小。这样网络刚开始更像恒等映射,训练再慢慢学会每层该加什么。

---

## 8.8 PyTorch 初始化骨架

```python
import torch.nn as nn

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

model.apply(init_weights)
```

---

## 8.9 本节核心要点

1. 初始化决定训练开始时信号和梯度的尺度。
2. 权重太大容易爆炸,太小容易消失。
3. Xavier 适合 tanh/sigmoid 一类激活。
4. Kaiming 适合 ReLU 系列激活。
5. 残差连接提供直接梯度路径。
6. 归一化、残差、初始化、学习率共同决定深层网络是否好训。

## 思考题

<details>
<summary>为什么 ReLU 网络常用 Kaiming 初始化而不是简单标准正态?</summary>

因为 ReLU 会把负半轴置 0,导致输出方差变化。Kaiming 初始化把这个因素考虑进去,让前向激活和反向梯度的尺度更稳定。

</details>
