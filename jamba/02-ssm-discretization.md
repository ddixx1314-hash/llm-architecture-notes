# 第 2 节:SSM 离散化

> 本节目标:把连续时间 SSM 变成可以处理 token 序列的离散递推,理解 $\Delta$、$\bar{A}$、$\bar{B}$ 的来源。

---

## 2.1 为什么需要离散化?

上一节的连续系统是:

$$
\frac{dx(t)}{dt}=Ax(t)+Bu(t)
$$

$$
y(t)=Cx(t)
$$

但语言模型看到的是 token 序列:

$$
u_1,u_2,\dots,u_L
$$

所以我们需要一个离散版本:

$$
x_t=\bar{A}x_{t-1}+\bar{B}u_t
$$

$$
y_t=Cx_t
$$

问题是:

> $\bar{A},\bar{B}$ 和连续参数 $A,B$ 到底是什么关系?

---

## 2.2 时间步长 $\Delta$

离散化需要指定每个 token 之间相隔多长时间:

$$
t_0, t_1, t_2, \dots
$$

如果步长固定:

$$
t_i - t_{i-1} = \Delta
$$

那么从 $t-\Delta$ 到 $t$ 的状态更新由连续解给出:

$$
x(t)=e^{A\Delta}x(t-\Delta)+\int_{0}^{\Delta}e^{A(\Delta-\tau)}Bu(t-\Delta+\tau)d\tau
$$

只要处理积分项,就能得到离散递推。

---

## 2.3 零阶保持(ZOH)

ZOH 的假设是:

> 在一个离散时间间隔内,输入保持常数。

也就是:

$$
u(t-\Delta+\tau)=u_t,\quad 0\leq \tau \leq \Delta
$$

代入积分:

$$
x_t=e^{A\Delta}x_{t-1}+\left(\int_0^\Delta e^{A(\Delta-\tau)}d\tau\right)Bu_t
$$

于是:

$$
\bar{A}=e^{A\Delta}
$$

$$
\bar{B}=\left(\int_0^\Delta e^{A\tau}d\tau\right)B
$$

如果 $A$ 可逆:

$$
\bar{B}=A^{-1}(e^{A\Delta}-I)B
$$

这就是常见的 ZOH 离散化公式。

---

## 2.4 直觉: $\Delta$ 控制记忆速度

看一维情况:

$$
\frac{dx}{dt}=ax+bu
$$

则:

$$
\bar{A}=e^{a\Delta}
$$

如果 $a<0$,状态会衰减。

- $\Delta$ 小:$e^{a\Delta}$ 接近 1,旧状态保留更多
- $\Delta$ 大:$e^{a\Delta}$ 更小,旧状态衰减更快

所以 $\Delta$ 可以理解成当前 token 的"时间跨度"。

Mamba 让 $\Delta$ 依赖输入,这会非常关键。

---

## 2.5 双线性变换(Bilinear / Tustin)

另一种常见离散化是双线性变换:

$$
\bar{A}=(I-\frac{\Delta}{2}A)^{-1}(I+\frac{\Delta}{2}A)
$$

$$
\bar{B}=(I-\frac{\Delta}{2}A)^{-1}\Delta B
$$

它可以看作一种梯形积分近似,数值稳定性较好。

S4 系列常讨论 bilinear transform；Mamba 实现中常见的是类似 ZOH 的 selective scan 离散化形式。

---

## 2.6 离散递推展开

离散 SSM:

$$
x_t=\bar{A}x_{t-1}+\bar{B}u_t
$$

展开几步:

$$
x_1=\bar{A}x_0+\bar{B}u_1
$$

$$
x_2=\bar{A}^2x_0+\bar{A}\bar{B}u_1+\bar{B}u_2
$$

$$
x_3=\bar{A}^3x_0+\bar{A}^2\bar{B}u_1+\bar{A}\bar{B}u_2+\bar{B}u_3
$$

如果 $x_0=0$,输出:

$$
y_t=Cx_t=\sum_{j=1}^{t} C\bar{A}^{t-j}\bar{B}u_j
$$

这就是离散卷积形式。

---

## 2.7 卷积核

定义:

$$
K_k=C\bar{A}^k\bar{B}
$$

那么:

$$
y_t=\sum_{j=1}^{t}K_{t-j}u_j
$$

这说明离散 SSM 等价于一个长卷积:

```
输入序列 u
↓
卷积核 K
↓
输出序列 y
```

如果 $\bar{A},\bar{B},C$ 固定,这个卷积核可以预先算出来,训练时可以并行。

---

## 2.8 Mamba 为什么不能只靠卷积?

Mamba 的选择性参数依赖输入:

$$
\bar{B}_t,\ C_t,\ \Delta_t
$$

这意味着每个位置的系统参数都不同。

于是卷积核不再是固定的:

$$
K_{t-j} \neq C\bar{A}^{t-j}\bar{B}
$$

不能简单预计算一个固定卷积核扫完整段序列。

这就是 Mamba 需要 **selective scan** 的原因:既保留输入依赖的灵活性,又尽量并行化递推。

---

## 2.9 代码骨架:一维离散 SSM

```python
import torch

def run_ssm(u, A_bar, B_bar, C):
    # u: (L,)
    # A_bar: (N, N)
    # B_bar: (N,)
    # C: (N,)
    N = A_bar.size(0)
    x = torch.zeros(N, device=u.device)
    ys = []

    for t in range(u.size(0)):
        x = A_bar @ x + B_bar * u[t]
        y = C @ x
        ys.append(y)

    return torch.stack(ys)
```

这个循环就是 SSM 最朴素的推理形式。

---

## 2.10 本节核心要点

1. 离散化把连续 ODE 变成 token 序列上的递推
2. ZOH 得到 $\bar{A}=e^{A\Delta}$
3. $\Delta$ 控制状态更新的时间尺度
4. 固定参数 SSM 可以写成卷积形式
5. Mamba 的输入依赖参数打破固定卷积核,需要 selective scan

---

## 2.11 下一节预告

下一节看 S4 和 HiPPO:

- 为什么普通 SSM 难以长程记忆?
- HiPPO 想解决什么问题?
- S4 如何把 SSM 变成长序列模型?
- Mamba 和 S4 的关系是什么?

→ [第 3 节:S4 与 HiPPO](03-s4-hippo.md)
