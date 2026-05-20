# 第 7 节:优化器与训练循环

> 本节目标:把"模型 forward 一次"扩展成完整训练流程,理解 SGD、Momentum、AdamW、learning rate schedule、gradient clipping 和 train/eval 模式。

---

## 7.1 一个完整训练 step

神经网络训练不是只写:

```python
loss.backward()
```

而是一套固定闭环:

```python
model.train()

for x, y in dataloader:
    logits = model(x)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

对应数学流程:

$$
\hat{y} = f_\theta(x)
$$

$$
\mathcal{L} = \ell(\hat{y}, y)
$$

$$
g = \nabla_\theta \mathcal{L}
$$

$$
\theta \leftarrow \text{Optimizer}(\theta, g)
$$

---

## 7.2 SGD

最朴素的随机梯度下降:

$$
\theta_{t+1} = \theta_t - \eta g_t
$$

其中:

- $\eta$ 是 learning rate
- $g_t$ 是当前 mini-batch 上的梯度

SGD 的优点是简单、泛化常常不错;缺点是如果 loss surface 弯弯绕绕,它可能走得很慢。

---

## 7.3 Momentum

Momentum 给更新方向加入"惯性":

$$
v_t = \beta v_{t-1} + g_t
$$

$$
\theta_{t+1} = \theta_t - \eta v_t
$$

直觉:

- 如果连续很多步方向一致,速度会积累
- 如果梯度来回震荡,正负方向会互相抵消

所以 Momentum 常能让训练更稳、更快。

---

## 7.4 Adam

Adam 同时维护梯度的一阶矩和二阶矩:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

参数更新近似为:

$$
\theta_{t+1}
=
\theta_t
-
\eta
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

直觉:

- $m_t$ 记录平均梯度方向
- $v_t$ 记录梯度尺度
- 梯度大的参数步子自动小一些,梯度小的参数步子相对大一些

---

## 7.5 AdamW

AdamW 是现代深度学习非常常用的默认选择。

它和 Adam 的关键区别是:**weight decay 解耦**。

AdamW 的更新可以理解成两部分:

$$
\theta \leftarrow \theta - \eta \cdot \text{AdamUpdate}
$$

$$
\theta \leftarrow \theta - \eta \lambda \theta
$$

第二项直接把参数往 0 拉,而不是混进 Adam 的自适应梯度里。

实践建议:

| 参数 | 常见起点 |
|------|----------|
| learning rate | $10^{-3}$ 到 $10^{-4}$ |
| betas | $(0.9, 0.999)$ |
| weight decay | $0.01$ 或 $0.1$ |

---

## 7.6 Learning Rate Schedule

learning rate 太重要了。常见策略:

| 策略 | 直觉 |
|------|------|
| constant | 始终固定 |
| step decay | 每隔一段时间降低 |
| cosine decay | 平滑下降到很小 |
| warmup | 训练开始时从小 lr 慢慢升高 |

Transformer/LLM 训练里常见:

1. 先 warmup,避免初期不稳定
2. 再 cosine decay,让后期慢慢收敛

---

## 7.7 Gradient Clipping

如果梯度爆炸,参数会被一次更新推飞。

Gradient clipping 限制梯度范数:

$$
g \leftarrow g \cdot \frac{c}{\max(c, \|g\|)}
$$

PyTorch:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

RNN/LSTM 训练中经常需要 clipping,因为 BPTT 的梯度链条很长。

---

## 7.8 train/eval 模式

`model.train()` 和 `model.eval()` 会影响某些层:

| 模块 | train 模式 | eval 模式 |
|------|------------|-----------|
| Dropout | 随机置零 | 关闭 |
| BatchNorm | 使用 batch 统计并更新 running stats | 使用 running stats |
| LayerNorm | 基本无差异 | 基本无差异 |

验证/测试时要写:

```python
model.eval()
with torch.no_grad():
    logits = model(x)
```

否则 dropout 还在随机工作,评估结果会乱。

---

## 7.9 一个更完整的训练骨架

```python
import torch
import torch.nn.functional as F

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            pred = logits.argmax(dim=-1)
            total_correct += (pred == y).sum().item()
            total += y.numel()

    acc = total_correct / total
    print(epoch, acc)
```

---

## 7.10 本节核心要点

1. 训练循环由 forward、loss、backward、optimizer step 组成。
2. SGD 直接沿负梯度方向更新。
3. Momentum 用历史梯度方向减少震荡。
4. Adam/AdamW 使用自适应学习率,AdamW 是现代常用默认选择。
5. learning rate schedule 决定训练不同阶段的步长。
6. gradient clipping 可以缓解梯度爆炸。
7. 验证和推理必须使用 `model.eval()` 与 `torch.no_grad()`。

## 思考题

<details>
<summary>为什么训练 loss 正常下降,验证时却波动很大?</summary>

可能原因包括:验证时忘了 `model.eval()`、batch size 太小导致 BatchNorm 统计不稳、验证集太小、learning rate 太大、模型过拟合。第一步通常先检查 train/eval 模式。

</details>
