# 第 8 节:Jamba 训练与推理

> 本节目标:理解 Jamba 训练和推理时的执行方式,尤其是 Attention KV Cache 与 Mamba 隐状态如何共存。

---

## 8.1 训练目标仍然是语言模型

Jamba 虽然架构混合,但训练目标仍然是 causal language modeling:

$$
P(x_1,\dots,x_L)=\prod_{t=1}^{L}P(x_t\mid x_{<t})
$$

损失:

$$
\mathcal{L}=-\sum_{t=1}^{L}\log P(x_t\mid x_{<t})
$$

也就是说,它仍然是 next-token prediction。

架构变了,目标没变。

---

## 8.2 训练时的并行

训练时输入完整序列:

```
x_1, x_2, ..., x_L
```

Attention 层用 causal mask 并行计算。

Mamba 层用 selective scan 处理整段序列,也尽量并行化。

MoE 层对 token 做路由,把不同 token 分发给不同专家。

所以训练路径大致是:

```
Embedding
↓
[Mamba/Attention + MoE] × N
↓
LM Head
↓
Cross Entropy
```

---

## 8.3 Prefill 阶段

推理开始时,用户给出 prompt:

```
prompt length = L
```

模型要先把 prompt 全部读一遍,这叫 prefill。

在 Jamba 中:

- Attention 层计算 prompt 的 K/V,写入 KV Cache
- Mamba 层扫描 prompt,得到最后的 SSM state
- MoE 层正常按 token 路由

Prefill 之后,模型得到:

1. 少量 Attention 层的 KV Cache
2. 多个 Mamba 层的最终隐状态
3. 最后一个位置的 logits

---

## 8.4 Decode 阶段

之后每次生成一个新 token。

### Attention 层

新 token 产生:

$$
q_t,k_t,v_t
$$

把 $k_t,v_t$ 追加到 KV Cache:

$$
K_{\text{cache}}\leftarrow[K_{\text{cache}};k_t]
$$

$$
V_{\text{cache}}\leftarrow[V_{\text{cache}};v_t]
$$

然后 $q_t$ attend 历史所有 K/V。

### Mamba 层

新 token 更新固定状态:

$$
x_t=\bar{A}_t x_{t-1}+\bar{B}_t u_t
$$

只需要保存新的 $x_t$。

---

## 8.5 缓存增长对比

全 Transformer:

$$
\text{cache} \propto L \times \text{num\_layers}
$$

Jamba:

$$
\text{cache} \approx L \times \text{attention\_layers} + \text{constant} \times \text{mamba\_layers}
$$

因为 Attention 层只占一部分,Jamba 的长上下文缓存压力更小。

这就是它适合长上下文的核心原因之一。

---

## 8.6 Recurrent Inference

Mamba 层天然支持 recurrent inference:

```
拿到一个新 token
↓
更新每层 SSM state
↓
输出当前 token 表示
```

状态大小不随历史长度增长。

这和 RNN 类似,但 Mamba 的训练又不是传统 RNN 那种完全串行训练,因为它有 parallel scan。

所以 Mamba 的理想形态是:

- 训练:并行 scan
- 推理:递推 state

---

## 8.7 混合架构的推理流程

概念伪代码:

```python
def decode_one_token(x, caches):
    for layer in layers:
        if layer.type == "attention":
            x, caches.kv[layer.id] = layer.attention_step(x, caches.kv[layer.id])
        else:
            x, caches.ssm[layer.id] = layer.mamba_step(x, caches.ssm[layer.id])

        x = layer.moe_or_ffn(x)

    logits = lm_head(x)
    return logits, caches
```

缓存里同时有:

```python
caches = {
    "kv": {...},    # attention layers
    "ssm": {...},   # mamba layers
}
```

---

## 8.8 工程取舍

Jamba 的收益:

1. 比全 Attention 更适合长上下文
2. 比纯 Mamba 更保留精确检索能力
3. MoE 增大总容量,激活成本可控

代价:

1. 实现复杂度高
2. 推理缓存有两套机制
3. MoE 路由和专家并行带来通信成本
4. Attention/Mamba 比例需要经验调参

混合架构的难点不只是论文公式,而是把不同计算模式高效拼在一起。

---

## 8.9 和前面 Transformer 笔记的连接

如果已经理解 Transformer:

- Jamba 的 Attention 层就是现代 causal attention
- Jamba 的 MoE 可以看作替换 FFN
- Jamba 的 Mamba 层替换了一部分 attention 序列混合

所以学习路线可以这样记:

```
Transformer: 所有层都用 Attention 混合序列
Mamba: 所有层都用 Selective SSM 混合序列
Jamba: 两者混合,再加 MoE 扩容
```

---

## 8.10 本节核心要点

1. Jamba 训练目标仍然是 next-token prediction
2. 训练时 Attention 用 causal mask,Mamba 用 selective scan
3. Prefill 阶段同时建立 KV Cache 和 SSM state
4. Decode 阶段 Attention 追加 K/V,Mamba 更新固定状态
5. 混合架构降低长上下文缓存压力,但实现复杂度更高

---

## 8.11 完结与下一步

到这里,Jamba 主线已经串起来:

```
为什么混合架构
→ SSM 基础
→ 离散化
→ S4/HiPPO
→ Mamba Selective SSM
→ Parallel Scan
→ MoE
→ Jamba 总体架构
→ 训练与推理
```

如果继续深入,下一步可以读:

1. Mamba 官方实现里的 `selective_scan`
2. Jamba 论文中的 architecture table
3. MoE 系统里的 expert parallel 和 all-to-all
4. 长上下文评测中精确检索与摘要任务的差异
