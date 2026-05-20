# 第 5 节:GNN 图神经网络

> 本节目标:理解图神经网络如何在节点、边和邻居之间传递信息,以及 message passing 与矩阵形式的基本推导。

---

## 5.1 为什么需要 GNN?

有些数据天然不是网格或序列,而是图:

- 社交网络:用户是节点,好友关系是边
- 分子结构:原子是节点,化学键是边
- 推荐系统:用户/商品是节点,交互是边
- 知识图谱:实体是节点,关系是边

图数据的关键是:**一个节点的含义往往取决于它的邻居。**

---

## 5.2 Message Passing

GNN 的基本套路:

1. 每个节点拿到邻居发来的 message
2. 聚合这些 message
3. 更新自己的表示

第 $k$ 层可以写成:

$$
m_v^{(k)} = \text{AGG}^{(k)}\left(\{h_u^{(k-1)}: u \in \mathcal{N}(v)\}\right)
$$

$$
h_v^{(k)} = \text{UPDATE}^{(k)}(h_v^{(k-1)}, m_v^{(k)})
$$

其中:

- $h_v^{(k)}$:第 $k$ 层后节点 $v$ 的表示
- $\mathcal{N}(v)$:节点 $v$ 的邻居集合
- AGG 必须对邻居顺序不敏感,常见是 sum/mean/max

---

## 5.3 GCN 的矩阵形式

设:

- $A$ 是邻接矩阵
- $I$ 是单位矩阵
- $\tilde{A}=A+I$ 加上自连接
- $\tilde{D}$ 是 $\tilde{A}$ 的度矩阵

GCN 一层常写成:

$$
H^{(k+1)} =
\sigma\left(
\tilde{D}^{-\frac{1}{2}}
\tilde{A}
\tilde{D}^{-\frac{1}{2}}
H^{(k)}W^{(k)}
\right)
$$

直觉拆开看:

| 部分 | 作用 |
|------|------|
| $H^{(k)}W^{(k)}$ | 对节点特征做可学习变换 |
| $\tilde{A}$ | 从邻居聚合信息 |
| $\tilde{D}^{-1/2}$ | 根据节点度数做归一化 |
| $\sigma$ | 非线性 |

---

## 5.4 为什么要加自连接?

如果只聚合邻居,节点自己的原始信息可能被冲淡。

加上自连接:

$$
\tilde{A}=A+I
$$

意味着每个节点在更新时也能看到自己。

这和 Transformer 里每个 token 可以 attend 自己有点像。

---

## 5.5 多层 GNN 的感受野

一层 GNN 聚合 1-hop 邻居。

两层后,节点可以间接接收到 2-hop 信息。

一般地,$K$ 层 GNN 后,节点表示包含 $K$-hop 邻域信息。

⚠️ 但层数不是越深越好。太深可能出现 over-smoothing:所有节点表示越来越像,区分度下降。

---

## 5.6 PyTorch 代码骨架

下面是一个不依赖 PyG 的最小 GCN 层。实际项目通常会用 PyTorch Geometric 或 DGL。

```python
import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adj):
        # x: (N, in_dim)
        # adj: (N, N), 0/1 adjacency matrix
        n = adj.size(0)
        eye = torch.eye(n, device=adj.device)
        adj_hat = adj + eye

        degree = adj_hat.sum(dim=1)
        degree_inv_sqrt = degree.pow(-0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        norm = degree_inv_sqrt[:, None] * adj_hat * degree_inv_sqrt[None, :]

        return norm @ self.linear(x)

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.gcn2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, x, adj):
        h = self.act(self.gcn1(x, adj))
        return self.gcn2(h, adj)
```

---

## 5.7 GNN 和 Attention 的关系

普通 GCN 常用固定归一化权重聚合邻居。

Graph Attention Network(GAT)则让模型学习邻居权重:

$$
h_v' = \sum_{u \in \mathcal{N}(v)} \alpha_{v,u} W h_u
$$

这和 self-attention 很像,只是 attention 范围被图结构限制在邻居集合里。

---

## 5.8 本节核心要点

1. GNN 适合节点之间存在显式关系的数据。
2. Message passing 是 GNN 的基本范式。
3. 聚合函数需要对邻居顺序不敏感。
4. GCN 用归一化邻接矩阵做邻居聚合。
5. 多层 GNN 扩大节点感受野,但太深可能 over-smoothing。
6. GAT 可以看作图结构约束下的 attention。

## 思考题

<details>
<summary>为什么 GNN 的聚合函数通常要 permutation invariant?</summary>

因为图的邻居集合没有天然顺序。同一个节点的邻居无论以什么顺序列出来,模型输出都应该一致,所以 sum/mean/max 这类顺序不敏感的聚合更合适。

</details>
