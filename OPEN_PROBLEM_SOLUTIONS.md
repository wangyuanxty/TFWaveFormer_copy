# 自适应小波尺度选择：开放问题解决方案

## 1. 论文开放问题分析

> "our analysis highlights that the optimal wavelet scales vary by dataset, underscoring the need for adaptive configuration based on graph characteristics."（Section 6）
>
> 我们的分析表明，最优的小波尺度因数据集而异，这凸显了基于图特征进行自适应配置的必要性。

**问题本质**：

原版 TFWaveFormer 使用固定核尺寸 `[3, 5, 7, 9]`，所有数据集共用同一套配置。然而不同图的时序特征差异巨大——社交网络需要高频短窗口，贸易网络需要低频长窗口。固定尺度无法适应这种多样性。

**关键矛盾**：
- 核尺寸是离散整数（Conv1d 要求 `kernel_size` 为奇数），不能直接作为可学习参数
- `round()` 截断梯度 → 如果只把 `kernel_size` 设成可学习量，梯度永远传不回来

---

## 2. 方案概览
**三种方案的逻辑链**：

**Continuous**：
```
核尺寸是整数，round() 截断梯度
  │
  └─→ 放弃输出整数，改用连续实数 s ∈ [3,15]
        │
        └─→ 在相邻奇数核之间线性插值:
              output = (1-α)·Conv_lo + α·Conv_hi
              α = s - floor(s)，∂α/∂s = 1，梯度通过
```

**Implicit**：
```
核尺寸是整数，round() 截断梯度
  │
  └─→ 放弃输出整数，改用连续实数 s ∈ [3,15]
        │
        ├─→ 不用模板，用 MLP(t, s, c) 动态生成核权重
        │     s 作为 MLP 输入，梯度经链式法则回传
        │
        └─→ 将通道 ID 也作为 MLP 输入
              每个通道独立查询 → 核形状按通道定制
```

**Gumbel**：
```
核尺寸是整数，round() 截断梯度
  │
  └─→ 保留离散选择，但让选择过程可微
        │
        ├─→ Gumbel-Softmax: 训练时候选加权混合（可微），推理时 argmax（硬选）
        │
        ├─→ 温度退火: τ: 5.0 → 0.5，训练逐步逼近推理
        │
        └─→ 多样性正则化: 最小化头选择的平均负熵，防止撞车
```

三种方案的核心区别在于**如何让离散的核尺寸选择变得可微分**：

| 方案 | 核尺寸 | 可微分策略 | 每通道独立？ |
|------|:---:|------|:---:|
| **Continuous** | 连续实数 (如 6.3) | 相邻整数核插值 | ✅ scale_weights (4,172) |
| **Implicit** | 连续实数 | MLP 连续函数 k(t,s,c) | ✅ 通道号输入 MLP |
| **Gumbel** | 离散整数 {3,5,7,9,11,13,15} | Gumbel-Softmax 近似 argmax | ✅ scale_weights (4,172) |

---

## 3. 方案一：Continuous — 连续尺度插值

### 核心思想

将每个头的核尺寸参数化为连续实数 $s_k \in [3, 15]$，前向传播时在相邻的两个奇数整数核之间线性插值。梯度不经过取整操作，而是通过插值系数 $\alpha$ 回传。

### 数学原理

设第 $k$ 个头的可学习尺度参数为 $s_k$。定义钳位运算 $s_k^{\text{c}} = \text{clamp}(s_k, 3, 15)$，上下界整数核为：

$$
k_{\lfloor} = \mathcal{O}(\lfloor s_k^{\text{c}} \rfloor), \quad
k_{\lceil}  = \mathcal{O}(\lceil s_k^{\text{c}} \rceil)
$$

其中 $\mathcal{O}(x) = x + (x \bmod 2 = 0)$ 将偶数调整为奇数。插值系数 $\alpha_k = s_k^{\text{c}} - \lfloor s_k^{\text{c}} \rfloor \in [0, 1)$。该头的输出为：

$$
\mathbf{z}_k = (1 - \alpha_k) \cdot \text{Conv}_{k_{\lfloor}}(\mathbf{x}) + \alpha_k \cdot \text{Conv}_{k_{\lceil}}(\mathbf{x})
$$

**梯度**：$\frac{\partial \alpha_k}{\partial s_k} = 1$（在非整数点），因此：

$$
\frac{\partial \mathcal{L}}{\partial s_k} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_k} \cdot \big(\text{Conv}_{k_{\lceil}}(\mathbf{x}) - \text{Conv}_{k_{\lfloor}}(\mathbf{x})\big)
$$

$k_{\lfloor}$ 和 $k_{\lceil}$ 仅用于索引预创建的卷积核，不参与计算图。尺度跨过整数边界时（如 6.9→7.1），上下界跳变但不影响 SGD 收敛。

### 实现细节

- 预创建 7 个 Conv1d（k=3,5,7,9,11,13,15），groups=d_model
- `scale_params`: (K=4,) 可学习参数，初始化为 U(3,15)
- `scale_weights`: (K, d_model) 每通道注意力，与原版一致
- 前向：每头根据 $s_k$ 找到 lo/hi → 插值 → 4 头 stack → attention 融合

```python
s = self.scale_params[k].clamp(3, 15)
lo, hi = snap_to_odd(int(floor(s))), snap_to_odd(int(ceil(s)))
alpha = s - lo
zk = (1 - alpha) * base_convs[str(lo)](x) + alpha * base_convs[str(hi)](x)
```

### 参考论文

**Dynamic Filter Networks** (NIPS 2016), Bert De Brabandere et al.
https://arxiv.org/abs/1605.09673

原论文为视频帧预测提出用小型网络根据输入动态生成卷积核权重。我们借鉴其"离散核连续化"思想，但仅学习尺度参数 $s_k$ 而非整个核权重矩阵——参数量从 $\mathcal{O}(D^2 \cdot k)$ 降至 $\mathcal{O}(K)$。

---

## 4. 方案二：Implicit — 隐式神经表示

### 核心思想

将卷积核建模为三维连续函数 $\phi(t, s, c): \mathbb{R}^3 \to \mathbb{R}$，用 MLP 实现。$t$ 为核内位置，$s$ 为尺度，$c$ 为通道 ID。任意 $(t,s,c)$ 处可查询核权重，梯度通过 MLP 链式法则回传。每通道独立查询意味着不同通道可学到不同核形状。

### 数学原理

对第 $k$ 个头，尺度 $s_k$，核尺寸 $k_k = \text{round}(s_k)$ 并调整为奇数。在 $[-1,1]$ 内均匀取 $k_k$ 个位置 $t_i$。对通道 $c$：

$$
w_{c,i} = \phi\left(t_i,\; \frac{s_k}{15},\; \frac{c}{D}\right), \quad
\tilde{w}_{c,i} = \frac{\exp(w_{c,i})}{\sum_j \exp(w_{c,j})}
$$

通道 $c$ 的卷积：

$$
\mathbf{z}_k[:, c, :] = \sum_{i=0}^{k_k-1} \tilde{w}_{c,i} \cdot \mathbf{x}[:, c, :+i-\lfloor k_k/2\rfloor]
$$

**梯度**：$s_k$ 经 MLP 输入 $\to$ 核权重 $\to$ 卷积 $\to$ loss 的完整链式法则：

$$
\frac{\partial \mathcal{L}}{\partial s_k} = \sum_{c=0}^{D-1} \sum_{i=0}^{k_k-1}
\frac{\partial \mathcal{L}}{\partial \tilde{w}_{c,i}} \cdot
\frac{\partial \tilde{w}_{c,i}}{\partial w_{c,i}} \cdot
\frac{\partial \phi}{\partial s} \cdot \frac{1}{15}
$$

$\phi$ 由 MLP + GELU 构成，$C^1$ 连续，$\frac{\partial\phi}{\partial s}$ 处处有定义。$\text{round}(s_k)$ 仅决定采样点数，不参与梯度。

### 实现细节

- `implicit_net`: 3 层 MLP（3→128→128→1），GELU 激活，Tanh 输出
- `scale_params`: (K=4,) 可学习参数
- **批量加速**：将 D×ks 次查询网格化为 `(D*ks, 3)` 的批量输入，MLP 一次前向完成，再用 `F.conv1d(groups=D)` 并行卷积

```python
# 网格化: 172 通道 × ks 位置 → (172*ks, 3) 批量输入
pos = linspace(-1, 1, ks).expand(D, ks)
ch  = arange(D).unsqueeze(1).expand(D, ks) / D
inp = stack([pos, full_like(pos, s/15), ch], dim=-1).reshape(D*ks, 3)
w = softmax(implicit_net(inp).view(D, ks), dim=-1)       # (D, ks)
zk = F.conv1d(pad(x), w.unsqueeze(1), groups=D)
```

### 参考论文

**SIREN** (NeurIPS 2020), Vincent Sitzmann et al.
https://arxiv.org/abs/2006.09661
— 用周期性激活函数的 MLP 表示连续信号，可在任意坐标查询。我们将其范式"离散信号用连续 MLP 表示"应用到卷积核。

**NeRF** (ECCV 2020), Ben Mildenhall et al.
https://arxiv.org/abs/2003.08934
— 用 MLP 表示 5D 函数 $(x,y,z,\theta,\phi) \to (\text{RGB}, \sigma)$，开创坐标网络范式。我们借鉴其将物理量与空间坐标关联的思想，将核权重与 $(t, s, c)$ 三维坐标关联。

---

## 5. 方案三：Gumbel — 离散架构搜索

### 核心思想

从候选集 $\{3,5,7,9,11,13,15\}$ 中为每个头离散选择一个核尺寸。用 Gumbel-Softmax 使离散选择可微分——训练时所有候选按概率加权混合（软），推理时 argmax 取最优（硬）。温度从高到低退火，使训练和推理逐步一致。引入多样性正则化鼓励不同头选不同候选。

### 数学原理

候选集 $\mathcal{C} = \{c_1,\ldots,c_C\}$，$C=7$。每头维护 logit $\mathbf{z}_k \in \mathbb{R}^{C}$。

**训练**：采样 Gumbel 噪声 $g_{k,j} = -\log(-\log(u_{k,j})), \; u_{k,j} \sim U(0,1)$，带温度 $\tau$ 的软选择：

$$
\tilde{p}_{k,j} = \frac{\exp((z_{k,j} + g_{k,j}) / \tau)}{\sum_m \exp((z_{k,m} + g_{k,m}) / \tau)}
$$

$$
\mathbf{z}_k = \sum_{j=1}^{C} \tilde{p}_{k,j} \cdot \text{Conv}_{c_j}(\mathbf{x})
$$

**推理**（$\tau \to 0$）：$\hat{c}_k = c_{j^*}, \; j^* = \arg\max_j z_{k,j}$，仅保留 argmax 支路。

**温度退火**：$\tau(t) = \max(0.5,\; 5.0 \times (1 - t/T))$，训练初期 $\tau=5$ 软探索，末期 $\tau=0.5$ 逼近硬选择。

**梯度**（Gumbel-Softmax）：

$$
\frac{\partial \tilde{p}_{k,j}}{\partial z_{k,m}} = \frac{1}{\tau} \cdot \tilde{p}_{k,j} \cdot (\delta_{jm} - \tilde{p}_{k,m})
$$

**多样性正则化**：令 $p_{k,j} = \text{softmax}(\mathbf{z}_k)_j$（无噪声的偏好概率），$\bar{p}_j = \frac{1}{K}\sum_k p_{k,j}$。最小化平均分布的负熵以鼓励分散：

$$
\mathcal{L}_{\text{div}} = \sum_{j=1}^{C} \bar{p}_j \log(\bar{p}_j + \varepsilon) \quad \text{（$\bar{p}$ 越均匀，值越小）}
$$

总损失：$\mathcal{L} = \mathcal{L}_{\text{BCE}} + 0.1 \cdot \mathcal{L}_{\text{div}}$。

### 实现细节

- `arch_params`: (K=4, C=7) 可学习 logit
- `gumbel_convs`: 7 个预创建 Conv1d（k ∈ {3,5,7,9,11,13,15}），groups=d_model
- `temperature`: 注册为 buffer，训练时调用 `anneal_temperature()` 退火
- 前向：GumbelSoftmax → 7 卷积全算 → 按概率加权 → K 头 stack → attention 融合
- `get_learned_scales()`：对 arch_params 取 argmax 映射回候选值，返回离散整数

```python
# 训练
probs = gumbel_softmax(arch_params, tau=temperature, hard=False, dim=-1)  # (K, C)
# 推理
probs = one_hot(arch_params.argmax(dim=-1), C).float()
# 加权融合
all_c = stack([conv(x) for conv in gumbel_convs], dim=0)   # (C, B, D, L)
zk = (all_c * probs[k].view(C, 1, 1, 1)).sum(dim=0)
```

### 参考论文

**Gumbel-Softmax** (ICLR 2017), Eric Jang et al.
https://arxiv.org/abs/1611.01144
— 用 Gumbel 噪声 + Softmax 将离散采样转化为可微操作。核心公式 $\tilde{p}_j = \text{softmax}((\log\pi_j + g_j)/\tau)$ 在 $\tau \to 0$ 时收敛到 one_hot(argmax)。

**DARTS** (ICLR 2019), Hanxiao Liu et al.
https://arxiv.org/abs/1806.09055
— 将 NAS 中的离散操作选择建模为连续优化：训练时 softmax 加权所有候选操作，训练后取 argmax。我们借鉴此范式将尺度选择视为架构搜索问题。

---

## 6. 实验结果

> 以下结果为 3 epochs × 1 run 的验证性实验，仅用于验证三种自适应方法的可行性。完整实验需更多 epochs 和 runs。

### Table 1: Transductive Link Prediction (AP / AUC)

| Dataset | Continuous | Implicit | Gumbel |
|------|------|------|------|
| **reddit** | 0.9916 / 0.9907 | 0.9922 / 0.9915 | 0.9914 / 0.9905 |
| **wikipedia** | 0.9926 / 0.9920 | 0.9928 / 0.9923 | 0.9924 / 0.9920 |
| **uci** | 0.9487 / 0.9366 | 0.9480 / 0.9380 | 0.9519 / 0.9410 |

### Table 2: Inductive Link Prediction (AP / AUC)

| Dataset | Continuous | Implicit | Gumbel |
|------|------|------|------|
| **reddit** | 0.9876 / 0.9859 | 0.9883 / 0.9869 | 0.9874 / 0.9856 |
| **wikipedia** | 0.9876 / 0.9872 | 0.9882 / 0.9876 | 0.9880 / 0.9873 |
| **uci** | 0.9263 / 0.9107 | 0.9246 / 0.9099 | 0.9288 / 0.9125 |

### 学到的卷积核尺寸

**Wikipedia**：

| Layer | Continuous | Implicit | Gumbel |
|------|------|------|------|
| 0 | [7.53, 9.29, 10.96, 14.30] | [7.53, 9.29, 10.96, 14.30] | [9, 5, 13, 13] |
| 1 | [5.20, 8.81, 9.56, 11.53] | [7.79, 9.76, 11.12, 12.60] | [11, 5, 13, 11] |

**Reddit**：

| Layer | Continuous | Implicit | Gumbel |
|------|------|------|------|
| 0 | [7.53, 9.29, 10.96, 14.30] | [7.53, 9.29, 10.96, 14.30] | [5, 15, 3, 9] |
| 1 | [5.20, 8.81, 9.56, 11.53] | [7.79, 9.76, 11.12, 12.60] | [11, 13, 15, 13] |

**UCI**：

| Layer | Continuous | Implicit | Gumbel |
|------|------|------|------|
| 0 | [7.53, 9.29, 10.96, 14.30] | [7.53, 9.29, 10.96, 14.30] | [11, 3, 13, 13] |
| 1 | [5.20, 8.81, 9.56, 11.53] | [7.79, 9.76, 11.12, 12.60] | [11, 11, 11, 11] |

