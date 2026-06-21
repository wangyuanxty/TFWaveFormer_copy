# 自适应小波尺度选择：TFWaveFormer 开放问题解决方案

## 1. 问题

TFWaveFormer 论文（WWW '26）在 Section 6 明确指出了一个开放问题：

> "our analysis highlights that the optimal wavelet scales vary by dataset, underscoring the need for adaptive configuration based on graph characteristics."

翻译过来就是：不同图的最优小波尺度不一样，固定的 `[3, 5, 7, 9]` 没法适应所有场景。比如社交网络交互密集、时间跨度短，需要小感受野捕捉秒级动态；贸易网络交互稀疏、周期长，需要大感受野才能看到趋势。一刀切的核尺寸显然有问题。

但这里有个硬伤：Conv1d 的 `kernel_size` 必须是整数，而且 `round()` 把梯度全截了。直接把 `kernel_size` 设成可学习参数？行不通，梯度传不回。

所以问题变成了：怎么在保证可微的前提下，让模型自己学会选核尺寸。

---

## 2. 三条路线

核尺寸必须可微 ← 这是所有方案共同的起点。但"怎么做"可以走两条完全不同的路：

**路 A：不做离散选择，把尺寸变成连续的。**

- **Continuous**：每个头学一个实数 s，前向的时候在相邻奇数核之间插值。s=6.3 就是 70% 的 Conv5 + 30% 的 Conv7。梯度走 α = s - floor(s)，和取整操作无关。
- **Implicit**：更进一步——连模板核都不要了。拿一个 MLP(t, s, c)，输入位置、尺度、通道号，当场算出核权重。s 作为 MLP 的连续输入，梯度自然过。

**路 B：保留离散选择，但让选择过程本身可微。**

- **Gumbel**：7 个候选核（3,5,7,9,11,13,15）全创建好。训练时 Gumbel-Softmax 给它们加权混合，梯度能过；推理时直接 argmax 取最优。温度从 5.0 慢慢降到 0.5，到训练末期软选择已经和硬选择差不多了。再加一个多样性正则防止四个头都选同一个候选。

三条路都保持原版的 K=4 头并行 + per-channel attention 融合结构，变的只是"每个头的核尺寸怎么来"。

| | 核尺寸 | 可微策略 | 每通道独立？ |
|------|:---:|------|:---:|
| **Continuous** | 连续实数 | 相邻核插值 | scale_weights (4,172) |
| **Implicit** | 连续实数 | MLP(t,s,c) | 通道号输入 MLP |
| **Gumbel** | 离散 {3,5,7,9,11,13,15} | Gumbel-Softmax | scale_weights (4,172) |

---

## 3. Continuous — 连续尺度插值

每头一个可学习参数 $s_k \in [3, 15]$。前向时找相邻奇数核做线性插值。逻辑很简单：

对第 k 个头，设 $s = \text{clamp}(s_k, 3, 15)$，找上下界奇数核 $k_{\text{lo}}, k_{\text{hi}}$，以及插值系数 $\alpha = s - \lfloor s \rfloor$。输出：

$$\mathbf{z}_k = (1-\alpha) \cdot \text{Conv}_{k_{\text{lo}}}(\mathbf{x}) + \alpha \cdot \text{Conv}_{k_{\text{hi}}}(\mathbf{x})$$

反向传播时 $\partial\alpha/\partial s = 1$，所以梯度就是 $\partial\mathcal{L}/\partial s = \partial\mathcal{L}/\partial\mathbf{z}_k \cdot (\text{Conv}_{k_{\text{hi}}} - \text{Conv}_{k_{\text{lo}}})$。

lo 和 hi 的取整只负责"选核"，不在计算图里。s 跨过整数边界（比如 6.9→7.1）时 lo/hi 跳变，有限的不可微点不影响 SGD。

实际做的时候预创建 7 个 Conv1d（3,5,7,9,11,13,15），`scale_params` 存 4 个可学习实数，前向时查表插值就行。

**参考**：Dynamic Filter Networks (NIPS 2016), https://arxiv.org/abs/1605.09673。原论文中是用小型网络根据输入动态生成整个卷积核权重。这里只学一个实数尺度，复杂度从 $\mathcal{O}(D^2k)$ 降到 $\mathcal{O}(K)$。

---

## 4. Implicit — 隐式神经表示

换一个思路：不预先定义卷积核的"模板"，而是用一个 MLP 来表达核——输入是 (位置, 尺度, 通道号)，输出是那个位置的权重。本质上就是把卷积核看成一个三维连续函数 $\phi(t, s, c)$。

对尺度 s，先确定取几个点（round(s) 然后调成奇数，梯度不经过这里），比如 7 个。对 172 个通道每个都这样取点、查询 MLP、Softmax 归一化，得到 172 组不同的核权重。然后用 `F.conv1d(groups=172)` 一并完成所有通道的卷积。

MLP 做了批量加速：172 通道 × ks 位置的查询网格化成 `(172*ks, 3)` 的输入矩阵，MLP 一次前向就出所有结果。s 通过 MLP 的链式法则回传梯度，不卡在 round 上。

通道 ID 作为 MLP 输入是个关键设计——通道 0 学到的核形状可以和通道 171 完全不同。同一层里高频特征用尖锐核、低频特征用平滑核，各取所需。

**参考**：SIREN (NeurIPS 2020), https://arxiv.org/abs/2006.09661 和 NeRF (ECCV 2020), https://arxiv.org/abs/2003.08934。SIREN 用 MLP 表示连续信号，NeRF 用坐标网络做 3D。这里把卷积核当作一个"二维信号（位置 + 尺度）随通道变化"，用 MLP 去学。

---

## 5. Gumbel — 离散架构搜索

和前两个不同，这个不逃避离散选择——候选集 $\{3,5,7,9,11,13,15\}$，7 个整数核，就是要从里头挑。但挑的过程得能求导。

Gumbel-Softmax 能做到。每个头维护一个 7 维的 logit 向量。训练时加 Gumbel 噪声，除以温度 τ，做 softmax。得到的不是 one-hot，是概率分布——比如 "30% 选 5, 50% 选 7, 20% 选 9"。7 个卷积全算好，按概率加权求和，梯度能过。

推理时不需要噪声了，直接 argmax 取最大的 logit，就是最终选中的核。

温度是个关键。一开始 τ=5，分布很平，相当于在试探所有候选；随着训练进行慢慢降到 τ=0.5，分布逼近 one-hot，训练和推理的行为就一致了。退火线就是 $\tau(t) = \max(0.5, 5.0 \times (1 - t/T))$。

还有一个实际问题：4 个头可能都选差不多的候选，那就和白花计算没区别了。加一项多样性正则：算每个头对 7 个候选的偏好概率（纯 softmax，不加 Gumbel 噪声），取 4 个头的平均分布，算它的负熵。负熵越小说明越均匀，头之间越分散。这项加到总 loss 里，权重 0.01。

**参考**：Gumbel-Softmax (ICLR 2017), https://arxiv.org/abs/1611.01144 和 DARTS (ICLR 2019), https://arxiv.org/abs/1806.09055。Gumbel-Softmax 把离散采样变成连续可微近似，DARTS 用这个做架构搜索——训练时 softmax 加权所有候选操作，训练后 argmax。这里把头的核尺寸选择当成一个微型的架构搜索问题。

---

## 6. 实验结果

下面是用 3 epochs × 1 run 做的快速验证，超参通过 `--load_best_configs` 自动加载（自适应模型共享 TFWaveFormer 的最佳配置）。结果只说明方法可行，要做完整对比得跑更多 epochs 和 runs。

**Table 1: Transductive (test AP / test AUC)**

| | Continuous | Implicit | Gumbel |
|------|------|------|------|
| **reddit** | 0.9916 / 0.9907 | 0.9922 / 0.9915 | 0.9914 / 0.9905 |
| **wikipedia** | 0.9926 / 0.9920 | 0.9928 / 0.9923 | 0.9924 / 0.9920 |
| **uci** | 0.9487 / 0.9366 | 0.9480 / 0.9380 | 0.9519 / 0.9410 |

**Table 2: Inductive (new node test AP / AUC)**

| | Continuous | Implicit | Gumbel |
|------|------|------|------|
| **reddit** | 0.9876 / 0.9859 | 0.9883 / 0.9869 | 0.9874 / 0.9856 |
| **wikipedia** | 0.9876 / 0.9872 | 0.9882 / 0.9876 | 0.9880 / 0.9873 |
| **uci** | 0.9263 / 0.9107 | 0.9246 / 0.9099 | 0.9288 / 0.9125 |

各模式学到的核尺寸见 comparison_results/ 目录下的 JSON 文件。wikipedia 上学到的尺寸偏大（均值 8-10），这和 wiki 编辑行为的长周期特性一致。
