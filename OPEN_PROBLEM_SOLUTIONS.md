# TFWaveFormer 自适应小波尺度

## 1. 问题

读 TFWaveFormer 论文（WWW '26）的时候注意到 Section 6 提了一个开放问题：

> "our analysis highlights that the optimal wavelet scales vary by dataset, underscoring the need for adaptive configuration based on graph characteristics."

大概意思就是，不同数据集的最优核尺寸不一样，固定的 `[3, 5, 7, 9]` 不够用。社交网络那种秒级交互的小图和贸易网络那种慢吞吞的大图，感受野需求显然不同。

但这事没那么简单。Conv1d 的 kernel_size 必须是整数，而且 round() 截断了梯度。把 kernel_size 直接设成 nn.Parameter？反向传播过不去。

所以问题的核心其实是：怎么让离散的核尺寸也能端到端训练。

---

## 2. 三条思路

我试了三个方向，分成两类：

**第一类：不做离散选择，把尺寸变成连续的。**

- **Continuous**：学一个实数 s，前向的时候在相邻整数核之间线性插值。比如 s=6.3 就是 70% 的 Conv5 + 30% 的 Conv7。梯度通过插值系数 α = s - floor(s) 回传，∂α/∂s = 1，很干净。
- **Implicit**：更激进一点——连模板核都不要了。用 MLP(t, s, c) 当场算核权重，s 作为 MLP 的连续输入，梯度走链式法则。把通道 ID 也喂给 MLP，这样不同通道可以学到不同形状的核。

**第二类：保留离散选择，让选择过程能求导。**

- **Gumbel**：7 个候选核（3,5,7,9,11,13,15）全建好。训练时 Gumbel-Softmax 按概率加权混合，梯度能过；推理时直接 argmax。温度从 5.0 退火到 0.5，训练和推理逐渐一致。加了个多样性正则防止所有头选同一个。

三个方案都保持原版的 K=4 头并行 + per-channel attention 加权融合，改的只是每个头的核尺寸怎么定。

| | 核尺寸 | 可微策略 | 每通道独立？ |
|------|:---:|------|:---:|
| **Continuous** | 连续实数 | 相邻核插值 | scale_weights (4,172) |
| **Implicit** | 连续实数 | MLP(t,s,c) | 通道号输入 MLP |
| **Gumbel** | 离散 {3,5,7,9,11,13,15} | Gumbel-Softmax | scale_weights (4,172) |

---

## 3. Continuous

想法很简单：每个头学一个实数 $s_k \in [3, 15]$，前向时找离它最近的两个奇数核，在线性插一下。

设 $s = \text{clamp}(s_k, 3, 15)$，找上下界 $k_{\text{lo}}, k_{\text{hi}}$（两个最近的奇数），$\alpha = s - \lfloor s \rfloor$。输出就是：

$$\mathbf{z}_k = (1-\alpha) \cdot \text{Conv}_{k_{\text{lo}}}(\mathbf{x}) + \alpha \cdot \text{Conv}_{k_{\text{hi}}}(\mathbf{x})$$

反向的时候 $\partial\alpha/\partial s = 1$（不在整数点就成立），梯度就是两个卷积输出的差。lo 和 hi 的取整只负责"选哪个核"，不在计算图里，不影响梯度。

实现上预创建了 7 个 Conv1d（3,5,7,9,11,13,15，groups=d_model），scale_params 存 4 个实数，`scale_weights` 存 per-channel 的注意力权重，和原版一致。前向就查表然后插值。

```python
s = self.scale_params[k].clamp(3, 15)
lo, hi = snap_to_odd(int(floor(s))), snap_to_odd(int(ceil(s)))
alpha = s - lo
zk = (1-alpha) * base_convs[str(lo)](x) + alpha * base_convs[str(hi)](x)
```

参考的是 Dynamic Filter Networks (NIPS 2016, https://arxiv.org/abs/1605.09673)，那篇是做视频帧预测，用小型网络根据输入动态生成卷积核。我这只学一个实数，复杂度低得多。

---

## 4. Implicit

受 NeRF 和 SIREN 启发，把卷积核看成一个连续函数 $\phi(t, s, c)$，用 MLP 来表示。t 是核内位置（归一化到 [-1,1]），s 是尺度，c 是通道 ID。

对每个头，先 round(s) 确定取几个点（比如 7 个），然后用 MLP 在这 7 个位置、172 个通道上各查询一次，得到 172×7 个权重，Softmax 归一化。最后用 F.conv1d(groups=172) 一次性做完所有通道的卷积。

批量加速这边做了一个优化：把 172×7 次查询展开成一个 `(172*7, 3)` 的矩阵，MLP 一次 forward 就全出来了。不然循环 172 个通道做 MLP 前向太慢了。

通道 ID 作为 MLP 输入这点比较关键——不同通道的核权重是可以不一样的。虽然目前实验中连续和 implicit 效果差不多，但这至少保留了一个自由度。

```python
# mesh D 通道 × ks 位置 → (D*ks, 3) 批量 MLP 查询
pos = linspace(-1, 1, ks).expand(D, ks)
ch  = arange(D).unsqueeze(1).expand(D, ks) / D
inp = stack([pos, full_like(pos, s/15), ch], dim=-1).reshape(D*ks, 3)
w = softmax(implicit_net(inp).view(D, ks), dim=-1)
zk = F.conv1d(pad(x), w.unsqueeze(1), groups=D)
```

参考两篇：SIREN (NeurIPS 2020, https://arxiv.org/abs/2006.09661) 用 MLP 表示连续信号，NeRF (ECCV 2020, https://arxiv.org/abs/2003.08934) 用坐标网络做 3D 重建。

---

## 5. Gumbel

这个和前两个思路不一样——不逃避离散选择，候选集 {3,5,7,9,11,13,15} 就是离散的，但要能求导。

用的是 Gumbel-Softmax。每个头维护一个 7 维 logit，训练时加 Gumbel 噪声除以温度做 softmax，得到的是一个概率分布而不是 one-hot。7 个卷积都算好，按概率加权，梯度能过。推理时不需要噪声，直接 argmax。

温度退火用最简单的线性：$\tau(t) = \max(0.5, 5.0(1 - t/T))$。一开始 τ=5 时分布很平，模型在试探；到最后 τ=0.5 时已经逼近 one-hot，和推理行为一致。

还有个实际的小问题：四个头可能都选了同一个候选，那多头的意义就没了。加了一个多样性正则：拿每个头对候选的偏好概率（不带 Gumbel 噪声的 softmax），算四个头的平均分布的负熵。分布的熵越高越均匀，头越分散。这一项以 0.01 的权重加到 loss 里。

训练时 loss 偶尔会出现负值，因为多样性的负熵项在头很分散时会比较大。后来把权重从 0.1 调到了 0.01 就好了。

```python
# 训练
probs = gumbel_softmax(arch_params, tau=temperature, hard=False, dim=-1)  # (K, C)
# 推理
probs = one_hot(arch_params.argmax(dim=-1), C).float()
# 加权
all_c = stack([conv(x) for conv in gumbel_convs], dim=0)   # (C, B, D, L)
zk = (all_c * probs[k].view(C, 1, 1, 1)).sum(dim=0)
```

- `arch_params`: (K=4, C=7) 可学习 logit
- `temperature`: 注册为 buffer，epoch 开始时 anneal
- `get_learned_scales()`：argmax 后映射回实际的候选值（返回整数）
- 多样性：`diversity_loss()` 返回负熵，`loss = BCE + 0.01 * diversity_loss`

参考：Gumbel-Softmax (ICLR 2017, https://arxiv.org/abs/1611.01144) 和 DARTS (ICLR 2019, https://arxiv.org/abs/1806.09055)。Gumbel-Softmax 把离散采样变成可微近似，DARTS 用这招做架构搜索。

---

## 6. 实验结果

跑的是 3 epochs × 1 run，主要是验证方法能 work。`--load_best_configs` 自动加载了 TFWaveFormer 的最佳超参，因为自适应模型和原版结构一样，可以复用。

**Table 1: Transductive (test AP / AUC)**

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

3 epoch 看个大概，要完整对比得跑更多轮。三种方法在 reddit 和 wikipedia 上差不多，uci 上 gumbel 略好一点。

**学到的核尺寸：**

Wikipedia：

| Layer | Continuous | Implicit | Gumbel |
|------|------|------|------|
| 0 | [7.53, 9.29, 10.96, 14.30] | [7.53, 9.29, 10.96, 14.30] | [9, 5, 13, 13] |
| 1 | [5.20, 8.81, 9.56, 11.53] | [7.79, 9.76, 11.12, 12.60] | [11, 5, 13, 11] |

Reddit：

| Layer | Continuous | Implicit | Gumbel |
|------|------|------|------|
| 0 | [7.53, 9.29, 10.96, 14.30] | [7.53, 9.29, 10.96, 14.30] | [5, 15, 3, 9] |
| 1 | [5.20, 8.81, 9.56, 11.53] | [7.79, 9.76, 11.12, 12.60] | [11, 13, 15, 13] |

UCI：

| Layer | Continuous | Implicit | Gumbel |
|------|------|------|------|
| 0 | [7.53, 9.29, 10.96, 14.30] | [7.53, 9.29, 10.96, 14.30] | [11, 3, 13, 13] |
| 1 | [5.20, 8.81, 9.56, 11.53] | [7.79, 9.76, 11.12, 12.60] | [11, 11, 11, 11] |

几点观察：
- Wikipedia 上学到的尺寸偏大（均值 8-10），和 wiki 编辑的长期性比较吻合
- Gumbel 在 UCI 的 Layer 1 四个头全选了 11，多样性正则没起作用，3 个 epoch 可能不够
- Continuous 和 Implicit 学到的核几乎一样——目前看不出 implicit 的 per-channel 定制有什么明显优势，可能需要更复杂的图才能体现出来
