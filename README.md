# TFWaveFormer — 自适应小波尺度选择

> 原论文：*TFWaveFormer: Temporal-Frequency Collaborative Multi-level Wavelet Transformer* (WWW '26)
>
> 原仓库：[SEUFHTong/TFWaveFormer](https://github.com/SEUFHTong/TFWaveFormer)

## 动机

TFWaveFormer 论文在 Section 6 指出，不同数据集上的最优小波尺度并不相同：

> "our analysis highlights that the optimal wavelet scales vary by dataset, underscoring the need for adaptive configuration based on graph characteristics."

原版使用固定的核尺寸 `[3, 5, 7, 9]`，所有数据集共用同一套配置。但不同图的时序特征存在明显差异——交互密集的图需要较小的感受野捕捉局部细节，交互周期较长的图需要较大的感受野覆盖长程趋势。如果能让模型根据数据自动选择合适的核尺寸，应该是一个值得探索的方向。

直接学习核尺寸面临一个技术问题：Conv1d 的 `kernel_size` 是奇数整数，取整函数不可导。将 `kernel_size` 设为可学习参数时，梯度会在 `round()` 处截断。

---

## 三条路线

绕过取整操作是让核尺寸可学习的关键。沿着这个思路，尝试了三条路线：

**路线 A：将尺寸参数化为连续变量，避免离散选择。**

- **Continuous**：每个头维护一个实数 s ∈ [3,15]，前向时在相邻奇数核之间做线性插值。s=6.3 等价于 70% Conv5 + 30% Conv7。梯度通过插值系数 α = s - floor(s) 回传，不经过取整。
- **Implicit**：进一步放弃预定义的卷积核模板，用 MLP(t, s, c) 直接表达核函数。输入为核内位置、尺度和通道号，输出为该位置的权重。s 作为 MLP 的连续输入，梯度沿链式法则回传。

**路线 B：保留离散选择，通过 Gumbel-Softmax 让选择过程可微。**

- **Gumbel**：候选集为 {3,5,7,9,11,13,15}。训练时对 logit 添加 Gumbel 噪声并通过温度 τ 控制 softmax 的"硬度"，使得所有候选按概率加权混合、梯度可回传。推理时直接 argmax 选取最优候选。温度从 5.0 逐步退火至 0.5，使训练末期的软选择逼近硬选择。

三条路线均保持原版 TFWaveFormer 的 K=4 多头并行和 per-channel attention 融合结构，仅修改了核尺寸的获取方式。

| | 核尺寸 | 可微策略 | 每通道独立？ |
|------|:---:|------|:---:|
| **Continuous** | 连续实数 | 相邻核插值 | scale_weights (4,172) |
| **Implicit** | 连续实数 | MLP(t,s,c) | 通道号输入 MLP |
| **Gumbel** | 离散 {3,5,7,9,11,13,15} | Gumbel-Softmax | scale_weights (4,172) |

---

## Continuous — 连续尺度插值

每头一个可学习参数 $s_k \in [3, 15]$。对第 k 个头，记 $s = \text{clamp}(s_k, 3, 15)$，取相邻奇数核 $k_{\text{lo}}, k_{\text{hi}}$ 作为上下界，插值系数 $\alpha = s - \lfloor s \rfloor$。该头的输出为：

$$\mathbf{z}_k = (1-\alpha) \cdot \text{Conv}_{k_{\text{lo}}}(\mathbf{x}) + \alpha \cdot \text{Conv}_{k_{\text{hi}}}(\mathbf{x})$$

由于 $\partial\alpha/\partial s = 1$，梯度为 

$$\partial\mathcal{L}/\partial s = \partial\mathcal{L}/\partial\mathbf{z}_k \cdot (\text{Conv}_{k_{\text{hi}}} - \text{Conv}_{k_{\text{lo}}})$$。
lo/hi 的取整仅用于索引预创建的卷积核，不参与计算图。

**实现**：预创建 7 个 Conv1d（k=3,5,7,9,11,13,15），groups=d_model。`scale_params` 为 (K=4,) 可学习参数，初始化为 U(3,15)。

```python
s = self.scale_params[k].clamp(3, 15)
lo, hi = snap_to_odd(int(floor(s))), snap_to_odd(int(ceil(s)))
alpha = s - lo
zk = (1 - alpha) * base_convs[str(lo)](x) + alpha * base_convs[str(hi)](x)
```

参考：Dynamic Filter Networks (NIPS 2016), https://arxiv.org/abs/1605.09673

---

## Implicit — 隐式神经表示

将卷积核看作三维连续函数 $\phi(t, s, c)$，t 为核内位置，s 为尺度，c 为通道号，用 MLP 学习该映射。对尺度 s，在 [-1,1] 内均匀取 ks 个点，对每个通道独立查询 MLP 获得核权重，Softmax 归一化后通过 `F.conv1d(groups=172)` 完成卷积。梯度沿 MLP 的链式法则回传，取整操作仅决定采样点数，不参与梯度。

通道 ID 作为 MLP 输入使得不同通道可以学出不同的核形状——高频通道可能倾向尖锐核，低频通道倾向平滑核。

**实现**：`implicit_net` 为 3 层 MLP（3→128→128→1），GELU 激活，Tanh 输出。批量加速：172 通道 × ks 位置的查询网格化为 `(D*ks, 3)`，MLP 一次前向完成。

```python
pos = linspace(-1, 1, ks).expand(D, ks)
ch  = arange(D).unsqueeze(1).expand(D, ks) / D
inp = stack([pos, full_like(pos, s/15), ch], dim=-1).reshape(D*ks, 3)
w = softmax(implicit_net(inp).view(D, ks), dim=-1)
zk = F.conv1d(pad(x), w.unsqueeze(1), groups=D)
```

参考：SIREN (NeurIPS 2020), https://arxiv.org/abs/2006.09661；NeRF (ECCV 2020), https://arxiv.org/abs/2003.08934

---

## Gumbel — 离散架构搜索

保留离散候选集 {3,5,7,9,11,13,15}，通过 Gumbel-Softmax 使选择过程可微。每个头维护一个 7 维 logit 向量。训练时添加 Gumbel 噪声，除以温度 τ 后做 softmax，得到概率分布而非 one-hot。7 个候选卷积全部计算，按概率加权求和，梯度可回传。推理时直接 argmax 选取最大 logit 对应的候选。

温度退火策略：$\tau(t) = \max(0.5, 5.0 \times (1 - t/T))$。训练初期 τ=5，分布较平，模型在各候选间探索；训练末期 τ=0.5，分布逼近 one-hot，与推理行为一致。

为防止多个头坍缩到同一候选，引入多样性正则：对每个头的偏好概率（纯 softmax，不含 Gumbel 噪声）取平均分布，计算其负熵并加入总损失，权重 0.01。负熵越小意味着分布越均匀，头之间选择越分散。

**实现**：`arch_params` 为 (K=4, C=7) 可学习 logit。`get_learned_scales()` 对 arch_params 取 argmax 后映射回候选值，返回离散整数。

```python
probs = gumbel_softmax(arch_params, tau=temperature, hard=False, dim=-1)  # 训练
probs = one_hot(arch_params.argmax(dim=-1), C).float()                     # 推理
all_c = stack([conv(x) for conv in gumbel_convs], dim=0)
zk = (all_c * probs[k].view(C, 1, 1, 1)).sum(dim=0)
```

参考：Gumbel-Softmax (ICLR 2017), https://arxiv.org/abs/1611.01144；DARTS (ICLR 2019), https://arxiv.org/abs/1806.09055

---

## 使用方法

```bash
# 单模式训练
python train_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TFWaveFormerContinuous \
    --num_epochs 50 --num_runs 5 --gpu 0

# 三种方法自动对比（默认跑 processed_data/ 下所有数据集）
python run_all_modes.py --epochs 50 --runs 5 --gpu 0

# 原版固定尺度
python train_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TFWaveFormer \
    --num_epochs 50 --gpu 0
```

三种自适应模式的 model_name：`TFWaveFormerContinuous` / `TFWaveFormerImplicit` / `TFWaveFormerGumbel`。

---

## 核心文件

```
TFWaveFormer_continuous.py   ← 连续尺度插值
TFWaveFormer_implicit.py     ← 隐式神经表示
TFWaveFormer_gumbel.py       ← Gumbel 离散选择
train_link_prediction.py     ← 训练（已集成三种自适应模型）
evaluate_link_prediction.py  ← 评估
run_all_modes.py             ← 自动化对比
```

---

## 初步实验

在三个数据集上用 3 epochs × 1 run 进行了快速验证，超参通过 `--load_best_configs` 自动加载（自适应模型共享 TFWaveFormer 的配置）。

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

**学到的卷积核尺寸（Wikipedia）**：

| Layer | Continuous | Implicit | Gumbel |
|------|------|------|------|
| 0 | [7.53, 9.29, 10.96, 14.30] | [7.53, 9.29, 10.96, 14.30] | [9, 5, 13, 13] |
| 1 | [5.20, 8.81, 9.56, 11.53] | [7.79, 9.76, 11.12, 12.60] | [11, 5, 13, 11] |

Wikipedia 上连续模式和隐式模式学到的尺寸偏大（均值 8-10），与该数据集编辑行为的长周期特性有一定关联。Gumbel 在 UCI 的 Layer 1 上四个头均选择了 11，多样性正则在此数据集上未拉开差距，可能需要调整权重或增加训练轮数。

---

## 🔗 引用

```bibtex
@inproceedings{tfwaveformer2026,
  title={TFWaveFormer: Temporal-Frequency Collaborative Multi-level Wavelet Transformer},
  author={Feng, Hantong and Wu, Yonggang and Chen, Duxin and Yu, Wenwu},
  booktitle={WWW},
  year={2026}
}
```
