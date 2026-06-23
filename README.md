# TFWaveFormer — 自适应小波尺度选择

> 原论文：*TFWaveFormer: Temporal-Frequency Collaborative Multi-level Wavelet Transformer* (WWW '26)
>
> 原仓库：[SEUFHTong/TFWaveFormer](https://github.com/SEUFHTong/TFWaveFormer)

## 出发点

TFWaveFormer 论文在 Section 6 提了一个很有意思的方向——作者在实验中发现不同数据集上最优的小波尺度不太一样：

> "our analysis highlights that the optimal wavelet scales vary by dataset, underscoring the need for adaptive configuration based on graph characteristics."

不同图的最优小波尺度应该不同，如果能根据图特征自适应配置就好了。原版用的核尺寸是固定的 `[3, 5, 7, 9]`，挺合理的。不过不同的图确实有不太一样的时序特点，比如交互很密集的图可能需要小一点的窗口看局部细节，交互周期长的图可能需要大一点的窗口看长程趋势。

然后遇到一个问题：Conv1d 的 `kernel_size` 得是奇数整数，取整函数不可导。所以不能直接把 `kernel_size` 设置成可学习参数——梯度回不来。

---

## 三条路线

把核尺寸变成可学习的关键是不经过取整操作。从这一点出发，试了三条路线：

**路线 A：不选择，把尺寸参数化为连续的。**

- **Continuous**：每个头维护一个实数 s ∈ [3,15]，前向的时候在相邻的奇数核之间做线性插值。比如 s=6.3 的话就是 70% Conv5 + 30% Conv7。梯度走插值系数 α = s - floor(s)，不会卡在取整上。
- **Implicit**：再进一步，不依赖预定义的卷积核模板，直接用 MLP 表达核。输入是 (位置, 尺度, 通道号)，输出是那个位置的权重值。s 作为 MLP 的连续输入，顺着链式法则回传。

**路线 B：保留离散选择，想办法让选择过程可导。**

- **Gumbel**：候选就是 {3,5,7,9,11,13,15} 这七个整数核。训练时用 Gumbel-Softmax 做近似——所有候选按概率加权混合，整个过程可导。推理时直接 argmax 取最大 logit。温度从大往小退火，训练到后期软选择已经很接近硬选择了。

三条路都保持了原版 TFWaveFormer 的 K=4 多头 + per-channel attention 融合的结构，区别只在于每个头的核尺寸是从哪来的。

| | 核尺寸 | 可微策略 | 每通道独立？ |
|------|:---:|------|:---:|
| **Continuous** | 连续实数 | 相邻核插值 | scale_weights (4,172) |
| **Implicit** | 连续实数 | MLP(t,s,c) | 通道号输入 MLP |
| **Gumbel** | 离散 {3,5,7,9,11,13,15} | Gumbel-Softmax | scale_weights (4,172) |

---

## Continuous — 连续尺度插值

每头一个可学习参数 $s_k \in [3, 15]$，前向时找相邻奇数核做线性插值。

对第 k 个头，记 $s = \text{clamp}(s_k, 3, 15)$，取上下界奇数核 $k_{\text{lo}}, k_{\text{hi}}$，和插值系数 $\alpha = s - \lfloor s \rfloor$。输出就是：

$$\mathbf{z}_k = (1-\alpha) \cdot \text{Conv}_{k_{\text{lo}}}(\mathbf{x}) + \alpha \cdot \text{Conv}_{k_{\text{hi}}}(\mathbf{x})$$

反向时 $\partial\alpha/\partial s = 1$，梯度是 $\partial\mathcal{L}/\partial s = \partial\mathcal{L}/\partial\mathbf{z}_k \cdot (\text{Conv}_{k_{\text{hi}}} - \text{Conv}_{k_{\text{lo}}})$。lo 和 hi 的取整只负责选核，不在计算图里。

**实现**：预创建 7 个 Conv1d（k=3,5,7,9,11,13,15），groups=d_model。`scale_params` 存 4 个可学习实数，前向时查表插值。

```python
s = self.scale_params[k].clamp(3, 15)
lo, hi = snap_to_odd(int(floor(s))), snap_to_odd(int(ceil(s)))
alpha = s - lo
zk = (1 - alpha) * base_convs[str(lo)](x) + alpha * base_convs[str(hi)](x)
```

参考：Dynamic Filter Networks (NIPS 2016), https://arxiv.org/abs/1605.09673

---

## Implicit — 隐式神经表示

换个角度看卷积核——不把它当成预先定义好的参数矩阵，而是当成一个连续函数 $\phi(t, s, c)$，t 是核内的位置，s 是尺度，c 是通道号。用一个 MLP 来学这个函数。

对某个尺度 s，先确定取多少个点（round(s) 调成奇数，梯度不参与），坐标均匀分布在 [-1,1]。对 172 个通道每个都这样取点、查 MLP、Softmax 归一化，得到 172 组核权重，然后用 `F.conv1d(groups=172)` 完成卷积。梯度通过 MLP 的链式法则回传。

通道 ID 作为 MLP 的输入参数是受了 NeRF 的启发——同一个函数在不同坐标查出来的值可以不一样，那同一个 MLP 在不同通道号上查出来的核形状也可以不一样。

**实现**：`implicit_net` 为 3 层 MLP（3→128→128→1），GELU 激活，Tanh 输出。批量加速：172 通道 × ks 位置的查询网格化成 `(D*ks, 3)`，MLP 一次前向完成。

```python
pos = linspace(-1, 1, ks).expand(D, ks)
ch  = arange(D).unsqueeze(1).expand(D, ks) / D
inp = stack([pos, full_like(pos, s/15), ch], dim=-1).reshape(D*ks, 3)
w = softmax(implicit_net(inp).view(D, ks), dim=-1)
zk = F.conv1d(pad(x), w.unsqueeze(1), groups=D)
```

参考：SIREN (NeurIPS 2020), https://arxiv.org/abs/2006.09661；NeRF (ECCV 2020), https://arxiv.org/abs/2003.08934

---

## Gumbel — 从候选里挑

和前两个思路不同，这个不把核尺寸变成连续值，而是直接从候选集 {3,5,7,9,11,13,15} 里挑。挑多挑少，训练完就知道了。

每个头维护一个 7 维的 logit 向量。训练时加 Gumbel 噪声，除温度 τ，做 softmax，得到的是一个概率分布而不是 one-hot。7 个候选卷积全算好，按概率加权加在一起，梯度能过。推理时不需要噪声，直接 argmax 取最大 logit 对应到候选集里。

温度和退火是跟 DARTS 学的——一开始 τ 大一点（比如 5.0），分布比较平，模型在各个候选之间试探；训练过程里 τ 慢慢降到 0.5，分布越来越尖，到后期软选择已经和硬选择差不多了：$\tau(t) = \max(0.5, 5.0 \times (1 - t/T))$。

另外加了多样性正则：算每个头对 7 个候选的偏好概率（纯 softmax，不混 Gumbel 噪声），取 4 个头平均，算它的负熵。负熵越小意味着分布越均匀，头之间越分散。这个项权重很小，0.01，只是轻轻推一下。

**实现**：`arch_params` 为 (K=4, C=7) 可学习 logit，7 个预创建 Conv1d。`temperature` 为 buffer，训练时退火。`get_learned_scales()` 对 arch_params 取 argmax 映射回候选值，返回离散整数。

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

用 3 epochs × 1 run 在三个数据集上跑了一轮验证。超参通过 `--load_best_configs` 自动加载，自适应模型共享了 TFWaveFormer 的配置。3 epochs 的结果还很初步，主要是看方法能不能 work。

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

wikipedia 上连续模式和隐式模式学到的尺寸偏大一些（均值 8-10），可能是这个数据集的交互周期比较长，模型倾向于用大一点的感受野。不过也就 3 个 epoch，还不能下结论。

---

## 🔧 环境

Python 3.8+ | PyTorch 1.8+ (CUDA 推荐) | NumPy, Pandas, scikit-learn, tqdm

数据需放在 `processed_data/{dataset}/` 下，包含 `ml_{dataset}.csv`, `ml_{dataset}.npy`, `ml_{dataset}_node.npy`。

## 🔗 引用

```bibtex
@inproceedings{tfwaveformer2026,
  title={TFWaveFormer: Temporal-Frequency Collaborative Multi-level Wavelet Transformer},
  author={Feng, Hantong and Wu, Yonggang and Chen, Duxin and Yu, Wenwu},
  booktitle={WWW},
  year={2026}
}
```
