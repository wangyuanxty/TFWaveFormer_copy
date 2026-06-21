# TFWaveFormer with Adaptive Wavelet Scale Selection

> 解决论文开放问题：最优小波尺度因数据集而异
>
> 原论文：*TFWaveFormer: Temporal-Frequency Collaborative Multi-level Wavelet Transformer* (WWW '26)
>
> 原仓库：[SEUFHTong/TFWaveFormer](https://github.com/SEUFHTong/TFWaveFormer)

## ⚡ 快速开始

```bash
# 1. 激活环境
conda activate py312

# 2. 训练自适应模型
python train_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TFWaveFormerContinuous \
    --num_epochs 10 --num_runs 1 --gpu 0

# 3. 所有模式自动对比
python run_all_modes.py --epochs 10 --runs 1 --gpu 0
```

## 🎯 三种自适应方案

| 模式 | model_name | 原理 | 推荐场景 |
|------|------|------|------|
| **Continuous** | `TFWaveFormerContinuous` | 连续尺度 + 相邻核插值 | 默认首选 |
| **Implicit** | `TFWaveFormerImplicit` | MLP 隐式核函数 k(t,s,c) | 每通道核定制 |
| **Gumbel** | `TFWaveFormerGumbel` | Gumbel-Softmax 离散选择 | 可解释分析 |

三种方法都保持原版 K=4 头多尺度融合结构，区别仅在核尺寸的学习方式。

## 📁 核心文件

```
TFWaveFormer_continuous.py   ← 连续尺度插值
TFWaveFormer_implicit.py     ← 隐式神经表示
TFWaveFormer_gumbel.py       ← Gumbel 离散选择
train_link_prediction.py     ← 训练（已集成三种自适应模型）
evaluate_link_prediction.py  ← 评估
run_all_modes.py             ← 自动化对比
OPEN_PROBLEM_SOLUTIONS.md    ← 方法详解文档
```

## 🚀 运行示例

```bash
# 单模式训练
python train_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TFWaveFormerContinuous \
    --num_epochs 50 --num_runs 5 --gpu 0

# 自动对比三种方法（默认跑 processed_data/ 下所有数据集）
python run_all_modes.py \
    --datasets wikipedia reddit uci \
    --epochs 50 --runs 5 --gpu 0

# 原版固定尺度
python train_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TFWaveFormer \
    --num_epochs 50 --gpu 0
```

## 📚 文档

- [OPEN_PROBLEM_SOLUTIONS.md](OPEN_PROBLEM_SOLUTIONS.md) — 开放问题分析、三种方案详解（含数学原理、参考论文、核心代码）
- [TFWaveFormer-Notes.md](TFWaveFormer-Notes.md) — 原论文笔记

## 🔧 环境

```
Python 3.8+ | PyTorch 1.8+ (CUDA 推荐) | NumPy, Pandas, scikit-learn, tqdm
```

数据需放在 `processed_data/{dataset}/` 下，包含 `ml_{dataset}.csv`, `ml_{dataset}.npy`, `ml_{dataset}_node.npy`。

## 📊 验证结果

3 epochs × 1 run 验证性实验（详见 [OPEN_PROBLEM_SOLUTIONS.md](OPEN_PROBLEM_SOLUTIONS.md) 第 6 节）：

| Dataset | Continuous | Implicit | Gumbel |
|------|:---:|:---:|:---:|
| **reddit** | 0.9916 | 0.9922 | 0.9914 |
| **wikipedia** | 0.9926 | 0.9928 | 0.9924 |
| **uci** | 0.9487 | 0.9480 | 0.9519 |

## 🔗 引用

```bibtex
@inproceedings{tfwaveformer2026,
  title={TFWaveFormer: Temporal-Frequency Collaborative Multi-level Wavelet Transformer},
  author={Feng, Hantong and Wu, Yonggang and Chen, Duxin and Yu, Wenwu},
  booktitle={WWW},
  year={2026}
}
```

**版本**: 2.0 | **日期**: 2026-06-21
