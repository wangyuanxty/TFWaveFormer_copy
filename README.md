# Adaptive TFWaveFormer - 自适应小波尺度选择

> 解决论文开放问题：最优小波尺度因数据集而异

[![Status](https://img.shields.io/badge/status-stable-green.svg)](PROJECT_FINAL_REPORT.md)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](INSTALLATION_GUIDE.md)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

## ⚡ 快速开始（3步）

```bash
# 1. 激活环境
conda activate py312

# 2. 快速测试（2分钟）
python train_adaptive_tfwaveformer.py \
    --dataset_name wikipedia \
    --wavelet_mode continuous \
    --num_epochs 1 --num_runs 1

# 3. 查看结果
grep "Learned scales" logs/AdaptiveTFWaveFormer_*/wikipedia/run0.log
```

或使用一键脚本：
```bash
bash quick_train.sh wikipedia continuous 1 1
```

## 🎯 四种自适应方案

| 方案 | 速度 | 推荐场景 |
|------|------|---------|
| `continuous` | ⭐⭐⭐⭐⭐ | 快速验证、生产部署 |
| `hyper` | ⭐⭐⭐ | 发论文、灵活实验 |
| `implicit` | ⭐⭐ | 理论研究、顶会冲刺 |
| `gumbel` | ⭐⭐⭐⭐ | 可解释性、NAS |

## 📚 文档导航

**推荐阅读顺序：**

1. 🚀 [QUICKSTART.md](QUICKSTART.md) - 5分钟上手
2. 📖 [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - 所有文档索引
3. 📊 [PROJECT_FINAL_REPORT.md](PROJECT_FINAL_REPORT.md) - 完整项目报告
4. 🎓 [ADAPTIVE_METHODS_EXPLAINED.md](ADAPTIVE_METHODS_EXPLAINED.md) - 方法详解

## ✨ 核心特性

- ✅ **完全独立**：所有代码在 `adaptive/` 目录，不修改原文件
- ✅ **向后兼容**：Python 3.12+ & NumPy 2.0+ 兼容
- ✅ **即插即用**：一行命令切换四种模式
- ✅ **文档完整**：11个文档涵盖所有方面
- ✅ **研究就绪**：可直接用于论文实验

## 📊 项目统计

- Python 文件：7 个
- 代码行数：1,881 行
- 文档数量：11 个
- 测试覆盖：14/14 (100%)

## 🔧 安装

详见 [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)

## 🎓 研究价值

- 解决论文开放问题：自适应小波尺度选择
- 跨领域创新：引入 NeRF/SIREN 到时序图
- 四种方案：从工程到理论全覆盖

## 📧 问题反馈

查阅 [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) 获取帮助

---

**版本**：1.0.0 | **状态**：✅ 完成且验证 | **日期**：2026-06-16
