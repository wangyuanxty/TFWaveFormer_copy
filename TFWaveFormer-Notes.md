# TFWaveFormer：时频协同多级小波Transformer

> 冯翰彤, 吴永刚, 陈都鑫\*, 虞文武\*
>
> *WWW '26*, 2026年4月, 迪拜 · 代码: github.com/SEUFHTong/TFWaveFormer

---

## 问题逻辑链

```
根本矛盾：动态链路预测需同时捕捉"秒级爆发+日/周周期+数月趋势"
  → 多尺度时间模式在同一序列中共存
        ↓
时域方法(RNN/TGN)→见树不见林，长程依赖丢失
频域方法(FFT/FreeDyG)→有频率但丢时间定位
固定窗口(GraphMixer)→一个窗口不能同时看快慢
        ↓
解法：小波变换→时频同时看到；可学习卷积核替代固定基→数据驱动尺度选择；
     时频加法融合→自注意力直接跨域学习
```

---

## Abstract

**小波变换（Wavelet Transform）**：不像傅里叶只看整段信号频率成分，小波用可变宽度的"窗口"在时间轴上滑动——窄窗看高频细节(时间精确)、宽窗看低频趋势(频率精确)。公式：`W(a,b) = (1/√a)∫ x(t)·ψ((t-b)/a) dt`，a=尺度(压缩/拉伸核)，b=平移(滑窗位置)。

**费曼拆解**：傅里叶=蛋糕成分表(不知道哪层是哪层)。小波=勺子一勺一勺尝——每口都告诉你时间和频率两个维度。

**可学习小波分解**：用K个不同kernel_size的可学习卷积核并行扫信号(kernel_size=3看秒级爆发，=15看长期趋势)。不串行迭代(DWT)，而是并行多尺度(CWT思路的离散近似)。

---

## 1. Introduction — 四类方法盲区

| 类型 | 代表 | 盲区 |
|------|------|------|
| RNN系 | JODIE/TGN | 梯度消失→长程依赖丢失 |
| Transformer | DyGFormer | 分不清不同频率的非平稳模式 |
| 频域(FFT) | FreeDyG | 有频谱但不知道"哪个频率何时出现" |
| 固定窗口 | GraphMixer | 一个窗口无法同时看秒级爆发和月级趋势 |

---

## 3-4. Method（按Algorithm 1逐行）

**Line 1: 特征融合**。四种特征拼接(H_node/H_edge/H_time/H_if)→X_v∈R^{L×d}, L=时间步数, d=特征维度。

**Line 5-7: 多尺度卷积分解**。K个不同kernel_size卷积核ψ_k并行在d个特征维度上独立做depthwise conv→Z_k∈R^{L×d}。kernel_size=3/5/7/.../2K_max+1。K是尺度种类数。

**费曼拆解**：同一时间序列被K把不同倍率的放大镜同时看——小核看"一秒内的波动"，大核看"十分钟以上的趋势"。各走各的，不依赖前一层输出。

**Line 9-10: 尺度注意力**。`S_k=softmax(w_k/τ), w_k∈R^d`——每个特征通道在K个尺度间独立分配注意力权重。τ是温度(越小越集中在最优尺度)。`Z̄=Σ S_k⊙Z_k`——K个尺度按注意力加权求和。

**费曼拆解**：混音台。K条音轨各一条，d个频道各一组独立的"哪条轨更好"判断。高频通道可能偏好小核(局部细节)，低频通道可能偏好大核(趋势)。

**Line 12-13: 门控去噪**。`G=σ(f₂(GELU(f₁(Z̄))))`→`Z_gated=G⊙Z̄`。自动把噪声频率压小音量。

**费曼拆解**：不是所有频率成分都有用。门控=自动调音器——给信噪比低的频段拉低音量。

**Line 15: 时频加法融合**。`Z⁰=LayerNorm(MLP(X_v)+Z_gated+PE)`。时域和频域特征在**同一向量空间**加法混合——自注意力直接跨维度看，不需额外的跨模态对齐。

**Line 16-17: Transformer**。MHSA→残差+LN→FFN→残差+LN。标准块。

**Line 19-21: 链路预测**。`ĥ_v=(1/L)Σ_t h_v[t]`(时间池化)→`s_uv=w^T(ĥ_u⊙ĥ_v)+b`→`ŷ_uv=σ(s_uv)`。

---

## 与CWT/DWT的关系

论文理论基础是**CWT**（原文公式1直接引用CWT），但实现上用K个离散kernel_size并行扫——**不是递归二进制DWT，也不是真正连续CWT**。K对应CWT离散抽样的尺度a值，|K|对应抽了多少个点。

---

## 开放问题与未来方向

### 1. 最优尺度自动配置

> "our analysis highlights that the optimal wavelet scales vary by dataset, underscoring the need for adaptive configuration based on graph characteristics."（Section 6）

### 2. K>5的探索

> "Due to time constraints, we only tested values of K up to 5. It is possible that more complex datasets may benefit from higher values, which we leave for future exploration."（消融段）

### 3. 可学习卷积核是否满足小波数学性质

零均值、紧支撑、正交性——论文无理论分析，仅实验验证有效性。可能是纯数据驱动的"多尺度特征提取器"，"小波"是名义类比。

### 4. 跨尺度交互

不同尺度间目前无直接信息交换——尺度注意力是单向加权。加跨尺度自注意力可让高频和低频互相修正。

---

## 概念速查

| 概念 | 费曼 |
|------|------|
| **小波变换** | 可变宽度的放大镜——窄窗看高频细节，宽窗看低频趋势 |
| **傅里叶vs小波** | 蛋糕成分表 vs 一勺一勺尝 |
| **K** | 尺度种类数=多少个不同大小的卷积核并行扫 |
| **尺度注意力** | 混音台——模型自己学哪个尺度的"频道"重要 |
| **门控去噪** | 自动调音器——噪声频率压小，有用频率保留 |
| **时频加法融合** | 时间域和频率域住在同一个向量里，自注意力直接跨维度看 |
| **CWT vs DWT** | CWT连续尺度(理论基础)，DWT二进制递归；TFWaveFormer=离散卷积核并行的CWT近似 |
