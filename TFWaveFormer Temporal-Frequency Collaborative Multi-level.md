
# TFWaveFormer: Temporal-Frequency Collaborative Multi-level Wavelet Transformer for Dynamic Link Prediction

Hantong Feng  
School of Cyber Science and Engineering, Southeast University  
Nanjing, China  
htfeng@seu.edu.cn

Yonggang Wu  
School of Mathematics, Southeast University  
Nanjing, China  
wuyg@seu.edu.cn

Duxin Chen\*  
School of Mathematics, Southeast University  
Nanjing, China  
chendx@seu.edu.cn

Wenwu Yu\*  
School of Mathematics, Southeast University  
Nanjing, China  
wwyu@seu.edu.cn

\* Correspondence to Duxin Chen and Wenwu Yu.

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.  
© 2026 Copyright held by the owner/author(s).  
ACM ISBN 979-8-4007-2307-0/2026/04  
https://doi.org/10.1145/3774904.3792422

Code: https://github.com/SEUFHTong/TFWaveFormer

---

## Abstract

Dynamic link prediction plays a crucial role in diverse applications including social network analysis, communication forecasting, and financial modeling. While recent Transformer-based approaches have demonstrated promising results in temporal graph learning, their performance remains limited when capturing complex multi-scale temporal dynamics.

In this paper, we propose **TFWaveFormer**, a novel Transformer architecture that integrates temporal-frequency analysis with multi-resolution wavelet decomposition to enhance dynamic link prediction. Our framework comprises three key components:

1. a temporal-frequency coordination mechanism that jointly models temporal and spectral representations;
2. a learnable multi-resolution wavelet decomposition module that adaptively extracts multi-scale temporal patterns through parallel convolutions, replacing traditional iterative wavelet transforms;
3. a hybrid Transformer module that effectively fuses local wavelet features with global temporal dependencies.

Extensive experiments on benchmark datasets demonstrate that TFWaveFormer achieves state-of-the-art performance, outperforming existing Transformer-based and hybrid models by significant margins across multiple metrics. The superior performance of TFWaveFormer validates the effectiveness of combining temporal-frequency analysis with wavelet decomposition in capturing complex temporal dynamics for dynamic link prediction tasks.

---

## CCS Concepts

- Theory of computation → Dynamic graph algorithms

## Keywords

Link Prediction; Continuous-Time Dynamic Networks; Wavelet Transforms; Complex Systems

## ACM Reference Format

Hantong Feng, Yonggang Wu, Duxin Chen, and Wenwu Yu. 2026. TFWaveFormer: Temporal-Frequency Collaborative Multi-level Wavelet Transformer for Dynamic Link Prediction. In *Proceedings of the ACM Web Conference 2026* (WWW ’26), April 13–17, 2026, Dubai, United Arab Emirates. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3774904.3792422

---

# 1. Introduction

Dynamic link prediction, as a core task in temporal graph analysis, derives its importance from the dynamic evolution inherent in real-world networks. Unlike static graph assumptions, real-world networks, such as social networks, biomolecular interaction networks, and information dissemination networks, exhibit significant temporal variability.

Accurately predicting the evolving connections in these networks over time not only holds theoretical value for understanding the dynamic behaviors of complex systems but also provides critical technical support for practical applications, such as recommender systems and epidemic forecasting. The advancements in dynamic graph neural networks have made it possible to learn evolutionary patterns from non-stationary temporal data. However, effectively capturing the multi-scale temporal dependencies in network dynamics remains an open challenge.

Dynamic link prediction faces fundamental challenges due to the multi-scale and complex temporal patterns in network evolution. These patterns include:

- periodic fluctuations in node interactions;
- long-range temporal dependencies influenced by global network states;
- abrupt topological changes caused by sudden events.

For example, in academic collaboration networks, two researchers may establish collaborations due to regular participation in periodic academic events, while their collaboration intensity is also influenced by long-term factors such as development trends in their research field. Existing methods frequently fail to model such cross-scale dependencies, resulting in systematic biases in predictions.

The prevailing methodologies are limited in several ways:

1. RNN-based architectures can handle sequential data but struggle to capture long-range dependencies due to gradient vanishing or explosion.
2. Conventional temporal attention mechanisms, such as TGAT and DyGFormer, are deficient in differentiating frequency patterns in nonstationary time intervals.
3. Purely frequency-domain methods capture global frequency characteristics but fail to preserve localized temporal details.
4. Fixed window-length mechanisms cannot adapt to variable-period patterns in real-world dynamic networks.

To address these bottlenecks, this paper proposes **Temporal-Frequency Collaborative Wavelet-Transformer (TFWaveFormer)**, a framework designed for multi-scale temporal modeling in dynamic link prediction.

The framework integrates temporal and frequency-domain perspectives. The time domain captures local dynamics and directly models microscopic evolution across consecutive timestamps. Wavelet transformation extracts local temporal-frequency features through multi-resolution analysis, overcoming limitations of traditional frequency-domain methods.

TFWaveFormer adopts a three-stage workflow:

1. Feature integration with positional encoding and Transformer-based interaction modeling.
2. A WaveletFilter module simulating wavelet transforms through multi-scale convolution kernels.
3. An attention-based temporal-frequency fusion framework for final node embeddings.

The main contributions are:

- We propose a learnable wavelet decomposition module that replaces traditional fixed-basis transforms with parallel multi-scale convolutional kernels.
- We design a temporal-frequency coordination mechanism that integrates time-domain and frequency-domain representations.
- We validate TFWaveFormer as a unified Transformer architecture and demonstrate state-of-the-art performance on ten benchmark datasets.

---

# 2. Related Work

Dynamic graphs have attracted significant attention due to their ability to model evolving relationships in real-world systems. Early methods extended static models with recurrent units or time-aware attention. Later approaches introduced specialized temporal architectures:

- JODIE uses coupled RNNs for interaction dynamics.
- TGAT and DySAT leverage temporal self-attention.
- GraphMixer adopts MLPs and pooling to simplify temporal modeling.
- TGN introduces a memory-based continuous-time framework.
- CAWN exploits anonymized temporal walks.
- DyGFormer and CorDGT enhance long-range temporal representation and spatiotemporal locality.

Despite these advances, most temporal models overlook frequency-domain representations. Spectral filtering and wavelet transforms can provide complementary insights into dynamic behaviors. FFT-based methods such as FreeDyG are effective at extracting periodic patterns, but existing models often fail to jointly capture micro-level temporal events and macro-level structural trends.

This motivates a unified framework integrating temporal and frequency perspectives through multi-scale modeling.

---

# 3. Preliminary

## Problem Formulation

This work considers a dynamic graph as a sequence of non-decreasing chronological interactions:

$$
X = \{(u_1, v_1, t_1), (u_2, v_2, t_2), \cdots \}
$$

with:

$$
0 \leq t_1 \leq t_2
$$

where \(u_i, v_i \in \mathcal{V}\) denote the source node and destination node of the \(i\)-th edge at timestamp \(t_i\).

The purpose of this algorithm is to learn node embeddings by leveraging both target nodes and their historical neighbors, and to predict the probability of a link existing between target nodes.

## Wavelet Transform Foundations

Traditional wavelet transforms provide a mathematical framework for multi-resolution analysis by decomposing signals into different frequency components. Given \(x(t)\), the continuous wavelet transform is:

$$
W(a,b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t)\psi^*\left(\frac{t-b}{a}\right)dt
\tag{1}
$$

where:

- \(\psi(t)\) is the mother wavelet;
- \(a\) is the scale parameter;
- \(b\) is the translation parameter;
- \(\psi^*\) denotes the complex conjugate.

Unlike traditional approaches relying on fixed wavelet bases, TFWaveFormer employs learnable convolutional kernels:

$$
\Psi = \{\psi_k^{(i)} \in \mathbb{R}^k \mid k \in \mathcal{K}, i \in [1,d]\}
$$

to achieve data-driven wavelet-like decomposition.

---

# 4. Methodology

The TFWaveFormer architecture consists of three key components:

1. feature integration;
2. multi-level wavelet transformation;
3. temporal-frequency hybrid Transformer.

The original data first undergoes feature integration, then dimension compression and multi-level wavelet transformation, and finally produces representations through a temporal-frequency hybrid Transformer.

![Figure 1: The proposed TFWaveFormer framework.](figure-1-placeholder)

**Figure 1:** The proposed TFWaveFormer framework. The architecture consists of three key components: feature integration, multi-level wavelet transformation, and temporal-frequency hybrid Transformer for generating final representations.

---

## 4.1 Feature Extraction

Dynamic graphs encode information through multiple modalities. For each node \(v\), historical interaction patterns are extracted to construct rich feature representations.

The node feature is:

$$
H_v^{node} \in \mathbb{R}^{d_n}
$$

The edge feature is:

$$
H_v^{edge} \in \mathbb{R}^{d_e}
$$

Temporal features are processed through a time encoder:

$$
H_v^{time} = \text{Time-Encoder}([t - t_1, t - t_2, ..., t - t_L])
\tag{2}
$$

Node Interaction Frequency (NIF) features capture topological relationships:

$$
H_v^{if} = \text{NIF-Encoder}([\text{count}(N(v) \cap N(u)), \text{freq}(N(v))])
\tag{3}
$$

where \(N(v)\) denotes the neighbor set of node \(v\).

To unify different modalities, feature alignment maps all features to a common space:

$$
F_v^{(m)} = H_v^{(m)}W^{(m)} + b^{(m)}
\tag{4}
$$

where:

$$
m \in \{node, edge, time, if\}
$$

The final fused features are obtained through concatenation-compression:

$$
X_v = \text{ReduceLayer}(\text{Concat}([F_v^{node}, F_v^{edge}, F_v^{time}, F_v^{if}]))
\tag{5}
$$

where \(X_v\) serves as the input for subsequent wavelet transformation and feature extraction.

---

## 4.2 Multi-Level Wavelet

Given input features:

$$
X_v \in \mathbb{R}^{L \times d}
$$

where \(L\) is the length of historical interactions and \(d\) is the feature dimension.

The temporal domain is defined as:

$$
\mathcal{T} = \{t_1, t_2, ..., t_L\}
$$

TFWaveFormer employs multi-level wavelet transformation to decompose dynamic graph data. Unlike traditional methods using predefined wavelet basis functions, this method realizes adaptive wavelet decomposition through learnable multi-scale convolutional kernels:

$$
\Psi = \{\psi_k^{(i)}\}_{i=1}^{d}
$$

---

## 4.2.1 Decomposition

The multi-level wavelet transform uses learnable convolutional filters \(\psi_k^{(i)}\) with different scales \(k\) to simulate wavelet-like behavior.

The scale space is:

$$
\mathcal{K} = \{k_1, k_2, ..., k_m\}
$$

The wavelet transform is formulated as:

$$
Z_k^{(i)}[t] =
\sum_{j \in R_k} \psi_k^{(i)}[j] \cdot X_v^{(i)}[t+j],
\quad
\forall k \in \mathcal{K}, \forall i \in [1,d]
\tag{6}
$$

The decomposition output is:

$$
Z = \{Z_{k_1}^{(1:d)}, Z_{k_2}^{(1:d)}, ..., Z_{k_m}^{(1:d)}\}
= \text{Decomp}(X_v, \Psi, m, \theta)
\tag{7}
$$

where \(\theta\) denotes learnable parameters.

Each filter is parameterized as a depth-wise separable convolutional filter with kernel size \(k\). The temporal receptive field is:

$$
R_k = \{r \mid |r| \leq \lfloor k/2 \rfloor\}
$$

Smaller kernels capture fine-grained short-term patterns, while larger kernels capture coarse-grained long-term trends.

---

## 4.2.2 Parallel Representation

Traditional iterative wavelet decomposition is replaced by parallelized multi-scale convolution.

The parameters are optimized by:

$$
\theta^* = \arg\min_{\theta \in \Theta} \mathcal{L}(\theta) + \lambda \|\Psi\|^2
\tag{8}
$$

where \(\lambda\) is the regularization coefficient.

Each feature channel \(i\) is processed independently:

$$
C_k^{(i)}(X_v) =
\sum_{j \in R_k} \psi_k^{(i,j)} \cdot X_v^{(t+j,i)},
\quad
\forall t \in [1,L]
\tag{9}
$$

The outputs of convolutional filters are combined into a unified multi-scale representation. Scale weights are normalized through softmax:

$$
S_k =
\frac{\exp(w_k/\tau)}
{\sum_{k' \in \mathcal{K}} \exp(w_{k'}/\tau)},
\quad \tau > 0
\tag{10}
$$

The resulting multi-scale wavelet representation is:

$$
Z_{ms} = \sum_{k \in \mathcal{K}} S_k \odot Z_k
\tag{11}
$$

where \(\odot\) denotes the Hadamard product.

---

## 4.2.3 Reconstruction

To adaptively assess feature importance across time and channels, a gating mechanism is introduced:

$$
G \in \mathbb{R}^{L \times d}
$$

Two MLP networks are defined:

$$
f_1, f_2 : \mathbb{R}^{d} \rightarrow \mathbb{R}^{d}
$$

The gate is computed as:

$$
G = \sigma(f_2(\text{GELU}(f_1(Z_{ms}) + b_1)) + b_2)
\tag{12}
$$

The reconstructed multi-level wavelet representation is:

$$
Z_{gated} = G \odot Z_{ms}
\tag{13}
$$

The refined representation is then passed to the temporal-frequency fusion module.

---

## Algorithm 1: TFWaveFormer Algorithm

```text
Require:
    Node v, length L, scales K, heads h

Ensure:
    Node embedding h_hat_v, prediction y_hat_uv

1: Feature Integration:
2:     X_v <- Fuse([H_v^node, H_v^edge, H_v^time, H_v^if])

3: Multi-Level Wavelet:
4:     // Decomposition
5:     for k in K, i in [1,d], t in [1,L] do
6:         Z_k^(i)[t] <- sum_{j in R_k} psi_k^(i)[j] * X_v^(i)[t+j]
7:     end for

8:     // Parallel Representation
9:     S_k <- Softmax(w_k / tau) for all k in K
10:    Z_ms <- sum_{k in K} S_k * Z_k

11:    // Reconstruction
12:    G <- sigmoid(f_2(GELU(f_1(Z_ms) + b_1)) + b_2)
13:    Z_gated <- G * Z_ms

14: Temporal-Frequency Hybrid Transformer:
15:    Z_0 <- LayerNorm(MLP(X_v) + Z_gated + PE)
16:    Z_1 <- LayerNorm(Z_0 + MHSA(Z_0))
17:    h_v <- LayerNorm(Z_1 + MLP(Z_1))

18: Dynamic Link Prediction:
19:    h_hat_v <- mean_pooling(h_v)
20:    s_uv <- w^T(h_hat_u * h_hat_v) + b
21:    y_hat_uv <- sigmoid(s_uv)

22: return h_hat_v, y_hat_uv
```

---

## 4.3 Temporal-Frequency Hybrid Transformer

The temporal-frequency features are fused based on multi-head self-attention.

The positional encoding is:

$$
PE(pos, 2i) = \sin(pos / 10000^{2i/d})
$$

$$
PE(pos, 2i+1) = \cos(pos / 10000^{2i/d})
\tag{14}
$$

The temporal feature is compressed by MLP:

$$
Z_t = MLP(X_v)
$$

Then temporal and frequency-domain features are fused:

$$
Z_0 = \text{LayerNorm}(Z_t + Z_{gated} + PE)
\tag{15}
$$

$$
Z_1 = \text{LayerNorm}(Z_0 + MHSA(Z_0))
\tag{16}
$$

$$
h_v = \text{LayerNorm}(Z_1 + MLP(Z_1))
\tag{17}
$$

The multi-head self-attention mechanism is:

$$
MHSA(Z) = \text{Concat}(head_1, ..., head_h)W^O
\tag{18}
$$

where:

$$
head_i =
\text{Softmax}
\left(
\frac{Q_iK_i^\top}{\sqrt{d/h}}
\right)V_i
\tag{19}
$$

and:

$$
Q_i = ZW_i^Q,\quad
K_i = ZW_i^K,\quad
V_i = ZW_i^V
$$

The full architecture comprises at least two stacked layers to ensure sufficient representational capacity.

---

## 4.4 Dynamic Link Prediction

After training, the learned model performs dynamic link prediction. Given node embeddings \(h_u\) and \(h_v\), temporal aggregation is first applied:

$$
\hat{h}_v = \frac{1}{L}\sum_{t=1}^{L} h_v[t]
$$

The prediction score is:

$$
s_{uv} = w^T(\hat{h}_u \odot \hat{h}_v) + b
\tag{20}
$$

The predicted probability is:

$$
\hat{y}_{uv} = \sigma(s_{uv})
\tag{21}
$$

The model is trained with cross-entropy loss:

$$
\mathcal{L} =
\frac{1}{|\mathcal{E}|}
\sum_{(u,v) \in \mathcal{E}}
\log(1 + \exp(-y^*_{uv} \cdot s_{uv}))
\tag{22}
$$

where:

$$
y^*_{uv} = 2y_{uv} - 1 \in \{-1, 1\}
$$

---

# 5. Experimental Design and Evaluation

## 5.1 Datasets and Benchmarks

All experiments are conducted on:

- Intel Xeon Gold 6326 CPU @ 2.90GHz
- NVIDIA RTX A6000 GPU

TFWaveFormer is benchmarked against several dynamic graph learning methods:

- DyRep
- TGN
- CAWN
- GraphMixer
- DyGFormer
- FreeDyG
- CorDGT
- CTAN
- DyGMamba

The evaluation covers ten real-world datasets:

| Dataset | Domain |
|---|---|
| Wikipedia | Online collaboration |
| Reddit | Social interaction |
| MOOC | Education |
| Social Evolution | Proximity |
| LastFM | Entertainment |
| Enron | Communication |
| UCI | Communication |
| Flights | Mobility |
| Contact | Physical proximity |
| UN Trade | International trade |

Performance is assessed using:

- Average Precision (AP)
- Area Under the ROC Curve (AUC)

Datasets are chronologically split into:

- 70% training
- 15% validation
- 15% testing

---

## 5.2 Performance Comparison Analysis

TFWaveFormer achieves the best balance between predictive accuracy and training efficiency. Compared with recent approaches such as FreeDyG, DyGFormer, CorDGT, CTAN, and DyGMamba, TFWaveFormer provides superior performance while maintaining comparable training efficiency.

Under random negative sampling, TFWaveFormer achieves average rankings:

| Setting | AP Rank | AUC Rank |
|---|---:|---:|
| Transductive | 1.20 | 1.40 |
| Inductive | 1.70 | 1.60 |

Notable results include:

- Wikipedia: 99.33% AP in transductive setting.
- Reddit: 99.32% AP in transductive setting.
- MOOC: 91.24% AP, outperforming the second-best method by a significant margin.
- LastFM: 94.64% AP.
- UN Trade: 66.06% AP, maintaining strong performance on sparse datasets.

The superior performance stems from:

1. Multi-resolution wavelet decomposition capturing short-term and long-term temporal patterns.
2. Adaptive gated fusion dynamically balancing temporal and frequency-domain features.
3. Transformer-based global dependency modeling.

---

## Table 1: Transductive Dynamic Link Prediction Results under Random Negative Sampling

| Dataset | Metric | TFWaveFormer |
|---|---|---:|
| Wikipedia | AP | 99.33 ± 0.01 |
| Reddit | AP | 99.32 ± 0.01 |
| MOOC | AP | 91.24 ± 0.05 |
| LastFM | AP | 94.64 ± 0.05 |
| Enron | AP | 92.70 ± 0.20 |
| Social Evo. | AP | 94.65 ± 0.01 |
| UCI | AP | 96.51 ± 0.03 |
| Flights | AP | 98.87 ± 0.01 |
| UN Trade | AP | 66.06 ± 0.24 |
| Contact | AP | 98.12 ± 0.01 |
| Average Rank | AP | 1.20 |
| Wikipedia | AUC | 99.31 ± 0.01 |
| Reddit | AUC | 99.29 ± 0.01 |
| MOOC | AUC | 91.89 ± 0.35 |
| LastFM | AUC | 94.52 ± 0.04 |
| Enron | AUC | 93.97 ± 0.09 |
| Social Evo. | AUC | 96.38 ± 0.02 |
| UCI | AUC | 95.67 ± 0.05 |
| Flights | AUC | 98.90 ± 0.01 |
| UN Trade | AUC | 69.96 ± 1.04 |
| Contact | AUC | 98.49 ± 0.01 |
| Average Rank | AUC | 1.40 |

---

## Table 2: Inductive Dynamic Link Prediction Results under Random Negative Sampling

| Dataset | Metric | TFWaveFormer |
|---|---|---:|
| Wikipedia | AP | 98.94 ± 0.01 |
| Reddit | AP | 98.98 ± 0.01 |
| MOOC | AP | 90.26 ± 0.36 |
| LastFM | AP | 95.49 ± 0.02 |
| Enron | AP | 88.25 ± 0.27 |
| Social Evo. | AP | 93.21 ± 0.02 |
| UCI | AP | 94.97 ± 0.06 |
| Flights | AP | 97.71 ± 0.02 |
| UN Trade | AP | 65.40 ± 0.59 |
| Contact | AP | 97.79 ± 0.02 |
| Average Rank | AP | 1.70 |
| Wikipedia | AUC | 98.93 ± 0.01 |
| Reddit | AUC | 98.89 ± 0.01 |
| MOOC | AUC | 91.81 ± 0.25 |
| LastFM | AUC | 95.31 ± 0.02 |
| Enron | AUC | 89.56 ± 0.13 |
| Social Evo. | AUC | 95.70 ± 0.02 |
| UCI | AUC | 93.58 ± 0.11 |
| Flights | AUC | 97.73 ± 0.02 |
| UN Trade | AUC | 67.49 ± 1.14 |
| Contact | AUC | 98.32 ± 0.02 |
| Average Rank | AUC | 1.60 |

---

## 5.3 Ablation Analysis

The ablation study evaluates the contribution of core components in TFWaveFormer. Two variants are analyzed:

- **w/o Temporal**: removes the temporal modeling branch including the NIF encoder.
- **w/o Frequency**: removes the multi-level wavelet transform module.

The results show that both components are essential. The frequency-domain module is particularly critical, especially on challenging datasets such as MOOC and LastFM. Removing the temporal component also causes consistent degradation, especially on sparse temporal graphs such as UN Trade and Contact.

The ablation results indicate that effective dynamic graph learning requires synergistic integration of temporal-domain and frequency-domain representations.

![Figure 4: Ablation study results.](figure-4-placeholder)

**Figure 4:** Results on ablation study for dynamic link prediction under different settings:  
(a) Transductive setting,  
(b) Inductive setting.

---

## 5.4 Robustness Analysis

Dynamic link prediction is evaluated under three negative sampling strategies:

1. random negative sampling;
2. historical negative sampling;
3. inductive negative sampling.

The historical strategy selects negative samples from nodes that previously interacted but not at the current timestamp. The inductive strategy selects negative samples from completely unseen node pairs during training.

TFWaveFormer maintains robust performance across all settings.

## Table 3: AP Results under Historical and Inductive Negative Sampling

| Dataset | Transductive Hist | Transductive Ind | Inductive Hist | Inductive Ind |
|---|---:|---:|---:|---:|
| Wikipedia | 86.21 | 73.36 | 70.08 | 70.07 |
| Reddit | 83.77 | 90.88 | 68.50 | 68.54 |
| MOOC | 88.31 | 82.46 | 79.97 | 79.99 |
| LastFM | 84.80 | 75.52 | 79.52 | 79.52 |
| Enron | 78.64 | 84.23 | 81.26 | 81.26 |
| Social Evo. | 96.96 | 97.19 | 95.85 | 95.86 |
| UCI | 90.19 | 86.95 | 87.57 | 87.59 |
| Flights | 65.31 | 71.20 | 56.68 | 56.72 |
| UN Trade | 60.16 | 61.37 | 53.88 | 53.90 |
| Contact | 97.85 | 96.10 | 95.27 | 94.69 |

---

## 5.5 Parameter Analysis

The sensitivity analysis focuses on the number of wavelet convolutional kernels \(m\).

Different datasets exhibit different optimal values:

- Wikipedia and Reddit achieve optimal performance at \(m = 5\).
- Enron and UCI achieve optimal performance at \(m = 3\).
- Datasets with more diverse temporal patterns require larger \(m\) values.

The results suggest a correlation between dataset temporal complexity and the optimal number of wavelet scales.

![Figure 3: Hyper-parameter sensitivity to the number of wavelet kernels.](figure-3-placeholder)

**Figure 3:** Results of hyper-parameter sensitivity to the number of wavelet convolution kernels \(m\) across datasets.

---

# 6. Conclusion

This work proposes **TFWaveFormer**, a temporal-frequency collaborative framework designed to overcome key limitations in dynamic link prediction.

By integrating micro-level temporal dynamics with macro-level evolutionary trends via multi-scale modeling, TFWaveFormer captures both transient events and periodic patterns in evolving graphs.

Extensive experiments on ten real-world datasets show that TFWaveFormer consistently achieves state-of-the-art results across both transductive and inductive settings. The analysis further shows that optimal wavelet scales vary across datasets, highlighting the need for adaptive configuration based on graph characteristics.

---

# Acknowledgments

This research was supported by:

- National Natural Science Foundation of China, Grants No. 62233004, 62273090, and T2541017;
- Youth Scientist Project of the Ministry of Science and Technology of China, Grant No. 2025YFF0524100;
- Zhishan Youth Scholar Program of Southeast University;
- Jiangsu Provincial Scientific Research Center of Applied Mathematics, Grant No. BK20233002;
- Basic Research Program of Jiangsu, Grant No. BK20253018;
- Open Research Project of the State Key Laboratory of Industrial Control Technology, China, Grant No. ICT2025B54.

---

# Appendix A: Implementation Details

## A.1 Details of Datasets

The evaluation includes ten datasets from diverse domains.

Small-scale datasets:

- Enron: 184 nodes, 125K edges.
- UN Trade: 255 nodes, 507K edges.
- Contact: 692 nodes, 2.4M edges.
- UCI: 1,899 nodes, 59K edges.

Large-scale datasets:

- LastFM: 1,980 nodes, 1.3M edges.
- MOOC: 7,144 nodes, 411K edges.
- Wikipedia: 9,227 nodes, 157K edges.
- Reddit: 10,984 nodes, 672K edges.
- Flights: 13,169 nodes, 1.9M edges.
- Social Evolution: 74 nodes, 2.1M edges.

## Table 4: Dataset Statistics

| Dataset | # Nodes | # Links | Time Span | Domain |
|---|---:|---:|---|---|
| Wikipedia | 9,227 | 157,474 | 1 month | Social |
| Reddit | 10,984 | 672,447 | 1 month | Social |
| MOOC | 7,144 | 411,749 | 17 months | Interaction |
| LastFM | 1,980 | 1,293,103 | 1 month | Interaction |
| Enron | 184 | 125,235 | 3 years | Social |
| Social Evo. | 74 | 2,099,519 | 8 months | Proximity |
| UCI | 1,899 | 59,835 | 196 days | Social |
| Flights | 13,169 | 1,927,145 | 4 months | Transport |
| UN Trade | 255 | 507,497 | 32 years | Economics |
| Contact | 692 | 2,426,279 | 1 month | Proximity |

---

## A.2 Details of Experiment Results

The appendix reports detailed AP and AUC-ROC results under historical and inductive negative sampling strategies. Across these challenging settings, TFWaveFormer maintains strong performance and demonstrates robust generalization to unseen links and evolving graph structures.

---

## A.3 Details of Parameter Analysis

Additional experiments on UN Trade and Contact further validate the robustness and generalizability of TFWaveFormer across different network structures and scales.

---

# References

[1] Seyed Mehran Kazemi, Rishab Goel, Kshitij Jain, Ivan Kobyzev, Akshay Sethi, Peter Forsyth, and Pascal Poupart. Representation learning for dynamic graphs: A survey. *J. Mach. Learn. Res.*, 21:70:1–70:73, 2020.

[2] Ocheme Anthony Ekle and William Eberle. Anomaly detection in dynamic graphs: A comprehensive survey. *ACM Transactions on Knowledge Discovery from Data*, 18(8):1–44, 2024.

[3] Unai Alvarez-Rodriguez, Federico Battiston, Guilherme Ferraz de Arruda, Yamir Moreno, Matjaž Perc, and Vito Latora. Evolutionary dynamics of higher-order interactions in social networks. *Nature Human Behaviour*, 5(5):586–595, 2021.

[4] E. Amiri Souri, Roman Laddach, S. N. Karagiannis, Lazaros G. Papageorgiou, and Sophia Tsoka. Novel drug-target interactions via link prediction and network embedding. *BMC Bioinform.*, 23(1):121, 2022.

[5] Muhan Zhang and Yixin Chen. Inductive matrix completion based on graph neural networks. In *ICLR*, 2020.

[6] Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal A. C. Xhonneux, and Jian Tang. Neural Bellman-Ford Networks: A general graph neural network framework for link prediction. In *NeurIPS*, pages 29476–29490, 2021.

[7] Jiaru Bai, Sebastian Mosbach, Connor J Taylor, Dogancan Karan, Kok Foong Lee, Simon D Rihm, Jethro Akroyd, Alexei A Lapkin, and Markus Kraft. A dynamic knowledge graph approach to distributed self-driving laboratories. *Nature Communications*, 15(1):462, 2024.

[8] Zeyang Zhang, Xin Wang, Ziwei Zhang, Haoyang Li, Yijian Qin, and Wenwu Zhu. LLM4DyG: Can large language models solve spatial-temporal problems on dynamic graphs? In *KDD*, pages 4350–4361, 2024.

[9] Srijan Kumar, Xikun Zhang, and Jure Leskovec. Predicting dynamic embedding trajectory in temporal interaction networks. In *KDD*, pages 1269–1278, 2019.

[10] Emanuele Rossi, Ben Chamberlain, Fabrizio Frasca, Davide Eynard, Federico Monti, and Michael Bronstein. Temporal graph networks for deep learning on dynamic graphs. In *ICML Workshop on Graph Representation Learning*, 2020.

[11] Yao Ma, Ziyi Guo, Zhaochun Ren, Jiliang Tang, and Dawei Yin. Streaming graph neural networks. In *SIGIR*, pages 719–728, 2020.

[12] Boris Weisfeiler and Andrei Leman. The reduction of a graph to canonical form and the algebra which appears therein. 1968.

[13] Anders Aamand, Justin Chen, Piotr Indyk, Shyam Narayanan, Ronitt Rubinfeld, Nicholas Schiefer, Sandeep Silwal, and Tal Wagner. Exponentially improving the complexity of simulating the Weisfeiler-Lehman test with graph neural networks. *NeurIPS*, 35:27333–27346, 2022.

[14] Xuhong Wang et al. APAN: Asynchronous propagation attention network for real-time temporal graph embedding. In *SIGMOD*, pages 2628–2638, 2021.

[15] Yuhong Luo and Pan Li. Neighborhood-aware scalable temporal network representation learning. In *Learning on Graphs Conference*, 2022.

[16] Da Xu, Chuanwei Ruan, Evren Körpeoglu, Sushant Kumar, and Kannan Achan. Inductive representation learning on temporal graphs. In *ICLR*, 2020.

[17] Le Yu, Leilei Sun, Bowen Du, and Weifeng Lv. Towards better dynamic graph learning: New architecture and unified library. *NeurIPS*, 36:67686–67700, 2023.

[18] Petar Veličković et al. Graph attention networks. *arXiv preprint arXiv:1710.10903*, 2017.

[19] Jiapu Wang et al. Large language models-guided dynamic adaptation for temporal knowledge graph reasoning. *NeurIPS*, 37:8384–8410, 2024.

[20] Han Shi, Haozheng Fan, and James T. Kwok. Effective decoding in graph auto-encoder using triadic closure. *AAAI*, 34(01):906–913, 2020.

[21] Amauri H. Souza, Diego Mesquita, Samuel Kaski, and Vikas Garg. Provably expressive temporal graph networks. In *NeurIPS*, 2022.

[22] Aravind Sankar et al. DySAT: Deep neural representation learning on dynamic graphs via self-attention networks. In *WSDM*, pages 519–527, 2020.

[23] Weilin Cong et al. Do we really need complicated model architectures for temporal networks? In *ICLR*, 2023.

[24] Yanbang Wang et al. Inductive representation learning in temporal networks via causal anonymous walks. In *ICLR*, 2021.

[25] Zhe Wang et al. Dynamic graph transformer with correlated spatial-temporal positional encoding. In *WSDM*, pages 60–69, 2025.

[26] Fan Xu et al. Revisiting graph-based fraud detection in sight of heterophily and spectrum. In *AAAI*, 2024.

[27] Bin Wu et al. SplitGNN: Spectral graph neural network for fraud detection against heterophily. In *CIKM*, 2023.

[28] Baisen Xiong, Sijie Wen, Ping Lu, and Kaibiao Lin. A spectral domain graph transformer model with position-encoded information. *IAENG International Journal of Computer Science*, 52(6), 2025.

[29] Esraa M. Shalby et al. A comprehensive guide to selecting suitable wavelet decomposition level and functions in discrete wavelet transform for fault detection in distribution networks. *Scientific Reports*, 15(1):1160, 2025.

[30] Xin Gao et al. Efficient multi-scale network with learnable discrete wavelet transform for blind motion deblurring. In *CVPR*, pages 2733–2742, 2024.

[31] Yuxing Tian, Yiyan Qi, and Fan Guo. FreeDyG: Frequency enhanced continuous-time dynamic graph model for link prediction. In *ICLR*, 2024.

[32] V. Necula, S. Klimenko, and G. Mitselmakher. Transient analysis with fast Wilson-Daubechies time-frequency transform. *Journal of Physics: Conference Series*, 363:012032, 2012.

[33] Rakshit Trivedi, Mehrdad Farajtabar, Prasenjeet Biswal, and Hongyuan Zha. DyRep: Learning representations over dynamic graphs. In *ICLR*, 2019.

[34] Alessio Gravina, Giulio Lovisotto, Claudio Gallicchio, Davide Bacciu, and Claas Grohnfeldt. Long range propagation on continuous-time dynamic graphs. In *ICML*, 2025.

[35] Zifeng Ding et al. DyGMamba: Efficiently modeling long-term temporal dependency on continuous-time dynamic graphs with state space models. *TMLR*, 2025.
