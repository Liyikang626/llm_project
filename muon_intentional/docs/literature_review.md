# 文献调研：Muon × intentional（函数空间）步长

检索时间：2026-11-20。检索途径：arXiv API 全库检索 + 网页检索（关键词：Muon / optimizer /
trust region / step size / intentional update / function-space / RLVR / GRPO）。

## 结论摘要

1. **"Muon 的步长选择"是一个已经拥挤的公开问题**，2026 年至少有 5 篇正面处理，
   而且是 ICLR/NeurIPS workshop 层面在讨论的题目。
2. **这些工作全部把步长定义在参数空间或谱空间**。把目标定义在**函数空间**
   （层输出的变化、log-prob 的变化）的，目前只在 intentional updates 原文里出现，而它只做流式 RL。
3. **`Muon 的正交化方向 + 函数空间步长` 这个具体组合没有检索到。** 组合同框的论文一篇也没有。
4. 但要注意 **Pion (2605.19282)** 已经指出 Muon 在 RLVR 上会失效，原因是谱形状（尾部方向被放大），
   而不是步长幅度。这直接限制了本 idea 在 RL 侧的叙事空间——见 §3。

## 1. Intentional Updates（本方案的步长规则来源）

**Intentional Updates for Streaming Reinforcement Learning** — Sharifnassab, Elsayed, De Asis,
Mahmood, Sutton. arXiv [2604.19033](https://arxiv.org/abs/2604.19033)（2026-04，19 页）。
代码：<https://github.com/sharifnassab/Intentional_RL>

核心：先指定更新的**结果**，再解出步长。

```
给定方向 d_t 和目标标量 y(w)、期望变化 Δ_t：
    y(w_{t+1}) ≈ y(w_t) + Δ_t
    α_t = Δ_t / ⟨∇y(w_t), d_t⟩          （用 Cauchy–Schwarz 下限做保守版）
```

实例：Intentional TD（让 TD 误差缩小固定比例 η）、Intentional Q、Intentional Policy Gradient
（让采样动作的 `log π` 每步变化 `η·A/Ā`，作为局部 KL 的廉价代理）。

**与本方案的关系**：原文的 `d_t` 是梯度（带 eligibility trace 和 RMSProp 对角缩放）。
本方案把 `d_t` 换成 Muon 的正交化矩阵方向 `U_l`，并把 `y` 从 RL 的 TD 误差/log-prob
换成"每一层线性层的输出向量"。这是唯一的新意所在，因此实验必须把这件事的价值测出来
（notebook 的 C1/C3 判据）。

原文没有涉及：矩阵结构的方向、预训练、逐层分配。

## 2. Muon 的步长：已被占掉的部分

| 论文 | 做法 | 分母/尺度所在空间 | 与本文差异 |
|---|---|---|---|
| [OrScale](https://arxiv.org/abs/2605.07815)（2605.07815） | 把 LARS/LAMB 的 trust ratio 移植到正交化更新，逐层动态标量；锚定在 1 使 Moonlight 的超参可直接迁移 | **参数空间** `‖W_l‖_F / ‖ΔW_l‖_F` | 逐层、自适应，但量的是权重变化不是输出变化 |
| [TrasMuon](https://arxiv.org/abs/2602.13498)（2602.13498，ICLR'26） | 全局 RMS 校准 + 能量型 trust-region 裁剪，抑制高能爆冲 | **参数空间**（全局标量 + 相对能量比） | 不是逐层；不含函数语义 |
| [Distance-Aware Muon](https://arxiv.org/abs/2605.18999)（2605.18999） | 把 Muon 表述成非欧 trust region：方向由 LMO 定、半径由"轨迹实际走过的距离"或下降证书定 | **优化轨迹空间** | 框架完全一致（方向已定、半径待定），但半径和模型当前函数行为无关 |
| [Equal Steps, Unequal Curvature](https://arxiv.org/abs/2611.24644)（2611.24644） | 用 Hessian-vector 测量每个奇异值方向的曲率，指出 Muon 对头部高曲率方向过步；给出 σ^{-1/4} 谱倾斜 | **谱空间 / 曲率** | 改的是**谱形状**，不是标量步长 |
| [Spectral Allocation / SAMuon](https://arxiv.org/abs/2608.25990)（2608.25990） | 逐奇异方向探测 loss 最优步长，得到各向异性的稳定步长剖面 | **谱空间 / loss** | 逐方向但仍是 loss 视角，需要额外探测开销 |

其他同族但更远的（按标题检索，ID 未逐一核对，引用前请自行确认）：
Scion / 范数约束统一框架（DeepMind）、
[Constant step sizes for Muon-type spectral optimizers](https://arxiv.org/abs/2610.27747)、
[Theoretical Analysis of Muon on Transformers for Next-Token Prediction](https://arxiv.org/abs/2611.04465)、
*Tune the Baseline First: Most of Muon's Scaled Weight Decay Gain Is a Tuning Gap*、
*Sharpness-Aware Muon*、*EnergyCal: Taming Loss Spikes in Muon-Family Optimizers*。

**判读**：这一列工作的共同分母都是"参数/谱/轨迹"，没有一个是"这一层的输出变了多少"。
如果本课题的新颖性成立，它必须立在这条缝上。

## 3. RL 后训练侧：一个必须知道的负面证据

**[Rethinking Muon Beyond Pretraining: Spectral Failures and High-Pass Remedies for VLA and RLVR](https://arxiv.org/abs/2605.19282)**
（2605.19282）

- Muon 的白光化把所有奇异值推到 1，在 pretraining 上是探索优势；
- 但在 **RLVR** 里梯度信噪比低，**尾部噪声方向被同样放大**，导致不稳定；
  在 VLA 里低秩动作模块的梯度也会被尾部噪声污染。
- 提出 Pion：高通版 Newton–Schulz（Promotion + Suppression），把主方向锚在 1、把尾部压向 0。

**对本课题的含义（重要）**：如果 Muon 在 RLVR 的失效是**谱形状**问题，
那么调步长幅度**修不好它**。所以 notebook 的 Stage C 被设计成一个"能不能救"的检验：

- 若 tuned Muon 在 GSM8K 上掉点、而 Intentional-L 追平 AdamW → 失效模式里有**幅度**成分，本 idea 有位置；
- 若两者都掉点 → 与 Pion 的谱形状解释一致，本 idea 不匹配这个场景 → 干净的负结论，及早止损。

这比"跑一遍看看涨不涨点"有信息量得多。

## 4. 其他相关但不相撞的线索

- ***bf16 Rounding Silently Discards Most RL Policy Updates: A Hidden Training-Inference Mismatch
  and an Unbiased One-Line Fix***（2026，arXiv 标题检索命中，ID 未核对）：
  bf16 下小的策略更新被舍入吃掉。本方案的 intentional 步长在小 η 时恰好落在这个区间，
  所以 notebook 里同时记录 `pred_gain` 与 `realized_gain`，可以直接看到"预测要变 X、实际只变了 Y"。
- **Grokking / 学习规则选择**（*Grokking Can Be Steered* 等 2026 年工作）：与本题无关。
- **μP / small-scale proxies**：`update-to-parameter ratio` 是判断小规模结论能否外推的标准工具，
  本 notebook 的 `rel_delta_h` 是它在**函数空间**的对应物。

## 5. 未检索到（= 潜在缝隙）

- 把 Muon 方向与"层输出变化量"绑定的步长规则；
- 把 intentional updates 用于预训练；
- 逐层函数空间归一化 vs 逐层参数空间归一化（OrScale）的**直接对照实验**——
  这一条本身就可以是一个小贡献，notebook 的 `orscale` arm 就是为它设的。

## 引用格式速查

```
@article{sharifnassab2026intentional,
  title={Intentional Updates for Streaming Reinforcement Learning},
  author={Sharifnassab, Arsalan and Elsayed, Mohamed and De Asis, Kris and Mahmood, A. Rupam and Sutton, Richard S.},
  journal={arXiv preprint arXiv:2604.19033}, year={2026}}
```
