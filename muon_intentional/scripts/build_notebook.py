#!/usr/bin/env python
"""Assemble the deliverable notebook from markdown + code cells.

Keeping the notebook generated means the code in it is byte-identical to the
smoke-tested package in src/muonint/.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = "/home/15Cu2Plus/muon_intentional_pilot.ipynb"

CELLS = []


def md(s):
    CELLS.append(("markdown", s.strip("\n")))


def code(s):
    CELLS.append(("code", s.strip("\n")))


# ===========================================================================
md(r"""
# Muon 定方向 + Intentional Updates 定步长：可行性初步论证

**课程 project 立项用的 pilot 实验。** 一个 notebook 跑完，用 5 个左右 GPU 小时回答一个问题：

> 把 Muon 的正交化方向和 "先指定函数空间的变化量、再反解步长" 的 intentional 更新结合起来，
> 在 **小模型预训练** 和 **LLM 后训练 RL** 两个场景里，有没有值得继续做的迹象？

---

## 怎么跑

| 事项 | 说明 |
|---|---|
| 依赖 | 公共 `llmenv`（torch 2.14 + transformers 5.17），无需额外安装 |
| 代码位置 | `~/muon_intentional/src/muonint/`（本 notebook `import` 它）；结果写到 `~/muon_intentional/results/` |
| 断点续跑 | 每个配置跑完就写一个 JSON，**重跑会自动跳过已完成的**。Notebook 单次连接 1 小时，分几次连就行 |
| 先试跑 | 把下面 `MODE = "quick"` 改掉之前，先用 `"quick"` 完整走一遍（约 6 分钟，CPU 也能跑 Stage 0/A） |
| 设备 | 自动选 CUDA；没有 GPU 时退到 CPU（只有 Stage 0/A 适合） |

跑完直接看 **最后一节「结论面板」**，里面会按事先定好的判据给出 go / 有条件 go / no-go。
""")

md(r"""
## 1. 判据先行：这次到底在验证什么

立项阶段最大的风险不是"实验做不出来"，而是"做完了也说不清结果意味着什么"。所以先把判据写死在这里，
后面的实验只是往这套判据里填数。**这几条判据是事先登记的，不是看完结果再补的。**

| 编号 | 问题 | 判据 | 如果不成立意味着 |
|---|---|---|---|
| **C1** | 前提：固定步长的 Muon 是否让不同层的**功能**变化差异很大？ | `log s_l − 2 log a_l` 的跨层离散度 > 0.7，或 activation 与 function 排序的 Spearman < 0.5 | 前提太弱，逐层 function-space 步长没东西可修 —— **直接 no-go** |
| **C2** | 等 token 预算下，逐层 Logit-KL 是否不差于 tuned Muon / OrScale？ | 与 tuned Muon 差值 ≤ −0.5σ 为"正向"，±1σ 内为"中性"，明显更差为"负向" | 中性可接受（见 C3），负向需 C3/C4 补救 |
| **C3** | **分配**是否有用？固定 KL 目标时，逐层分配是否优于全局配额？ | `intent_kl_layer` 相对 `intent_kl_global` 的差值 ≤ −0.3σ | 若两者打平，说明"逐层"这个细化没有价值，课题退化成"用 KL 定步长"这个更弱的命题 |
| **C4** | **目标**是否有用？固定逐层时，KL 目标是否优于 activation 目标？ | `intent_kl_layer` 相对 `intent_act` 差值 ≤ −0.3σ | 若打平，说明函数空间校准没有换来实际收益，只剩理论意义 |
| **C5** | 校准后预算是否真的被满足？ | `kl_achieved_ratio` ∈ [0.9, 1.1]（Stage 0 已实测 0.995） | 偏离说明校准失效；此时两个 KL 臂的数字不可解释 |

**为什么是这几条。** 你的想法里，"adaptive step size for Muon" 本身已经不是空白（见第 2 节的文献表）；
真正可能新的是**把步长定义在函数空间**。所以判据的重心放在 C1（前提）和 C3（迁移性）上，
而不是"能不能涨点"。如果 C2 只是打平但 C3 成立，这个课题仍然成立，题目会变成
"用函数空间的步长换取更少调参"，而不是"新的 SOTA 优化器"。
""")

md(r"""
## 2. 文献：哪些已经被做了，缝在哪里

截至 2026-11，"Muon 需要更好的步长"已经是一个公开的活跃问题，2026 年至少有五篇论文正面处理它。
**必须区分的是：它们全部在参数空间里定义步长，没有人把目标定义在函数空间。**

| 工作 | 做了什么 | 和本想法重叠的部分 | 没有做的部分 |
|---|---|---|---|
| **OrScale** (arXiv 2605.07815) | 把 LARS/LAMB 的 trust ratio 搬到正交化更新上，逐层缩放 | 逐层自适应步长 | 分母是参数范数 `‖W‖/‖ΔW‖`，**不是** 输出的变化 |
| **TrasMuon** (arXiv 2602.13498) | 全局 RMS 校准 + 能量型 trust region 裁剪 | 抑制能量爆冲 | 全局标量，非逐层；仍是参数空间 |
| **Distance-Aware Muon** (arXiv 2605.18999) | 把 Muon 解释成非欧 trust region，半径由轨迹距离/下降证书定 | "方向已定、半径待定"的框架与我们完全一致 | 半径来自解的距离，跟模型当前的函数行为无关 |
| **Equal Steps, Unequal Curvature** (arXiv 2611.24644) | 按奇异值方向测曲率，指出 Muon 对高曲率的头部过步 | 说明固定步长确实"一刀切" | 修的是谱形状（σ^{-1/4} 倾斜），不是步长目标 |
| **Spectral Allocation / SAMuon** (arXiv 2608.25990) | 逐奇异方向探测 loss 最优步长 | 逐方向步长 | 同样是 loss/参数视角 |
| **Intentional Updates** (arXiv 2604.19033) | 在**函数空间**里先指定结果再解步长（NLMS 的推广），用于**流式 RL** | 步长定义本身 | 只在流式 RL 的 TD / PG 上做，没有和矩阵结构的方向结合，没有预训练 |
| **Rethinking Muon Beyond Pretraining** (arXiv 2605.19282) | 指出 Muon 在 **RLVR/VLA** 上失效：低保真梯度的尾部被白光化放大，提出高通的 Pion | Muon 在 RL 后训练会出问题 | 修的是**谱形状**；没处理步长幅度 |

**结论**：`Muon × 逐层自适应步长` 是拥挤的；`Muon × 函数空间步长` 这个具体组合没有出现在上述任何一篇里。
但这也意味着**新颖性完全押在"函数空间"这三个字上**，所以 C1/C3 必须是实验的主角——否则审稿人会说
"这不就是 OrScale 换了个分子吗"。本 notebook 的第 4 节就是围绕这一点设计的。

> 与参考文献的差别要讲清楚的一点：intentional updates 原文的公式是
> `α = Δ_target / ⟨∇y, d⟩`，其中 y 是我们关心的标量、d 是更新方向。原文用 d = 梯度（含 trace、RMS 对角缩放）。
> **本工作的全部新意在于 d 换成 Muon 的正交化矩阵方向 U**，以及在 LLM 上如何定义 y。
""")

md(r"""
## 3. 方法：两个实例化

### 3.1 先用一句话说清楚

- **Muon** 决定方向：对第 l 层的动量做正交化，得到 `U_l`（奇异值都≈1 的矩阵），更新 `ΔW_l = -α_l·U_l`。
- **Intentional** 决定 α：不直接给 α，而是先说"这一步我希望第 l 层的**输出**变化多少"，再反解 α。

### 3.2 为什么在 Muon 上这件事特别干净（这是本方案最好的一点）

线性层 `H = W X`。参数更新 `ΔW` 引起的输出变化是

```
ΔH = ΔW · X          ← 这是精确的等式，不是一阶近似（因为 W→H 是线性的）
```

代入 Muon 的方向 `ΔW_l = -α_l U_l`：

```
ΔH_l = -α_l · (X_l U_lᵀ)      ⇒     ‖ΔH_l‖_F = α_l · ‖X_l U_lᵀ‖_F
```

所以只要取

```
α_l = η · ‖H_l‖_F / ‖X_l U_lᵀ‖_F
```

第 l 层的输出就**恰好**改变 η 的相对幅度。这件事对 NLMS 在单层线性回归上是精确的；对 Muon 这个矩阵方向，
**在层这个粒度上同样是精确的**（`src/muonint/diagnostics.py::test_exactness` 验证误差 ~5e-6）。
这就是我们认为这个组合比"参数空间 trust ratio"更有内容的技术理由：它不是一个代理指标，是一个可实现的等式。

实践中 `‖X_l U_lᵀ‖_F` 用 512 个 token 的子采样估计（相对误差 ~4%），额外的矩阵乘法约占总计算量的 2–3%。

**一个必须正面回答的反驳：Muon 不是已经约束了输出变化吗？**

有人会说：Muon 的更新谱范数是 `√(max(1, m/n))`，所以 `‖ΔH_l‖ ≤ α·√(max(1,m/n))·‖X_l‖`，
这不已经是函数空间的约束了？不完全是，有三点：

1. **它是不等式，上界由 `‖X_l‖` 决定**。不同层的激活尺度可以差好几倍，而且会随训练漂移。
   固定 α 下**实际**的 `‖ΔH‖/‖H‖` 因此既不可控、也不跨层可比——这正是 Stage A 要量出来的东西。
2. **谱范数管的是"最坏方向"，不是"数据实际走的方向"**。真正发生的变化是 `X_l U_lᵀ`，
   即 U 在**数据张成的子空间**上的作用，两者差一个对齐因子 `‖XU‖/(‖X‖·‖U‖)`；
   当激活只集中在少数方向上时，这个因子可以远小于 1，上界就完全松掉了。
3. **它对 batch 不敏感**。同样的 α 在流式/小批量下造成的功能变化抖动大得多，而谱约束看不出来。

`intent_act`（逐层 activation 规则）做的正是把这三件事一起消掉：直接量**实际发生的** `X_l U_lᵀ`，而不是它的最坏情况上界。

### 3.3 两个实例化 + 两个基线

六个 arm 构成一个**消融阶梯**，每一级只改一件事：

| 记号 | 步长规则 | 这一级新增了什么 |
|---|---|---|
| `adamw` | AdamW | 参照锚点（若连它也崩，说明代码有问题） |
| `muon` | `α_l = lr · √(m/n)` | ① **基线**：经典 Muon，各层同一个 α |
| `orscale` | `α_l = lr · clip((‖W_l‖/‖ΔW_l‖)/median)` | 逐层自适应，但在**参数空间**（OrScale 风格） |
| `intent_act` | `α_l = η·‖H_l‖_F / ‖X_l U_lᵀ‖_F` | ③ 逐层自适应，量的是 **activation** 变化 |
| `intent_kl_global` | `α = sqrt(ε / s_joint)` | ② 目标换成 **KL**，但仍是**全局配额** |
| `intent_kl_layer` | `α_l = sqrt(ε/L / s_l)` | ④ **逐层 + KL**（本方案） |

两条关键对照，各自只变一个维度：

- **②vs④**：目标都是 KL，只变分配粒度（全局 → 逐层）→ 回答"逐层有没有用"（C3）
- **③vs④**：都是逐层，只变目标（activation → KL）→ 回答"KL 校准有没有用"（C4）

其中 `s_l = ½·Var_p[v_l]`，`v_l = dz/dη` 是该层单位 Muon 更新引起的 logits 变化，
由 logits 有限差分测出；`s_joint` 则把全部层**同时**扰动一次测得，天然包含交叉项。

**⚠ 一个必须先说的实测结果：逐层目标不可加。** 各层都按 `ε_l = ε/L` 调好之后，
联合更新产生的 KL 是目标的 **3.4 倍**（实测，见 Stage 0），因为所有层沿同一个下降方向移动，
logits 的变化**同向叠加**。所以两个 KL 臂都加了一次**校准**：先按敏感度定**形状**（相对分配），
再实测联合 KL，用对数空间的割线法把**总量**缩放到真正的 ε。
实测残差 < 1%（`kl_achieved_ratio ≈ 0.995`）。

这条经验本身就有价值：它说明"每层各自达标"和"整体达标"是两件事，
朴素的逐层加性假设会让实际步长偏大 3 倍以上——这可能是别人实现类似想法时踩的坑。

### 3.4 三个必须交代的实现细节

1. **哪些参数给 Muon**：只给隐藏层矩阵（q/k/v/o、gate/up/down）。embedding / lm_head / RMSNorm 给 AdamW，
   且 AdamW 的 lr 在各 arm 之间**固定**，保证比较只发生在矩阵层上。
2. **护栏**：`‖X_l U_lᵀ‖` 接近 0 时 α 会爆，用 `α ≤ 20×median(α)` 截断，并记录**被截断的比例**
   （`frac_clipped`，如果长期很高说明这个定义不稳，本身就是一个负结果）。
3. **η 是「每个矩阵」的预算，不是「每个 block」的。** 一个 block 有 7 个 Muon 矩阵
   （q/k/v/o/gate/up/down），其中 `wo` 和 `w_down` 直接写进残差流。
   `intent_act` 让**每一个**矩阵的输出都变化 η，于是 block 级的累积远大于 η。
   实测：η=0.02 稳定，η=0.05 起在一个 384×6 的模型上发散（loss → NaN）。
   所以扫参范围必须落在稳定区，且 NaN 要被显式识别而不是当成一个很差的分数。
4. **KL 臂的日程加在 ε 上**：因为 `α ∝ √ε`，把 cosine 日程乘到 ε 上，步长按 √sched 衰减。
   直接对 α 加日程会让它不再是"intentional"（目标就不再是固定 KL）。
5. **测量频率**：`s_l` 每 20 步重算一次（512-token 探测批），中间复用。
   敏感度是慢变量，这带来约 5–7% 额外算力，代价是 α_l 呈阶梯状——报告里要写明。
""")

md(r"""
## 4. 实验设计

六个阶段：Stage 0 保证实现正确，A 回答「前提成立吗」，B1/B2 回答「有用吗」。**Stage A 是省钱的关键**：它几乎不花 GPU 时间，
但如果不成立，后面全部可以不做。

| 阶段 | 回答什么 | 规模 | 估计 GPU 时长 |
|---|---|---|---|
| **0 正确性** | 公式对不对、六个 arm 能不能跑、KL 预算是否被满足 | 微型模型 / CPU | < 1 min |
| **A 前提诊断** | 固定步长的 Muon 是否已经"一刀切"？（C1） | d=256×L4, 300 步 × 3 个 batch size | ~5 min |
| **B1 扫参** | 每个 arm 的合理 lr / η / ε | d=384×L6, 15M tokens × 6 arms × 4 点 | ~37 min |
| **B2 主实验** | 等 token 下 val loss（C2/C3/C4） | d=512×L8, 100M tokens × 6 arms × 2 seeds | ~122 min |
| **KL 测量开销** | 每 20 步一次 × 512-token 探测 | 已含在 B2 内 | +6 min |
| | | **核心合计** | **≈ 2.8 GPU·h** |
| B3 迁移（可选） | 同一 ε 换形状/批量还灵吗 | d=320×L10, 60M tokens | +40 min |
| C RLVR（可选） | RL 下可用吗 | Qwen3-0.6B, GSM8K | +110 min |

**并发数：1。** 单卡串行。Notebook 每次连接上限 1 小时，所以实际会分 4–5 次连接跑完；
配置级缓存让中断不会浪费已完成的部分。如果希望缩短墙钟时间，本 notebook 末尾附了一个
**可选的 sbatch 生成器**——把 B2 的各 seed 拆成独立作业并发，墙钟可压到 ~1 小时，
但会占用多张卡（个人并发额度 48 张，完全够）。

**几个刻意的设计选择，防止结果被"工程细节"污染：**

- **KL 臂与其它臂等 token，但有约 5–7% 额外开销**（探测前向）。这个开销按 token 算是不公平的，
  所以墙钟时间单独报告；如果 KL 臂只是微弱领先而墙钟明显更慢，这个代价要算进去。
- **所有 arm 等 token**（不是等墙钟）。Muon 的 Newton–Schulz 开销**取决于 batch**：
  预训练（每步 ~16k token）时只占 3% 左右，但在 RL 这种小 batch（每步 ~10k token、
  但矩阵本身很大）下可以到 30–50%。按 token/等 rollout 比是优化器文献的通行做法，
  但墙钟时间同时记录并报告——如果 RL 侧 AdamW 明显更快而分数接近，这个差异要算进去。
- **随机源可控**：同 seed 下数据顺序、初始权重一致；RL 的采样用独立 Generator。
- **每个 arm 都同样预算地调参**（B1 里每个 arm 扫同样多的点），避免"我们的 arm 调得更狠"。
- **各 arm 扫的是各自定义里的那个量**：Muon/OrScale/Activation 扫 lr/η，KL 两臂扫 **ε**
  （因为 ε 才是"指定结果"的那一端，η 由它反解出来；调 η 就失去 intentional 的意义了）。
  代价是 ε 的量纲不直观，所以扫参会**同时打印换算出的实际 η**，用来检查搜索范围
  是否和 Muon 的 lr 区间重叠——不重叠的话，"KL 更差"可能只是范围没覆盖到。
- **2 个 seed**（开题前只做趋势判断），报 mean 与 seed 间差；2 seed 无法做显著性检验，
  所以所有结论只按事先登记的阈值读，不做统计宣称。立项后可加到 4–5 seed。
""")

md(r"""
## 5. 代码

下面开始跑。第一个 cell 只做配置和 import，**不花 GPU**。
""")

code(r'''
import os, sys, json, math, time, glob
import numpy as np
import torch

ROOT = "/home/15Cu2Plus/muon_intentional"
sys.path.insert(0, os.path.join(ROOT, "src"))

from muonint import pretrain, rl, diagnostics, optim
from muonint.pretrain import RunConfig, run as run_pretrain
from muonint.rl import RLConfig, run as run_rl
from muonint.common import load_json, save_json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| torch", torch.__version__)
if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")

# ---------------------------------------------------------------- 配置
MODE = "full"        # "quick" = 全流程小规模试跑（~6 min，用于验证 notebook 能跑通）；"full" = 正式跑

# 实验矩阵来自 muonint/expconfig.py —— 与 scripts/run_experiments.py 共用同一份定义，
# 所以 Slurm 上跑出的缓存这里能直接复用（tag 必须完全一致）。
from muonint.expconfig import (ARMS, ARM_LABEL, ARM_SHORT, SWEEP_PARAM,
                               SWEEP as SWEEP_BASE, params as exp_params,
                               paths as exp_paths, DATA as EXP_DATA)

DATA = EXP_DATA
P = exp_params(MODE)
_pp = exp_paths(MODE)
DIAG, PRE, RLD = _pp["DIAG"], _pp["PRE"], _pp["RLD"]
RES = _pp["RES"]
for d in (RES, DIAG, PRE, RLD):
    os.makedirs(d, exist_ok=True)

SWEEP = SWEEP_BASE if MODE == "full" else {k: list(v)[:2] for k, v in SWEEP_BASE.items()}

print("MODE =", MODE)
''')

md(r"""
### 5.1 数据

用集群共享的 ClimbMix（`/mnt/oss/datasets/climbmix/`，GPT-2 tokenizer 已预先分好词）。
只取前 4 亿 token 缓存成 uint16（约 800 MB，读一次约 30 秒）。
用真实预训练数据而不是合成数据，是为了让 val loss 曲线有可比性。
""")

code(r'''
from muonint.data import build_token_cache
t0 = time.time()
build_token_cache(DATA, 400_000_000)
print(f"{os.path.getsize(DATA)/1e9:.2f} GB, {time.time()-t0:.0f}s")
''')

md(r"""
### 5.2 Stage 0：先确认公式和六个 arm 是对的

三件事：(a) `intent_act` 是否真的把每层输出的相对变化精确地打成 η；
(b) KL 规则算出的步长是否被数字上验证过；(c) 六个 arm 在微型模型上是否都能跑。
这一步不成立的话，后面所有结论都没意义。
""")

code(r'''
ex = diagnostics.test_exactness()
print("A1 线性层精确性（目标 η=0.01，三种形状）：")
for k, v in ex.items():
    print(f"   {k:10s} {v:.8f}")
assert ex["max_rel_err"] < 2e-2, ex
print(f"   → 最大相对误差 {ex['max_rel_err']:.2e}  PASS\n")

from muonint.model import GPT, GPTConfig
from muonint.optim import MuonIntentional, split_muon_params

cfg = GPTConfig(vocab_size=50257, n_layer=2, n_head=2, d_model=64, seq_len=32)
# 这里直接测优化器内部的 mode（fixed=Muon, orscale, intent_l=逐层 activation）
for mode in ["fixed", "orscale", "intent_l"]:
    torch.manual_seed(0); m = GPT(cfg); m._n_sub = 16
    mu, ad = split_muon_params(m.named_parameters())
    o = MuonIntentional(mu, ad, mode=mode, lr=0.02)
    o.register_names({id(p): n for n, p in m.named_parameters()})
    x = torch.randint(0, 200, (2, 32)); y = torch.randint(0, 200, (2, 32))
    for _ in range(2):
        o.zero_grad(); _, l = m(x, y, capture=(mode == "intent_l")); l.backward()
        if mode == "intent_l":
            o.set_layer_inputs(m.layer_inputs)
        o.step()
    extra = ""
    if mode == "intent_l":
        r = torch.tensor(list(o.stats.rel_delta_h.values()))
        extra = f" | 实测 rel_dH ∈ [{r.min():.4f}, {r.max():.4f}]"
    print(f"   {mode:9s} ok  loss={l.item():.3f}{extra}")

# KL 规则：单步预算是否真的被满足？（含交叉项）
from muonint.kl_tuner import measure_step_sizes, check_joint_kl
al, kd = measure_step_sizes(m, o, x[:1], eps=1e-3, layerwise=True)
ck = check_joint_kl(m, o, x[:1], al)
print(f"   KL-layer ok  n={kd['n']}  η中位={kd['eta_med']:.4f}  "
      f"ε={1e-3:g}  联合KL={ck['kl_joint']:.2e}  交叉项比={ck['cross_term_ratio']:.2f}")
al, kd = measure_step_sizes(m, o, x[:1], eps=1e-3, layerwise=False)
print(f"   KL-global ok  η={kd['eta']:.4f}  s_joint={kd['s_joint']:.4f}")
# (c) 初始损失必须 ≈ ln(vocab)：embedding 初始化写错时这里会是 ~127 而不是 ~10.8
import numpy as np
mm = GPT(GPTConfig(vocab_size=50257, n_layer=2, n_head=2, d_model=128, seq_len=64))
x = torch.randint(0, 50257, (2, 64)); y = torch.randint(0, 50257, (2, 64))
_, l0 = mm(x, y)
print(f"\n   初始 loss = {l0.item():.3f}  （均匀基线 ln(V) = {np.log(50257):.3f}）")
assert l0.item() < 12.0, "初始 loss 过高：embedding / 初始化有问题"
print("\nStage 0 通过。")
''')

md(r"""
### 5.3 Stage A：固定步长的 Muon 到底有多"一刀切"？—— **C1**

这是最便宜、也最能决定要不要继续的一步。做法：用**经典固定步长 Muon** 训一个小模型，
每一步记录每一层实际发生的 `‖ΔH_l‖/‖H_l‖`。注意这个量跟 intentional 规则要固定的正是同一个量，
所以它直接测量了"给定固定 α 时，各层的功能变化差多少"。

要看两个维度：

- **跨层**：同一步里，不同层的比值差几倍？差得多 → 逐层设步长有空间；都差不多 → 没空间。
- **跨时间**：同一个层，比值随训练漂移多少？漂移大 → 固定 α 不是稳态的，自适应有意义。

另外跑三个 batch size，检验"batch 越小参数空间步长越不可控"这个标准直觉是否成立——
这决定了这个 idea 更适合预训练还是 RL。
""")

code(r'''
diag = diagnostics.run_layer_heterogeneity(
    DATA, batch_sizes=P["a_batches"], steps=P["a_steps"], seq_len=P["a_seq"],
    d_model=P["a_d"], n_layer=P["a_L"], n_head=P["a_H"], device=DEVICE,
    out_path=os.path.join(DIAG, "heterogeneity.json"))

print(f"{'batch':>7} {'跨层 CV':>9} {'跨层 p90 倍数':>13} {'时间 CV':>9} {'比值中位':>9}")
for k, v in diag["results"].items():
    print(f"{k:>7} {v['across_layer_cv']:9.3f} {v['across_layer_spread']['p90']:13.2f} "
          f"{v['temporal_cv_mean']:9.3f} {v['ratio_median']:9.4f}")
''')

code(r'''
import matplotlib as mpl
import matplotlib.pyplot as plt

# 分类色板（已用 dataviz 的校验脚本验证：明色 5 槽相邻对 CVD ΔE≥9.1，正常视觉 ΔE≥19.6）
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ARM_COLOR = dict(zip(ARMS, SLOT))          # 颜色跟随实体，永不按排名重排
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "figure.dpi": 120,
})

def style_axes(ax, title=None, xlabel=None, ylabel=None):
    if title: ax.set_title(title, color=INK, loc="left", pad=10)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax

print("plot style ready — 图内文字用英文：集群没有中文字体，中文会变成方框")
''')

code(r'''
# Fig A: 固定步长下，各层相对输出变化的分布（跨 batch size）
res = diag["results"]
fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 3.6), sharey=True)
if len(res) == 1: axes = [axes]
for ax, (k, v) in zip(axes, res.items()):
    m = np.array(v["mean_by_layer"]); names = [n.split(".")[-2] for n in v["layer_names"]]
    med = np.median(m)
    ax.barh(range(len(m)), m, color=SLOT[0], height=0.7)
    ax.axvline(med, color=INK2, lw=1, ls="--")
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(names, fontsize=6)
    style_axes(ax, f"{k}  (CV={v['across_layer_cv']:.2f})", "||dH_l|| / ||H_l||  (fixed step, measured)")
    ax.grid(axis="y", visible=False)
fig.suptitle("Stage A: per-layer relative output change under fixed-step Muon", x=0.01, ha="left", color=INK)
plt.tight_layout(); plt.show()

print("解读要点：如果各层的条几乎一样长（CV 很小），intentional 的逐层归一化就没有可修的东西；")
print("若差异明显且随 batch 变小而变剧烈，说明逐层函数空间步长有内容，且更偏向小 batch / RL 场景。")
''')

md(r"""
### 5.4 Stage B：小模型预训练

三件事：扫参（B1）→ 主实验（B2）→ 迁移测试（B3）。

**关于"公平调参"**：每个 arm 在 B1 里扫同样多的点，用最好的那个进 B2。
B2 用 3 个 seed。val loss 在固定的 8 个验证 batch 上算，seed 相同、数据相同。
""")

code(r'''
def make_cfg(arm, value, d, L, tokens, batch, seq, seed, tag_prefix, out_dir, adam_lr=3e-4):
    """按 arm 决定这个数是 lr 还是 kl_eps；另一个保持默认。"""
    kw = dict(name=f"{tag_prefix}_{arm}", opt_mode=arm, adam_lr=adam_lr,
              n_layer=L, n_head=max(2, L), d_model=d, seq_len=seq,
              batch_size=batch, tokens=tokens, seed=seed,
              log_every=10**9, eval_every=10**9, data_path=DATA, out_dir=out_dir)
    if SWEEP_PARAM[arm] == "kl_eps":
        kw.update(lr=0.02, kl_eps=value, kl_every=20, kl_check_every=200)
    else:
        kw.update(lr=value)
    return RunConfig(**kw)

def sweep(arms, sweep_grid, d, L, tokens, batch, seq, seeds, tag_prefix,
          out_dir=PRE, adam_lr=3e-4):
    """对每个 arm × 每个候选值跑一遍，返回 {arm: [(值, val_loss), ...]}。"""
    table = {}
    for arm in arms:
        rows = []
        for val in sweep_grid[arm]:
            vals, eta_seen = [], []
            for sd in seeds:
                c = make_cfg(arm, val, d, L, tokens, batch, seq, sd, tag_prefix, out_dir, adam_lr)
                r = run_pretrain(c, device=DEVICE, verbose=False)
                # a diverged run has no final_val_loss -- record nan, never KeyError
                if r.get("status") != "done":
                    print(f"      seed={sd}: DIVERGED at step {r.get('diverged_at_step')}")
                vals.append(float(r.get("final_val_loss", float("nan"))))
                kd = r.get("last_kl_diag") or {}
                # old cache entries predate the nominal-schedule correction and
                # would report a meaningless ~0 eta, so skip them
                if kd.get("sched_seen"):
                    for key in ("eta_med", "eta"):
                        if kd.get(key):
                            eta_seen.append(kd[key]); break
            mean_v = float(np.mean(vals))
            rows.append((val, mean_v))
            unit = "ε" if SWEEP_PARAM[arm] == "kl_eps" else "lr/η"
            tag = "  ← 发散" if not np.isfinite(mean_v) else ""
            # KL 臂的 ε 会换算成实际步长；不打印出来就无法判断它的搜索范围
            # 和 Muon 的 lr 区间是否重叠（否则"KL 更差"可能只是范围没覆盖到）。
            note = ""
            if SWEEP_PARAM[arm] == "kl_eps" and eta_seen:
                note = f"  实测η中位≈{np.mean(eta_seen):.4f}"
            print(f"   {tag_prefix} {arm:16s} {unit}={val:<9g} "
                  f"val_loss={mean_v:.4f}{tag}{note}")
        table[arm] = rows
    save_json(os.path.join(out_dir, f"{tag_prefix}_sweep.json"), table)
    return table

def best_of(table):
    """Pick the best finite value. A diverged point (nan) must never win.

    Python's min() with NaN is order-dependent: comparisons against NaN are
    always False, so if the *first* candidate diverges it is kept forever.
    """
    out = {}
    for arm, rows in table.items():
        good = [(v, l) for v, l in rows if np.isfinite(l)]
        if not good:
            raise RuntimeError(f"{arm}: 所有候选都发散了，需要调大/调小搜索范围")
        if len(good) < len(rows):
            print(f"   ! {arm}: {len(rows)-len(good)}/{len(rows)} 个候选发散，已排除")
        out[arm] = min(good, key=lambda t: t[1])[0]
    return out
''')

code(r'''
t0 = time.time()
print("B1 扫参（小模型）")
b1 = sweep(ARMS, SWEEP, P["b1_d"], P["b1_L"], P["b1_tokens"], P["b1_batch"],
           P["b1_seq"], seeds=(0,), tag_prefix="b1")
BEST = best_of(b1)
print("\n选定：", {k: f"{v:g}" for k, v in BEST.items()})
print(f"B1 用时 {(time.time()-t0)/60:.1f} min  ≈ {(time.time()-t0)/3600:.2f} GPU·h")
''')

code(r'''
# Fig B1: 扫参曲线 —— 看"最优点有多陡"，这本身回答 C3 的一部分
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for arm in ARMS:
    xs, ys = zip(*b1[arm])
    ax.plot(xs, ys, "-o", color=ARM_COLOR[arm], lw=2, ms=5, label=ARM_SHORT[arm])
ax.set_xscale("log")
style_axes(ax, "B1: hyperparameter sensitivity (steeper = more tuning-dependent)",
           "lr / eta  (log scale)", "val loss @ 15M tokens")
ax.legend(frameon=False, fontsize=8, loc="upper left")
plt.tight_layout(); plt.show()
''')

code(r'''
t0 = time.time()
print(f"B2 主实验（{len(P['b2_seeds'])} seeds）")
b2 = {}
for arm in ARMS:
    vals = []
    for sd in P["b2_seeds"]:
        c = make_cfg(arm, BEST[arm], P["b2_d"], P["b2_L"], P["b2_tokens"],
                     P["b2_batch"], P["b2_seq"], sd, "b2", PRE)
        c.log_every = 50
        c.eval_every = max(50, (P["b2_tokens"] // (P["b2_batch"] * P["b2_seq"])) // 8)
        r = run_pretrain(c, device=DEVICE, verbose=False)
        if r.get("status") != "done":
            print(f"   {arm:16s} seed={sd} DIVERGED at step {r.get('diverged_at_step')}")
        vals.append(r.get("final_val_loss", float("nan")))
    b2[arm] = vals
    print(f"   {arm:16s} {np.mean(vals):.4f} ± {np.std(vals):.4f}")
save_json(os.path.join(PRE, "b2_summary.json"), b2)
print(f"B2 用时 {(time.time()-t0)/60:.1f} min ≈ {(time.time()-t0)/3600:.2f} GPU·h")
''')

code(r'''
# Fig B2: val loss vs tokens（每个 arm 的 seed 均值；条带为 ±1 std）
def load_curves(arm, tag, seeds, out_dir=PRE):
    """按 seed 取回已完成的曲线（不重建配置，直接读 cfg）。"""
    curves, finals = [], []
    for f in glob.glob(os.path.join(out_dir, f"*{tag}_{arm}_*.json")):
        d = json.load(open(f))
        if d.get("status") != "done" or d["cfg"]["seed"] not in seeds:
            continue
        ev = [(h["tokens"], h["val_loss"]) for h in d["history"] if "val_loss" in h]
        if ev:
            curves.append(ev); finals.append(ev[-1][1])
    curves.sort(key=lambda c: c[-1][1])
    return curves, finals

fig, ax = plt.subplots(figsize=(6.8, 4.2))
for arm in ARMS:
    curves, finals = load_curves(arm, "b2", P["b2_seeds"])
    if not curves: continue
    n = min(len(c) for c in curves)
    xs = np.array([t for t, _ in curves[0][:n]]) / 1e6
    Y = np.array([[v for _, v in c[:n]] for c in curves])
    mu, sd = Y.mean(0), Y.std(0)
    ax.plot(xs, mu, "-o", color=ARM_COLOR[arm], lw=2, ms=4, label=ARM_SHORT[arm])
    if n > 1:
        ax.fill_between(xs, mu - sd, mu + sd, color=ARM_COLOR[arm], alpha=0.12, lw=0)
style_axes(ax, "B2: validation loss vs tokens (mean +- 1 std)", "training tokens (M)", "val loss")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()
''')

md(r"""
#### B3：迁移测试 —— **C3，本 idea 最重要的判据**

做法：B2 里在 `d=512/批量 32` 上调好的 lr/η，**原样**搬到 `d=320×L10 / 批量 8` 上跑；
再在目标配置上重新调一次（小规模 3 点扫参）。两者的差距之比就是"迁移惩罚"。

如果 intentional 的理论动机成立（η 是无量纲的功能空间量，和宽度、激活尺度、批量无关），
`intent_kl_layer` 的迁移惩罚应该显著小于 `muon`。注意：ε 的量级依赖词表大小和归约方式，
所以 KL 臂在"跨规模迁移"这一点上**先天弱于** activation 臂——这是它的一处真实退步，要如实报告。
""")

code(r'''
# 如果 B3 已经由 Slurm 跑过（scripts/run_experiments.py --stage b3），直接读汇总，
# 否则才在这里算。两边去重逻辑一致，所以不会重复消耗 GPU。
import json as _json
_b3p = os.path.join(PRE, "b3_summary.json")
if os.path.exists(_b3p) and json.load(open(_b3p)).get("__complete__", True):
    penalty = json.load(open(_b3p))
    print("读取 Slurm 产出的 b3_summary.json（未重新训练）")
    for a in ARMS:
        p = penalty[a]
        print(f"   {a:16s} 迁移={p['transferred']:.4f}  重调={p['retuned']:.4f}  gap={p['gap']:+.4f}")
    _SKIP_B3 = True
else:
    _SKIP_B3 = False

if not _SKIP_B3:
    t0 = time.time()
    T3 = dict(d=P["b3_d"], L=P["b3_L"], tokens=P["b3_tokens"], batch=P["b3_batch"], seq=P["b3_seq"])
    small = {a: [BEST[a]] for a in ARMS}      # 迁移：只用 B2 选出的那一个值
    transfer = sweep(ARMS, small, T3["d"], T3["L"], T3["tokens"], T3["batch"], T3["seq"],
                     seeds=P["b3_seeds"], tag_prefix="b3_transfer")

    retune_grid = {a: ([BEST[a]] + [v for v in SWEEP[a] if v != BEST[a]][:2]) for a in ARMS}
    retuned = sweep(ARMS, retune_grid, T3["d"], T3["L"], T3["tokens"], T3["batch"], T3["seq"],
                    seeds=P["b3_seeds"], tag_prefix="b3_retuned")
    BEST3 = best_of(retuned)

    penalty = {}
    for a in ARMS:
        t_loss = dict(transfer[a])[BEST[a]]
        r_loss = dict(retuned[a])[BEST3[a]]
        penalty[a] = dict(transferred=t_loss, retuned=r_loss, gap=t_loss - r_loss)
    save_json(os.path.join(PRE, "b3_summary.json"), penalty)
    print("\n迁移惩罚（越小越好）：")
    for a in ARMS:
        p = penalty[a]
        print(f"   {a:16s} 迁移={p['transferred']:.4f}  重调={p['retuned']:.4f}  gap={p['gap']:+.4f}")
    print(f"B3 用时 {(time.time()-t0)/60:.1f} min ≈ {(time.time()-t0)/3600:.2f} GPU·h")
''')

code(r'''
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.0, 3.4))
gaps = [penalty[a]["gap"] for a in ARMS]
bars = ax.bar(range(len(ARMS)), gaps, color=[ARM_COLOR[a] for a in ARMS], width=0.62)
for i, (b, a) in enumerate(zip(bars, ARMS)):
    ax.annotate(f"{gaps[i]:+.3f}", (b.get_x() + b.get_width()/2, max(gaps[i], 0)),
                ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_xticks(range(len(ARMS)))
ax.set_xticklabels([ARM_SHORT[a] for a in ARMS], fontsize=8)
ax.axhline(0, color=INK2, lw=1)
style_axes(ax, "B3: transfer penalty = loss(reused lr/eta) - loss(retuned)", None, "delta val loss (lower is better)")
ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.show()
''')

md(r"""
### 5.5 Stage C：LLM 后训练 RL（RLVR / GRPO-lite）

这里的对应关系最直接：语言模型的交叉熵本身就是 log-likelihood 的 policy gradient，
而 Intentional Policy Gradient 原文就是"让 log π 每步变化一个受控的量"。所以 RL 后训练是这个 idea 的天然主场。

**但也有一个明确的负面预警。** 文献里的 Pion（arXiv 2605.19282）已经报告 Muon 在 RLVR 上会失效，
原因是低保真梯度被白光化后**尾部方向被放大**——这是**谱形状**问题，不是步长幅度问题。
本方案修的是步长，所以如果 C4 显示 Muon 在 RLVR 上掉点、而 intentional 救不回来，
那不是实验做砸了，而是**这个 idea 和那个失效模式不匹配**——这本身就是一个干净的负结论。

**设置**：Qwen3-0.6B（集群本地），GSM8K，GRPO-lite：8 prompt × 6 采样，序列级 group 归一化优势，
单次梯度步（不做 PPO clip 多轮），reward = 最终答案精确匹配，最长 192 token。

**机制指标**（比准确率灵敏得多，是本次 RL 部分的重点）：

- `pred_gain`：优化器用一阶模型**预测**的"平均 log π 会涨多少"；
- `realized_gain`：更新后在同一批数据上**实测**的涨幅。
  `realized / pred` 接近 1 说明一阶近似成立、intentional 的步长是可信的；远离 1 说明"intentional"在自欺欺人。
""")

code(r'''
RUN_STAGE_C = (DEVICE == "cuda")     # CPU 上跳过（0.6B 太慢）
RL_MODEL_PATH = rl.MODEL_PATH        # 换成任意本地模型目录即可换模型
if RUN_STAGE_C:
    t0 = time.time()
    model0, tok0 = rl.load_model(DEVICE, RL_MODEL_PATH)
    _, test = rl.load_prompts(RLConfig(n_eval=P["c_eval"]))
    acc0 = rl.evaluate(model0, tok0, test,
                       RLConfig(n_eval=P["c_eval"], max_new_tokens=P["c_max_new"]))
    print(f"GSM8K 子集（{len(test)} 题）上 Qwen3-0.6B 的初始准确率：{acc0:.3f}")
    if acc0 < 0.10:
        print("  ⚠ 初始准确率过低：整组奖励几乎全 0 -> 优势恒为 0 -> 没有梯度。"
              "需要换更简单的任务或加部分分奖励，否则 RL 部分无效。")
    elif acc0 > 0.90:
        print("  ⚠ 初始准确率过高：整组奖励几乎全 1 -> 同样没有梯度。")
    del model0; torch.cuda.empty_cache()
    print(f"用时 {(time.time()-t0)/60:.1f} min")
else:
    acc0 = None
    print("跳过 Stage C（无 GPU）")
''')

code(r'''
# RL 阶段默认不做（见预算表）。若要做，只保留最能说明问题的三个 arm。
C_SWEEP = {
    "adamw":    [1e-5, 3e-5, 6e-5],
    "muon":     [0.005, 0.01, 0.02],
    "intent_kl_layer": [1e-5, 3e-5, 1e-4],   # 单位在 RL 里是 log-prob/KL 预算，需重新标定
}
if MODE == "quick":
    C_SWEEP = {k: v[:1] for k, v in C_SWEEP.items()}

def rl_run(arm, lr, seed, steps, tag, tokens_every=500):
    c = RLConfig(name=f"{tag}_{arm}", opt_mode=arm, lr=lr,
                 adam_lr=(lr if arm == "adamw" else 1e-5),
                 steps=steps, seed=seed, n_eval=P["c_eval"],
                 prompts_per_step=P["c_prompts"], group_size=P["c_group"],
                 max_new_tokens=P["c_max_new"], model_path=RL_MODEL_PATH,
                 eval_every=max(1, steps), log_every=max(1, steps // 5),
                 out_dir=RLD)
    return run_rl(c, device=DEVICE, verbose=False)

if RUN_STAGE_C:
    t0 = time.time()
    print("C1 扫参（短步数）")
    cbest = {}
    for arm, grid in C_SWEEP.items():
        rows = []
        for lr in grid:
            r = rl_run(arm, lr, 0, max(2, P["c_steps"] // 5), "c1")
            accs = [h["eval_acc"] for h in r["history"] if "eval_acc" in h]
            rows.append((lr, accs[-1] if accs else 0.0))
            print(f"   {arm:9s} lr/η={lr:<8g} acc={rows[-1][1]:.3f}")
        cbest[arm] = max(rows, key=lambda t: t[1])[0]
    save_json(os.path.join(RLD, "c1_sweep.json"), cbest)
    print(f"C1 用时 {(time.time()-t0)/60:.1f} min\n选定:", cbest)
else:
    cbest = {k: v[0] for k, v in C_SWEEP.items()}
''')

code(r'''
if RUN_STAGE_C:
    t0 = time.time()
    print("C2 主实验（3 seeds × 250 步）")
    for arm in C_SWEEP:
        for sd in P["c_seeds"]:
            r = rl_run(arm, cbest[arm], sd, P["c_steps"], "c2")
            accs = [h["eval_acc"] for h in r["history"] if "eval_acc" in h]
            print(f"   {arm:9s} seed={sd} final_acc={accs[-1] if accs else float('nan'):.3f}")
    print(f"C2 用时 {(time.time()-t0)/60:.1f} min ≈ {(time.time()-t0)/3600:.2f} GPU·h")
''')

code(r'''
# Fig C: RL 准确率曲线 + 机制校验（pred vs realized）
if RUN_STAGE_C:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for arm in C_SWEEP:
        runs = []
        for f in glob.glob(os.path.join(RLD, f"*c2_{arm}_*.json")):
            d = json.load(open(f))
            if d.get("status") != "done": continue
            ev = [(h["step"], h["eval_acc"]) for h in d["history"] if "eval_acc" in h]
            if ev: runs.append(ev)
        if not runs: continue
        n = min(len(r) for r in runs)
        xs = [s for s, _ in runs[0][:n]]
        Y = np.array([[a for _, a in r[:n]] for r in runs])
        ax.plot(xs, Y.mean(0), "-o", color=ARM_COLOR[arm], lw=2, ms=5, label=ARM_SHORT[arm])
        ax.fill_between(xs, Y.mean(0)-Y.std(0), Y.mean(0)+Y.std(0), color=ARM_COLOR[arm], alpha=0.12, lw=0)
    if acc0 is not None: ax.axhline(acc0, color=INK2, lw=1, ls="--")
    style_axes(ax, "C: GSM8K subset accuracy (dashed = before training)", "optimizer step", "accuracy")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for arm in C_SWEEP:
        P_, R_ = [], []
        for f in glob.glob(os.path.join(RLD, f"*c2_{arm}_*.json")):
            d = json.load(open(f))
            for h in d.get("history", []):
                if "realized_gain" in h and "pred_gain" in h:
                    P_.append(h["pred_gain"]); R_.append(h["realized_gain"])
        if P_:
            ax.scatter(P_, R_, s=14, color=ARM_COLOR[arm], label=ARM_SHORT[arm])
    lim = ax.get_xlim()
    ax.plot(lim, lim, color=INK2, lw=1, ls="--")
    style_axes(ax, "C: predicted vs realised gain (on diagonal = trustworthy)", "predicted dlog-pi", "realised dlog-pi")
    ax.legend(frameon=False, fontsize=7)
    plt.tight_layout(); plt.show()
''')

md(r"""
## 6. 结论面板

把上面所有数字按事先登记的判据过一遍。
""")

code(r'''
report = {}

# C1 -------------------------------------------------------------------
a = diag["results"]
c1_cv   = np.mean([v["across_layer_cv"] for v in a.values()])
c1_p90  = np.mean([v["across_layer_spread"]["p90"] for v in a.values()])
c1_tcv  = np.mean([v["temporal_cv_mean"] for v in a.values()])
c1 = (c1_cv >= 0.3 and c1_p90 >= 2.5) or c1_tcv >= 0.15
report["C1 前提（固定步长是否一刀切）"] = dict(
    value=f"跨层CV={c1_cv:.3f}, p90倍数={c1_p90:.2f}, 时间CV={c1_tcv:.3f}",
    verdict="PASS" if c1 else "FAIL")

# C2 -------------------------------------------------------------------
if "b2" in dir():
    mu_f = np.mean(b2["muon"]); sd_f = np.std(b2["muon"])
    rows = {arm: np.mean(b2[arm]) - mu_f for arm in ARMS}
    report["C2 预训练 val loss（相对 tuned Muon）"] = dict(
        value=", ".join(f"{k}:{v:+.4f}" for k, v in rows.items()),
        verdict=("正向" if rows["intent_kl_layer"] <= -0.5*sd_f else
                 "中性" if abs(rows["intent_kl_layer"]) <= 1.0*sd_f else "负向"))

    sd_glob = np.std(b2["intent_kl_global"])
    d_alloc = np.mean(b2["intent_kl_layer"]) - np.mean(b2["intent_kl_global"])
    report["C3 分配是否有用（逐层 vs 全局，同一 KL 目标）"] = dict(
        value=f"{d_alloc:+.4f}（层={np.mean(b2['intent_kl_layer']):.4f}, "
              f"全局={np.mean(b2['intent_kl_global']):.4f}）",
        verdict="PASS" if d_alloc <= -0.3 * max(sd_glob, 1e-6) else "FAIL")

    d_target = np.mean(b2["intent_kl_layer"]) - np.mean(b2["intent_act"])
    sd_act = np.std(b2["intent_act"])
    report["C4 目标是否有用（KL vs activation，同一逐层粒度）"] = dict(
        value=f"{d_target:+.4f}（KL={np.mean(b2['intent_kl_layer']):.4f}, "
              f"act={np.mean(b2['intent_act']):.4f}）",
        verdict="PASS" if d_target <= -0.3 * max(sd_act, 1e-6) else "FAIL")

    # C3 ---------------------------------------------------------------
    g_fixed = penalty["muon"]["gap"]; g_l = penalty["intent_kl_layer"]["gap"]
    if abs(g_fixed) < 1e-6:
        report["C6 迁移惩罚比（KL-layer / Muon）"] = dict(
            value=f"分母≈0，无法计算（Muon gap={g_fixed:+.5f}）；通常意味着该 arm 在两种设定下"
                  f"选到了同一个超参——需要更宽的 retune 网格",
            verdict="N/A")
    else:
        ratio = g_l / g_fixed
        report["C6 迁移惩罚比（KL-layer / Muon）"] = dict(
            value=f"{ratio:.2f}（Muon gap={g_fixed:+.4f}, KL-layer gap={g_l:+.4f}）",
            verdict="PASS" if ratio < 0.5 else "FAIL")

# C4 -------------------------------------------------------------------
if RUN_STAGE_C:
    ratios = []
    for f in glob.glob(os.path.join(RLD, "*c2_*.json")):
        d = json.load(open(f))
        for h in d.get("history", []):
            if "realized_gain" in h and h.get("pred_gain"):
                ratios.append(h["realized_gain"] / h["pred_gain"])
    if not ratios:
        report["C7a 一阶近似可信度（realized/pred 中位数）"] = dict(
            value="没有非零梯度步（多半是整组奖励全同 -> 优势为 0），无法评估", verdict="N/A")
    else:
        med = float(np.median(ratios))
        report["C7a 一阶近似可信度（realized/pred 中位数）"] = dict(
            value=f"{med:.2f}", verdict="PASS" if 0.5 <= med <= 1.5 else "FAIL")
    accs = {}
    for arm in C_SWEEP:
        v = []
        for f in glob.glob(os.path.join(RLD, f"*c2_{arm}_*.json")):
            d = json.load(open(f))
            e = [h["eval_acc"] for h in d.get("history", []) if "eval_acc" in h]
            if e: v.append(e[-1])
        accs[arm] = float(np.mean(v)) if v else float("nan")
    report["C7b RL 最终准确率"] = dict(
        value=", ".join(f"{k}:{v:.3f}" for k, v in accs.items()),
        verdict=("PASS" if accs.get("intent_kl_layer", 0) >= accs["adamw"] - 0.03 else "FAIL"))

print("=" * 78)
for k, v in report.items():
    print(f"{v['verdict']:>5}  {k}\n        {v['value']}")
print("=" * 78)

n_pass = sum(1 for v in report.values() if v["verdict"] in ("PASS", "正向"))
if MODE == "quick":
    print("\n[注意] 当前 MODE='quick'：步数/模型都缩到最小，上面的判据只是占位演示，"
          "**不要据此下任何结论**。把 MODE 改回 'full' 再跑一次。")
elif not c1:
    print("→ NO-GO：前提不成立。固定步长的 Muon 已经把各层的功能变化压得比较齐，"
          "逐层函数空间步长没有抓手。建议换题。")
else:
    c2 = report["C2 预训练 val loss（相对 tuned Muon）"]["verdict"]
    c3 = report.get("C3 分配是否有用（逐层 vs 全局，同一 KL 目标）", {}).get("verdict")
    c4 = report.get("C4 目标是否有用（KL vs activation，同一逐层粒度）", {}).get("verdict")
    c6 = report.get("C6 迁移惩罚比（KL-layer / Muon）", {}).get("verdict")

    if c2 == "负向" and c3 != "PASS" and c4 != "PASS":
        print("→ NO-GO：KL 规则既不涨点，拆分后也看不出是「分配」还是「目标」在起作用。")
        print("  前提（C1）成立但落地无效，建议把课题转向理论分析或换方向。")
    elif c3 == "PASS" and c4 == "PASS":
        print("→ GO（最强）：逐层分配和 KL 目标都比各自的控制组好。")
        print("  课题成立，主线 = Layer-wise Logit-KL Intentional Muon。")
    elif c4 == "PASS":
        print("→ 有条件 GO：KL 目标本身有价值，但「逐层」相对「全局」没有增量。")
        print("  建议把课题收缩成「全局 Logit-KL 步长」，故事更小但成立。")
    elif c3 == "PASS":
        print("→ 有条件 GO：逐层分配有价值，但未必非得是 KL 目标。")
        print("  这时要回去看 C1——如果 activation 与 function 排序确实无关，")
        print("  那「逐层」的价值可能来自别的机制，需要补实验定位。")
    else:
        print("→ 中性：结论不足以支持立项。若 C6（迁移性）成立，可把主线改成")
        print("  「换更少调参」；否则建议把这个 topic 降级为课程作业而非研究项目。")
''')

md(r"""
## 6.5 如果 C2 是负向，最可能是这三个原因（以及对应的改法）

写在这里是为了防止"结果不好看就临时编一个解释"。三条都在实验设计里可以预见：

1. **`‖H_l‖` 被公共大分量主导。** 如果某层激活里有一个跨 token 共享的巨大偏移，
   `‖H_l‖` 主要由它决定，`η‖H_l‖` 就会远大于"真正有用方向"上该走的距离，
   于是早期步长偏大、后期偏小。**改法**：把目标改成 `‖ΔH_l‖ = η‖H_l − mean_token(H_l)‖`，
   或者只对残差分支的增量做归一化。这是一个一行代码的改动，值得作为第二轮。
2. **超参是从 15M token 迁到 150M token 的。** 小预算下的最优 lr 在大预算下未必最优，
   这对所有 arm 一样，但 intentional 的 η 语义（每步相对变化）可能对预算更敏感。
   **改法**：在 B2 的规模上对每个 arm 再做一次 2 点局部扫参。
3. **逐层等相对变化本身可能就是错的目标。** Equal Steps/Unequal Curvature 那篇的结论是
   不同谱方向的曲率差很大；同理，不同*层*该走多远也可能差很多。把 η 强行拉平
   可能是在给某些层喂过大的步长。**改法**：回到 `intent_kl_global` 那样的全局配额，
   或者让 ε 由一个可信的全局量（如 loss 的相对下降）来定，而不是逐层拉平。

如果这三条都不奏效、而 C1 又成立，那说明问题不在实现而在想法本身——这也是一个不错的结题理由。
""")

md(r"""
## 7. 附录：用 Slurm 跑（不受 notebook 1 小时限制）

Notebook 单次连接上限 1 小时、30 分钟后可能被抢占，**正式实验请用 Slurm**。
项目里已经准备好脚本（与 notebook 共用同一份配置，结果互通）：

```bash
cd ~/muon_intentional
sbatch --exclude=llm-node02 slurm/b2.sbatch   # 主实验，~2.1 GPU·h
sbatch --exclude=llm-node02 slurm/b3.sbatch   # 迁移测试，~2.0 GPU·h
squeue -u $USER
tail -f slurm-<JobID>.out
```

`--exclude=llm-node02` 是必须的：该节点上 home 挂载异常，作业会以 exit 126 秒失败。
全部跑完后回到本 notebook，B2/B3 单元会命中缓存、几秒完成。

断点续跑：每个配置独立落盘，作业被抢占或超时后重新 `sbatch` 会跳过已完成的部分。
""")

# ===========================================================================
nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for kind, src in CELLS:
    if kind == "markdown":
        nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})
    else:
        nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {},
                            "outputs": [], "source": src.splitlines(keepends=True)})

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"wrote {OUT}: {len(nb['cells'])} cells")
