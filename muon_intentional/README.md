# Muon × Intentional Updates —— 立项 pilot

用 Muon 的正交化方向配合 intentional updates 的函数空间步长，在**小模型预训练**与**LLM 后训练 RL**
两个场景做一次廉价、可判读的可行性检验。

**交付物**：

| notebook | 作用 | GPU 时长 |
|---|---|---|
| `~/muon_intentional_pilot.ipynb` | 主 pilot：六个 arm 的消融阶梯（Muon / OrScale / Activation / 全局KL / 逐层KL），带事先登记的判据 | ≈2.8 h |
| `~/muon_intentional_sensitivity.ipynb` | 逐层 activation sensitivity vs function (logit-KL) sensitivity | ≈4 min |

本目录是它们 `import` 的实现、文档和结果。本分支内的 notebook 副本在 `notebooks/`；
notebook 里写死了 `ROOT = "/home/15Cu2Plus/muon_intentional"`，换机器运行时需改成自己的路径。

## 一句话方法

线性层上 `ΔH = ΔW·X` 是**精确**等式（不是一个近似）。代入 Muon 的方向 `ΔW_l = -α_l U_l`：

```
‖ΔH_l‖_F = α_l · ‖X_l U_lᵀ‖_F     ⇒     α_l = η · ‖H_l‖_F / ‖X_l U_lᵀ‖_F
```

于是"让第 l 层的输出变化 η 的相对幅度"这个目标可以被**精确实现**，而不是靠代理指标逼近。
这既是本方案相对 OrScale / TrasMuon / Distance-Aware Muon（都在参数空间定步长）的技术差异，
也是 `diagnostics.test_exactness()` 验证过的性质（相对误差 ~5e-6）。

## 目录

```
muon_intentional/
├── src/muonint/            # 实现（notebook 直接 import）
│   ├── optim.py            # Muon / OrScale / Intentional-Act / 外部步长 / AdamW
│   ├── kl_tuner.py         # 逐层 Logit-KL 步长 + 联合 KL 校准（割线法）
│   ├── expconfig.py        # 实验矩阵的唯一来源（notebook 与 CLI 共用）
│   ├── model.py            # 小型 GPT（可上报每层输入/输出范数）
│   ├── pretrain.py         # 预训练循环 + 结果缓存
│   ├── rl.py               # GRPO-lite / RLVR（Qwen3-0.6B + GSM8K）
│   ├── diagnostics.py      # Stage A：精确性 + 逐层异质性
│   ├── sensitivity.py      # 逐层 activation-vs-function 敏感性（float64 探测）
│   ├── data.py             # ClimbMix 分片缓存、GSM8K 读取
│   └── common.py           # 设备自适应 autocast 等
├── slurm/                  # sbatch 作业（b1/b2/b3/all）
├── scripts/
│   ├── run_experiments.py  # 无头驱动（Slurm 用），与 notebook 共用 expconfig
│   ├── smoke_test.py       # CPU 端到端冒烟测试（全部 arm）
│   ├── build_notebook.py   # 由 markdown + 代码生成交付 notebook
│   ├── build_sensitivity_notebook.py   # 敏感性 notebook 的生成脚本
│   └── run_notebook.py     # 逐 cell 执行 notebook 的测试外壳
├── docs/
│   ├── literature_review.md   # 已有工作与缝隙（含链接）
│   └── experiment_plan.md     # 预算、并发、判据
├── notebooks/              # 两个交付 notebook 的副本（原文件在 ~/）
├── data/                   # GSM8K 与 ClimbMix token 缓存（自动生成，不入库）
└── results/                # 每个配置一个 JSON，可断点续跑
```

## 运行方式

### 方式 A：Slurm（推荐，不受 notebook 1 小时限制）

```bash
cd ~/muon_intentional
sbatch slurm/b1.sbatch      # 扫参，~45 min
sbatch slurm/b2.sbatch      # 主实验，~2 h（依赖 b1）
sbatch slurm/b3.sbatch      # 迁移测试，~40 min
sbatch slurm/all.sbatch     # 或一次跑完全部，~3 h
squeue -u $USER             # 查看
tail -f slurm/out-*-mi-b1.log
```

每个配置跑完即落盘，作业被抢占或超时后**重新 sbatch 会从断点继续**。
全部跑完后打开 notebook，分析单元会命中缓存、瞬间完成。

### 方式 B：notebook

Notebook 单次连接上限 1 小时，只适合 quick 模式试跑或看结果；
正式配置请用方式 A，否则会在 30 分钟后被抢占。

```bash
source /etc/profile.d/llm-conda.sh && conda activate llmenv
# 冒烟测试（CPU，约 1 分钟）
OMP_NUM_THREADS=4 python -I scripts/smoke_test.py
# 重建 notebook
python -I scripts/build_notebook.py
```

然后在 Jupyter 中打开 `~/muon_intentional_pilot.ipynb`：先把 `MODE` 设为 `"quick"` 跑一遍验证，
再改回 `"full"`。全流程约 4.5 GPU·h（单卡串行，可分多次连接）。

## 状态

- [x] 实现 + CPU 冒烟测试通过（5 个 arm 在预训练与 RL 两条链路上都能跑）
- [x] 线性层精确性验证通过（相对误差 5e-6）
- [x] Intentional-L 实测逐层 `‖ΔH‖/‖H‖` 与目标 η 一致（预训练与 RL 两条链路）
- [x] notebook 全部 35 个 cell 端到端跑通（quick 模式，CPU，约 3 分钟；
      Stage C 用 `/tmp` 下的小 Qwen 验证过胶水代码，未在大模型上跑）
- [x] 初始 loss 断言 = 10.86 ≈ ln(50257)，确认初始化正确
- [ ] Stage A–C 在 GPU 上的正式结果（需你手动运行 notebook，`MODE="full"`）

## 几个已知的坑（写在这里省你时间）

- **图里的文字是英文的**：集群环境没有 CJK 字体，matplotlib 画中文会变成方框。
  notebook 的 markdown 是中文，图内标签是英文。
- **quick 与 full 的结果目录是分开的**（`results/<stage>/<MODE>/`），不会串。
- **`RUN_STAGE_C` 需要 GPU**；没有 GPU 时该阶段自动跳过，其余照常。
- **GSM8K 初始准确率要在 0.1–0.9 之间**：如果 Qwen3-0.6B 几乎全对或全错，
  整组奖励相同 → 优势为 0 → 根本没有梯度，RL 部分会看起来"什么都没发生"。
  notebook 会在 (5.5) 打印出来并给出警告。
