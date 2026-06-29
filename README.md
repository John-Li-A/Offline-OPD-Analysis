# Offline On-Policy Distillation under Multi-Turn Agentic Distribution Shift

A lightweight, **exact (sampling-free)** study of when and why offline on-policy
distillation (Lightning OPD, arXiv 2604.13010) breaks in a multi-turn agentic
setting — and a theory-driven patch that fixes it cheaply.

Everything runs on CPU in pure NumPy. The information-state process is small and
fully enumerable, so state visitation, policy gradients, the χ² divergence, and
the Theorem 3.5 bound are all computed **in closed form** — no Monte Carlo, no
GPU, no LLM. That is the whole point: we can read the offline/online gradient gap
off exactly instead of arguing about estimator variance.

> 中文版在文末 (Chinese version below).

## TL;DR — the four findings

1. **Offline OPD collapses under multi-turn shift; online recovers.** On a
   retrieval-QA POMDP with an emergent pollution trap, offline OPD barely lifts
   off the SFT floor (0.42) while online OPD/RL recover to the teacher ceiling
   (0.96–1.00).
2. **Generalisation is the *accomplice*, not the rescuer.** The linear
   (generalising, ≈ real-model) student collapses *harder* than the tabular one
   (0.42 < 0.58), because it extrapolates the on-path "commit the evidence peak"
   behaviour straight into the polluted trap states. The competitor's report has
   the opposite ordering (tabular fails, generalisation rescues linear).
3. **The failure is a smooth spectrum, not a caricature.** Sweeping the pollution
   rate traces a continuous curve: as the off-support ratio rises 0.37 → 0.82 the
   offline/online gap widens 0.13 → 0.68, with `off_support_ratio` as the
   controlled x-axis. With *no* shift, offline = online (Theorem 3.6).
4. **The Theorem 3.5 bound is ~10× vacuous**, and worse: the measured gradient
   gap grows while success stays flat — offline OPD is *successfully optimising
   the wrong objective*, because the feature that would trigger recovery is
   structurally starved of gradient.

Patch (Q-vii): a **χ²-staleness-triggered dataset refresh** reaches the online
ceiling at **~3 % of the environment-access cost** (a handful of refreshes vs one
per step) and is **robust across learning rates** where fixed-period refresh
explodes.

## The task in one paragraph

A hidden answer is drawn from K candidates. An agent issues *retrieve* actions
against M > K noisy sources, then *commits* an answer. With probability ρ(noise)
the episode is **polluted**: a wrong answer ã is silently substituted and governs
*all* reads, so the agent accumulates a clean, confident evidence signature for
the wrong answer — an early derailment that compounds over every turn. The agent
observes only a *polluted* phase bit (never which wrong answer). Clean collection
(noise ≈ 0) is essentially never polluted, so offline data never exercises the
polluted bit; deployment noise makes pollution common. The only reliable recovery
is a costly `reconcile` action (an authoritative re-read). This is the
distribution shift the study turns on — emergent from the noise gap, not a
hand-wired secret action.

## Results

### Q-iv: four methods × four metrics (teacher@deploy = 0.973)

| method | linear success | tabular success |
|---|---|---|
| SFT | 0.417 | 0.388 |
| **offline OPD** | **0.420** | **0.576** |
| online OPD | 0.960 | 0.789 |
| online RL | 1.000 | 0.828 |

![baseline](results/fig1_anatomy.png)

### Q-v: when does offline fail? (coverage spectrum)

![coverage](results/fig2_coverage.png)

### Gradient layer: the ~10× vacuous bound

![bound](results/fig3_bound.png)

### Q-vii: χ²-triggered refresh — online performance at ~3 % env cost

![patch](results/fig4_patch.png)

The honest ablation story (verified by a learning-rate sweep, not just the
headline run): χ²-refresh is **not** simply "more accurate than a periodic
schedule." At a small learning rate periodic works fine too. The defensible
claims are (1) the **Pareto win** — χ² hits the online ceiling at ~3 % env cost
at *every* learning rate; (2) **robustness** — χ² bounds the student↔data drift
by construction (it fires when χ² crosses a threshold and resets it, capping the
importance weights), so it never blows up, whereas fixed-period refresh lets
drift accumulate unchecked and explodes to the 1/K floor at larger learning
rates; (3) χ² **self-tunes** its refresh budget with the learning rate.

## Repository structure

```
opd_toy/
  env.py        RetrievalQAEnv: enumerable POMDP, undirected-spoof pollution trap
  policies.py   exact Boltzmann teacher; linear-softmax & tabular students; features
  exact.py      sampling-free occupancy, OPD gradient, χ², Theorem 3.5 bound terms
  methods.py    SFT / offline OPD / online OPD / online RL / χ²-refresh patch
experiments (each is a standalone, kept script):
  baseline_table.py   Q-iv: 4 methods × 4 metrics, both students
  coverage_sweep.py   Q-v: success & off-support vs pollution rate
  patch_ablation.py   Q-vii: χ² vs periodic vs random placement
  patch_lr_sweep.py   Q-vii control: robustness across learning rates
  minimal_collision_test.py   isolated proof of the feature-collision mechanism
  gen_data.py         caches all figure arrays to results/figdata.npz
  plots.py            draws the four research-grade figures from the cache
```

## Reproduce

```bash
pip install numpy matplotlib
python baseline_table.py     # Q-iv table
python coverage_sweep.py     # Q-v spectrum
python patch_ablation.py     # Q-vii ablation
python patch_lr_sweep.py     # Q-vii robustness control
python gen_data.py           # cache figure data -> results/figdata.npz
python plots.py              # draw four figures from the cache -> results/
```

Pure NumPy, CPU, runs in minutes. Exact gradients from a fixed init are
deterministic, so single-seed numbers are reproducible to floating point.

## Key design choices (and why)

- **Exact, not Monte Carlo.** The information-state graph is a DAG under a
  potential `φ = depth + (M+1)·[verified]`, so the teacher is one backward pass
  and every occupancy/χ²/σ quantity is one forward pass. We can state the
  offline gradient coefficient *is* zero, not "is high-variance."
- **Undirected spoof + observable pollution bit.** Makes a polluted state
  *feature-identical* to a clean confident state for the wrong answer, so a
  generalising student is actively misled — the mechanism that makes even the
  linear student collapse.
- **`reconcile_cost` knob.** Tuned (0.8) so the teacher never reconciles on the
  clean path (keeping it off the reference demonstrations, so the polluted-bit
  weight stays untrained offline) yet reconciles when polluted.

---

# 离线 On-Policy 蒸馏在多轮 Agentic 分布漂移下的失效（中文版）

一个轻量、**精确（无采样）** 的研究：离线 on-policy 蒸馏（Lightning OPD，
arXiv 2604.13010）在多轮 agentic 场景下**何时、为何**失效，以及一个理论驱动、
代价极低的修补方案。

全部在 CPU 上用纯 NumPy 运行。信息状态过程小且可完全枚举，所以状态访问分布、
策略梯度、χ² 散度、Theorem 3.5 的 bound 全部**闭式计算**——无 Monte Carlo、
无 GPU、无 LLM。这正是关键：我们能把离线/在线梯度差**精确读出来**，而不是争论
估计量的方差大小。

## 一句话结论 —— 四个发现

1. **离线 OPD 在多轮漂移下崩溃，在线恢复。** 在带涌现污染陷阱的检索 QA POMDP 上，
   离线 OPD 几乎贴着 SFT 地板（0.42），而在线 OPD/RL 恢复到 teacher 天花板
   （0.96–1.00）。
2. **泛化是「帮凶」而非「救星」。** 线性（会泛化、≈真实模型）学生崩得**比**表格学生
   **更狠**（0.42 < 0.58），因为它把在轨学到的「commit 证据峰」行为直接外推进污染
   陷阱态。对方报告的 ordering 恰好相反（表格崩、泛化救回线性）。
3. **失效是连续谱，不是 caricature。** 扫描污染率得到一条连续曲线：off-support 比例
   从 0.37 升到 0.82 时，离线/在线 gap 从 0.13 拉大到 0.68，off_support_ratio 是受控
   自变量。**无**漂移时离线 = 在线（Theorem 3.6）。
4. **Theorem 3.5 的 bound 约 10× 空泛**，更糟的是：实测梯度差在涨而成功率钉死不动
   ——离线 OPD 在「成功地优化错误目标」，因为能触发恢复的那个特征结构性地拿不到梯度。

修补（Q-vii）：**χ²-过期触发的数据集刷新**，用约 **3% 的环境访问代价**（数次刷新 vs
每步刷新）达到在线天花板，且在固定周期刷新会爆炸的学习率范围内**保持稳健**。

## 任务一段话说明

从 K 个候选里抽一个隐藏答案。智能体对 M > K 个含噪源执行 *retrieve*，然后 *commit*
一个答案。以概率 ρ(noise) 该回合被**污染**：一个错答 ã 被悄悄替换并主导**所有**读取，
于是智能体攒出一份对错答自信的干净证据签名——早期脱轨沿每一轮累积。智能体只观测到
一个*污染相位位*（不知是哪个错答）。干净采集（noise≈0）几乎从不污染，所以离线数据从不
训练污染位；部署噪声让污染常见。唯一可靠的恢复是代价较高的 `reconcile`（权威重读）。
这就是本研究依托的分布漂移——从噪声鸿沟**涌现**，而非手焊的秘密动作。

## 结果

### Q-iv：四方法 × 四指标（teacher@deploy = 0.973）

| 方法 | 线性成功率 | 表格成功率 |
|---|---|---|
| SFT | 0.417 | 0.388 |
| **离线 OPD** | **0.420** | **0.576** |
| 在线 OPD | 0.960 | 0.789 |
| 在线 RL | 1.000 | 0.828 |

见 `results/fig1_anatomy.png`、`fig2_coverage.png`、`fig3_bound.png`、`fig4_patch.png`。

## 关键设计选择（及原因）

- **精确而非 Monte Carlo。** 信息状态图在势函数 `φ = depth + (M+1)·[verified]` 下是
  DAG，所以 teacher 一次反向扫描、所有 occupancy/χ²/σ 量一次前向扫描即可精确算出。
  我们能断言离线梯度系数**就是**零，而非「高方差」。
- **不定向 spoof + 可观测污染位。** 让污染态与「对错答自信的干净态」**特征全等**，
  从而主动误导会泛化的学生——这是让线性学生也崩的机制。
- **`reconcile_cost` 旋钮。** 调到 0.8，使 teacher 在干净路径上从不 reconcile（从而它
  不出现在参考示范里，离线时污染位权重保持未训练），但在污染态会 reconcile。

## 复现

```bash
pip install numpy matplotlib
python baseline_table.py     # Q-iv 表
python coverage_sweep.py     # Q-v 谱
python patch_ablation.py     # Q-vii 消融
python patch_lr_sweep.py     # Q-vii 鲁棒性对照
python gen_data.py           # 缓存图数据 -> results/figdata.npz
python plots.py              # 从缓存画四张图 -> results/
```

纯 NumPy、CPU、分钟级。固定初值的精确梯度是确定性的，单 seed 数值可复现到浮点精度。
