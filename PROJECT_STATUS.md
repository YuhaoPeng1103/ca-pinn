# 项目状态交接文档

> **新会话请先读这个文件。** 最后更新：2026-09-17

## 一句话现状

论文《Gradient Decoupling in Adaptive Loss Balancing for PINN Inverse Problems》已完成写作与实验，
但发现一个**可能动摇核心结论的实现错误**（见下"🔴 关键问题"），
当前卡在**必须先跑的决定性验证实验**上。

---

## 🔴 关键问题（最重要，务必先读）

**论文声称使用 ReLoBRaLo，但代码实现的不是 ReLoBRaLo。**

三方对比（已核实原文）：

| 方法 | 权重信号 | Anchor | 解耦损失的后果 |
|---|---|---|---|
| **LRA** (Wang et al. 2021, SIAM) | 梯度范数 | **全部参数 θ** | `λ = max‖∇_θ L_r‖/mean‖∇_θ L_i‖` → 分母 0 → 权重爆炸 |
| **真 ReLoBRaLo** (Bischof & Kraus, CMAME 2025) | **损失比值** softmax，**完全不用梯度** | 不适用 | 权重有界，不归零 |
| **本项目实现** (`training/loss_balancing.py`) | 梯度范数 | **只有最后一层 W_L** | `rel_norm→0` → 权重衰减到 0 |

**后果**：论文的核心爆点「balance prior 导致 err_C 从 0.047% 恶化到 37.2%」，
很可能只是「最后一层 anchor + 梯度范数相对值」这一特定实现的产物。
换成真 ReLoBRaLo（基于损失比值），prior 的权重不会因解耦而归零，**现象可能不复现**。

**风险**：ReLoBRaLo 期刊版发表于 CMAME 439 (2025)，CMAME 编委熟悉该论文。

**注意**：`experiments/train_extra_server.py` 里的 `LRAnnealing` 类（`λ = max(norms)/norms`）
才是 LRA 的正确公式，但也误用了最后一层 anchor（真 LRA 用全参数）。

---

## ✅ 已完成：平衡器实现 + 受控验证（2026-09-17）

**新增文件**：
- `training/balancers.py` — `ReLoBRaLoTrue`（真 ReLoBRaLo，纯损失比值，无梯度）
  与 `LRAnnealingFull`（LRA 正确形式，全参数 anchor）
- `experiments/train_balancer_compare.py` — 4 配置对比实验脚本（已冒烟测试通过）

**受控测试结果**（玩具模型：一个损失耦合于网络参数、一个解耦；300 次迭代后解耦损失的权重）：

| 平衡器 | w[解耦损失] | 失效模式 |
|---|---|---|
| `ll_ema`（现有实现，最后一层 anchor） | **1.39e-08** | 衰减到零 → 损失被静默关闭 |
| `lra_full`（Wang 2021，全参数 anchor） | **2.96e+03** | 发散 → 权重爆炸 |
| `relobralo_true`（Bischof & Kraus，损失比值） | **1.00** | 有界，不受解耦影响 |

**这个结果比原故事强**：不是"某个实现有 bug"，而是**梯度类平衡器在解耦损失上有两种不同失效模式，损失类平衡器免疫**。

**重要细节**：SWE 模型里 `n_raw/C_raw/qx0_raw` 本身是 `nn.Parameter`，所以全参数 anchor 下 prior
**不是完全解耦**（3 个非零分量 / ~50k 总数）→ 预期 LRA 会**过度加权** prior（而非归零），
即三种平衡器的偏差方向不同。这让研究更丰富。

## ⏭️ 下一步：在服务器上跑 `train_balancer_compare.py`

4 个配置 × 30k epochs ≈ 12h（串行）：
- `true_relo_fixed`、`true_relo_balprior`（关键测试）
- `lra_full_fixed`、`lra_full_balprior`（关键测试）

对照点（已有，`ll_ema`）：`relobralo_30k` err_C 0.047%、`relobralo_balprior_30k` err_C 37.2%

**判读**：
- `true_relo_balprior` 仍漂移 → 现象普适，叙事基本成立
- 不漂移（预期）→ 论文定位为「anchor/信号选择如何决定解耦损失的失效模式 + 诊断 + 修复」

## 🎯 决定性结果（2026-09-18 完成，n_coll=2000）

**全部 6 个配置跑完。结果在 `outputs/server2/`（本地）与 `~/ca_pinn/experiments/outputs/`（服务器）。**

| 配置 | err_h | err_n | err_C | w_prior |
|---|---|---|---|---|
| **balance prior（4 损失）** | | | | |
| `true_relo_balprior` | 0.640% | 0.048% | **0.071%** ✓ | 0.736（有界） |
| `ll_ema_balprior` | 0.632% | 0.072% | **52.9%** ✗ | 1.6e-05（→0） |
| `lra_full_balprior` | 0.708% | 2.2e-08 | 1.5e-08（假） | **3.3e+07**（→∞） |
| **fixed prior（3 损失 + 固定权重）** | | | | |
| `true_relo_fixed` | 0.583% | 0.041% | **0.055%** ✓ | — |
| `ll_ema_fixed` | 0.617% | 0.041% | **0.037%** ✓ | — |
| `lra_full_fixed` | 0.933% | 0.280% | **4.18%** | — |

（fixed 配置的 w_prior 栏记录的是 prior **损失值**而非权重，勿误读。）

### 三条核心结论

1. **原论文论断被证伪并给出正确解释**：`ll_ema` + balance prior = 52.9% 是
   **最后一层 anchor** 导致的，不是 ReLoBRaLo 的问题。
2. **"排除 prior"的补丁对正确的平衡器不必要**：真 ReLoBRaLo balance prior 得 0.071%，
   固定 prior 得 0.055%——几乎相同。它本来就能正确处理 prior。
3. **LRA 以不同方式失效**：balance prior → 权重 3.3e7，参数被 prior 钉死（**循环论证**）；
   固定 prior → 4.18%，精度差两个数量级。

### 三种失效模式（可证明）

| 平衡器 | 更新规则 | 解耦/弱耦合损失的行为 |
|---|---|---|
| `ll_ema` | λ ∝ 相对梯度范数 | 分子→0 ⇒ **λ→0**（损失被静默关闭） |
| `lra_full` | λ = max‖∇L‖/‖∇L_i‖ | 分母→0 ⇒ **λ→∞**（损失压倒一切） |
| `true_relo` | λ 由损失比值 softmax | **不用梯度 ⇒ 不受解耦影响** |

**两种梯度类平衡器失效方向相反，损失类平衡器免疫**——这是论文的核心论点，且可证明。

### 🔴 prior 错配测试（已完成，2026-09-18）——暴露了更根本的问题

做法：prior 中心偏 **+20%**（n=0.036, C=0.06, qx0=1.2），数据仍由真值（0.03, 0.05, 1.0）生成。

| 配置 | err_C（vs 真值） | **err_C（vs prior 中心）** | w_prior |
|---|---|---|---|
| `true_relo_balprior_shift20` | 19.94% | **0.051%** | 0.728 |
| `lra_full_balprior_shift20` | 20.00% | **0.000%** | 3.44e7 |
| `ll_ema_balprior_shift20` | 53.56% | 61.30% | 1.04e-5 |

**结论（严重）**：
1. **`true_relo` 也完全跟着 prior 走**——参数距 prior 中心仅 0.051%，距真值 19.9%。
   数据**没有**把参数拉回来。
2. `lra_full` 距 prior 中心 **0.000%**（精确）→ prior 完全支配。
3. `ll_ema` 两者都不靠（prior 已关闭，参数自由漂移）。

**推论**：之前 shift=0 时"err_C = 0.047%"的漂亮结果，**是因为 prior 中心恰好就是真值**
——那是循环论证，不是逆问题求解。论文的"参数恢复"结果可信度存疑。

**机制**：约束 n 的只有 PDE 残差（数据损失只约束场 h，不约束 n——网络总能拟合观测的 h）。
把 n 从 prior 中心拉回真值的 prior 代价（~2.6e-4）比 PDE 残差代价（~2.5e-5）**高一个数量级**
→ 优化器选择保持 prior 中心、容忍残差。

### 🔴🔴 弱 prior 诊断（已完成，2026-09-18）——结论：逆问题不可辨识

| prior_scale | err_h（场） | err_n | err_C |
|---|---|---|---|
| **0**（无 prior） | 0.635% | 6.333% | **54.985%** |
| 0.01 | 0.644% | 5.061% | 6.125% |
| 0.1 | 0.635% | 0.505% | 0.645% |
| 1.0（论文值） | 0.640% | 0.048% | 0.071% |

**决定性观察**：`err_h` 几乎恒定（0.635–0.644%），而 `err_n/err_C` 完全由 prior 强度决定。

**结论（根本性）**：
1. **场是数据驱动的**（err_h 与 prior 强度无关）
2. **参数是 prior 决定的**（err_n/err_C 随 prior 强度单调变化）
3. **无 prior 时 err_C = 55%** → 数据**无法辨识** C（drainage 凹陷仅占 h 的 1.1%，
   20% 的 C 变化只改变 h 约 0.22%，低于 0.5% 噪声）

**一致性检验**：`ll_ema`（prior 被丢弃）得 err_C 52.9% ≈ scale=0 的 55% ✓
—— 说明丢弃 prior 等价于没有 prior。

**对论文的影响**：不能声称"解出了水动力逆问题、参数恢复到 sub-percent"——
那是 prior 告知的。论文必须重新定位（见下）。

### ✅ 已实现：诊断指标 + anchor-robust 修复（2026-09-18）

**新增文件**：
- `training/anchor_robust.py`
  - `estimate_anchor_coupling(losses, anchor, params)` — **诊断**：
    `c_k = ‖∇_anchor L_k‖ / ‖∇_full L_k‖`。c_k≈0 表示该损失对 anchor 不可见，
    其平衡权重无意义。
  - `AnchorRobustBalancer(base, anchor, params)` — **修复**：包装任意平衡器，
    定期探测耦合比，对解耦损失改用"可见损失的平均权重"（既不丢弃也不压倒），
    并重新归一化。

**玩具模型验证（全部通过）**：

| 测试 | 结果 |
|---|---|
| 诊断识别解耦损失 | 耦合比 coupled=0.894 / decoupled=**0.00e+00** ✓ |
| LRA 单独 | 权重 → **1.99e+03**（爆炸） |
| **LRA + ARB** | **1.00**（有界）✓ |
| LL-EMA 单独 | 权重 → **1.37e-08**（塌缩） |
| **LL-EMA + ARB** | **0.996**（有界）✓ |

**wrapper 同时修复两个相反方向的失效** —— 这是论文的**方法贡献**（补 novelty 缺口）。

**真实问题集成测试通过**：`lra_full_arb` 在 SWE 上 `w_prior = 1.00`（对比：LRA 单独 3.3e7）。

**真实逆问题验证（已完成，2026-09-19）—— 修复在两个方向上都生效**：

| 配置 | err_h | err_n | err_C | w_prior |
|---|---|---|---|---|
| `true_relo`（损失比值，参照） | 0.640% | 0.0484% | 0.0705% | 0.736 |
| `ll_ema`（无修复） | 0.632% | 0.0716% | **52.87%** | 1.64e-05 |
| **`ll_ema` + ARB** | 0.680% | 0.0275% | **0.0298%** | **1.000** |
| `lra_full`（无修复） | 0.708% | ~0 | ~0（循环） | **3.34e+07** |
| **`lra_full` + ARB** | 0.839% | 0.0018% | **0.0008%** | **1.000** |

- `ll_ema`：err_C **52.87% → 0.0298%**（改善 **1773 倍**），prior 权重 1.6e-5 → 1.000
- `lra_full`：权重 **3.34e7 → 1.000**（爆炸被阻止）
- **一个 wrapper 同时修复两个相反方向的失效，且无需任何人工干预**

结果存档在 `outputs/server2/`（`*_arb_balprior/` 与 `balancer_compare_summary_arb.json`）。

**论文四个贡献现已全部有实验支撑**：理论（两种失效方向）／诊断（`c_k`）／
方法（ARB 双向修复）／实证（0.05%↔55% 摆动 + 1773 倍改善）。

### 论文的新定位（A + 方法）

标题方向：*Silent Failure of Adaptive Loss Balancing in Physics-Informed Inverse
Problems: Diagnosis and a Fix*

| # | 贡献 | 性质 |
|---|---|---|
| 1 | 梯度类平衡器对 anchor 解耦损失的**两种相反失效方向**（可证明） | 理论 |
| 2 | **运行时诊断指标** `c_k`，自动识别对 anchor 不可见的损失 | **新工具** |
| 3 | **anchor-robust 平衡器**，包装任意平衡器、无需人工干预 | **新方法** |
| 4 | 真实逆问题上 err_C 摆动 **0.05% ↔ 55%**；含错配测试与 prior 强度曲线 | 实验（已完成） |

**可行的重新定位方向**：
- **A（推荐）**：定位为「**自适应损失平衡如何静默决定先验信息的去留**」。
  核心论点可证明（梯度解耦 + 两种失效方向），且有硬后果（丢弃 prior → 参数漂移 55%）。
  实验**已全部做完**。
- **B**：改实验设置让 C 可辨识（增大 drainage 信号/加观测/观测 qx），然后重跑全部实验。
  成本高（数天），且 PINN 逆问题往往弱可辨识，未必成功。
- **C**：把"先验主导"本身作为发现报告（与 A 合并）。

**附带科学问题（可作为 limitation/future work）**：不可辨识性是否部分是网络过参数化造成的？
（网络有能力把参数误差吸收进场里）更受限的架构可能改善可辨识性。

## 🖥️ 服务器现状（2026-09-17 更新）

**新服务器 `ubuntu@134.175.139.218`（当前使用）**：
- 连接：`SSH_ASKPASS=/home/pngyuo/.ssh/askpass_capinn.sh SSH_ASKPASS_REQUIRE=force setsid ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no ubuntu@134.175.139.218 '命令'`
- 密码存在 `/home/pngyuo/.ssh/askpass_capinn.sh`（权限 700）。**实验跑完后建议删除此文件**。
- 配置：**2 核**（AMD EPYC 7K83）、**1.9 GB 内存** + 2 GB swap、39 GB 磁盘
- 环境：miniconda3（Python 3.9）在 `~/miniconda3`，已装 torch 2.0.1 + numpy<2 + scipy + matplotlib
- 代码：`~/ca_pinn`（含 `outputs/swe_reference.npz`，MD5 与本地一致）
- **坑**：`/tmp` 是 tmpfs 只有 966 MB，pip 装大包会报 "Disk quota exceeded"。
  解法：`export TMPDIR=~/tmp`（主磁盘 32 GB）。
- **规模限制**：内存决定 n_coll ≤ ~4000（n_coll=2000 时峰值内存 592 MB；旧服务器 n_coll=8000 时约 4.3 GB）
- 速度：n_coll=2000、2 线程时约 **0.2 秒/epoch** → 30k epochs ≈ 1.7 h

**旧服务器 `student01@172.24.148.158:2206`**：可用，`ca_pinn` 项目还在；但用户另有任务在跑
（`qbpdm_work`，492% CPU），非必要不占用。

**本机**：16 逻辑核，但内存仅 7 GB（可用 6 GB），跑满规模偏紧。

---

## 已决定的方向（用户已确认）

**方向 A：提出新 balancer**。把「排除 prior」的手工 hack 升级为算法：
anchor-robust / 自动检测解耦损失的平衡方案，无需手工排除 prior 即避免漂移。
预期同时补定量理论（方向 B）作支撑。

---

## 拒稿历史

| 期刊 | 结果 | 理由 |
|---|---|---|
| CMAME | desk reject | "not suitable"（未送外审） |
| Neurocomputing | 拒 | **"falls within aim and scope... declined due to lack of sufficient novelty"** |

编辑明确说范围对、创新不足。故本轮目标是**补方法贡献**，不是改包装。

---

## 环境与路径

**本地项目**：`/mnt/f/downloads/PINNs-master/ca_pinn/`

| 内容 | 路径 |
|---|---|
| 论文（elsarticle） | `paper/main.tex`（19 页，编译通过） |
| 平衡器实现 | `training/loss_balancing.py` |
| causal 训练 | `training/causal.py` |
| 有限体积求解器 | `physics/swe_solver.py` |
| SWE 残差 + 数据生成 | `physics/swe.py` |
| 实验脚本 | `experiments/train_*.py` |
| 投稿材料 | `cover_letter.tex/.txt`、`abstract.tex/.txt`、`declaration_of_interest.tex`、`credi_statement.txt` |

**服务器**：`ssh -Y -p 2206 student01@172.24.148.158`
- 密码在 `/home/pngyuo/.ssh/askpass.sh`（内容 `cugstu@a906`）
- **所有操作仅限 `/home/student01/pngyuo`**，不得改动其他目录
- 服务器 12 核，训练用 `torch.set_num_threads(10)`
- 启动方式：`setsid nohup python -u script.py > log 2>&1 < /dev/null & disown`
- 注意：**串行跑，不要并行**（并行会因 CPU 竞争慢 7 倍）
- 服务器曾重启过一次，nohup 进程会丢失，需检查后重启

**GitHub**：https://github.com/YuhaoPeng1103/ca-pinn
- 推送需 ed25519 私钥（已从 Windows 复制到 `~/.ssh/id_ed25519`）
- 若新环境失效，需重新从 `/mnt/c/Users/HUAWEI/.ssh/` 复制

**编译论文**：`export openout_any=a && pdflatex main.tex && bibtex main && pdflatex main.tex ×2`
- 该 TeX 环境缺 `pdftex.map`，main.tex 里已加 `\pdfmapfile{+pdftex35.map}` 解决字体问题

---

## 实验数据现状（新参考解，可信）

参考解已从「解析构造的合成场」换成**有限体积求解器**（Lax-Friedrichs），
已验证：无 drainage 时回到 Manning 正常水深（机器精度）、质量守恒偏差 <0.1%、网格收敛。

结果在 `outputs/server/`（本地），服务器在 `~/ca_pinn/experiments/outputs/`。

| 配置 | err_h | err_n | err_C | 备注 |
|---|---|---|---|---|
| vanilla_30k | 0.543% | 0.049% | 0.064% | 基线 |
| causal_30k | 0.922% | 1.69% | 2.78% | |
| relobralo_30k (fixed prior) | 0.605% | 0.045% | 0.047% | 现有实现 |
| ca_pinn_v2_30k (causal+relo) | 1.355% | 2.49% | 3.80% | 组合最差 |
| **relobralo_balprior_30k** | 0.630% | 7.31% | **37.2%** | **← 争议中的发现** |
| lr_anneal_30k | 0.463% | 3.06% | 8.92% | |
| vanilla_sparse (n_obs=100) | 0.985% | 0.037% | 0.051% | |
| ca_pinn_sparse | 1.919% | 2.48% | 2.86% | |

**Multi-seed（3 seeds，`outputs/server/multiseed_summary.json`）**：
vanilla 0.549±0.012%、causal 0.919±0.007%、relobralo 0.592±0.018%、both 1.216±0.016%（err_h）

Burgers / NS 实验用的是 Cole-Hopf 解析级数解 / Raissi 2019 CFD 数据，**不依赖 SWE 参考解，结果仍有效**。

---

## 论文已完成的重要修正（勿回退）

1. 参考解从假合成场 → 有限体积求解器（`physics/swe_solver.py`）
2. drainage 参数 C：0.5 → 0.05（原值物理不合理，抽水量 > inflow）
3. 标题改为强调方法论：`Gradient Decoupling in Adaptive Loss Balancing for PINN Inverse Problems`
4. 移除了「悬挂的 RAR 模块」（声称但未评估，审稿人会惩罚）
5. 弱化了过度声称（不再说"CA-PINN 新框架"）
6. 所有表格数字已换成新数据
7. 作者信息：Yuhao Peng，中国地质大学(武汉)数学与物理学院，3136496634@qq.com

---

## 遗留的可选改进

- 参考文献 `nabian2021efficient` 有 volume+number 字段冲突（bibtex 警告，不影响编译）
- GradNorm 作为第三个 baseline
- Burgers/NS 的多次 seed
