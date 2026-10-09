# 项目状态交接文档

> **新会话请先读这个文件。** 最后更新：2026-10-09

## 一句话现状

论文《Silent Failure of Adaptive Loss Balancing in Physics-Informed Inverse
Problems: Diagnosis and a Fix》已完成写作与全部实验，但**连续被 3 个期刊拒稿**
（CMAME / Neurocomputing / Neural Networks），最近一次编辑明确说"**偏 ML 范围、
但神经网络创新不是主要贡献**，建议投 ML 期刊"。

用户已确认：**强化 ML 角度、投 Q1 非 OA 期刊**。当前正在做这件事。

---

## 🎯 当前工作：把论文从"PINN 应用"升级为"ML 方法论"

### 已完成的阶段①：定量理论（关键突破）

**新增文件**：
- `paper/theory_anchor_bias.md` — 完整推导 + 数值验证（183 行）
- `experiments/verify_theory.py` — 玩具模型验证（K=2 与 K=4）
- `experiments/mtl_anchor_benchmark.py` — MTL 基准（共享 trunk + 任务头 + 可调耦合的辅助损失）
- `experiments/train_mtl_anchor.py` — 真实 MTL 训练循环验证

**核心结果**（设 `c_k = ‖∇_φ L_k‖/‖∇_θ L_k‖` 为耦合比，`G_k = ‖∇_θ L_k‖`，
`S = Σ_j G_j`，`S_c = Σ_j c_j G_j`，`M = max_j(c_j G_j)`，`M_G = max_j G_j`）：

| 平衡器 | 偏差公式 | c→0 极限 |
|---|---|---|
| **相对范数** | **λ_k/λ_k⁰ = c_k · S/S_c** | → 0（损失被关闭） |
| **倒数比** | **λ_k/λ_k⁰ = M/(c_k · M_G)** | → ∞（损失压倒一切） |

- 相对范数的公式**无条件成立**（数值验证：K=2 和 K=4 均 0.00% 误差）
- 倒数比的公式**一般形式**含因子 M/M_G；只在"最大值由完全耦合损失取得"时才化为 1/c
- **重要区分**：全参数 anchor 下所有 c_k ≡ 1，倒数比**不存在 anchor 偏差**；
  SWE 里 prior 权重 3.3e7 的成因是"**参数支撑稀疏**"（另一机制，见理论文档 §6）

**MTL 训练循环验证结果**（`experiments/train_mtl_anchor.py`）：

| γ | c_aux | 相对范数 | 倒数比 |
|---|---|---|---|
| 0（完全解耦） | 0.000 | **w = 0**（关闭） | **w = 2.5e10**（压倒） |
| 0.02–10 | 0.06–1.00 | **0.00%** 误差 | **0.00%** 误差 |

即：**同一解耦损失，两种平衡器给出方向相反、量级极端的结果，且公式精确预测。**

### ⏭️ 未完成（新会话从这里继续）

1. **真实数据集验证** — 目前只有合成 MTL 任务，需在标准 benchmark 上复现
   - 建议：Multi-MNIST（小、快、CPU 可跑）或 CelebA 子集
   - 目标：证明"解耦损失被静默错误加权"在真实任务上确实发生
2. **修复方法验证** — 把 `training/anchor_robust.py` 的 ARB wrapper 用到 MTL 上
   - 目标：ARB 能修复偏移，**且不损伤正常任务**
3. **论文重写** — 以 MTL 为主体、PINN 作为一个应用；把定量理论写进 §3

---

## 📄 论文现状（`paper/main.tex`，未改动，仍可编译）

**规格**：13 页，4 图，6 表，38 条参考文献（22 条带 DOI），0 未解析引用

**当前定位**（提交给 Neural Networks 的版本）：
"自适应损失平衡的静默失效：诊断与修复"

**结构**：
1. Introduction（2 种失效方向）
2. Related Work（5 个小节）
3. Theory: Anchor Decoupling（命题 + 推论，**二元版本**）
4. Diagnostic: Anchor-Coupling Ratio
5. Method: Anchor-Robust Balancing (ARB)
6. Experimental Setup（SWE + 有限体积参考解）
7. Experiments（受控实验 / 主表 / prior 固定 / 错配测试 / prior 强度曲线 / 修复 / multi-seed / 泛化）
8. Discussion / 9. Conclusion

**⚠️ 重写时需做的**：把 §3 的二元命题**替换为理论文档里的定量公式**，
并把 Related Work 扩到 MTL 文献。

---

## 💾 备份与数据（三重保障，不会丢）

| 内容 | 位置 |
|---|---|
| Neural Networks 提交版备份 | `paper_backup_NN_submitted_20261009/` |
| git 历史 | 12 个提交（`git log --oneline`） |
| SWE 实验数据 | `outputs/server/`（23M）、`outputs/server2/`（3.5M） |
| 代码仓库 | https://github.com/YuhaoPeng1103/ca-pinn |

---

## 🖥️ 服务器与环境

**新服务器 `ubuntu@134.175.139.218`**（2 核 / 1.9 GB，n_coll ≤ 2000）：
```bash
SSH_ASKPASS=/home/pngyuo/.ssh/askpass_capinn.sh SSH_ASKPASS_REQUIRE=force \
  setsid ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no ubuntu@134.175.139.218 '命令'
```
- 密码在 `/home/pngyuo/.ssh/askpass_capinn.sh`（权限 700）
- 环境：`~/miniconda3`（Python 3.9 + torch 2.0.1 + numpy<2）
- 代码在 `~/ca_pinn`
- **坑**：`/tmp` 是 tmpfs 仅 966 MB，装大包须 `export TMPDIR=~/tmp`
- 启动：`setsid nohup python -u 脚本 > 日志 2>&1 < /dev/null & disown`
- **必须串行跑**（并行会因 CPU 竞争慢 7 倍）

**旧服务器 `student01@172.24.148.158:2206`**：可用（12 核 / 31 GB），
但用户另有任务在跑；训练脚本里用 `torch.set_num_threads(10)`。

**本机**：16 逻辑核，内存 7 GB（可用 6 GB）。MTL 合成实验在本机跑即可。

**编译论文**（`ca_pinn/paper/`）：
```bash
export openout_any=a && pdflatex main.tex && bibtex main && pdflatex main.tex ×2
```
- 用 **Elsevier CAS 模板**（`cas-sc.cls` + `cas-common.sty` + `cas-model2-names.bst`）
- 需 `thumbnails/` 目录（社交图标），否则编译报缺文件
- 该 TeX 环境缺 `pdftex.map`；CAS 模板本身会产生一个 117pt overfull 警告
  （**官方 sample 同样有，非本文问题**）

---

## 🔬 实验数据现状（已完成，可信）

参考解用**有限体积求解器**（`physics/swe_solver.py`），已验证质量守恒 <0.1%。

**Balance-prior 主表**（n_coll=2000，30k epochs）：

| 配置 | err_h | err_n | err_C | w_prior |
|---|---|---|---|---|
| `true_relo_balprior`（真 ReLoBRaLo） | 0.640% | 0.048% | **0.071%** | 0.736 |
| `ll_ema_balprior`（最后一层 anchor） | 0.632% | 0.072% | **52.9%** | 1.6e-05 |
| `lra_full_balprior`（全参数 LRA） | 0.708% | ~0 | ~0（循环） | 3.3e+07 |
| **`ll_ema_arb_balprior`** | 0.680% | 0.028% | **0.030%** | **1.000** |
| **`lra_full_arb_balprior`** | 0.839% | 0.002% | **0.001%** | **1.000** |

**Prior 错配测试**（prior 中心偏 +20%）：
- `true_relo`：距真值 19.94%、距 prior 中心 **0.051%** → 跟着 prior 走
- `lra_full`：距真值 20.00%、距 prior 中心 **0.000%**
- `ll_ema`：两者都不靠（prior 已关闭）

**Prior 强度曲线**（`err_h` 恒定 ~0.64%，`err_C` 从 0.07% 变到 55%）：
σ=0 → 54.99%，0.01 → 6.13%，0.1 → 0.65%，1.0 → 0.071%
→ **场是数据驱动的，参数是 prior 决定的**（该逆问题弱可辨识）

---

## 📌 关键设计原则（勿回退）

1. **论文声称的"ReLoBRaLo"曾与实现不符** — 已在论文中改名为 `LL-EMA`，
   并把真正的损失比值法实现为 `ReLoBRaLoTrue`（`training/balancers.py`）
2. **不要删除任何旧成果** — 用户明确关心这点；定量理论是纯新增
3. **Burgers 实验数据无效**（Cole-Hopf 参考解违反物理约束），已从论文删除；
   泛化实验只剩 NS
4. **drainage 参数 C = 0.05**（原值 0.5 物理不合理，抽水量 > inflow）

---

## ⚠️ 需要用户核实的两条参考文献

`wang2022improved` 与 `he2023physics` 在 Crossref 按卷/页查不到，
**原始 bib 的卷号/页码可能有误**，投稿前建议在 ScienceDirect 上核对。
