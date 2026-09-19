# 观测 SpecPrune → 未来帧剪枝，Early-exit hidden 补全

本轮由用户确认：在真实观测上选区，剪枝应用到全部未来帧；被删除时保存 hidden，经正常输出头预测近似 velocity，以便继续完整去噪。十任务沿用之前 8 simple + 2 moderate 清单。**不是完整原版 SpecPrune-VLA 复现，不预设迁移必然失败。**

## 输入、计划、执行

```text
当前真实 RGB / L0 + 上一 chunk 观测历史
                  ↓
step0 conditional：观测空间评分，生成逐层计划
  Local/Global：L0 Q → instruction K
  层级重要性：action q1..q32 → L0 K
                  ↓
每 step / 每 CFG branch 从完整当前 latent 编码
  B0/B1 静态筛选 → B10/15/20/25 进一步剪枝
  同一层 L1..L8 共享空间 mask，L0/action/text 始终计算
  被删位置：保存本 forward 中退出层的 hidden
                  ↓
最后 block 后：按原坐标拼回全部 hidden
                  ↓
最终 RMSNorm → llm2vae（含 bias）→ 完整 velocity
                  ↓
conditional/unconditional CFG → 原生完整 UniPC 更新
                  ↓
下一 step 从完整更新后的 latent 开始，不复用旧 hidden
```

对在层 ℓ 退出的位置 p，保存的是该层输出、最终 RMSNorm 之前的 hidden：

\[
\widehat v_s^b(p)=\mathrm{llm2vae}(\mathrm{RMSNorm}(H_{s,\ell,\mathrm{out}}^b(p))),\qquad b\in\{c,u\}.
\]

未退出位置使用 B27 输出。分别补全两分支后按原公式 \(v_u+g(v_c-v_u)\) guidance，完整交给 UniPC。条件 latent、条件 action 和无效 patch scalar 沿用原生屏蔽。

这是“提前退出 hidden 经过原头”近似，不是直接把 hidden 当 velocity，也不是缓存上一步 velocity。它保证数组/积分坐标完整，不保证生成质量或动作不变。退出位置不参与后续 attention/MLP，因此有效输入确实变短；输出头、VAE latent 与 solver 不变短。

## 冻结参数与官方差异

| 项目 | 本轮定义 |
|---|---|
| 历史 Global | 上个 chunk step0 conditional 的 L0 B13/B27，各 Top40 并集 |
| Local | B0 Top64 预筛，保留其 Top32；B1 在当前候选中 Top32，与 Global/Dynamic/B0 Top32 合并 |
| Dynamic 图像比较 | 原 RGB patch cosine ≥0.986 的低变化位置最多313，保护其补集；首 chunk 无历史，仅 Local |
| 一致空间网格 | 名义 VAE crop/patchify 对齐17×20；真实 RGB 544×736，原图540×640，比较528×640有效区域 |
| 层级重要性 | 当前候选的 action→L0 概率均值排名；sigmoid(-rank) 归一化；乘逆归一化熵置信度 |
| 熵置信度 | 每任务首次访问该层计算，后续 chunk 复用；任务切换清除 |
| EMA | S ← 0.8S + 0.2s；每 chunk 初始化零 |
| 分数更新层 | B14、B19、B24 |
| 动态剪枝层 | B10、B15、B20、B25；均在对应 block 输出后 |
| 剪枝预算 | 对虚拟 `[当前观测候选 + 全部文本/action]` 长度保留0.9；保留全部非视觉项，动态剪枝下限60。若静态阶段已少于60，不额外补点；再广播8帧 |
| 平分规则 | 按原始空间ID稳定排序；明确记录，不宣称与官方设备上的不稳定同分排序逐位一致 |
| 逐步计划 | 每 chunk 的 step0 conditional 生成各层 mask；全部四步、两分支逐层重放同一计划。缓存的是 mask，不是 hidden/velocity |
| 动作控制器 | 关闭；不拿关节角替代末端位移，不引入 FK 或新阈值 |
| 模型原始计算 | 不改权重、attention kernel、attention mask；RoPE 用原始坐标 |

官方参考快照：`8091adc4b574ce9008d49a1dc9a210f4eec314c1`，本机 `/tmp/specprune-review.CpeUHW/repo`。保留官方首次 B10 剪枝在首次 B14 评分之前的调度，因此 **B10 为全零重要性同分选择，不能解读成 action-aware 排名**。

评分访问边界：达到动态预算下限后，停止本 chunk 后续重要性和置信度更新，与官方停止逻辑一致。仍可采集诊断用 action→L0 热力图，但这些数据不进入选择。gate_v2 后复核修正了这一停止条件；修正时尚未运行任何 Sparse episode，Dense episode 不重跑，Sparse 前重新进行 gate_v3。各阶段代码指纹独立保存。

本轮保留原动态层编号（都在28层内），Global 沿用此前确认 B13/B27；静态预算延续已有 Edge 对照，不将其冒称为原生 OpenVLA 多视角超参数。虚拟观测预算、共享未来 mask、早退出头补全均是公开标注的 WAM 适配。

L0 的初始 latent 来自真实观测，但其深层 hidden 通过 joint attention 读取 future/action。只从 L0 位置评分不等于模型上下文与未来噪声无关。

## 验证与测试

- 31 项 CPU 测试通过（24项 SpecPrune/坐标恢复测试 + 7项 attention 测试），包括动态下限停止和显式时间轴切片。
- 最新 GPU gate_v4：全保留 action 与 vision latent 的形状和数值均与原生 Dense 一致，MSE=0、cosine=1；所有四步完整3060视觉patch状态、224 block调用均通过检查。
- 首次 GPU gate 发现观测patch留在CPU的问题，修复设备迁移后重跑门禁；未因此启动、删除或重试任何闭环 episode。失败日志仍保留。
- 指令/动作注意力重算均与实际 attention kernel 的对应输出行比较；finite 检查逐层执行。

模拟器 seed=0，policy seed=0，非 deterministic_seed，每任务重置请求RNG；shift=5、guidance=3、4步，eager，compile/CUDA Graph关闭。两策略各10个episode，共20个，不重试失败。官方任务步数上限、不保存仿真视频。

十任务：BananaInBowlTask、BananaOnPlateTask、ButterAboveRaisinTask、BowlStackingLeftOnRightTask、GrabABagelTask、LargerObjectRaisinBoxInBinTask、MustardInLeftBinTask、RubiksCubeTask、RubiksCubeLeftOfBowlTask、MarkerInMugTask。

每个 Sparse chunk 额外计算同输入、同seed Dense reference，计算32×8 action（除condition行）与L1–L8完整latent误差；该参考不返回仿真、不参与计划或历史。各策略闭环沿各自轨迹运行。

本轮包含诊断、保存和额外 Dense reference，**不是优化后的稳定单chunk性能测试**。generation日志不能直接当加速比。

## 本轮执行状态（2026-09-19，尚未完成 Sparse 十任务）

用户已确认重新启动。新尝试位于 `closed_loop_restart_v2/`，tmux 会话 `specprune_exit_smoke10_v2`；仅跑 Sparse，`dense/` 为指向原已完成结果的软链接。新 manifest 保存 Dense 原始结果 SHA256 和旧尝试路径，不覆盖旧日志。状态查看：`tmux attach -t specprune_exit_smoke10_v2`，打印 completed、success、score、chunks。

- Dense 已完成10/10，success=4/10。官方显示 mean score=0.4667；JSONL 原始 score 均值=0.3667。区别来自 RoboLab `get_avg_score` 将 success=True 的 score 强制记作1，本轮 ButterAboveRaisin 原始 score=0。保留两种口径，不修改原始记录。
- Sparse 首个请求在 paired vision 指标处异常，尚未返回动作，0个完成 episode；不能计为任务失败，也不能声称完成十任务比较。
- 根因：原生返回 `[1,48,9,33,40]`，适配器返回 `[48,9,33,40]`。直接对两者 `[:,1:]`，分别切到了通道和时间。原 gate 展平比较漏掉了形状差异；不是 NaN/Inf，也不是积分丢失位置。
- 修复：适配器按原始 shape 恢复返回接口；指标先去除可选 singleton batch，再明确截取 L1–L8；门禁增加返回形状相等检查。仅改变接口与指标索引，不改变选择或去噪数值。
- gate_v4 已通过。失败时的运行源码保存在 `source_failed_sparse/`，错误日志和既有 Dense 结果均保留。本次重启已取得用户确认，不启用自动失败重试。
- 单个门禁输入（非十任务统计）的 future-only cosine=0.21182、relative-L2=1.00702；该结果说明 early-exit hidden 补全没有保证预测质量，不能只凭 action cosine 较高判断安全。

验证记录：`cpu_tests_v4.log`、`cpu_attention_v4.log`、`shape_diagnostic.log`、`gate_v4/gate.json` 均位于本轮实验根目录。原 `closed_loop/sparse_manifest.json` 记录失败尝试的源码指纹，不代表修复后版本；不要将旧 manifest 改写成新源码指纹。

## 路径与复现

工作树 `/root/robolab/worktrees/specprune-future`，分支 `experiment/specprune-future`。全部新增在该实验树，旧三任务实现和结果保留；未改 Baseline 或生产部署默认路径，未自动push。

入口：

- `cosmos_framework/inference/specprune_exit_plan.py`：观测选择与exit缓存。
- `cosmos_framework/inference/specprune_observation_exit.py`：真实稀疏及完整头/CFG/UniPC。
- `cosmos_framework/scripts/specprune_exit_server.py`：实验服务。
- `tools/run_specprune_exit_smoke10.py`：成对十任务，不重试。

```bash
cd /root/robolab/worktrees/specprune-future
export PYTHONPATH="$PWD:/root/robolab/cosmos-edge-overlay"
export COSMOS_TRAINING=0 CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/specprune-mpl
EDGE_PY=/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python
# 新实验必须使用新目录，禁止覆盖已有结果。
RUN=experiments/observation_specprune_exit_smoke10_s0_p0_v1
"$EDGE_PY" tools/validate_specprune_exit_gpu.py --output "$RUN/gate_NEW"
"$EDGE_PY" tools/run_specprune_exit_smoke10.py --output "$RUN/closed_loop_NEW" --gate "$RUN/gate_NEW/gate.json" --port 8041
"$EDGE_PY" tools/summarize_specprune_exit.py --run "$RUN/closed_loop_NEW"
```

本轮：[状态](../experiments/observation_specprune_exit_smoke10_s0_p0_v1/closed_loop/status.json)、[配置及源文件SHA256](../experiments/observation_specprune_exit_smoke10_s0_p0_v1/closed_loop/manifest.json)、[完整结果报告](../experiments/observation_specprune_exit_smoke10_s0_p0_v1/closed_loop/report_cn.md)。结果报告只在20个episode完整后生成。
