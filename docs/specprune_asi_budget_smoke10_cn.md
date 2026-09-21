# SpecPrune 对齐 ASI future 保留比例：十任务

## 目标与边界

分支 `experiment/specprune-asi-budget`，从冻结的 SpecPrune 提交 `3945720` 创建。
目标是对齐 **全部 denoise step、CFG forward、Transformer block 上平均的 future token 保留率**。
不是仅匹配最后层，也不是匹配总 FLOPs、L0/UND 缓存行为或速度。

ASI 每个 chunk：step0 conditional 全量一次，其余7次每帧保留184/340：

`r_ASI = (340 + 7 * 184) / (8 * 340) = 59.85294%`。

## 冻结预算

| 参数 | 原 SpecPrune | 本轮 |
|---|---:|---:|
| Local K | 32 | 58 |
| Global K（各历史层） | 40 | 72 |
| 动态收缩下限 | 60 | 184 |
| keep_ratio | 0.9 | 0.9 |

`configs/specprune_asi_budget.json` 是本轮 runner 实际加载的配置。
不改 Local/Global/Dynamic 的评分、层位置、mask 嵌套、L1–L8 空间共享、early-exit hidden 恢复、输出头和 UniPC。
基础 `ExitConfig()` 默认值仍是原版，旧策略没有被覆盖。

**下限不是强制补点**：若前两层静态选择已少于184，保持原算法，不补齐。
首个 chunk 没有 Global/Dynamic 历史，因此可能更稀疏；这不是固定K184策略。

## 校准方法

使用原十任务保存的168个 chunk attention score 和 Dynamic mask，离线推算不同预算下的层输入长度。
只改变 Local K（32..128，步长2）和按原比例1.25缩放的 Global K，下限固定184。
依据所有 chunk 的平均 future token 保留率距离 ASI 目标最小来选参数；不读取成功率标签。

- 选定参数的旧评分代理估计：**59.99450%**，与 ASI 差 **+0.14156 个百分点**。
- 对每个任务先平均后再平均：60.22998%。
- 代理估计最终每帧平均178.18 token，范围80–184。

旧 attention 来自旧策略轨迹，预算变化会影响后续 attention 和闭环轨迹；代理估计不是新实验的实测结果。
新十任务逐 chunk 保存真实保留率，不在看见结果之后再调参。
详细扫描：`experiments/asi_budget_calibration_v1/calibration.json`。

## 验证与运行

- CPU：30项 SpecPrune 测试和7项 instruction-attention 测试通过。
- GPU：全保留与 Dense 的 action、vision 逐位相同；新预算第一/第二次请求有限值、attention重算验证及完整solver布局通过。
- 测试前清空门禁生成的历史和policy RNG，正式任务不继承重复输入的门禁历史。
- eager，4steps，shift=5，guidance=3，policy seed=0、deterministic_seed=False；每个新任务重置历史/RNG。
- 同此前10任务，8简单+2普通，各一次，官方step上限；不重试失败，不保存视频或大型原始张量。
- 只重跑本轮 SpecPrune。旧 Dense 4/10、旧 SpecPrune 1/10 仅作历史参照，不声称是本轮重新配对的实测。

```bash
cd /root/robolab/worktrees/specprune-asi-budget
tmux attach -t specprune_asi_budget_smoke10
```

后台命令：

```bash
env LD_LIBRARY_PATH= \
  /root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  tools/run_specprune_asi_budget.py \
  --output experiments/specprune_asi_budget_smoke10_s0_p0_v1
```

输出目录包含：

- `manifest.json`：参数、任务、命令、推理源码hash。
- `gate.json`：GPU门禁。
- `requests.jsonl`：每chunk实际层预算、平均future/GEN保留率、generation时间。
- `status.json`：实时完成数、成功率、官方score及原始score。
- `results.json`：仅在全部十任务完成且seed检查通过后生成。
- `simulator/episode_results.jsonl`：原始任务结果。

generation时间含评分和检查，不含RPC/仿真/文件保存；去掉每任务首chunk后报告mean/median/P90。
它是闭环输入上的描述性延迟，不能替代同输入交替基准来宣布相对ASI/Dense加速。

## 固定版本的最终结果

十任务全部完成，129个chunk，无重试。配置固定为 Local58 / Global72 / 收缩下限184 / keep_ratio0.9。
本配置是显式实验 preset，不修改原 SpecPrune 的默认预算。上述 runner 读取 JSON；
直接启动 `action_policy_server_specprune` 时必须添加：

```bash
--specprune-local-k 58 --specprune-global-k 72 \
--specprune-min-observation-tokens 184 --specprune-keep-ratio 0.9
```

| 指标 | 本轮实测 |
|---|---:|
| Success | 4/10 (40%) |
| 官方汇总平均 score | 0.466667 |
| 原始 episode score 平均 | 0.366667 |
| 成功任务平均步数 | 305 |
| Future token 平均保留率，全部chunk/step/branch/block | 60.963862% |
| 不含每任务首chunk的 future 保留率 | 63.742762% |
| GEN保留率，含L0/action、不含文本 | 65.671421% |
| Generation mean / median / P90 | 0.669972 / 0.666847 / 0.719798 s |

generation统计排除每任务首chunk，共119个样本；本轮为eager，不能与其他版本的compile/Graph测速直接比较。
60.96%是闭环观测均值，不是每层强制比例，也不保证其他模型/轨迹得到相同比例。

| Task | Success | 原始 score | Steps |
|---|---:|---:|---:|
| BananaInBowlTask | 1 | 1 | 367 |
| BananaOnPlateTask | 1 | 1 | 453 |
| ButterAboveRaisinTask | 1 | 0 | 210 |
| BowlStackingLeftOnRightTask | 0 | 0 | 300 |
| GrabABagelTask | 0 | 0 | 450 |
| LargerObjectRaisinBoxInBinTask | 0 | 0 | 450 |
| MustardInLeftBinTask | 0 | 0 | 450 |
| RubiksCubeTask | 1 | 1 | 190 |
| RubiksCubeLeftOfBowlTask | 0 | 0.666667 | 450 |
| MarkerInMugTask | 0 | 0 | 600 |

ButterAboveRaisin的记录为success=True但原始score=0；官方汇总按成功计1，因此列出两个score口径，不修改原始记录。
数据保留于被Git忽略的 `experiments/specprune_asi_budget_smoke10_s0_p0_v1/`；
提交仅包含代码、配置、测试和本文的关键结果，不包含实验原始数据。
