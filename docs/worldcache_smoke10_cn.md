# WorldCache D/D/D/C：与固定 ToCa 同十任务

已完成。WorldCache：**5/10，score 0.4667，单chunk macro median 0.670914 s，同轮Dense 0.863966 s，加速1.287743×**。
18项CPU测试和真实GPU gate通过；600个配对计时样本，10任务FP64输出误差均完成。
完整结果：[十任务报告](../experiments/worldcache_dddc_smoke10_v2/report_cn.md)。

## 提交的固定配置（2026-09-11）

规范参数记录：[configs/worldcache_baseline.json](../configs/worldcache_baseline.json)。本页、该 JSON、源码和测试随 `experiment/worldcache` 提交；experiments、权重及日志不进 Git。参数核对原十任务 manifest/runtime，推理核心 SHA256 与该次测试一致；本次没有修改缓存算法或重新运行仿真。

JSON 的 `config` 可用于低层 API：`WorldCacheConfig(**settings["config"])`。现有服务不直接读取此 JSON；服务 CLI 仍使用 `--worldcache-dddc --eager --worldcache-stable-percentile 0.30 --worldcache-chaotic-percentile 0.70 --worldcache-n-max 6`，并按 `protocol` 指定 steps=4、shift=5、guidance=3、policy seed=0。D/D/D/C 是当前适配器固定调度，不是可任意替换的 JSON 调度器。

十任务专用 runner 使用本机的 ToCa 配置和历史 Dense chunk 作为 GPU gate 输入，这些输入没有上传。换机器时先配置本地 checkpoint/环境/路径并提供对应 gate 输入；不能仅 clone 分支就假设历史 artifacts 也存在。无需历史 gate 的通用模型服务启动命令见 [适配说明](worldcache_dddc_cn.md#现有环境与启动命令)。

## 固定对照与范围

ToCa 固定为 `D/C/D/C，r=0.25，spatial_bonus=0，CFG independent`。
配置在 `../toca-future/configs/toca_baseline.json`（相对 worktrees 目录）；推理源码 SHA256 已记录并核对。
已停止旧扫描并关闭 `toca_scan10`；保留原 169 个完整 episode，不补跑其他超参。

本次只运行已经约定的 WorldCache 四步适配：前三步 FULL，第四步 CACHE；两个 CFG 分支独立历史，缓存同时覆盖未来 video 和预测 action。
`percentile_stable=0.30, percentile_chaotic=0.70, n_max=6`。
每次请求重新积累三个 FULL 历史，不跨 chunk 复用。CFG 与四次 UniPC 积分都保持原实现。
它不是 ToCa 的 future-only 稀疏，也不是官方多步 WorldCache 调度；详见 [移植说明](worldcache_dddc_cn.md)。

## 闭环协议

与 ToCa 扫描逐项一致：policy seed=0；每个新任务重置请求 RNG；`deterministic_seed=False`；simulator seed=0；shift=5；guidance=3；4 denoise steps；结构化 prompt；eager，关闭 compile/CUDA graphs。
每任务一次 episode，保留官方任务时限；失败不重试、不筛选。仿真实际返回 WorldCache action，不返回 Dense action。

任务顺序（8 simple + 2 moderate）：

1. BananaInBowlTask
2. BananaOnPlateTask
3. ButterAboveRaisinTask
4. BowlStackingLeftOnRightTask
5. GrabABagelTask
6. LargerObjectRaisinBoxInBinTask
7. MustardInLeftBinTask
8. RubiksCubeTask
9. RubiksCubeLeftOfBowlTask
10. MarkerInMugTask

## 先验实现验证

实际 Edge 的单样本 vision 张量带 batch 维：`[1,48,9,33,40]`。
原生 patchifier 将 H=33 补至 34，得到 17×20 spatial grid；FULL 步捕获原生 llm2vae 的 `[2720,192]` 输出，保留真实 padding 输出值。
CACHE 在此原生向量空间预测，再按原生 unpatchify 顺序恢复并裁回 H=33；不以零值伪造预测历史。

GPU gate 使用以前 Dense 的 Banana chunk3，同 data_batch、noise seed=1097657232：

- Dense 当前输出与历史 Dense action、vision 都逐位一致。
- Dense 实际 Transformer forward=8；WorldCache=6。
- 前三步 conditional/unconditional 对应输出逐位一致。
- 全部中间缓存及最终 action/video 都检查 NaN/Inf。
- 此审计带 CPU copy，审计耗时不用于加速比较。

第一次 GPU 预检发现原生 vision 带 batch 维，未运行任何 episode 即退出；修复后新建 v2 目录验证通过。v1 保留失败日志，不是任务失败，也不构成失败任务重试。

## 计时与 future-frame 误差

闭环每任务只暂存第三个 chunk 的输入。仿真结束且进程退出后，独占 GPU 做 Dense/WorldCache 同输入、同初始 noise 的交替计时；每种方法预热5次、测30次。包含全部生成计算、历史维护与有限值检查；不含模型加载、VAE decode、CPU input clone 和磁盘 I/O。
各任务分别计算 mean/median/P90，再在10任务之间等权平均；加速为配对 Dense macro median / WorldCache macro median。

注意：此处采样的是 **WorldCache 闭环轨迹** 第三 chunk。旧 ToCa 扫描计时采样 **Dense 轨迹**，不是这批相同 observation；两轮秒数不能冒充同轮次严格三方配对。
future latent 和解码 RGB 都排除首个条件帧；动作误差排除 q0，比较生成空间动作，不将其称为物理关节执行误差。解码使用固定 `clamp[-1,1]→[0,1]`，不逐图重新归一化。
指标计算使用完整张量展平后的 FP64 norm/dot，防止长 RGB 向量 FP32 归约误差导致 cosine 越界。临时输入在全部指标复核之后才清理；不保存仿真视频或大批预测图片。

首轮配对的前两个 RGB cosine 出现 FP32 归约错误；这些结果被归档到 `invalid_fp32_metrics/`，不得引用。用户批准补采 BananaInBowl/BananaOnPlate 第三 chunk；辅助运行在两个输入得到后停止，不计入成功率。修正后全部10任务重新配对，原 `simulator/episode_results.jsonl` 以 SHA256 验证未改动，补采及代码哈希记在 `benchmark_metric_revision.json`。前两个 observation 来自同设置辅助补采，其他八个来自原闭环。

## 操作

所有运行都使用 `/root/robolab` 已有环境，不创建或下载环境。
在 GPU 无其他实验占用时运行（输出目录必须不存在）：

```bash
cd /root/robolab/worktrees/worldcache
/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python -u \
  tools/run_worldcache_smoke10.py \
  --output "$PWD/experiments/worldcache_dddc_smoke10_v2" --port 8028
```

runner 记录完整 server/simulator/benchmark 命令至 `commands.json`，只停止自己启动的子进程；不占用 tmux。

结果目录：`experiments/worldcache_dddc_smoke10_v2/`。

- `gpu_gate.json`：真实 GPU 正确性验证。
- `simulator/episode_results.jsonl`、`episodes.csv`：任务级 success/score/step。
- `requests.jsonl`：chunk seed、时长及6 FULL/2 CACHE核验。
- `paired/<task>.json`：GPU独占的配对计时及预测误差。
- `summary.json`：当前完成数和最终聚合。`phase=complete` 才表示闭环和配对测量均完成。
- `temporary_cleanup.jsonl`：临时输入清理记录。

本实验是固定 seed 的小规模对照，不声明多 seed 泛化、任务成功率不变或理论4/3加速一定兑现。
