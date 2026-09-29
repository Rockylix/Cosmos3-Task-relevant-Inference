# 最新 ASI Compile 十任务测试

2026-09-13。模型源码提交：`87caf74`。
运行目录：`experiments/asi_compile_smoke10_s0_p0_v1/`。

## 配置与结果

使用生产 `Version1PolicyService`，`--asi-execution compile`：Torch Compile 开启、
CUDA Graph 关闭、compile_dynamic=True。RTX 4090 / Torch 2.10.0+cu130；
现有项目 Edge 与 RoboLab venv，没有新建环境或下载权重。

复用原来八简单、两普通的固定十任务，每任务一个 episode。
simulator seed=0；policy seed=0，每任务重置 request RNG，deterministic_seed=False；
4 denoise steps、guidance=3、shift=5。使用各任务官方步数上限，不加额外截断，失败不重试。

**完成 10/10，成功 5/10（50%），平均 score=0.500。**

| Task | success | score | 最终步数 | 请求数 |
|---|---|---:|---:|---:|
| BananaInBowlTask | true | 1.0 | 304 | 10 |
| BananaOnPlateTask | true | 1.0 | 140 | 5 |
| ButterAboveRaisinTask | false | 0.0 | 600 | 19 |
| BowlStackingLeftOnRightTask | false | 0.0 | 300 | 10 |
| GrabABagelTask | true | 1.0 | 337 | 11 |
| LargerObjectRaisinBoxInBinTask | false | 0.0 | 450 | 15 |
| MustardInLeftBinTask | false | 0.0 | 450 | 15 |
| RubiksCubeTask | true | 1.0 | 131 | 5 |
| RubiksCubeLeftOfBowlTask | true | 1.0 | 439 | 14 |
| MarkerInMugTask | false | 0.0 | 600 | 19 |

以 RoboLab `episode_results.jsonl` 的布尔 success 为准，原始 reason 原样保留。
本轮没有重新跑配对 Dense 或旧 ASI，不从此十任务结果推断相对成功率改善。

## 除 Torch Compile 外是否改变计算

相对原仿真 ASI `7d3c424`，这不是只加一个 compile 开关。也包含：

- 八个 future frame 的 action attention 评分批量化；
- Profile 通过 decoder 输出 metadata 返回，能够进入编译图；
- request 内 token layout 与稀疏 SequencePack metadata 缓存；
- CFG 实际文本长度和 padding 区分、Profile 输出独立存储修复。

Core80、Stable104、Top6、动作权重 `[1/6,1/3,1/3,1/6]`、一次 dense/seven sparse、
当前 stack 输入恢复、CFG/UniPC 公式没有修改。没有新增 velocity cache 或更改 token 预算。
公式不变不等于数值逐位相同：此前 Compile 配对已发现部分 Top-K mask 与输出变化。
本轮只新增测试 runner，没有修改生产模型代码；运行前后源码 SHA256 核对一致。

runner 经子 agent 只读 review，确认沿用生产入口、旧十任务与 seed 协议，不改变模型计算。

## 运行验证与时间口径

123 个请求全部完成，最终 action/vision latent 均有限，无 NaN/Inf。
每任务 chunk 编号从 1 连续递增；seed 序列按生产 `_next_seed` 的
`np.random.default_rng(0).integers(0, 2**31)` 逐项验证一致。
任务清单及 episode 无重复或缺项；tmux runner 正常退出 code=0，服务器关闭，GPU 已释放。
未保存仿真视频或模型预测帧；保留模拟器自动输出的记录和小型统计日志。

另附在线观测延迟：剔除每任务前三个请求后的 93 个 generation-wrapper 样本，
mean=0.475139 s、median=0.474672 s、P90=0.480259 s。
此时 Isaac 同时占用 GPU，计时包含生产 controller wrapper，但不含 RPC、环境步进或 VAE decode；
**不把这些值当作独占 GPU 稳态单 chunk 加速比**。
之前同输入、独占 GPU 交替测速见 [优化验证报告](robolab_asi_optimized_cn.md)。

## 文件与复现

- `summary.json`：汇总成功率、score、逐任务结果；
- `simulator/episode_results.jsonl`：原始 success、score、步数和原因；
- `requests.jsonl`：逐 chunk seed、耗时和有限性检查；
- `manifest.json`：任务、配置、源码哈希；
- `runtime.json`：实际 Python、Torch、GPU、compile/graph 状态；
- `commands.json`、`server.log`、`simulator.log`：原命令及日志。

```bash
cd /root/robolab/cosmos-framework-edge-core80-stable104-action-weighted
PYTHONPATH="$PWD" .venv/bin/python -u -m tools.run_asi_optimized_smoke10 \
  --execution compile --port 8017 \
  --output "$PWD/experiments/asi_compile_smoke10_s0_p0_retest"
```

新测试必须用新输出目录。已有 tmux `asi_compile_smoke10` 保留了完成界面：

```bash
tmux attach -t asi_compile_smoke10
```
