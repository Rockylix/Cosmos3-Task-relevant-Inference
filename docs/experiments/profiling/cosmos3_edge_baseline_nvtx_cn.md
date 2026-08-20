# Cosmos3 Edge eager baseline NVTX 采集命令

固定基线：不启用 `torch.compile`，不启用 CUDA Graphs；4-step UniPC、shift=5、guidance=3、seed=0。`--nvtx-profile` 会在源码中强制这两个优化关闭。

当前版本把 Generator 拆成 17 条独立 NVTX domain 轨道：1 条 Summary、8 条 `step × branch × Stages` 和 8 条 `step × branch × Blocks`。conditional 为绿色、unconditional 为橙色。请优先打开文末的 `nvtx_split_tracks_v4` 报告；v3 虽有 domain，但仍会在 Timeline 折叠成一个整体。

## 1. 启动被 profile 的服务

```bash
cd /root/robolab/worktrees/cosmos3-edge-baseline-nvtx

mkdir -p /root/robolab/cosmos-framework-edge/experiments/preliminary/profiling/baseline_nvtx/edge_eager_shift5_banana_nvtx_split_tracks_v4

env \
  PYTHONPATH=/root/robolab/cosmos-edge-overlay:. \
  HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home \
  HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0 \
nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=none \
  --cpuctxsw=none \
  --cuda-memory-usage=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --force-overwrite=true \
  --output=/root/robolab/cosmos-framework-edge/experiments/preliminary/profiling/baseline_nvtx/edge_eager_shift5_banana_nvtx_split_tracks_v4/edge_baseline_eager_nvtx_split_tracks_chunk1 \
  /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID \
  --host 127.0.0.1 \
  --port 8000 \
  --format-prompt-as-json True \
  --guidance 3.0 \
  --num-steps 4 \
  --shift 5.0 \
  --deterministic-seed \
  --seed 0 \
  --no-guardrails \
  --nvtx-profile \
  --nvtx-warmup-chunks 1 \
  --nvtx-capture-chunks 1
```

## 2. 启动 RoboLab

```bash
cd /root/robolab/RoboLab

env CUDA_VISIBLE_DEVICES=0 .venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 \
  --remote-port 8000 \
  --task BananaInBowlTask \
  --num-envs 1 \
  --num-runs 1 \
  --headless \
  --video-mode none \
  --output-folder-name edge_baseline_eager_nvtx_split_tracks_trigger_v4
```

chunk 0 是预热。chunk 1 完成后 nsys 的 `stop-shutdown` 会主动关闭 policy server，RoboLab 客户端随后出现连接断开属于预期行为。

## 3. 导出统计

```bash
cd /root/robolab/cosmos-framework-edge/experiments/preliminary/profiling/baseline_nvtx/edge_eager_shift5_banana_nvtx_split_tracks_v4

nsys stats \
  --force-export=true \
  --force-overwrite=true \
  --report nvtx_pushpop_sum,nvtx_gpu_proj_sum,nvtx_pushpop_trace,cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  --format csv \
  --output split_track_stats \
  edge_baseline_eager_nvtx_split_tracks_chunk1.nsys-rep
```

生成 `step × conditional/unconditional` 和逐 block 精简表：

```bash
cd /root/robolab/worktrees/cosmos3-edge-baseline-nvtx

/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m cosmos_framework.scripts.summarize_edge_nvtx \
  /root/robolab/cosmos-framework-edge/experiments/preliminary/profiling/baseline_nvtx/edge_eager_shift5_banana_nvtx_split_tracks_v4/split_track_stats_nvtx_gpu_proj_sum.csv \
  --output-dir /root/robolab/cosmos-framework-edge/experiments/preliminary/profiling/baseline_nvtx/edge_eager_shift5_banana_nvtx_split_tracks_v4
```

Generator 内部 range 采用以下层级，因此可在 Nsight GUI 中独立筛选任意 step、CFG branch 和 block：

```text
edge.generator.step_0.conditional.total
edge.generator.step_0.conditional.transformer
edge.generator.step_0.conditional.block_0
...
edge.generator.step_3.unconditional.block_27
```

打开 `.nsys-rep`：

```bash
nsys-ui /root/robolab/cosmos-framework-edge/experiments/preliminary/profiling/baseline_nvtx/edge_eager_shift5_banana_nvtx_split_tracks_v4/edge_baseline_eager_nvtx_split_tracks_chunk1.nsys-rep
```

完整流程、逐 step/branch 时间与瓶颈结论见：

`/root/robolab/cosmos-framework-edge/experiments/preliminary/profiling/baseline_nvtx/edge_eager_shift5_banana_nvtx_split_tracks_v4/report_cn.md`
