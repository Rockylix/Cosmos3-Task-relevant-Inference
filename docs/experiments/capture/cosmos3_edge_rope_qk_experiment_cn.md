# Cosmos3 Edge RoPE 前后 Q/K 补充实验

## 1. 实验目的

本实验用于判断三维 mRoPE 是否显著改变不同未来 vision latent 之间的 Q/K 相似度。

采集点严格位于 GEN attention 内：

```text
Q/K projection
→ QK Norm
→ 保存 q_raw / k_raw
→ apply_rotary_pos_emb
→ 保存 q_rope / k_rope
→ attention
```

因此 `raw` 和 `rope` 之间唯一的算子是 RoPE，不包含 LayerNorm、QK Norm、attention softmax、残差或 MLP 的差异。

首轮固定采集：

```text
step:   0 3
block:  0 10 20 27
branch: conditional
chunk:  3
task:   BananaInBowlTask
```

原计划的 `chunk=5` 在首轮任务成功结束前未被调用，因此正式实验改为采集
`chunk=3`（第 4 次请求，约从环境 step 96 开始）。它处于任务中段且能稳定命中；
下文命令和结果均以 `chunk=3` 为准。

## 2. 代码位置

```text
/root/robolab/cosmos-framework-edge/cosmos_framework/model/generator/mot/unified_mot.py
/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/robolab_rope_qk_capture.py
/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/action_policy_server_robolab.py
/root/piper-cosmos3/scripts/analyze_edge_rope_qk.py
```

对应服务端参数：

```text
--rope-qk-capture-dir PATH
--rope-qk-capture-chunks INT [INT ...]
--rope-qk-capture-steps INT [INT ...]
--rope-qk-capture-blocks INT [INT ...]
--rope-qk-capture-branches conditional|unconditional [...]
--rope-qk-capture-disk-reserve-gib FLOAT
```

启用 Q/K 采集时，服务端自动关闭 `torch.compile` 和 CUDA graphs。Q/K 采集和全 block hidden-state 采集不能在同一个服务进程中同时开启。

## 3. 输出格式

实验根目录：

```text
/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/rope_qk/
└── edge_rope_qk_banana_bowl_c3_s03_b0_10_20_27_v1/
    ├── experiment.json
    ├── manifest.jsonl
    └── task_<task>/
        └── chunk_000003/
            ├── q_raw.bin
            ├── k_raw.bin
            ├── q_rope.bin
            ├── k_rope.bin
            ├── rope_qk_profile.json
            ├── position_ids.pt
            ├── rope_cos_sin.pt
            ├── metadata.json
            ├── conditioning_observation.png
            ├── denoised_vision_latent.pt
            ├── future_vision_latent.pt
            └── predicted_future_frames/
```

Edge DROID checkpoint 的预期 Q/K shape：

```text
Q: [2 selected_step, 4 selected_block, N_gen, 16 q_head, 128 head_dim]
K: [2 selected_step, 4 selected_block, N_gen,  8 kv_head, 128 head_dim]
```

`rope_qk_profile.json` 保存：

- step、timestep、branch、block；
- action/video 的精确 GEN-relative token indexes 和连续 ranges；
- vision token 的 `[latent_t, patch_h, patch_w]`；
- 四个 raw 文件的 shape、dtype、axis；
- position IDs 与 RoPE cos/sin 文件名。

## 4. 启动服务端

终端 A：

```bash
cd /root/robolab/cosmos-framework-edge

export EXP_ID=edge_rope_qk_banana_bowl_c3_s03_b0_10_20_27_v1
export EXP_ROOT=/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/rope_qk/$EXP_ID
export COSMOS_EDGE_CKPT=/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID
export HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home
export HF_HUB_OFFLINE=1
export PYTHONPATH=/root/robolab/cosmos-edge-overlay:/root/robolab/cosmos-framework-edge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=

test -f "$COSMOS_EDGE_CKPT/checkpoint.json"
test ! -e "$EXP_ROOT"
df -h /root/robolab

/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path "$COSMOS_EDGE_CKPT" \
  --format-prompt-as-json True \
  --no-guardrails \
  --rope-qk-capture-dir "$EXP_ROOT" \
  --rope-qk-capture-chunks 3 \
  --rope-qk-capture-steps 0 3 \
  --rope-qk-capture-blocks 0 10 20 27 \
  --rope-qk-capture-branches conditional \
  --rope-qk-capture-disk-reserve-gib 2 \
  --seed 0 \
  --host 0.0.0.0 \
  --port 8000
```

预期日志：

```text
[robolab-rope-qk-capture] enabled ... chunk_indices=[3] steps=[0, 3] blocks=[0, 10, 20, 27] branches=['conditional']
```

## 5. 启动 RoboLab

终端 B：

```bash
cd /root/robolab/RoboLab
export OMNI_KIT_ACCEPT_EULA=Y
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

.venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 \
  --remote-port 8000 \
  --task BananaInBowlTask \
  --num-envs 1 \
  --num-runs 1 \
  --output-folder-name edge_rope_qk_banana_bowl_c3_s03_b0_10_20_27_v1 \
  --video-mode viewport \
  --headless
```

chunk 使用从 0 开始的 policy request index。命中第 4 次请求时采集
`chunk_000003`。当前机器设置了 `HTTP_PROXY`；如果不设置上述
`NO_PROXY/no_proxy`，本地 WebSocket 握手也会经过代理并超时。

## 6. 产物检查

```bash
export EXP_ROOT=/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/rope_qk/edge_rope_qk_banana_bowl_c3_s03_b0_10_20_27_v1

find "$EXP_ROOT" -maxdepth 4 -type f -printf '%p %s bytes\n' | sort
du -sh "$EXP_ROOT"

ARTIFACT=$(find "$EXP_ROOT" -type d -name chunk_000003 | head -n 1)
test -n "$ARTIFACT"
test -f "$ARTIFACT/rope_qk_profile.json"
test -f "$ARTIFACT/q_raw.bin"
test -f "$ARTIFACT/q_rope.bin"
test -f "$ARTIFACT/k_raw.bin"
test -f "$ARTIFACT/k_rope.bin"
test "$(find "$ARTIFACT/predicted_future_frames" -name 'frame_*.png' | wc -l)" -eq 32
```

## 7. 执行分析

```bash
cd /root/piper-cosmos3

export EXP_ID=edge_rope_qk_banana_bowl_c3_s03_b0_10_20_27_v1
export EXP_ROOT=/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/rope_qk/$EXP_ID
export ARTIFACT=$(find "$EXP_ROOT" -type d -name chunk_000003 | head -n 1)
export ANALYSIS_ROOT=/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/offline_analysis/$EXP_ID

/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  scripts/analyze_edge_rope_qk.py \
  "$ARTIFACT" \
  --output-dir "$ANALYSIS_ROOT"
```

分析输出：

```text
analysis_cn.md
analysis_summary.json
adjacent_future_latent_qk.csv
step_block_qk_summary.csv
temporal_qk_logits.csv
rope_norm_invariance.csv
rope_effect_step_block_heatmap.png
adjacent_future_latent_cosine.png
```

主指标：

```text
delta_cosine = cosine(q/k_rope[Lt], q/k_rope[Lt+1])
             - cosine(q/k_raw[Lt],  q/k_raw[Lt+1])
```

只统计 `L1-L2` 到 `L7-L8`，不把条件 latent `L0` 纳入主平均。若 `delta_cosine` 在多个 step/block 中稳定为负，说明 RoPE 直接降低了对齐空间位置上相邻未来 latent 的 Q/K 相似度。

`rope_norm_invariance.csv` 用于验证 RoPE 的范数保持性质；若相对范数误差明显大于 bf16 数值误差，需要先排查采集点或张量布局，不能直接解释相似度结果。

## 8. 已完成的静态验证

```text
13 passed
Ruff: passed
py_compile: passed
git diff --check: passed
```

CPU 测试覆盖参数校验、step/branch/block 选择、Q/K raw 文件 shape、token ranges、position IDs、future-frame artifact。真实 GPU shape 仍以 `rope_qk_profile.json` 为最终依据。
