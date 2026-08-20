# Cosmos3 Edge RoboLab 中间 Chunk Hidden State 采集实验

本文档用于在已经跑通的 Cosmos3 Edge + RoboLab 闭环仿真中，选择多个任务的中间 action chunk，采集完整 GEN Transformer hidden state，并同时保存 VAE 解码后的未来帧。

本文档只描述代码验证和实验操作。正式仿真实验需在确认任务与 chunk 索引后再启动。

## 1. 实验目标和数据语义

RoboLab 的 `Cosmos3Client.OPEN_LOOP_HORIZON = 32`。客户端每向 policy server 发起一次请求，server 会生成一个包含 32 个动作的 action chunk；客户端随后开环执行这些动作，再请求下一个 chunk。

采集以“同一个精确 prompt 下的 WebSocket 请求序号”为 chunk 编号，编号从 0 开始。不同任务的 prompt 分别计数，因此多个任务不会互相挤占 chunk 索引。本实验要求：

- `--num-envs 1`：否则同一 prompt 的多个环境会交错请求，无法只根据服务端请求确定 env。
- `--num-runs 1`：否则同一任务的第二个 episode 会延续第一个 episode 的 prompt 计数。
- 默认采集建议为 chunk `3 5 7`，对应仿真 action step `96–127`、`160–191`、`224–255`；它们跳过最初的冷启动和初始静止段，又能覆盖已跑通的 259-step BananaInBowlTask 中段。

每个被选中的 chunk 生成一个独立 artifact。raw hidden state 的物理布局与参考实验
`/root/piper-cosmos3/experiments/vision_experiments3/1784979586711922873_req000003`
一致：

```text
[forward_call, transformer_block, temporal_latent, spatial_token, hidden_channel]
```

默认 `num_steps=4`、`guidance=3.0` 时，一个去噪步包含 conditional 和 unconditional 两次 GEN forward，因此 Edge 预期 raw shape 为：

```text
[8, 28, 9, 360, 2048]
```

其中 profile 中保存：

```text
sampler_step_by_call = [0, 0, 1, 1, 2, 2, 3, 3]
cfg_branch_by_call   = [conditional, unconditional, ...]
```

所以可无损重解释为逻辑布局：

```text
[4 denoising_step, 2 cfg_branch, 28 block, 9 latent, 360 spatial_token, 2048 hidden]
```

保留 `forward_call` 物理轴是为了让现有分析脚本无需修改即可读取。

## 2. 已完成的代码改动

采集实现位于：

```text
/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/robolab_hidden_state_capture.py
```

服务入口修改位于：

```text
/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/action_policy_server_robolab.py
```

新增参数：

```text
--hidden-state-capture-dir PATH
--hidden-state-capture-chunks INT [INT ...]
--hidden-state-capture-disk-reserve-gib FLOAT
```

行为边界：

- 未设置 capture 参数时，正常推理行为不变。
- 只有命中选中 chunk 的请求才注册 Transformer block hook、保存 raw hidden state并执行额外的 VAE decode。
- 开启采集时服务端自动关闭 `torch.compile` 和 CUDA graphs，确保逐层 hook 能观测到真实 eager forward。
- `num_steps` 必须为 4，`guidance` 必须不等于 1，以兼容现有 FIS-DiT 分析中的四步、双 CFG 分支语义。
- artifact 先写入 `chunk_XXXXXX.partial`，全部文件完成后才改名为 `chunk_XXXXXX`；异常时保留 `FAILED.txt`。

CPU 单元测试和静态检查已执行：

```text
11 passed
Ruff: All checks passed
```

这些测试不包含真实 GPU 模型推理；第一次正式采集仍需观察首个目标 chunk 的实际 `raw_shape`。

## 3. 磁盘预算

Edge checkpoint 配置为 28 个 Transformer block、hidden size 2048。若实际视觉 token geometry 为参考配置的 `9 × 360`，单个 chunk 的 bf16 raw hidden state 大约为：

```text
8 × 28 × 9 × 360 × 2048 × 2 bytes = 2.77 GiB
```

预计：

| 任务数 | 每任务 chunk 数 | raw hidden 总量 |
|---:|---:|---:|
| 2 | 3 | 约 16.6 GiB |
| 3 | 2 | 约 16.6 GiB |
| 3 | 3 | 约 24.9 GiB |

2026-07-29 检查本机根分区只剩约 31 GiB，所以首轮建议采用“2 个任务 × 3 个 chunk”，并保留默认 10 GiB safety reserve。不要直接执行 3 个任务 × 3 个 chunk。

运行前检查：

```bash
df -h /root/robolab
du -sh /root/robolab/cosmos-framework-edge/experiments 2>/dev/null || true
```

## 4. 启动采集版 Cosmos3 Edge server

先选择一个全新的实验 ID。capture 根目录必须不存在或为空；这样能防止重启服务后 chunk 计数归零并覆盖旧结果。

终端 A：

```bash
cd /root/robolab/cosmos-framework-edge

export EXP_ID=edge_hidden_banana_bowl_plate_c357_v1
export EXP_ROOT=/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/hidden_states/$EXP_ID
export COSMOS_EDGE_CKPT=/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID
export HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home
export HF_HUB_OFFLINE=1
export PYTHONPATH=/root/robolab/cosmos-edge-overlay:/root/robolab/cosmos-framework-edge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -f "$COSMOS_EDGE_CKPT/checkpoint.json"
test -f "$HF_HOME/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
test ! -e "$EXP_ROOT"
df -h /root/robolab

/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path "$COSMOS_EDGE_CKPT" \
  --format-prompt-as-json True \
  --no-guardrails \
  --hidden-state-capture-dir "$EXP_ROOT" \
  --hidden-state-capture-chunks 3 5 7 \
  --hidden-state-capture-disk-reserve-gib 10 \
  --host 0.0.0.0 \
  --port 8000
```

预期启动日志包含：

```text
[robolab-hidden-capture] enabled output=... chunk_indices=[3, 5, 7]
```

另开一个终端监控 GPU 与磁盘：

```bash
watch -n 1 nvidia-smi
```

```bash
watch -n 2 'df -h /root/robolab; find /root/robolab/cosmos-framework-edge/experiments/preliminary/representation/hidden_states -maxdepth 4 -type f -name gen_hidden_state_raw.bin -printf "%p %s bytes\n" 2>/dev/null'
```

## 5. 启动两个任务的 RoboLab 仿真

终端 B：

```bash
cd /root/robolab/RoboLab
export OMNI_KIT_ACCEPT_EULA=Y

.venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 \
  --remote-port 8000 \
  --task BananaInBowlTask BananaOnPlateTask \
  --num-envs 1 \
  --num-runs 1 \
  --output-folder-name cosmos3_edge_hidden_banana_bowl_plate_c357_v1 \
  --video-mode viewport \
  --headless
```

说明：

- `--video-mode viewport` 保存完整仿真第三人称 MP4，便于把 hidden artifact 对齐到完整 episode。
- policy server 的 artifact 只保存选中 chunk 的 conditioning observation 和 VAE 预测未来帧。
- 首轮选两个简单任务，控制 raw 数据约 16.6 GiB。确认磁盘和实际 shape 后，再决定是否增加第三个任务。
- 如果某个 episode 在 chunk 7 之前结束，该任务不会生成 `chunk_000007`；这不是文件损坏，根目录 `manifest.jsonl` 会准确列出实际完成的 artifact。

命中目标 chunk 后，server 预期打印：

```text
[robolab-hidden-capture] completed prompt='...' chunk_index=3 artifact=... raw_shape=[8, 28, 9, 360, 2048]
```

## 6. 输出目录规范

```text
/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/hidden_states/
└── edge_hidden_banana_bowl_plate_c357_v1/
    ├── experiment.json
    ├── manifest.jsonl
    ├── task_pick_up_the_banana_and_place_it_in_the_bowl_<hash>/
    │   ├── chunk_000003/
    │   │   ├── metadata.json
    │   │   ├── gen_hidden_state_profile.json
    │   │   ├── gen_hidden_state_raw.bin
    │   │   ├── denoised_vision_latent.pt
    │   │   ├── future_vision_latent.pt
    │   │   ├── conditioning_observation.png
    │   │   └── predicted_future_frames/
    │   │       ├── frame_001.png
    │   │       └── ...
    │   ├── chunk_000005/
    │   └── chunk_000007/
    └── task_pick_up_the_banana_and_put_it_on_the_plate_<hash>/
        ├── chunk_000003/
        ├── chunk_000005/
        └── chunk_000007/
```

文件含义：

- `experiment.json`：整个实验的 checkpoint、采集 chunk、采样配置和 action chunk size。
- `manifest.jsonl`：每个已完整落盘的 artifact 一行；分析时以它为准。
- `gen_hidden_state_raw.bin`：bf16/float16 mmap raw tensor，不做时空 pooling。
- `gen_hidden_state_profile.json`：shape、dtype、去噪步、CFG 分支、timestep 和 latent-to-frame 映射。
- `denoised_vision_latent.pt`：包含 conditioning temporal latent 的完整视觉 latent。
- `future_vision_latent.pt`：去掉 temporal latent 0 后的 future-only latent。
- `predicted_future_frames/`：VAE 解码帧 1 到末帧；帧 0 是 conditioning，不放入 future 目录。
- `conditioning_observation.png`：该 chunk 发起推理时的 RoboLab 拼接观测。

## 7. 实验完成后的完整性检查

以下检查不会加载 2.77 GiB raw tensor到内存：

```bash
export EXP_ROOT=/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/hidden_states/edge_hidden_banana_bowl_plate_c357_v1

find "$EXP_ROOT" -type d -name '*.partial' -print
wc -l "$EXP_ROOT/manifest.jsonl"
find "$EXP_ROOT" -type f -name gen_hidden_state_raw.bin -printf '%p %s bytes\n'
find "$EXP_ROOT" -type f -path '*/predicted_future_frames/frame_*.png' | wc -l
```

两个任务都完成三个目标 chunk 时：

```text
manifest.jsonl 行数 = 6
完整 chunk 目录数 = 6
```

逐 artifact 校验 metadata 中记录的字节数与实际文件一致：

```bash
/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path('/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/hidden_states/edge_hidden_banana_bowl_plate_c357_v1')
entries = [json.loads(line) for line in (root / 'manifest.jsonl').read_text().splitlines()]
for entry in entries:
    artifact = Path(entry['artifact_dir'])
    profile = json.loads((artifact / 'gen_hidden_state_profile.json').read_text())
    actual = (artifact / profile['files']['raw']).stat().st_size
    expected = profile['raw_num_bytes']
    assert actual == expected, (artifact, actual, expected)
    assert profile['axis_names'] == [
        'forward_call', 'transformer_block', 'latent_frame', 'spatial_token', 'hidden_channel'
    ]
    assert profile['sampler_step_by_call'] == [0, 0, 1, 1, 2, 2, 3, 3]
    print(artifact, profile['raw_shape'], f'{actual / 2**30:.2f} GiB')
PY
```

## 8. 使用参考脚本进行逐 Chunk 分析

分析入口：

```text
/root/piper-cosmos3/scripts/analyze_fis_dit_hidden_states.py
```

先分析单个 artifact：

```bash
export ARTIFACT=/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/hidden_states/edge_hidden_banana_bowl_plate_c357_v1/task_<实际目录>/chunk_000005

/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  /root/piper-cosmos3/scripts/analyze_fis_dit_hidden_states.py \
  "$ARTIFACT"
```

分析结果默认写入：

```text
$ARTIFACT/fis_dit_analysis/
```

批量分析 manifest 中的所有完成 artifact：

```bash
export EXP_ROOT=/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/hidden_states/edge_hidden_banana_bowl_plate_c357_v1

/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python - <<'PY'
import json
import subprocess
from pathlib import Path

root = Path('/root/robolab/cosmos-framework-edge/experiments/preliminary/representation/hidden_states/edge_hidden_banana_bowl_plate_c357_v1')
script = Path('/root/piper-cosmos3/scripts/analyze_fis_dit_hidden_states.py')
python = '/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python'
for line in (root / 'manifest.jsonl').read_text().splitlines():
    artifact = Path(json.loads(line)['artifact_dir'])
    subprocess.run([python, str(script), str(artifact)], check=True)
PY
```

后续跨 task、跨 chunk 对比时，主键应至少包含：

```text
task_key, prompt, chunk_index, sampler_step, cfg_branch, transformer_block, latent_pair
```

不要把 GEN block hidden output 当作 attention K/V，也不要仅凭离线 cosine/L2 直接推导可跳层或可复用；若后续要验证加速，需要再做运行时干预实验并比较 action、velocity、VAE 输出和真实耗时。

## 9. 首轮实验确认项

正式运行前建议确认：

1. 任务使用 `BananaInBowlTask` 和 `BananaOnPlateTask`。
2. 采集 chunk 使用 0-based `3 5 7`。
3. 每任务只跑 `num-envs=1, num-runs=1`。
4. 完整仿真视频只保存 viewport。
5. 输出实验 ID 使用 `edge_hidden_banana_bowl_plate_c357_v1`。

确认后再启动第 4、5 节的 server 和 RoboLab 命令。
