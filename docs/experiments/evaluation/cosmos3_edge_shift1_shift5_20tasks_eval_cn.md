# Cosmos3 Edge shift=1 / shift=5 的 20-task RoboLab 成功率对照

## 1. 实验目标和统计口径

本实验在同一套 Cosmos3 Edge、RoboLab 和 Isaac Sim 环境中，仅改变
UniPC sampler 的 `shift`：

- A 组：`shift=1`
- B 组：`shift=5`
- 固定 20 个不同 RoboLab benchmark task
- 每个 task 运行 1 个 episode：`num_envs=1`、`num_runs=1`
- 仿真环境 seed：`0`（RoboLab 实际启动日志确认）
- 策略 seed：`579362556`
- `--deterministic-seed`：每个策略请求使用相同的初始噪声 seed
- `guidance=3.0`
- `num_steps=4`
- 默认 instruction
- `video-mode=none`：策略仍然接收相机 observation，只是不把评测视频写盘
- 成功率口径：`episode_results.jsonl` 中 `success=true` 的 episode 数除以完整 episode 数

服务端 `--shift` 最终传给 `generate_samples_from_batch(..., shift=self.cfg.shift)`；
不是只修改日志字段。代码位置：

- `/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/action_policy_server_robolab.py:363-372`
- `/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/action_policy_server_robolab.py:467-475`
- `/root/robolab/cosmos-framework-edge/cosmos_framework/scripts/action_policy_server_robolab.py:759-766`

RoboLab 的 `--task` 接受多个任务，`num_runs * num_envs` 决定每个任务的
episode 数；相同 `output-folder-name` 会跳过已经完成的 episode：

- `/root/robolab/RoboLab/robolab/eval/runner.py:45-60`
- `/root/robolab/RoboLab/robolab/eval/runner.py:82-93`
- `/root/robolab/RoboLab/robolab/eval/runner.py:191-200`
- `/root/robolab/RoboLab/robolab/eval/runner.py:237-244`

> 注意：每个任务只有一个 episode，因此只能得到“20 个任务上的 pooled
> 成功比例”，不能估计每个任务自身的稳定成功率。若要报告正式的 per-task
> 成功率，应增加 `num_runs` 或使用 adaptive sampling。

## 2. 本次使用的软件和硬件

记录时间：2026-07-31。

```text
GPU: NVIDIA GeForce RTX 4090 24 GB
Driver: 580.173.02
Workspace commit: 21b04ad
cosmos-framework-edge commit: f45d2fe
RoboLab commit: 34be120
Checkpoint: /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID
Wan2.2 cache: /root/cosmos3/cosmos/checkpoints/hf_home
Policy port: 8000
```

RoboLab Cosmos3 runner会启用相机并通过 WebSocket client 连接策略服务：

- `/root/robolab/RoboLab/policies/cosmos3/run.py:15-29`
- `/root/robolab/RoboLab/policies/cosmos3/run.py:34-48`

## 3. 固定的 20 个任务

两组必须使用完全相同的顺序：

1. `BananaInBowlTask`
2. `BananaOnPlateTask`
3. `RubiksCubeTask`
4. `RubiksCubeAndBananaTask`
5. `RubiksCubeLeftOfBowlTask`
6. `RubiksCubeRightOfBowlTask`
7. `RubiksCubeInFrontOfBowlTask`
8. `RubiksCubeBehindBowlTask`
9. `SpoonInMugTask`
10. `MarkerInMugTask`
11. `YogurtInBowlTask`
12. `WoodSpatulaToBowlTask`
13. `BowlInBinTask`
14. `SmartphoneInBinTask`
15. `CoffeePotInBinTask`
16. `PickUpBluePitcherTask`
17. `PickDrillTask`
18. `PickGlassesTask`
19. `GrabABagelTask`
20. `GrabAFruitTask`

## 4. shift=1

### 4.1 终端 A：启动策略服务器

```bash
cd /root/robolab/cosmos-framework-edge

export HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home
export HF_HUB_OFFLINE=1
export PYTHONPATH=/root/robolab/cosmos-edge-overlay:/root/robolab/cosmos-framework-edge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID \
  --format-prompt-as-json True \
  --no-guardrails \
  --host 127.0.0.1 \
  --port 8000 \
  --guidance 3 \
  --num-steps 4 \
  --shift 1 \
  --seed 579362556 \
  --deterministic-seed
```

看到以下信息后再启动仿真：

```text
[robolab-policy-server] ready ... guidance=3.0 num_steps=4 shift=1.0
server listening on 127.0.0.1:8000
```

可另开终端检查：

```bash
curl -fsS http://127.0.0.1:8000/healthz
nvidia-smi
```

### 4.2 终端 B：运行 20 个任务

```bash
cd /root/robolab/RoboLab

export OMNI_KIT_ACCEPT_EULA=Y

TASKS=(
  BananaInBowlTask
  BananaOnPlateTask
  RubiksCubeTask
  RubiksCubeAndBananaTask
  RubiksCubeLeftOfBowlTask
  RubiksCubeRightOfBowlTask
  RubiksCubeInFrontOfBowlTask
  RubiksCubeBehindBowlTask
  SpoonInMugTask
  MarkerInMugTask
  YogurtInBowlTask
  WoodSpatulaToBowlTask
  BowlInBinTask
  SmartphoneInBinTask
  CoffeePotInBinTask
  PickUpBluePitcherTask
  PickDrillTask
  PickGlassesTask
  GrabABagelTask
  GrabAFruitTask
)

.venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 \
  --remote-port 8000 \
  --num-envs 1 \
  --num-runs 1 \
  --task "${TASKS[@]}" \
  --output-folder-name cosmos3_edge_shift1_20tasks_v1 \
  --video-mode none \
  --headless
```

输出：

```text
/root/robolab/RoboLab/output/cosmos3_edge_shift1_20tasks_v1/
```

本次已经中止过一次。使用完全相同的 `output-folder-name` 重跑上述命令时，
RoboLab 会读取 `episode_results.jsonl`，跳过已经完成的 5 个任务，从未完成
的任务继续。不要更换任务顺序。

## 5. shift=5

先等待 shift=1 仿真完全退出，再在终端 A 使用 `Ctrl-C` 停止 shift=1 server。
确认端口释放：

```bash
ss -ltnp | grep ':8000' || true
```

### 5.1 终端 A：启动 shift=5 策略服务器

启动命令与 4.1 相同，仅将：

```text
--shift 1
```

改为：

```text
--shift 5
```

完整命令：

```bash
cd /root/robolab/cosmos-framework-edge

export HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home
export HF_HUB_OFFLINE=1
export PYTHONPATH=/root/robolab/cosmos-edge-overlay:/root/robolab/cosmos-framework-edge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID \
  --format-prompt-as-json True \
  --no-guardrails \
  --host 127.0.0.1 \
  --port 8000 \
  --guidance 3 \
  --num-steps 4 \
  --shift 5 \
  --seed 579362556 \
  --deterministic-seed
```

### 5.2 终端 B：运行相同 20 个任务

重新声明第 4.2 节的同一 `TASKS` 数组，然后运行：

```bash
cd /root/robolab/RoboLab
export OMNI_KIT_ACCEPT_EULA=Y

.venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 \
  --remote-port 8000 \
  --num-envs 1 \
  --num-runs 1 \
  --task "${TASKS[@]}" \
  --output-folder-name cosmos3_edge_shift5_20tasks_v1 \
  --video-mode none \
  --headless
```

输出：

```text
/root/robolab/RoboLab/output/cosmos3_edge_shift5_20tasks_v1/
```

## 6. 自动汇总结果

两组完成后执行：

```bash
/root/robolab/RoboLab/.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("/root/robolab/RoboLab/output")
groups = {
    1: root / "cosmos3_edge_shift1_20tasks_v1" / "episode_results.jsonl",
    5: root / "cosmos3_edge_shift5_20tasks_v1" / "episode_results.jsonl",
}

all_rows = {}
for shift, path in groups.items():
    if not path.exists():
        print(f"shift={shift}: missing {path}")
        continue
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    all_rows[shift] = {row["task_name"]: row for row in rows}
    success = sum(bool(row["success"]) for row in rows)
    print(f"shift={shift}: {success}/{len(rows)} = {success / len(rows):.1%}")

if set(all_rows) == {1, 5}:
    names = sorted(set(all_rows[1]) | set(all_rows[5]))
    print("\ntask,shift1,shift5")
    for name in names:
        value1 = all_rows[1].get(name, {}).get("success", "MISSING")
        value5 = all_rows[5].get(name, {}).get("success", "MISSING")
        print(f"{name},{value1},{value5}")
PY
```

必须确认两组都显示 `20` 条完整结果后，才能报告 20-task A/B 成功率。

## 7. 2026-07-31 中止时的部分结果

用户要求停止后，仿真和 server 已关闭。停止后的状态：

```text
TCP 8000: no listener
GPU memory: 42 MiB / 24564 MiB
GPU utilization: 0%
```

本次只完成 shift=1 的前 5 个任务：

| Task | Success | Episode step | Score |
|---|---:|---:|---:|
| BananaInBowlTask | true | 177 | 1.000 |
| BananaOnPlateTask | true | 178 | 1.000 |
| RubiksCubeTask | true | 149 | 1.000 |
| RubiksCubeAndBananaTask | true | 524 | 1.000 |
| RubiksCubeLeftOfBowlTask | false | 450 | 0.667 |

部分成功率：

```text
shift=1: 4/5 = 80.0%（仅部分结果）
shift=5: 尚未运行
```

`RubiksCubeRightOfBowlTask` 在 episode 中途停止，没有写入
`episode_results.jsonl`，因此不计入分母。当前结果文件：

```text
/root/robolab/RoboLab/output/cosmos3_edge_shift1_20tasks_v1/episode_results.jsonl
/root/robolab/cosmos-framework-edge/experiments/preliminary/evaluation/shift_success_20tasks_v1/shift1_partial_episode_results.jsonl
/root/robolab/cosmos-framework-edge/experiments/preliminary/evaluation/shift_success_20tasks_v1/shift1_server_console.log
```

这 5 个 episode 不能用于判断 shift=1 和 shift=5 谁的成功率更高；必须完成
同一 20 个 task 的两组配对实验。

## 8. 运行注意事项

- 不要同时启动 shift=1 和 shift=5 server；两者默认都绑定 TCP 8000。
- 每组开始时在 server 日志中确认实际的 `shift`。
- 两组必须使用相同任务顺序、环境 seed、策略 seed、guidance 和 num_steps。
- `--video-mode none` 适合成功率批量测试；需要观察失败轨迹时，可以单独复现失败
  task 并使用 `--video-mode sensor`，不要改变主实验的统计口径。
- 当前磁盘剩余约 13 GiB；保持 `video-mode=none`，避免批量视频占满磁盘。
- Headless 下的 GLFW/X Server warning 不等于失败。本次 5 个完整 episode 均在这些
  warning 存在时正常运行和落盘。
- 主动 `Ctrl-C` 终止 server 后出现 `destroy_process_group()` 或 asyncio pending-task
  warning 属于中止清理信息，不应计为 task 失败。
