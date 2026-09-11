# Edge 五分支：源码迁移与 RoboLab 配置启动

更新：2026-09-11。本次发布只保存源码、配置、测试和文档；不上传模型权重、venv、RoboLab 资产、视频或实验原始数据，不合并 main、不删除其他分支。

## 1. 保留的版本

仓库：`git@github.com:Rockylix/Cosmos3-Task-relevant-Inference.git`。

| 远端分支 | 内容 | 新服务器目录 |
|---|---|---|
| `baseline` | Dense，全 future vision/action 计算 | `cosmos-framework-edge` |
| `version/core80-stable104-action-weighted` | 当前 ASI：Core80 + Stable104，action 对齐权重 | `cosmos-framework-edge-core80-stable104-action-weighted` |
| `experiment/asi-velocity-cache` | ASI + request 内被移除 token 的 guided velocity cache；可选 task 首 chunk Top7 layer 固定 | `worktrees/asi-velocity-cache` |
| `experiment/toca-future` | 固定 ToCa-Future：D/C/D/C，r=0.25，spatial bonus=0，CFG 独立，joint attention/score | `worktrees/toca-future` |
| `experiment/worldcache` | 固定 WorldCache：D/D/D/C，两 CFG 分支历史独立，joint video/action prediction cache | `worktrees/worldcache` |

这里的 Baseline 是项目已经固定的 Dense 推理分支，不等于 NVIDIA 上游未经任何部署适配的源码。迁移不改五种策略的算法。旧机器 Baseline 的本地分支名是 `cache`，对应远端 `baseline`，不要把顶层工作区仓库误推到推理仓库。

在新服务器空目录执行；目录已存在时先检查，不覆盖：

```bash
mkdir -p /root/robolab/worktrees
git clone --branch baseline git@github.com:Rockylix/Cosmos3-Task-relevant-Inference.git /root/robolab/cosmos-framework-edge
cd /root/robolab/cosmos-framework-edge
git worktree add -b version/core80-stable104-action-weighted ../cosmos-framework-edge-core80-stable104-action-weighted origin/version/core80-stable104-action-weighted
git worktree add -b experiment/asi-velocity-cache ../worktrees/asi-velocity-cache origin/experiment/asi-velocity-cache
git worktree add -b experiment/toca-future ../worktrees/toca-future origin/experiment/toca-future
git worktree add -b experiment/worldcache ../worktrees/worldcache origin/experiment/worldcache
git worktree list
```

使用 worktree 只共享 Git 对象，不复制模型和 Python 环境。不要单独复制旧 worktree 的 `.git` 文件：它指向原主仓库的绝对路径。上述新 clone 的 `origin` 是个人仓库；旧机器的个人仓库 remote 名为 `project`。

## 2. 环境分为两套

已用环境记录（不是新服务器的重新验证结果）：Ubuntu 22.04/x86_64、RTX 4090，驱动 580.178.04；Edge Python 3.13.14 / Torch 2.10.0+cu130；RoboLab Python 3.11.15 / Isaac Sim 5.0.0 / Isaac Lab 2.2.0。新 GPU/驱动组合必须重新跑 smoke，不能只凭 Python import 判定仿真成功。

先在 GPU 节点检查 `nvidia-smi`。需要管理员安装驱动时单独处理，不在 Python 环境里更新内核驱动；驱动变更后如 NVML mismatch，核对已加载内核模块和用户态库版本，必要时重启。

下面只是在新服务器执行的安装命令，不要求重装已有的有效环境。假设系统已有 `uv`、`git-lfs`、`ffmpeg`、常用 OpenGL/Vulkan/X11 运行库；Ubuntu 管理员可先安装：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs ffmpeg libgl1 libglib2.0-0 libx11-6 libxext6 libvulkan1
```

### 2.1 Edge policy server 环境：五分支共用

```bash
cd /root/robolab/cosmos-framework-edge-core80-stable104-action-weighted
uv venv --python 3.13.14 .venv
uv sync --frozen --all-extras --group=cu130-train --group=policy-server
.venv/bin/python -c 'import torch, flash_attn; from openpi_server.websocket_policy_server import WebsocketPolicyServer; print(torch.__version__, torch.version.cuda, flash_attn.__version__)'
```

使用仓库 `uv.lock` 和 `pyproject.toml` 的依赖组；这里的 `cu130-train` 是官方 setup 使用的依赖集合名称，并不运行训练。大包下载超时先看具体 URL/网络，再续装，不重复创建多套环境。锁文件的部分 wheel 位于 NVIDIA/PyTorch 索引，设置 HF 镜像不能代理这些下载。

`policy-server` 已声明 `openpi-server`（锁定 0.1.0）和其 `openpi-client` 依赖。新安装不依赖旧机器 `cosmos-edge-overlay/` 中的未跟踪 Python 包。以后执行其他分支时使用这一个 Python，并让 `PYTHONPATH` 指向所需 worktree；不要在每个 worktree 再运行一次 uv sync。

### 2.2 RoboLab / Isaac Sim 环境

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVlabs/RoboLab.git /root/robolab/RoboLab
cd /root/robolab/RoboLab
git checkout --detach 0aef241
git lfs install --local
git lfs pull
git lfs fsck
uv venv --python 3.11.15 .venv
uv sync --extra isaac50
uv pip install --python .venv/bin/python 'openpi-client==0.1.2'
```

`0aef241` 是旧机器 RoboLab 所基于的 v0.2.1 上游提交。资产通过 Git LFS 单独下载；只有 LFS 指针而没有 USD/图像实体时不能跑仿真。Isaac50 和 Isaac51 不能装进同一个 venv。旧机器另有可选 `--max-episode-steps` 本地扩展；本文不依赖它，不改变任务原有时限，新拉的上游也不支持该扩展参数。

阅读并同意 NVIDIA 相关许可后，再设置许可环境变量并测试：

```bash
cd /root/robolab/RoboLab
export OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PYTHONPATH=/root/robolab/RoboLab
.venv/bin/python examples/run_empty.py --headless
```

这一步只验证仿真器，不是 policy 成功率测试。若空场景也崩溃，先排查驱动/Vulkan/Isaac/资产，不调整模型策略。一个 GPU 只启动一套 server + simulator，避免占用叠加。

## 3. 模型资产：自己迁移或下载，不进 Git

已有权重直接指定路径，不需要重新下载。推荐新机器布局：

```text
/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID/
/root/robolab/checkpoints/wan2.2/Wan2.2_VAE.pth
```

确实没有文件时才执行：

```bash
# 需要镜像时先 export HF_ENDPOINT=https://hf-mirror.com
uvx --from huggingface-hub hf download nvidia/Cosmos3-Edge-Policy-DROID --local-dir /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID
uvx --from huggingface-hub hf download Wan-AI/Wan2.2-TI2V-5B --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e --include Wan2.2_VAE.pth --local-dir /root/robolab/checkpoints/wan2.2
```

公开仓库可匿名下载；镜像不改变仓库许可/访问权限。本文的 `--no-guardrails` 用于这个已获授权的仿真实验，不下载 Guardrail 资产。

## 4. 终端 A：选择一个服务器

先设置共同变量。`EDGE_VAE` 也可以直接指向原 HF cache 中已有的文件。下面全部按 eager 运行，不混入 compile/graph benchmark。

```bash
export EDGE_ROOT=/root/robolab
export EDGE_PY="$EDGE_ROOT/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python"
export EDGE_VAE="$EDGE_ROOT/checkpoints/wan2.2/Wan2.2_VAE.pth"
export COSMOS_TRAINING=0 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export LD_LIBRARY_PATH=''
test -f "$EDGE_VAE"
test -f "$EDGE_ROOT/RoboLab/Cosmos3-Edge-Policy-DROID/model_index.json"
COMMON_ARGS=(--checkpoint-path "$EDGE_ROOT/RoboLab/Cosmos3-Edge-Policy-DROID"
  --host 127.0.0.1 --port 8000 --seed 0 --no-deterministic-seed
  --num-steps 4 --shift 5 --guidance 3 --no-guardrails
  --format-prompt-as-json True
  --experiment-overrides "model.config.tokenizer.vae_path=$EDGE_VAE"
  model.config.tokenizer.object_store_credential_path_pretrained=
  model.config.tokenizer.bucket_name=)
```

只运行下面的一种 server。更换策略先正常退出自己的前一 server，并用新的输出目录；不要同时占用同一个端口。Bash 数组只在当前终端有效。

### A. Baseline：显式 Dense eager

旧 Baseline CLI 没有 `--eager` 参数，其 setup 的 compile/graph 默认值为 True。以下启动包装仅覆盖 setup 两个布尔值，不改模型、attention、输出头或采样公式，也不安装稀疏 controller：

```bash
cd "$EDGE_ROOT/cosmos-framework-edge"
export PYTHONPATH="$PWD"
"$EDGE_PY" - "${COMMON_ARGS[@]}" --output-dir "$PWD/experiments/baseline_server" <<'PY'
from cosmos_framework.scripts import action_policy_server_robolab as server
class DenseEager(server.RobolabPolicyService):
    def _build_setup_args(self, args):
        return super()._build_setup_args(args).model_copy(
            update={"use_torch_compile": False, "use_cuda_graphs": False})
server.RobolabPolicyService = DenseEager
server.main()
PY
```

### B. ASI Core80 + Stable104

```bash
cd "$EDGE_ROOT/cosmos-framework-edge-core80-stable104-action-weighted"
export PYTHONPATH="$PWD"
"$EDGE_PY" -m cosmos_framework.scripts.action_policy_server_robolab_version1 \
  "${COMMON_ARGS[@]}" --output-dir "$PWD/experiments/core80_server"
```

默认不保存逐请求 mask 原始数据。该服务显式关闭 compile/graph。每个 chunk 的 step0 conditional 全量 profile，其余 7 次 GEN forward 稀疏；不是整个 step0 两分支都全量。

### C. 固定 ToCa-Future

```bash
cd "$EDGE_ROOT/worktrees/toca-future"
export PYTHONPATH="$PWD"
"$EDGE_PY" -m cosmos_framework.scripts.action_policy_server_toca_scan \
  "${COMMON_ARGS[@]}" --scan-config "$PWD/configs/toca_baseline.json" \
  --scan-output "$PWD/experiments/toca_fixed/server" \
  --output-dir "$PWD/experiments/toca_fixed/model_output"
```

此入口读取已测试配置，不要用早期 `action_policy_server_robolab_toca_future` 的默认值代替：早期入口的评分后端/CFG 选择默认值不是这次固定版本。Scan 入口只允许 `configs/toca_baseline.json` 所列的十个任务；不传 `--scan-capture-root` 就不保存大体积 paired inputs。对同一输出目录拒绝启动第二次，需要换新 run 名。

### D. WorldCache D/D/D/C

```bash
cd "$EDGE_ROOT/worktrees/worldcache"
export PYTHONPATH="$PWD"
"$EDGE_PY" -m cosmos_framework.scripts.action_policy_server_robolab \
  "${COMMON_ARGS[@]}" --eager --worldcache-dddc \
  --worldcache-stable-percentile 0.30 --worldcache-chaotic-percentile 0.70 \
  --worldcache-n-max 6 --output-dir "$PWD/experiments/worldcache_server"
```

参数对应 `configs/worldcache_baseline.json`；此 server CLI 不自动读取该 JSON。D/D/D/C 是固定调度，缓存/近似范围包括预测 video 和 action，而不只是 future vision tokens。

### E. ASI velocity-cache：实验入口的额外输入

该分支保留已测试的实验服务 `cosmos_framework.scripts.asi_velocity_cache_smoke`，不是把普通 Version1 服务改成默认启用 velocity cache。当前单 full-forward 版本缓存 step0 CFG 后的排除区域 velocity，在当前 chunk 的 step1–3 复用；L0、action 和选中区域不使用这个缓存。每个 request 清空 velocity cache。

启动前需要：

- 一个可信的普通 ASI（无 velocity cache、Top6）DROID 调用捕获文件，包含 `args`、`kwargs` 和 `outputs`（action/vision）。启动 gate 会重新计算普通 ASI 并与保存输出逐元素核对，因此不能拿 Dense 输出充当 ASI 输出。不要加载来源不明的 pickle/PT。
- 一个新 `eval_root/manifest.json`，包含 `tasks` 列表和 `task_core_top7` 布尔值。例如三个已测试任务：`BananaInBowlTask`、`BananaOnPlateTask`、`ButterAboveRaisinTask`。
- 这些实验输入不随源码上传，不能把旧绝对路径当成新服务器上已存在的数据。

没有旧 gate 输入时，可在新机器先用普通 ASI 捕获第三个 chunk。在已经设置共同变量的终端 A 中执行以下包装，终端 B 运行 BananaInBowlTask；保存完成后正常退出两端。捕获开销不纳入 benchmark：

```bash
export EDGE_GATE=/root/robolab/worktrees/asi-velocity-cache/experiments/gate/banana_c3_asi.pt
mkdir -p /root/robolab/worktrees/asi-velocity-cache/experiments/gate
cd "$EDGE_ROOT/cosmos-framework-edge-core80-stable104-action-weighted"
export PYTHONPATH="$PWD"
"$EDGE_PY" - "${COMMON_ARGS[@]}" <<'PY'
import os
from pathlib import Path
import torch
from cosmos_framework.scripts import action_policy_server_robolab_version1 as server
from cosmos_framework.scripts.asi_smoke import cpu_tree
destination = Path(os.environ["EDGE_GATE"])
if destination.exists():
    raise FileExistsError(destination)
class CaptureService(server.Version1PolicyService):
    def __init__(self, args):
        super().__init__(args)
        original = self.model.generate_samples_from_batch
        count = 0
        def generate(*positional, **kwargs):
            nonlocal count
            count += 1
            saved = cpu_tree({"args": positional, "kwargs": kwargs}) if count == 3 else None
            result = original(*positional, **kwargs)
            if saved is not None:
                saved["outputs"] = {k: result[k][0].detach().cpu() for k in ("action", "vision")}
                torch.save(saved, destination)
                print(f"Saved ordinary ASI chunk 3 gate: {destination}", flush=True)
            return result
        self.model.generate_samples_from_batch = generate
server.Version1PolicyService = CaptureService
server.main()
PY
```

这只是新服务器的一次启动验证采样，不纳入原成功率；不要使用相同路径覆盖旧 gate。若换 GPU/精度/kernel 后旧 gate 无法逐元素重现，应在该环境重新捕获，而不是放宽 gate 掩盖配置差异。

准备好 `EDGE_GATE` 指向该 PT 后，使用分支内已保存的三任务 manifest。目录已存在时换一个 run 名，不覆盖：

```bash
export EDGE_EVAL=/root/robolab/worktrees/asi-velocity-cache/experiments/velocity_smoke3_new_server
mkdir "$EDGE_EVAL"
cp /root/robolab/worktrees/asi-velocity-cache/configs/asi_velocity_cache_smoke3.json "$EDGE_EVAL/manifest.json"
```

随后启动服务：

```bash
cd "$EDGE_ROOT/worktrees/asi-velocity-cache"
export PYTHONPATH="$PWD"
test -f "$EDGE_GATE"
test -f "$EDGE_EVAL/manifest.json"
"$EDGE_PY" -m cosmos_framework.scripts.asi_velocity_cache_smoke \
  "${COMMON_ARGS[@]}" --eval-root "$EDGE_EVAL" --gate-capture "$EDGE_GATE" \
  --task-core-top7 --output-dir "$EDGE_EVAL/model_output"
```

此例要求 manifest 的 `task_core_top7=true`。这会在每个任务首 chunk 选一次 Top7 Core 来源层；后续 chunk 仅固定这七个层号，仍重新计算当前 Core/mask，Stable 仍在全部 28 层取 profile。去掉 flag 并令 manifest 为 false 才是每 chunk 重选 Top6。相同 prompt 的新 episode 不会自动被识别为新任务，做独立 episode 对比时重启服务，不能把旧任务 cache 带过去。

`tools/run_asi_velocity_cache_smoke3.py` 等历史 runner 的默认 gate/VAE 路径仍是旧机器布局；新服务器优先使用上面的显式路径，不直接照抄历史绝对路径。其他历史实验报告中的本地图片/原始数据链接也不会随 Git 自动迁移。

## 5. 终端 B：仿真客户端

server 完成加载并监听后才启动；`ss -ltnp` 可核对 8000 端口，勿误停其他用户进程。以下使用任务原有 episode step limit、每任务一次，不保存视频：

```bash
cd /root/robolab/RoboLab
export PYTHONPATH=/root/robolab/RoboLab
export CUDA_VISIBLE_DEVICES=0
export OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
.venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --task BananaInBowlTask --num-envs 1 --num-runs 1 \
  --headless --video-mode none --output-folder-name edge_smoke_new_server
```

扩为三个任务时将 `--task` 后改成 `BananaInBowlTask BananaOnPlateTask ButterAboveRaisinTask`。需要仿真视频时将 `none` 改为 `viewport`；模型预测 future frames 和仿真摄像机视频不是同一数据，不能混称。

RoboLab 同名输出目录会恢复/跳过已完成 episode；全新测试要使用新名称。`success` 来自 `output/<run>/episode_results.jsonl`，`score` 单独统计，不用 score=1 替代 success=True。

本 checkout 的 runner 调用 `create_env` 时使用默认 seed=0；注册配置里 seed=1 不代表实际运行值。policy `seed=0, deterministic_seed=False` 则表示从基准 RNG 逐请求生成 seed，不是所有请求都使用相同 noise。正式对比应核对每个 `env_cfg.json` 和请求 seed，并明确是否按任务重置 policy RNG；不同实验入口的 task reset 逻辑不完全相同，不能把默认启动命令直接当成严格配对实验。

## 6. 迁移核验与发布边界

- 用 `git branch --show-current` / `git rev-parse HEAD` 记录运行版本；用 `import cosmos_framework; print(cosmos_framework.__file__)` 确认 PYTHONPATH 没有导入另一个 worktree。
- 先 `run_empty`，再一个真实 policy episode；环境导入成功不等于任务测试成功。
- 本次仅重新运行 CPU contract tests：Core80 6 passed；ASI velocity-cache 26 passed；ToCa 21 passed / 6 GPU skipped；WorldCache 18 passed。未重跑 GPU benchmark、闭环成功率或新机器安装。
- 旧报告的数值保留为历史实测，不因本次源码发布成为新测试；不要用整任务 wall time 除法代替相同输入、预热交替的单 chunk median 加速。
- `.venv/`、`experiments/`、HF cache、权重和仿真 output 不提交。只将后续配置/结论写入各分支 `docs/` 并提交；不要 `git add .` 或 `git push --all`。
- ToCa 详情在其分支 `configs/toca_baseline.json`；WorldCache 在 `configs/worldcache_baseline.json`；ASI velocity-cache 在其分支 `docs/asi_single_dense_velocity_cache_smoke3_cn.md` 与 `docs/task_core_top7_global_stable_cn.md`。
