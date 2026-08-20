# Cosmos 3 + 官方 RoboLab 仿真配置手册

> 更新日期：2026-07-28
> 目标：只跑 NVIDIA 官方 RoboLab / Isaac Lab 仿真和官方 Cosmos 3 DROID 策略实验，不接 Piper14，不改 `/root/piper-cosmos3`。

## 1. 先明确系统结构

RoboLab 是基于 NVIDIA Isaac Lab 的机器人操作仿真基准，不是一个名为 “robolab” 的独立 NVIDIA 仿真引擎。实际分为两部分：

```text
RoboLab 客户端
  Isaac Sim + Isaac Lab + RoboLab 任务/场景
  生成三路相机图像和 DROID 机器人状态
                |
                | WebSocket，默认 8000 端口
                v
Cosmos 3 策略服务器（可选）
  Cosmos3-Edge-Policy-DROID（4B）
  返回 32 步、每步 8 维的 joint_pos 动作
```

建议按两个阶段配置：

1. 先只装 RoboLab，跑通空动作、官方 HDF5 录制轨迹回放和夹爪测试。
2. 仿真稳定后，再启动 Cosmos3 Edge 策略服务器，并用 RoboLab 客户端做闭环实验。

纯仿真阶段不需要下载 Cosmos 3 权重。官方示例所需场景、资产和演示 HDF5 由 RoboLab 仓库的 Git LFS 内容提供。

## 2. 本机现状和硬件边界

本文生成时检测到：

- Ubuntu 22.04.5，满足官方 Ubuntu 22.04+ 要求。
- `uv 0.11.28`、`ffmpeg 4.4.2`、Docker 29.5.2 已安装。
- 根分区当前剩余约 32 GB（2026-07-28 权重整理后复查）。
- 机器是 RTX 4090 24 GB。
- 当前 NVIDIA 驱动为 `580.173.02`，`nvidia-smi` 显示 CUDA 13.0。
- 内核模块、NVML 用户态库和 DKMS 版本均为 `580.173.02`，之前的 `Driver/library version mismatch` 已经在重启后解决。
- Isaac Sim 5.0 / Isaac Lab 2.2.0 已能创建 RTX `SceneRenderer`，并完成 50 步带相机 headless 仿真。
- `git-lfs 3.0.2` 已安装，RoboLab 资产已拉取；新机器仍必须在克隆后执行 `git lfs pull`。
- RoboLab 的 Cosmos3 客户端依赖 `openpi-client`；本机已补装并完成导入测试。

官方 RoboLab 要求/建议：

- Python 3.11。
- 默认 Isaac Sim 5.0 + Isaac Lab 2.2.0。
- 也支持 Isaac Sim 5.1 + Isaac Lab 2.3.2.post1，但两套环境不能装在同一个 venv。
- 仓库和资产约 8 GB，其中资产约 7 GB。
- 必须有 NVIDIA RTX GPU，官方建议 48 GB 以上显存。
- 官方 Docker 镜像约 42 GB。

24 GB 4090 可用于单环境 RoboLab。最新 Edge Policy 是 4B、仓库约 9.17 GB，比 16B Nano 更适合边缘部署；但 NVIDIA 模型卡没有把 Ada/RTX 4090 列入已测试微架构，因此“Edge 服务端 + Isaac Sim 共用同一张 24 GB 卡”只能作为实验路径，不能视为官方保证。

## 3. 固定本次可复现版本

本文验证的官方提交为：

```text
NVLabs/RoboLab main:
0aef241fb088ca21bb4ebd24448940ed56620d17

NVIDIA/cosmos-framework main:
e7420de81ce87fdf6ce4604a9a0e16e070ebf1f5

nvidia/Cosmos3-Edge-Policy-DROID:
3ea407af3e156c0af3b4bb6edd85842cc9a58777
```

先用这些提交跑通。后续若升级，只升级一个组件并重新执行测试，避免 Isaac Sim 物理版本变化与代码变化混在一起。

推荐目录布局：

```text
/root/robolab/
├── doc/                 # 本文档
├── RoboLab/             # 官方 RoboLab 仓库
│   └── Cosmos3-Edge-Policy-DROID/ # 唯一的本地权重目录
├── cosmos-framework-edge/ # 官方干净源码，策略服务器
└── cosmos-edge-overlay/ # OpenPI 轻量依赖
```

## 4. 第零阶段：修复并验证 GPU

首次配置时，必须先保证宿主机 GPU 驱动正常。若 `nvidia-smi` 失败，或者没有 `/dev/nvidia*`，先重启，让内核模块和设备节点完整重建：

```bash
sudo reboot
```

重连机器后检查：

```bash
nvidia-smi
ls -l /dev/nvidia*
lsmod | grep '^nvidia'
cat /proc/driver/nvidia/version
```

合格标准：

- `nvidia-smi` 能显示 RTX 4090、驱动版本和显存。
- 至少存在 `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm`。

若重启后仍失败，先收集诊断，不要继续装 RoboLab：

```bash
journalctl -k -b | grep -iE 'nvrm|nvidia|xid' | tail -n 100
dmesg -T | grep -iE 'nvrm|nvidia|xid' | tail -n 100
dkms status
dpkg -l | grep -E 'nvidia-driver|libnvidia-compute|nvidia-container'
```

先解决日志中具体的驱动加载、DKMS、Secure Boot 或设备节点问题，再进行下一步。不要在原因未确认时同时升级驱动、CUDA 和内核。

本机最初安装 `580.173.02` 后未重启，出现：

```text
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 580.173
```

这是新用户态库与仍在运行的旧内核模块不一致。重启后，以下三个版本已统一为 `580.173.02`：

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
cat /proc/driver/nvidia/version
modinfo -F version nvidia
```

不要仅凭 `nvidia-smi` 的 `CUDA Version: 13.0` 判断本机安装了完整 CUDA Toolkit；它表示驱动可支持的最高 CUDA 运行时版本。当前 `R580.173.02 + Isaac Sim 5.0` 已经实测可用，不需要再切换驱动。

## 5. 推荐方案：原生 uv 环境安装 RoboLab

在当前仅剩约 32 GB 空间的机器上，继续优先使用原生 `uv` 环境；不要直接再构建约 42 GB 的 Docker 镜像。

### 5.1 安装基础工具和 Git LFS

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs ffmpeg
git lfs install
git lfs version
```

### 5.2 克隆官方仓库并拉取大文件

```bash
mkdir -p /root/robolab
cd /root/robolab

git clone https://github.com/NVLabs/RoboLab.git RoboLab
cd RoboLab
git checkout 0aef241fb088ca21bb4ebd24448940ed56620d17
git lfs pull
```

确认提交和 LFS：

```bash
git rev-parse HEAD
git lfs ls-files | head
du -sh assets examples
git status --short
```

如果 `assets/` 下的 `.usd`、`.png` 或 `examples/` 下的 `.hdf5` 文件只有一百多字节，并且内容以 `version https://git-lfs.github.com/spec/v1` 开头，说明拿到的是 LFS 指针而不是真实数据。重新执行：

```bash
git lfs install
git lfs pull
```

### 5.3 创建 Python 3.11 环境

默认选择官方主路径 `isaac50`：

```bash
cd /root/robolab/RoboLab
uv venv --python 3.11
source .venv/bin/activate
uv sync --extra isaac50
```

设置 EULA：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

建议加入当前用户的 shell 配置，避免每次忘记：

```bash
echo 'export OMNI_KIT_ACCEPT_EULA=Y' >> ~/.bashrc
```

不要在同一个 `.venv` 中再执行 `uv sync --extra isaac51`。如确实要测试 Isaac Sim 5.1，单独创建：

```bash
cd /root/robolab/RoboLab
UV_PROJECT_ENVIRONMENT=.venv-51 uv sync --extra isaac51
```

## 6. 纯仿真验证流程

所有命令都在 `/root/robolab/RoboLab` 执行。以下命令使用 `uv run`，因此即使忘记激活 venv 也能使用正确环境。

### 6.1 官方完整安装测试

```bash
cd /root/robolab/RoboLab
export OMNI_KIT_ACCEPT_EULA=Y
uv run pytest tests/
```

该测试会检查 Isaac Lab 导入、任务定义、环境注册和完整 episode。首次启动 Isaac Sim 会比较慢。

### 6.2 最小空动作冒烟

先只跑一个任务、一个环境、50 步：

```bash
uv run python examples/run_empty.py \
  --task BananaInBowlTask \
  --num_envs 1 \
  --num-steps 50 \
  --headless
```

在已经完成 `uv sync` 的稳定环境中，也可直接调用 venv，避免 `uv run` 在启动前重新解析或同步依赖：

```bash
OMNI_KIT_ACCEPT_EULA=Y .venv/bin/python examples/run_empty.py \
  --task BananaInBowlTask \
  --num_envs 1 \
  --num-steps 50 \
  --headless
```

输出默认位于：

```text
/root/robolab/RoboLab/output/run_empty_env/
```

检查结果：

```bash
find output/run_empty_env -maxdepth 3 -type f | sort | head -n 50
```

#### 6.2.1 本机 2026-07-28 的实测结果

切换到 `R580.173.02` 并重启后，原生 `isaac50` 环境已通过：

| 检查项 | 实测结果 |
|---|---|
| `nvidia-smi` / NVML | 通过，驱动 `580.173.02` |
| CUDA PhysX device 0 | 创建成功 |
| RTX `SceneRenderer` | 创建成功 |
| 三路策略相机 | 创建成功 |
| `run_empty.py` 50 步 | 50/50，进程 exit 0 |
| Isaac Sim 关闭 | 正常 |

本次命令：

```bash
cd /root/robolab/RoboLab
OMNI_KIT_ACCEPT_EULA=Y .venv/bin/python examples/run_empty.py \
  --task BananaInBowlTask \
  --num_envs 1 \
  --num-steps 50 \
  --headless
```

本次 Kit 日志：

```text
/root/robolab/RoboLab/.venv/lib/python3.11/site-packages/omni/logs/Kit/Isaac-Sim/5.0/kit_20260728_193604.log
```

本次输出：

```text
/root/robolab/RoboLab/output/run_empty_env/
```

`run_empty` 没有策略，最终 `success=false` 或 score 0 是预期的任务结果，不代表仿真启动失败。判断冒烟成功应看进程 exit 0、步数完成、结果文件生成和 Isaac Sim 正常关闭。

仍会看到一些非致命警告，例如 headless GLFW/display、Warp `cuDeviceGetUuid`、低分辨率 DLSS、旧 actuator 配置以及个别 USD reference 警告；本次它们均未阻止场景、相机、物理或输出生成。只有出现 `SIGSEGV`、`CUDA error`、`Xid`、Python traceback 或非零退出码时才按故障处理。

### 6.3 回放官方 HDF5 演示数据

这一步验证场景、机器人、控制器、HDF5 输入和物理回放链路：

```bash
uv run python examples/run_recorded.py \
  --task RubiksCubeAndBananaTask \
  --num_envs 1 \
  --headless
```

默认输入目录是：

```text
examples/demo/recorded_data/RubiksCubeAndBananaTask/data.hdf5
```

默认输出目录名称以 `output/playback_` 开头。查看：

```bash
find output -maxdepth 3 -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.mp4' -o -name '*.hdf5' \) \
  | sort
```

注意：这里的 HDF5 是 RoboLab 官方录制轨迹，用于在仿真中恢复初始状态并开环回放，不是直接送入 Cosmos 3 策略服务器的输入。

### 6.4 验证夹爪动作

```bash
uv run python examples/run_gripper_toggle.py \
  --task BananaInBowlTask \
  --headless
```

输出位于 `output/run_gripper_toggle/`。该步骤能把“环境可启动”和“机器人动作控制链正常”区分开。

### 6.5 有桌面时启动 GUI

去掉 `--headless`：

```bash
uv run python examples/run_empty.py \
  --task BananaInBowlTask \
  --num_envs 1 \
  --num-steps 200
```

远程服务器通常没有可用显示服务，应继续使用 headless，并通过结果视频和 Dashboard 查看。

## 7. 查看实验结果

启动官方 Dashboard，并显式指定只监听本机：

```bash
cd /root/robolab/RoboLab
uv run robolab-dashboard \
  --host 127.0.0.1 \
  --port 8080 \
  --output-dir /root/robolab/RoboLab/output
```

从自己的电脑建立 SSH 端口转发：

```bash
ssh -L 8080:127.0.0.1:8080 <user>@<server>
```

浏览器打开：

```text
http://127.0.0.1:8080
```

## 8. 可选方案：官方 Docker

### 8.1 Docker 不能替换宿主机驱动

没有“自带合适 NVIDIA 内核驱动”的 RoboLab/Isaac Lab 普通 Docker 镜像。NVIDIA Container Toolkit 会把宿主机 GPU 设备和驱动能力提供给容器；容器可以携带不同版本的 Isaac Sim、Kit、CUDA 用户态库，但仍然使用宿主机加载的 NVIDIA 内核驱动。

所以：

- `nvcr.io/nvidia/isaac-lab:2.2.0` 是 Isaac Lab 2.2.0 / Isaac Sim 5.0，与本机已通过的原生环境基本同栈。
- `nvcr.io/nvidia/isaac-lab:2.3.0` 是 Isaac Lab 2.3.x / Isaac Sim 5.1，只在需要做版本对照时使用。
- 当前 `R580.173.02` 已跑通原生环境，没有必要为了修复崩溃再构建 RoboLab Docker。

本机已于 2026-07-28 使用 `docker manifest inspect` 确认 `2.2.0` 和 `2.3.0` 标签均可从 NGC 获取。二者基础镜像压缩层合计约 7.9 GiB，但 RoboLab 官方构建结果约 42 GB，还会需要解压、构建层和缓存空间。

### 8.2 当前本机不应立即构建

当前现场状态：

```text
根分区可用空间：约 32 GB
Docker NVIDIA runtime：已安装
现有 Docker 镜像：约 50.6 GB
现有 build cache：约 40.9 GB
```

32 GB 可用空间不足以安全构建完整 RoboLab 镜像。建议先把可用空间提升到至少 70 GB；不要直接执行 `docker system prune -a`，因为当前存在活动容器和用户镜像，应先逐项确认哪些镜像、容器和构建缓存可以删除。

只读检查：

```bash
docker info
docker info --format '{{json .Runtimes}}'
docker system df
docker system df -v
docker ps --size
docker image ls
```

如果提示无权访问 `/var/run/docker.sock`，先检查：

```bash
ls -l /var/run/docker.sock
systemctl status docker --no-pager
```

不要使用 `chmod 777 /var/run/docker.sock`。应修复 Docker 服务的 socket group，或在普通用户场景下把用户加入 `docker` 组后重新登录。

### 8.3 先验证官方标签和 GPU 透传

```bash
docker manifest inspect nvcr.io/nvidia/isaac-lab:2.2.0 >/dev/null
docker manifest inspect nvcr.io/nvidia/isaac-lab:2.3.0 >/dev/null

docker run --rm --gpus all \
  --entrypoint nvidia-smi \
  nvcr.io/nvidia/isaac-lab:2.3.0
```

容器内 `nvidia-smi` 显示的驱动仍应是宿主机的 `580.173.02`，这是正常现象，也证明 Docker 没有替换宿主机驱动。

### 8.4 推荐的 Isaac Sim 5.1 容器对照

释放足够磁盘空间后，使用 RoboLab 已提供的 `--isaac51` 构建路径：

```bash
cd /root/robolab/RoboLab
./docker/build_docker.sh robolab-0aef241-isaac51 --isaac51
```

直接运行 headless 冒烟，不需要 X11 或 `xhost`：

```bash
docker run --rm \
  --gpus all \
  --network=host \
  --entrypoint /workspace/isaaclab/_isaac_sim/python.sh \
  -e ACCEPT_EULA=Y \
  -e OMNI_KIT_ACCEPT_EULA=Y \
  robolab:robolab-0aef241-isaac51 \
  examples/run_empty.py \
  --task BananaInBowlTask \
  --num_envs 1 \
  --num-steps 50 \
  --headless
```

判断：

- 成功：5.1 容器也可作为隔离的对照环境，但不要与原生 5.0 的结果混在同一组实验中。
- 报 CUDA/driver API 不兼容：先检查宿主机驱动和 NVIDIA Container Toolkit，不要在容器内安装 `nvidia-driver-*`。
- 出现渲染或物理差异：分别记录 Isaac Sim、Isaac Lab 和 RoboLab commit；不要直接归因于策略。

## 9. 部署官方 Cosmos3-Edge-Policy-DROID

Edge Policy 是当前推荐路径：

- 模型：`nvidia/Cosmos3-Edge-Policy-DROID`
- 参数量：4B；官方仓库 34 个文件，dry-run 合计约 9.2 GB
- 精度：官方仅测试 BF16
- DROID 动作：8 维
- RoboLab WebSocket：默认 `0.0.0.0:8000`
- 当前官方 PyTorch RoboLab server 默认输出 `32 × 8` 动作块、4 步 UniPC、guidance 3.0、conditioning FPS 15

本机 4090 是 Ada。模型卡列出的已测试微架构是 Ampere、Hopper 和 Blackwell，并未列出 Ada。因此可以试跑，但不要把同卡显存可容纳和延迟当作官方保证。若同卡启动 RoboLab 时 OOM，保留 4090 跑仿真，把 Edge server 移到另一张 GPU 或另一台机器。

### 9.1 本机推荐：复用现有 cu130 venv，但隔离源码

本机已有：

```text
/root/cosmos3/cosmos/packages/cosmos3/.venv
Python 3.13.14
torch 2.10.0+cu130
transformer-engine 2.12+cu130.torch210
CUDA 可用，RTX 4090 capability 8.9
```

这个 venv 可以复用，但对应的 `/root/cosmos3/cosmos/packages/cosmos3` 工作树是自定义提交，带有 KV cache/offload 改动，且旧 server 脚本没有 Edge 的 JSON prompt 支持。不要在该脏工作树中直接 `git pull` 或 `uv sync`。

本机已经按以下隔离布局准备好：

```text
/root/robolab/cosmos-framework-edge/  # 官方干净源码 e7420de
/root/robolab/cosmos-edge-overlay/    # openpi-server/client，约 188 KB
/root/cosmos3/cosmos/packages/cosmos3/.venv/  # 复用的 12 GB cu130 环境
```

验证组合环境：

```bash
export COSMOS_EDGE_SRC=/root/robolab/cosmos-framework-edge
export COSMOS_EDGE_OVERLAY=/root/robolab/cosmos-edge-overlay
export COSMOS_EDGE_PYTHON=/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python
export PYTHONPATH="$COSMOS_EDGE_OVERLAY:$COSMOS_EDGE_SRC"

"$COSMOS_EDGE_PYTHON" -c \
  "from cosmos_framework.scripts.action_policy_server_robolab import _load_openpi_websocket_policy_server; print(_load_openpi_websocket_policy_server())"
git -C "$COSMOS_EDGE_SRC" rev-parse HEAD
```

预期分别看到 `WebsocketPolicyServer` 和：

```text
e7420de81ce87fdf6ce4604a9a0e16e070ebf1f5
```

如果需要从零重建这两个隔离目录：

```bash
cd /root/robolab
git clone --filter=blob:none --depth 1 \
  https://github.com/NVIDIA/cosmos-framework.git \
  cosmos-framework-edge

uv pip install \
  --target /root/robolab/cosmos-edge-overlay \
  --no-deps \
  openpi-server==0.1.0 \
  openpi-client==0.1.2
```

这里使用 `--no-deps`，因为复用 venv 已有 `dm-tree`、`msgpack`、NumPy、Pillow 和 websockets。

### 9.2 使用已经下载好的本地权重

本机已经把完整 HF snapshot 物化到唯一的本地目录：

```text
/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID
```

原 `/root/robolab/hf-cache` 和不完整的 local-dir 已删除，当前只保留这一份约 8.6 GiB 的权重。四个主要文件已经通过 safetensors header、权重索引和官方 server checkpoint 校验：

```text
transformer/diffusion_pytorch_model-00001-of-00002.safetensors  5,000,041,944
transformer/diffusion_pytorch_model-00002-of-00002.safetensors  1,748,779,008
vae/diffusion_pytorch_model.safetensors                         1,409,400,600
vision_encoder/model.safetensors                                  978,739,880
```

Edge policy 主权重直接读取本地目录，不需要重新执行 `hf download`，也不需要传 `--hf-revision`。但配置中的 Wan2.2 tokenizer VAE 仍通过 Hugging Face checkpoint registry 解析，因此要把已有的公共缓存根目录设为 `HF_HOME`：

```bash
export COSMOS_EDGE_CKPT=/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID
export HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home

test -f "$COSMOS_EDGE_CKPT/checkpoint.json"
test -f "$COSMOS_EDGE_CKPT/transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
test -f "$COSMOS_EDGE_CKPT/transformer/diffusion_pytorch_model-00002-of-00002.safetensors"
test -f "$COSMOS_EDGE_CKPT/vae/diffusion_pytorch_model.safetensors"
test -f "$COSMOS_EDGE_CKPT/vision_encoder/model.safetensors"
test -f "$HF_HOME/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
du -sh "$COSMOS_EDGE_CKPT"
```

注意 `HF_HOME` 要指向 `hf_home`，不要指向末级的 `hf_home/hub`；Hugging Face 客户端会自动在其下使用 `hub/`。

### 9.3 启动 Edge policy server

终端 A：

```bash
cd /root/robolab/cosmos-framework-edge

export COSMOS_EDGE_CKPT=/root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID
export HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home
export PYTHONPATH=/root/robolab/cosmos-edge-overlay:/root/robolab/cosmos-framework-edge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/root/cosmos3/cosmos/packages/cosmos3/.venv/bin/python \
  -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path "$COSMOS_EDGE_CKPT" \
  --format-prompt-as-json True \
  --no-guardrails \
  --host 0.0.0.0 \
  --port 8000
```

注意：

- 参数是 `--checkpoint-path`，不是 Python 字段名形式的 `--checkpoint_path`。
- `--no-guardrails` 关闭文本和视频安全过滤，适合本地封闭仿真实验，并避免下载需要审批的 `nvidia/Cosmos-Guardrail1`。
- 不要用 `--offload-guardrail-models` 替代；offload 只改变模型驻留位置，仍然会下载和启用 guardrails。
- `HF_HOME` 会让 Wan2.2 VAE 解析命中现有缓存；本机已用 `HF_HUB_OFFLINE=1` 验证解析到 revision `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`，没有重新下载。
- 本机已用当前源码执行 `--help` 和单元测试验证这些参数。

若希望严格禁止缺失文件时访问网络，可在启动前额外设置：

```bash
export HF_HUB_OFFLINE=1
```

设置后，缺失的依赖会直接报错而不会下载。主模型从 `COSMOS_EDGE_CKPT` 读取，Wan2.2 VAE 则从 `HF_HOME` 缓存读取。首次加载 BF16 模型时另开终端监控：

```bash
watch -n 1 nvidia-smi
```

服务完成加载后检查：

```bash
curl -fsS http://127.0.0.1:8000/healthz
ss -ltnp | grep ':8000'
```

若服务端和 RoboLab 分处两台机器，只放通可信来源到 TCP 8000；该服务没有面向公网的鉴权层。

### 9.4 启动 RoboLab 闭环客户端

本机 RoboLab venv 已安装并验证：

```text
openpi-client 0.1.2
Cosmos3Client.OPEN_LOOP_HORIZON = 32
```

终端 B 先跑一个任务、一个环境，并关闭视频以降低额外开销：

```bash
cd /root/robolab/RoboLab
export OMNI_KIT_ACCEPT_EULA=Y

.venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 \
  --remote-port 8000 \
  --task BananaInBowlTask \
  --num-envs 1 \
  --num-runs 1 \
  --output-folder-name cosmos3_edge_banana \
  --video-mode all \
  --headless
```

两台机器部署时，把 `127.0.0.1` 改成策略服务器 IP。确认单环境稳定后再增加 `--num-envs`，不要第一次就照官方示例开 10 个环境。

同卡 4090 的启动顺序建议是：

1. 先启动 Edge server，等待模型完全加载并记录稳定显存。
2. 再启动 RoboLab 单环境。
3. 若 OOM，停止 RoboLab；不要降低模型到未经官方验证的 FP16/FP8，改为分卡部署。
4. 服务和仿真都稳定后，再把 `--video-mode none` 改为 `sensor` 或 `all` 保存实验视频。

### 9.5 当前已经验证和仍待完成的边界

已经验证：

- R580.173.02、Isaac Sim 5.0、Isaac Lab 2.2.0 和 `run_empty`。
- RoboLab `openpi-client` 导入和 Cosmos3 客户端 CLI。
- 现有 cu130 venv 可加载最新官方 Edge server 源码。
- 隔离 overlay 可加载 `WebsocketPolicyServer`。
- 完整权重已合并到唯一的本地目录，HF cache 和部分下载目录已删除。
- 本地 checkpoint 已通过官方 server `_validate_checkpoint()` 和 safetensors/index 校验。

尚未完成：

- Edge BF16 模型在 RTX 4090 上的完整加载。
- Edge server 与 Isaac Sim 同卡闭环 episode。

当前根盘剩余约 32 GB。按 9.3–9.4 的命令即可直接从本地路径继续，不再下载权重。

## 10. HDF5、权重和仿真数据之间的关系

不要混淆三类数据：

| 类型 | 来源 | 用途 |
|---|---|---|
| RoboLab 资产 | RoboLab Git LFS | USD 场景、物体、纹理、机器人仿真 |
| RoboLab HDF5 | `examples/demo/recorded_data/` 或实验输出 | 恢复仿真状态、开环回放、结果分析 |
| Cosmos3 Policy 权重 | Hugging Face `nvidia/Cosmos3-Edge-Policy-DROID` | 在线接收 RoboLab observation，生成动作 |

官方闭环推理不是“把一个 HDF5 文件直接喂给 Edge Policy”。正常链路是：

```text
仿真当前状态
  -> 三路 RGB 相机 + 7 维关节 + 1 维夹爪 + 指令
  -> WebSocket 策略服务
  -> [32, 8] 动作块
  -> RoboLab 执行动作
  -> 记录结果/HDF5/视频
```

HDF5 回放用于先验证仿真系统；策略闭环则由 RoboLab 客户端实时构造 observation。

## 11. 推荐实验顺序

每一步成功后再进入下一步：

1. `nvidia-smi` 和 `/dev/nvidia*` 正常。
2. `git lfs pull` 后资产和 HDF5 是真实大文件。
3. `uv run pytest tests/` 通过。
4. `run_empty.py` 单任务、单环境通过。
5. `run_recorded.py` 能回放官方 HDF5。
6. `run_gripper_toggle.py` 能生成结果。
7. Dashboard 能查看日志和视频。
8. 启动 Edge Policy server；同卡 4090 仅按实验路径验证。
9. RoboLab 单环境连接策略服务器。
10. 最后增加任务数、episode 数和并行环境数。

建议每次实验记录：

```text
RoboLab commit
Isaac Sim / Isaac Lab 版本
GPU 与驱动版本
任务名
num_envs
随机种子
策略 checkpoint revision
是否 headless
输出目录
成功率、score、运行耗时
```

Isaac Sim 5.0 与 5.1 的 PhysX 不同，接触、抓取和物体沉降结果不能假定完全一致。不同实验应固定相同仿真栈。

## 12. 常见问题

### `nvidia-smi` 失败

先停在第 4 节。Isaac Sim 和策略服务都依赖 CUDA，继续安装不能绕过驱动问题。

### `git: 'lfs' is not a git command`

```bash
sudo apt-get install -y git-lfs
git lfs install
cd /root/robolab/RoboLab
git lfs pull
```

### USD、PNG、HDF5 文件很小或解析失败

这是 Git LFS 指针未下载。执行 `git lfs pull`，并检查网络和 GitHub LFS 配额/访问。

### Isaac Sim 首次启动卡住

确认已设置：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

首次运行还会构建 shader/cache，明显慢于后续启动。

### CUDA OOM

先保持：

```bash
--num_envs 1
```

关闭同卡上的其他 CUDA 进程：

```bash
nvidia-smi
```

先使用 `--num-envs 1 --video-mode none`。如果 Edge server 与 RoboLab 共用单张 24 GB 4090 时 OOM，改为分卡/分机部署；不要擅自切换到模型卡未验证的 FP16、FP8 或 FP4。

### `Cosmos-Guardrail1` 提示 `Access denied`

这是策略服务默认启用 guardrails 后访问受审批模型导致的，与本地 Edge Policy 权重无关。对于本地封闭仿真实验，在 server 命令中加入：

```bash
--no-guardrails
```

当前 RoboLab server 源码已补充该开关，并会把 `guardrails=False` 传给 `OmniSetupOverrides`。异常退出后出现的 `destroy_process_group() was not called` 是前述加载错误触发的次生清理警告，不是独立的 NCCL 故障。

### 回放结果与官方演示不完全一致

检查 RoboLab commit、Isaac Sim/Isaac Lab 版本、环境配置和 `num_envs`。官方说明接触物理会随仿真器版本变化，录制与回放最好使用同一仿真栈且单环境运行。

### Docker 无权访问 socket

```bash
ls -l /var/run/docker.sock
id
getent group docker
systemctl status docker --no-pager
```

修复 Docker 服务或用户组设置，不要把 socket 改成全局可写。

## 13. 官方资料

- RoboLab 仓库与安装说明：<https://github.com/NVLabs/RoboLab>
- RoboLab 文档：<https://github.com/NVLabs/RoboLab/tree/main/docs>
- RoboLab Cosmos 3 客户端：<https://github.com/NVLabs/RoboLab/tree/main/policies/cosmos3>
- Cosmos Framework：<https://github.com/NVIDIA/cosmos-framework>
- 官方 RoboLab Policy Server：<https://github.com/NVIDIA/cosmos-framework/blob/main/cosmos_framework/scripts/action_policy_server_robolab.py>
- Cosmos Framework DROID Policy Server 手册：<https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_server.md>
- Edge Policy DROID 模型卡：<https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID>
- Isaac Sim 5.0 系统与驱动要求：<https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html>
- Omniverse 已验证驱动版本：<https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html>
- NVIDIA Container Toolkit 架构：<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/arch-overview.html>
