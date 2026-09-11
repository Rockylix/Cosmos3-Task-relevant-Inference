# Edge C³ache 首版

工作树：`/project/peilab/xiexinling/.worktrees/edge-experiment-c3ache`

分支：`experiment/c3ache`

基线：`cosmos-framework-edge-baseline` 的 `baseline`，提交
`324574a454a989f9b8f5392f7673ace487243e8d`。所有本次代码均在此 worktree；
原 baseline 和共享 RoboLab 源码不需要修改。

依据 [C³ache 方法](https://arxiv.org/html/2606.08962v1) 实现 Edge 移植：
同一个 episode、CFG 分支和去噪步，保存整个 GEN Transformer 栈的
`R = h_L - h_0`，命中时使用当前 embedding 的 `h_0 + R`。
边界位于 final norm **之前**；GEN 中视频、动作和条件位置全部包含。
命中仍执行当前观测/动作 embedding、位置编码、final norm、输出头、
CFG 混合、条件 mask 和原采样更新。不会复用上一轮的输出动作或噪声。

## 开关与调度

默认 `--no-c3ache`，保持 dense 行为。开启时限定**单 GPU / 单 rank、4 步 UniPC**。
默认刷新周期为 2、末尾 dense 步数为 2。零起始 chunk 0 总是完整计算；
正常连续请求在 `chunk_id % refresh_period == 0` 时刷新。

本轮 RoboLab1200 固定配置见 [configs/c3ache_baseline.json](configs/c3ache_baseline.json)：
10 个并行环境时设置 `--c3ache-max-sessions 10`。该 JSON 用于记录参数，
服务端不自动读取它，启动时须显式传入相同 CLI 参数。

| 对照 / 实验 | 参数 | 刷新 chunk | 其他 chunk |
| --- | --- | --- | --- |
| 原 baseline | `--no-c3ache` | D D D D | D D D D |
| 同执行模式 dense 对照 | `--c3ache --c3ache-refresh-period 1` | D D D D | D D D D |
| 首选首轮 | `--c3ache --c3ache-refresh-period 2` | D D D D | C C D D |
| 较长周期 | `--c3ache --c3ache-refresh-period 4` | D D D D | C C D D |
| 扩展配置 | `--c3ache --c3ache-dense-tail-steps 1 --c3ache-refresh-period 4` | D D D D | C C C D |

周期支持 1/2/4/8 等正整数。首版启用缓存时关闭 `torch.compile` 和 CUDA graphs，
因此速度比较应至少包含 **period=1 的 eager dense 对照**，同时报告原 baseline。
不能把与 compiled baseline 的差值直接归因为残差缓存本身。

服务端入口仍为 `cosmos_framework.scripts.action_policy_server_robolab`。
在已获分配的计算节点，复用 baseline 的 Cosmos 环境/依赖和权重，改变源码路径并加开关：

```bash
ROOT=/project/peilab/xiexinling
WT="$ROOT/.worktrees/edge-experiment-c3ache"
VAE="$ROOT/assets/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
# COSMOS_PYTHON 必须指向当前作业账户可访问的、与 baseline 依赖一致的解释器。
export PYTHONPATH="$WT"
export LD_LIBRARY_PATH=''
"$COSMOS_PYTHON" -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path "$ROOT/assets/Cosmos3-Edge-Policy-DROID" \
  --sampler unipc --num-steps 4 --guidance 3 --shift 5 \
  --seed 0 --no-deterministic-seed --port 8000 \
  --format-prompt-as-json True --no-guardrails \
  --c3ache --c3ache-refresh-period 2 --c3ache-dense-tail-steps 2 \
  --c3ache-max-sessions 10 \
  --experiment-overrides "model.config.tokenizer.vae_path=$VAE" \
  model.config.tokenizer.object_store_credential_path_pretrained= \
  model.config.tokenizer.bucket_name=
```

这是计算节点运行示例。当前 HPC 实验使用 `envs/cosmos_shared_py313/bin/python`，
Python 3.13.15、Torch 2.10.0+cu128，复用原 Cosmos site-packages。
不要使用仍指向私有目录的旧 `envs/cosmos/bin/python` 链接。
轻量 CPU 测试使用现有可访问的 `envs/dreamzero_scan_py311/bin/python`。

## RoboLab 客户端

配套客户端在 `integrations/robolab/c3ache_client.py`，继承共享 baseline 客户端的
图像处理、动作转换和动作 chunk 播放逻辑；每个环境单独标记：

```text
session_id = 客户端 UUID + 环境编号
episode_id = 每次 reset 后新的 UUID
chunk_id   = 当前 episode 内的规划序号，从 0 开始
```

`reset(env_id=...)` 会发送 `c3ache_control="reset"` 请求并清空本地状态。
若 reset 消息丢失，新的 episode UUID 仍确保隔离。任务指令改变时也立即 reset。
重连重试保留原请求 ID；服务端遇到重复或跳号的 chunk 会丢弃旧残差并完整计算。
reset 不消耗生成随机数。普通请求仍沿用 baseline 的 `_next_seed()` 和观测处理。

`integrations/robolab/run.py` 是 RoboLab `policies/cosmos3/run.py` 的配套入口，
保留原 evaluator 和参数，仅替换客户端 import（以及函数说明/必要 lint 注释）。
在已分配的仿真节点/作业内使用：

```bash
ROOT=/project/peilab/xiexinling
WT="$ROOT/.worktrees/edge-experiment-c3ache"
cd "$ROOT/RoboLab"
PYTHONPATH="$WT/integrations/robolab:$ROOT/RoboLab" \
  "$ROOT/envs/robolab/bin/python" "$WT/integrations/robolab/run.py" \
  --remote-host SERVER_HOST --remote-port 8000 --task TASK_NAME \
  --num-envs 10 --num-runs 1 --video-mode none --headless
```

缓存开启时拒绝没有标识的旧客户端请求，避免静默串用。缓存关闭时仍接受旧客户端，
也接受配套客户端的 reset 控制消息。默认最多保留 4 个 session/episode；
用 `--c3ache-max-sessions N` 匹配并行环境数量，避免 LRU 淘汰降低命中率。

## 正确性与观测

- 按 `(session_id, episode_id)` 隔离，内部按 `(CFG branch, denoising step)` 存储。
- 精确核对 timestep、dtype、device、token 角色/顺序、几何、条件 mask、文本、
  position IDs、域和采样配置。新 episode、任务/配置改变、布局变化和不连续
  chunk 会重建缓存；失败请求不提交部分缓存。
- baseline 的文本 KV 仍限于单次请求。缓存命中的早期步骤可以保持它为空，
  后续 dense 步按原逻辑初始化；不跨 chunk 复用文本 KV。
- 当前仅用于 action/video policy，拒绝文本预测、音频和 TaylorSeer 组合；
  也不与已有 hidden-state/block-residual/RoPE capture 模式叠加。
- 每个响应包含 `c3ache`：`dense_forwards`、`cache_hits`、`cached_bytes`、
  `chunk_id`、`reason`；服务器打印相同统计。计数单位是一次 CFG 分支前向，
  默认 guidance=3 时，刷新 chunk 为 8 dense / 0 hit，复用 chunk 为 4 dense / 4 hit。

CPU 测试直接执行实际修改的 Transformer 栈、采样 velocity closure 和协议入口，
用小型替代层/传输验证边界和状态流；不加载权重，不运行仿真或 CUDA。

```bash
ROOT=/project/peilab/xiexinling
WT="$ROOT/.worktrees/edge-experiment-c3ache"
timeout 180 env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$ROOT/envs/dreamzero_scan_py311/bin/python" -m unittest discover \
  -s "$WT/.sisyphus/test" -p 'test_c3ache*.py' -v
```

2026-09-11：18 项 CPU 测试通过。H800 上已使用真实 Edge 权重完成 RoboLab
闭环批次，验证了跨 chunk 缓存命中。前两个完整任务的日志共 1000 次规划请求：
20 次新 episode、490 次周期刷新、490 次复用；累计 6040 次 dense 分支前向、
1960 次缓存命中，符合刷新 chunk 8/0、复用 chunk 4/4 的调度。

RoboLab1200 评测仍在进行中，不发布最终成功率或配对加速结论。
真实权重的 period=1 与关闭缓存的严格数值对照、残差误差和同输入配对性能测试
仍待完成。CPU 替代层数值一致性不能视为完整模型等价证明。

主要落点：`cosmos_framework/inference/c3ache.py` 管理缓存；
`model/generator/omni_mot_model.py` 显式标记去噪步和分支；
`model/generator/mot/cosmos3_vfm_network.py` 记录输入布局；
`model/generator/mot/unified_mot.py` 跳过整段层循环并保留 final norm；
`scripts/action_policy_server_robolab.py` 处理开关和请求生命周期。

保守实现的性能代价见 [modification.md](modification.md)。
