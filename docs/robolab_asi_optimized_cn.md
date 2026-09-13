# RoboLab ASI 真机优化接入与验证

日期：2026-09-13。仿真基线提交 `7d3c424`；导入真机优化 `abb8f7d`。
集成分支 `experiment/robolab-asi-optimized`。

## 计算约束

所有执行模式都使用同一数学策略：step0 conditional 的 28 层全量 Profile，
Top6 block quality、逐帧 Core80、共享 Stable104、每帧 K184、动作权重
`[1/6,1/3,1/3,1/6]`。剩余七次 Transformer forward 从 B0 至 B27
保持稀疏输入；L0、文本和全部 action 始终保留；未选 future hidden 恢复
当前 stack 的输入。CFG、UniPC、初始噪声、去噪步、shift 和输出头公式均不改。

这里不包含 ASI velocity cache、跨 chunk mask cache、人工 ROI 或新的 token 预算。

## 入口与开关

仍使用 `cosmos_framework.scripts.action_policy_server_robolab_version1`。

| `--asi-execution` | Controller | Compile | CUDA Graph 请求 |
|---|---|---|---|
| `legacy`（默认） | 原仿真 Controller | 关 | 关 |
| `optimized-eager` | 新 Controller，8 帧批量 Profile、metadata 缓存 | 关 | 关 |
| `compile` | 新 Controller，Profile 在 decoder 内返回 | 开 | 关 |
| `compile-graph` | 同上 | 开 | 开 |

保持默认 legacy，已有启动命令行为不变。推荐先用 optimized-eager 进行严格旧输出回归。
Compile 两种模式是显式选择：**公式相同不意味着 BF16 数值和 Top-K mask 逐位相同**。
不把单次 smoke 或高 action cosine 当作任务成功率不变的证据。

编译范围遵循现有模型配置：28 个 decoder 分别 `fullgraph=True` 编译，
并编译 encode/decode heads；不是把整个四步 UniPC 包进一个 CUDA Graph。
Graph 模式通过 Inductor `reduce-overhead` 请求捕获；存在分区提示，不能只凭
开关称所有算子都 replay。此次 Graph 模式未比 compile-only 提供额外可测收益。

## 相对真机提交的补充

1. 仿真 server 显式选择新 Controller 和编译开关，不再仅导入旧脚本。
2. 稀疏 metadata 按 `(真实 UND 长度, GEN 长度, device)` 缓存，允许 CFG
   两分支的实际文本长度不同。CUDA Graph padding 不进入真实 attention key 长度。
3. 保存 compiled Profile 时 clone，防止之后的图运行复用输出存储。
4. 编译模式与 eager hidden/block/RoPE 采集互斥，提前报错。
5. request 每次新建 Controller；不复用旧 request 的 mask、Profile 或隐藏状态。

token 索引和 RoPE 切片相同；RoPE 值仍按原位置取得，不重排 temporal ID。
`cache_layout` 只在同一 request 的同一 packed 对象上命中；这里没有额外移植
Piper 客户端的 `reuse_denoise_packing` monkey patch 或重复输入单帧化。
原生 sampler 自身已有的 packing/text-KV 优化保持原样。

## GPU 单 chunk 实测

RTX 4090 / Torch 2.10.0+cu130；项目本地 Edge venv。一个加载模型内切换五种
执行路径，相同 BananaInBowlTask 第三个 chunk 输入和 seed `1097657232`；
4 steps / guidance3 / shift5。每模式预热 4 次，随后正序/逆序交替测 10 次。
计时仅包 `generate_samples_from_batch`，前后 CUDA synchronize。
Controller summary、tensor CPU copies、指标计算、VAE decode、RPC、Isaac step 均不计时。

| 模式 | Mean (s) | Median (s) | P90 (s) | 对 Dense 加速 | 对旧 ASI 加速 |
|---|---:|---:|---:|---:|---:|
| Dense eager | 0.854009 | 0.854345 | 0.857308 | 1.000x | — |
| 旧 ASI eager | 0.583121 | 0.582345 | 0.586684 | 1.467x | 1.000x |
| optimized-eager | 0.551402 | 0.551195 | 0.553302 | 1.550x | 1.057x |
| compile | 0.463099 | 0.462451 | 0.465678 | 1.847x | 1.259x |
| compile-graph | 0.463823 | 0.463908 | 0.466473 | 1.842x | 1.255x |

该表不包括首次编译/capture，不代表整个闭环任务加速。显存峰值是所有模式在同一
进程测试的合计（allocated 8,837,456,896 bytes，reserved 9,479,127,040 bytes），
不是各模式独立显存需求，也不是与 Isaac 共存时的总量。

## 相对旧 ASI 的数值边界

Action 指标只使用真实发给 RoboLab 的 32 个预测 horizon：剔除历史 action、
按实际 action_dim 截断、执行 gripper 翻转。Vision 指标只用 L1–L8：
原 tensor shape `[1,48,9,33,40]`，取 `[:, :, 1:]`。
action 各维包含不同语义，max-abs 是模型输出单位，不能统一解释为角度或毫米。

| 模式 | Action MSE | Action max-abs | Future latent cosine | Future rel-L2 | mask 不同位置数 |
|---|---:|---:|---:|---:|---:|
| optimized-eager | 0 | 0 | 1.000000 | 0 | 0 |
| compile | 0.00215725 | 0.150194 | 0.961593 | 0.276209 | 52 |
| compile-graph | 0.000974120 | 0.177964 | 0.969478 | 0.246458 | 38 |

不同位置数为执行 mask XOR 总和，移出/移入各计一次；Top6 block 列表相同。
同 seed 重复运行输出与 mask 稳定；另测 seed+1，optimized-eager 仍保持输出/mask
完全一致，compiled 路径仍有数值偏移且没有返回旧 seed 的 latent。
批量 Profile 自身相对旧 Profile 最大差约 `3.73e-8`，本次没有改变 eager mask。

额外检查 `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` 和
`TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING=1` 未消除 compiled 输出差异，未设为生产默认。
目前只能确认编译路径和旧路径不是逐位等价；未完成逐算子归因，不能把全部偏移
都断言为某一个融合算子引起。需要严格旧输出一致时选 optimized-eager。

## 审查与测试

每轮均经过子 agent 只读审查。首轮发现异长 CFG metadata 和 Profile 生命周期问题；
第二轮复核修复，并要求剔除条件帧/历史动作后报告误差及禁止 compiled+eager capture。
CPU 28 tests 覆盖新旧选择一致、全 key LSE、GQA、动作加权、K184、L0/action 保留、
optimized pack/restore、CFG 分支不同长度、原始 RoPE、Profile 独立存储及 server 接线。
最终子 agent 审查结论：可合并默认 legacy、优化显式开启的集成；不能将 compiled
称为严格数值等价优化，不能由 smoke 推断任务成功率不变。

真实 RoboLab 接线 smoke 使用 `compile-graph`，simulator seed0、policy seed0（逐请求推进），
BananaInBowlTask 一次运行，共 24 个请求正常完成，无推理异常。结果为 **success=false，
score=1.0，750 steps**（官方任务时限，无额外评测截断）；任务最终没有满足香蕉留在碗中
且脱离夹爪的条件。score 不能代替 success；本次为 0/1，不能报告为成功。
没有重跑筛选，也没有用此单次结果声称编译与旧 ASI 的任务成功率相同。
证据：`experiments/optimized_server_smoke/simulator/episode_results.jsonl`。

```bash
cd /root/robolab/worktrees/robolab-asi-optimized
export EDGE_PY=/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python
export PYTHONPATH="$PWD" LD_LIBRARY_PATH='' COSMOS_TRAINING=0
"$EDGE_PY" -m pytest -q --num-gpus=0 \
  tests/test_robolab_version1.py tests/test_robolab_asi_server.py \
  cosmos_framework/inference/edge_core_stable_test.py
```

## 启动

复用项目现有权重，无需下载。VAE 变量可以指向项目已有 symlink 或实际文件。

```bash
cd /root/robolab/worktrees/robolab-asi-optimized
export EDGE_PY=/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python
export PYTHONPATH="$PWD" LD_LIBRARY_PATH='' COSMOS_TRAINING=0 CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export EDGE_VAE=/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth
"$EDGE_PY" -m cosmos_framework.scripts.action_policy_server_robolab_version1 \
  --asi-execution optimized-eager \
  --checkpoint-path /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID \
  --host 127.0.0.1 --port 8017 --no-guardrails \
  --seed 0 --num-steps 4 --shift 5 --guidance 3 \
  --output-dir experiments/optimized_server \
  --experiment-overrides "model.config.tokenizer.vae_path=$EDGE_VAE" \
  model.config.tokenizer.object_store_credential_path_pretrained= model.config.tokenizer.bucket_name=
```

使用 `compile` 或 `compile-graph` 替换 execution 值即可测试编译路径，注意上表数值偏移。
另一个终端启动仿真器：

```bash
cd /root/robolab/RoboLab
export PYTHONPATH="$PWD:/root/robolab/cosmos-edge-overlay"
export OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y PRIVACY_CONSENT=Y
.venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 --remote-port 8017 --task BananaInBowlTask \
  --num-envs 1 --num-runs 1 --headless --video-mode none \
  --output-folder-name /root/robolab/worktrees/robolab-asi-optimized/experiments/simulator_smoke
```

## 复现单 chunk 对比

在上面的 Edge 环境变量基础上执行。capture 是旧实验已保存的真实输入，
SHA256、prompt、seed、所有 timing samples 和指标写入 results.json。

```bash
"$EDGE_PY" tools/benchmark_robolab_asi_optimized.py \
  --capture /root/robolab/worktrees/toca-future/experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/dense/server/captures/request_000002/sample.pt \
  --checkpoint /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID --vae "$EDGE_VAE" \
  --output experiments/optimized_retest --warmup 4 --repeats 10
```

已测结果在本 worktree `experiments/optimized_final/results.json`。
完整输入/输出留在 Git ignored experiments，不提交原始数据。
真机 Piper14 的动作/末端位置正确性仍须在真机项目规定的容器内用录制输入复验；
本次 RoboLab/DROID 结果不等价于真机部署验证。
