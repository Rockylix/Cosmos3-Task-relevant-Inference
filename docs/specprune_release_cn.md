# SpecPrune-Future：冻结十任务策略与远端部署

本轮已完成的 [FLOPs、单chunk速度与Graph验证](specprune_compile_speed_cn.md)。

分支：`experiment/specprune-future`，远端：`git@github.com:Rockylix/Cosmos3-Task-relevant-Inference.git`。仅推送源码、配置、测试、文档；权重、环境、原始实验数据忽略。没有合并 main、ASI、ToCa 或 WorldCache。

## 1. 策略和复现边界

冻结 eager 原始提交 `d5c8fc6` 保存已测十任务版本。配置快照：[specprune_baseline.json](../configs/specprune_baseline.json)，与 `ExitConfig` 默认值由单元测试核对；它是描述性快照，不是服务器自动加载的超参数文件。

- 真实观测 L0 上做 Local/Global：L0 Q→instruction K，Local32，上一 chunk B13/B27 Global40；Dynamic 为相邻 chunk 真实 RGB patch 变化，阈值0.986、低变化最多313位置。
- step0 conditional 生成逐层嵌套计划。B0/B1 静态筛选，B14/19/24 用 action Q→L0 K 更新重要性，B10/15/20/25 按虚拟观测序列保留率0.9收缩。B10在首次重要性更新之前，零分并列按原位置ID稳定排序。动态下限60，静态已少于60时不补点。
- 各层 L1–L8 共用空间 mask，L0/text/action 全保留；真实缩短 attention/MLP 输入，原始 RoPE 坐标不变。
- 被删位置缓存本 forward 的退出层 hidden，最终 RMSNorm/llm2vae 前恢复完整位置；两支 CFG 和 UniPC 始终更新完整 latent。下一 step 从更新后的完整 latent 开始。不跨 step 缓存 hidden 或 velocity。
- 未启用原论文动作粗细控制器。观察域到 future 域共享 mask、虚拟预算与 early-exit 输出头是 WAM 适配，不能称完全不变的论文复现。

## 2. 已测成功率

seed0/seed0、shift5、guidance3、4步、每任务一次，8简单+2普通。两策略各十次，失败不重试。一次首请求的指标维度异常未返回动作，保留日志后经用户授权重启 Sparse。

| 策略 | success | 官方平均score | 原始JSONL平均score | 平均完成/终止steps |
|---|---:|---:|---:|---:|
| Dense eager | 4/10 | 0.4667 | 0.3667 | 347.8 |
| SpecPrune eager | 1/10 | 0.2667 | 0.2667 | 518.6 |

唯一稀疏成功任务 BananaOnPlate，536步。官方汇总将 success=True 的 score 记为1，与原始 score 独立字段不同。本轮没有评估编译版闭环成功率。不能把下面可选编译开关的结果称作上述十任务实测。

## 3. 获取源码和环境

```bash
git clone --branch experiment/specprune-future --single-branch \
  git@github.com:Rockylix/Cosmos3-Task-relevant-Inference.git cosmos-specprune
cd cosmos-specprune
```

已有环境与权重直接使用，不重新下载。完整项目环境和 RoboLab/Isaac/权重准备见同仓库 [迁移配置指南](robolab_setup_and_launch_cn.md)。该指南的历史五分支表不含本分支；本实验服务器用下面的新入口。环境均安装在项目目录中，不使用 Piper 真机容器。

```bash
export EDGE_SOURCE=/absolute/path/cosmos-specprune
export EDGE_PY=/absolute/path/project/edge-env/.venv/bin/python
export EDGE_CHECKPOINT=/absolute/path/Cosmos3-Edge-Policy-DROID
export EDGE_VAE=/absolute/path/Wan2.2_VAE.pth
export ROBOLAB_ROOT=/absolute/path/RoboLab
export PYTHONPATH="$EDGE_SOURCE"
export COSMOS_TRAINING=0 CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
test -x "$EDGE_PY"
test -f "$EDGE_VAE"
"$EDGE_PY" -c 'import torch; from openpi_server.websocket_policy_server import WebsocketPolicyServer; print(torch.__version__, torch.cuda.is_available())'
```

如果旧机器的 openpi-server 只在 overlay 中，额外将其目录加入 PYTHONPATH；新环境按指南安装项目的 policy-server 依赖组即可。

## 4. 启动服务

复现十任务请用 **eager**。新的入口直接返回稀疏 action，不运行同输入 Dense 对照、不保存逐 chunk 原始数组；评分、有限值检查及策略自身 CPU 诊断仍保留。

```bash
cd "$EDGE_SOURCE"
"$EDGE_PY" -m cosmos_framework.scripts.action_policy_server_specprune \
  --checkpoint-path "$EDGE_CHECKPOINT" \
  --host 127.0.0.1 --port 8000 \
  --specprune --specprune-backend eager \
  --seed 0 --no-deterministic-seed --num-steps 4 --shift 5 --guidance 3 \
  --no-guardrails --format-prompt-as-json True \
  --experiment-overrides "model.config.tokenizer.vae_path=$EDGE_VAE" \
  model.config.tokenizer.object_store_credential_path_pretrained= model.config.tokenizer.bucket_name=
```

Dense 对照：相同命令改 `--specprune` 为 `--no-specprune`。可选 `--specprune-backend graph` 仅编译非评分层并让 compiler 管理 CUDA Graph；评分回调层保持 eager，排序/索引/solver/head 保持原实现。不是整 chunk 单一 Global Graph；首次编译不计稳定延迟，动态 token 长度可能重新编译/捕获。是否加速及数值偏差以配套测速记录为准。

## 5. 十任务模拟器

在另一终端设好相同 ROBOLAB_ROOT 和 EDGE_SOURCE。以下一次只跑一个 env，每任务一个 rollout，默认仿真seed沿用已测路径为0。RoboLab版本变化时先核对 run.py/create_env 默认值，不把policy seed当simulation seed。

```bash
cd "$ROBOLAB_ROOT"
export OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PYTHONPATH="$ROBOLAB_ROOT"
.venv/bin/python policies/cosmos3/run.py \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --task BananaInBowlTask BananaOnPlateTask ButterAboveRaisinTask \
  BowlStackingLeftOnRightTask GrabABagelTask LargerObjectRaisinBoxInBinTask \
  MustardInLeftBinTask RubiksCubeTask RubiksCubeLeftOfBowlTask MarkerInMugTask \
  --num-envs 1 --num-runs 1 --headless --video-mode none \
  --output-folder-name "$EDGE_SOURCE/experiments/specprune_remote_smoke10_v1"
```

结果使用新目录，不覆盖、筛选或偷偷重试失败。扩展任务可替换 `--task` 列表；本策略仅支持同一 DROID 9 latent/33 action、17×20 patch 网格。

**历史 reset 限制**：继承十任务协议，prompt 改变时清空 Global/Dynamic/confidence 并重置 policy RNG；原客户端不传可靠 episode ID。同任务连续多个 rollout 必须每个 episode 重启服务，或先实现双方明确的 reset 协议。不要直接多 env 共享一个 history，也不要将 `--num-runs 10` 误称严格独立的1200次评估。

## 6. 本地校验与测速

发布前34项CPU测试通过（27项SpecPrune/release + 7项attention），CLI帮助解析通过；GPU全保留精确回归、20次稳定重复、更换seed防陈旧输出和Graph replay均通过。编译路径相对eager有浮点偏差，详见测速报告；没有声称编译版已完成十任务成功率验证。

```bash
cd "$EDGE_SOURCE"
PYTHONPATH="$EDGE_SOURCE" "$EDGE_PY" -m unittest discover -s tests -p 'test_specprune*.py'
PYTHONPATH="$EDGE_SOURCE" "$EDGE_PY" -m unittest discover -s tests -p test_future_instruction_attention.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$EDGE_PY" tools/benchmark_specprune_exit.py \
  --checkpoint "$EDGE_CHECKPOINT" --vae "$EDGE_VAE" \
  --capture /absolute/path/BananaInBowlTask.pt \
  --output experiments/specprune_compile_speed_NEW --warmups 5 --repeats 20
```

录制输入/权重不随Git上传。benchmark 要求本项目已有的 args/kwargs torch capture，不直接读取任意 HDF5。计时含在线评分、检查、packing、实际稀疏与恢复；不减去内部CPU诊断成本。无历史和相同录制输入预置两次的历史分别报告，后者不是实际闭环 c1/c2 观测。
