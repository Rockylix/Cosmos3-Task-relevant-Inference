# WorldCache 四步适配：D/D/D/C，video/action 联合预测

## 状态与 Git 隔离

- 分支：`experiment/worldcache`
- worktree：`/root/robolab/worktrees/worldcache`
- 基点：Dense Baseline `324574a454a989f9b8f5392f7673ace487243e8d`，没有叠加 ASI/ToCa。
- 创建前 Baseline：`/root/robolab/cosmos-framework-edge`，branch `cache`，上述 HEAD，`git status --short` 为空。
- 创建前当前 ASI：`version/core80-stable104-action-weighted`，HEAD `62a09f7f95f1f0a54d5af9c2b8c530c862db9025`，`git status --short` 为空。
- 所有修改及测试限于此 worktree。没有 merge、push、删除其他 worktree，未自动提交。

初版只完成代码、CPU 单元/集成接缝测试和服务器参数检查；2026-09-10 已补齐原生 batch/padding 布局并通过真实权重 GPU gate：Dense 8 次、WorldCache 6 次 Transformer forward，前三步两分支逐位一致。后续十任务状态与计时参见 [十任务记录](worldcache_smoke10_cn.md)，不能将 CPU 测试作为 GPU 加速证据。

验证记录：初版12/12 CPU测试通过；补充原生 batch、33→34 padding 和直接执行真实网络 unpatchify 的验证后，15/15通过。新增模块/测试和服务器脚本 Ruff 检查通过；CLI `--help` 及非法 step/sampler/guidance/n_max 拒绝检查通过。Baseline 和当前 ASI 未修改，保持干净。

## 方法及与官方实现的区别

参考 [论文](https://arxiv.org/abs/2603.06331) 和 [官方代码](https://github.com/FofGofx/WorldCache)，锁定提交：
`b921368f7dfbd7ca5d7cfcd0276fec1d1cfd7d91`。

官方 Voyager 是模型输出级缓存，不是按 token 选择性运行 Transformer。默认预热 5 步，最后一步强制 FULL。因此原版在四步设置下是 D/D/D/D。

本实验明确修改为：

| 0-based step | conditional | unconditional | UniPC |
|---|---|---|---|
| 0 | FULL，记录历史 | FULL，记录历史 | 正常更新 |
| 1 | FULL，记录历史 | FULL，记录历史 | 正常更新 |
| 2 | FULL，记录历史 | FULL，记录历史 | 正常更新 |
| 3 | CACHE，预测输出 | CACHE，预测输出 | 正常更新 |

这相当于预热降至 3 步，取消末步保护。只有一次最终 CACHE，所以没有后续 FULL 决策；**本版本不实现/不声称验证 CAS 自适应调度，也没有无效的 error-threshold 扫描参数**。

缓存对象：最终投影后的模型 velocity prediction，不是 B27 hidden、block residual、最终 denoised latent 或反归一化动作。

完整路径：

```text
每个 request 新建 WorldCacheRequest
  VAE/条件编码、初始噪声：原路径
  对每个 UniPC step：
    对 conditional / unconditional 分别处理：
      按当前 noisy latent 更新原有 packed sequence
      step 0/1/2：self.denoise -> Transformer + 输出头
        保存当前分支的真实模型输出
      step 3：不调用 self.denoise
        三次 FULL 历史 -> token 曲率分组 -> 预测 video/action 输出
      原有 condition velocity mask、action padding mask
    原有 CFG：u + guidance * (c - u)
    原有 UniPC 积分
  原有 action 后处理、可选 VAE decode
  输出轻量 report，释放全部历史 tensor
```

step 3 仍有 packed metadata 更新、历史预测和输出重建成本，不等于整步耗时为零。L0/state 的采样值固定，但 CACHE 时不声称其 hidden 仍执行 Transformer。

## Token layout 与公式

- 首版仅支持单卡单 rank、batch=1、policy、L0 条件帧 + L1..L8、q0 条件状态 + q1..q32。
- H/W/C、patch size、action 有效维度均来自实时张量/metadata，不硬编码 2048 hidden dimension 或 action padding 维数。
- future video 按 Cosmos 的 `thwpqc` 顺序重排为 `[8*H/p*W/p, p*p*C]`；当前常见 grid 为 `17*20=340` token/帧。
- action 使用 `[32, raw_action_dim]`。两种模态分别保存和计算曲率，合并每 token 标量曲率后求分位数，不混合不同维数的向量。
- L0/q0 不参与曲率统计；重建时条件输出速度和 action padding 精确置零，之后仍执行原始 velocity mask。
- Edge 原始 H=33 会补到 34。FULL 步直接捕获原生 llm2vae 输出（包括真实 padding 输出），CACHE 在完整 patch grid 上预测，再按原生 unpatchify 顺序还原并裁回 H=33；不会给历史补零伪造 padding。

最近三次 FULL 的 step index 为 `s0 < s1 < s2`：

\[
v_1=(Y_{s1}-Y_{s0})/(s1-s0),\quad
v_2=(Y_{s2}-Y_{s1})/(s2-s1)
\]
\[
a_2=(v_2-v_1)/(s2-s1),\quad
\kappa_i=\|a_{2,i}\|_2/(\|v_{2,i}\|_2^2+\epsilon).
\]

使用执行序号，不是 999/937 等 timestep 值；这里的斜率 `v` 不要混淆为被近似的 diffusion velocity。
分位阈值默认 `p_stable=0.30`、`p_chaotic=0.70`。

Stable：`kappa < q_stable`，复用 `Y_s2`；Linear：中间区间，`Y_s2+k*v2`；Chaotic：`kappa >= q_chaotic`，

\[
\widehat Y=Y_{s2}+k[(1-\alpha)v_2+\alpha v_1],\quad
\alpha=3x^2-2x^3,\quad x=\min(k/n_{max},1).
\]

本版 `k=1`，`n_max=6` 是阻尼参数，不会增加 CACHE 步数。曲率相同的 token 可能使分组比例偏离 30/40/30，保留官方不等式规则，不强行凑比例。

历史使用模型输出 dtype 的独立 clone；斜率、曲率、quantile 和预测在 FP32 执行，返回原输出 dtype。开启 NaN/Inf 检查，包括 cast 后溢出检查。FULL 输出原样返回，不用重建张量替代真实输出。

两个 CFG 分支历史独立，调度均为固定 D/D/D/C。每个新 chunk 重新建立三次 FULL 历史，不能把前一个 chunk 或另一分支当作预热数据。近似输出不进入 FULL 历史。

## 代码位置

- `cosmos_framework/inference/worldcache.py`：曲率/预测、token adapter、request controller。
- `cosmos_framework/model/generator/omni_mot_model.py`：`generate_samples_from_batch(worldcache_config=...)` 创建请求；`_get_velocity` 在原速度屏蔽前接入。
- `cosmos_framework/scripts/action_policy_server_robolab.py`：默认关闭的 CLI 开关，强制 eager，拒绝与 dense hidden/residual/QK collector 混用。
- `tests/worldcache_dddc_test.py`：CPU 测试。

成功生成后，`model._last_worldcache_report` 包含配置、每步分支 FULL/CACHE、两模态的分组 token 数，及 `full_forwards=6` / `cache_forwards=2`。report 无原始 tensor，不改变默认 `vision`/`action` 返回格式。

CUDA 上的新增 NVTX 范围：`worldcache/step{0..3}/{conditional|unconditional}/{FULL|CACHE}`。未来真实 profile 应确认两个 CACHE 范围没有 Transformer/attention/MLP kernel，而不是仅依赖日志计数。

## 现有环境与启动命令

使用项目已有环境，不执行 uv sync、不下载依赖或权重。先确认 GPU/端口没有被其他实验占用；下列命令为手动启动。十任务使用带每任务 RNG 重置和审计的专用 runner，见 [操作说明](worldcache_smoke10_cn.md)。

```bash
cd /root/robolab/worktrees/worldcache
export PYTHONPATH=/root/robolab/worktrees/worldcache:/root/robolab/cosmos-edge-overlay
export HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export COSMOS_TRAINING=0
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=''

/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID \
  --host 127.0.0.1 --port 8028 \
  --sampler unipc --num-steps 4 --guidance 3 --shift 5 \
  --seed 0 --no-deterministic-seed --format-prompt-as-json True \
  --no-guardrails --eager --worldcache-dddc \
  --worldcache-stable-percentile 0.30 \
  --worldcache-chaotic-percentile 0.70 --worldcache-n-max 6 \
  --output-dir /root/robolab/worktrees/worldcache/experiments/dddc_server \
  --experiment-overrides \
  model.config.tokenizer.vae_path=/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth \
  model.config.tokenizer.object_store_credential_path_pretrained= \
  model.config.tokenizer.bucket_name=
```

Dense 对照：同一命令移除 `--worldcache-dddc`，仍保留 `--eager`；使用不同 output-dir。
不要在单张 4090 上同时启动两个模型服务。若需要返回预测视频，添加 `--decode-video`；该选项返回视频数组，不自动生成 MP4。

低层 API：

```python
from cosmos_framework.inference.worldcache import WorldCacheConfig

samples = model.generate_samples_from_batch(
    data_batch, seed=[0], guidance=3.0, num_steps=4, shift=5.0,
    worldcache_config=WorldCacheConfig(),
)
report = model._last_worldcache_report
```

低层 API 同样要求模型已配置为 eager。Dense 传 `worldcache_config=None`，原默认行为不变。

## 验证与后续实验边界

```bash
cd /root/robolab/worktrees/worldcache
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
PYTHONPATH=/root/robolab/worktrees/worldcache:/root/robolab/cosmos-edge-overlay \
/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  -m unittest discover -s tests -p worldcache_dddc_test.py -v
```

CPU 测试覆盖：patch 顺序及往返、L0/q0/padding、三点历史、三类预测、分位 ties、BF16、branch/step gating、无 alias、NaN/Inf、六次真实回调与两次 CACHE。
集成接缝测试直接提取并执行当前 `_get_velocity` 的网络分派/速度屏蔽/flatten 代码，加原生 UniPC CPU solver；在常量预测模型上验证 Dense/CACHE 完全一致、CFG 正常、四次积分未减少。它没有加载真实 Transformer，不能代替 GPU 模型验证。

后续建议先做同输入 chunk 3 配对输出检查，再做计时/十任务闭环。主时间口径是 warmed、交替、同输入/seed 的 `generate_samples_from_batch`，报告 mean/median/P90；包括每次请求的三次 FULL、缓存维护与有限值检查，不能排除算法预热。VAE 解码和首次启动单列。

在各步全量成本相同的近似下，D/D/D/C 的 backbone 理想上限是 `4/3=1.333x`，不是端到端保证。已完成十任务：5/10，score 0.4667，单chunk 0.670914 s，对同轮Dense 0.863966 s为1.287743×；完整保真度和统计边界见 [十任务记录](worldcache_smoke10_cn.md)。
