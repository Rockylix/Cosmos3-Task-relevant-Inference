# 固定任务 Core Top-7 是否加速：实测 profile

日期：2026-09-11。结论：**没有测到可辨认的单 chunk 加速；缓存的是 Core 来源层编号，不是 block quality 或 attention profile。**

## 同轮稳定 chunk 时间

RTX 4090，项目现有 Python/PyTorch 2.10.0+cu130；无仿真或其他计算进程争用。BananaInBowlTask 的既有 chunk3 输入，扩散 seed=1097657232，guidance=3、shift=5、4 steps。模型只加载一次，各组使用相同输入副本并重置 RNG。每组冷启动参考后预热 4 次，六种执行顺序轮换，30 轮 × 3 组 = 90 个完整计时样本，无剔除。

三组都关闭 compile/CUDA Graph，采用 1 次全量 conditional + 7 次稀疏 forward，Core80 + Stable104，每帧 184/340 token，保留 ASI Step0 guided velocity cache。

| 策略 | Mean | Median | P90 | 相对旧版 Median 加速 |
|---|---:|---:|---:|---:|
| 旧版：每 chunk 重选 Top-6 | 0.582970 s | 0.582620 s | 0.586172 s | 1.00000× |
| 对照：每 chunk 重选 Top-7 | 0.582760 s | 0.582477 s | 0.585752 s | 1.00025× |
| 当前：任务首 chunk 固定 Top-7 | 0.582842 s | 0.582760 s | 0.585329 s | 0.99976× |

主计时直接调用未插桩的 `Service.generate_fast`：CUDA synchronize 后开始，完整 generation、mask 选择、velocity cache、最终有限值检查及 synchronize 后结束；不包含模型加载、输入克隆、VAE、RPC、score 转 CPU 或 CSV 输出。不是完整服务请求时间。

固定层编号来自刚才三任务测试的 BananaInBowl **实际首个 chunk**：`[21,15,19,20,23,18,12]`。当前输入为既有 chunk3，用于离线速度测量，不是这次闭环的 chunk3 重新采集。动态 Top-7 与固定 Top-7 的实际层集合/mask 可不同，但预算、张量尺寸、其余算法保持一致；辅助的分段和相同 profile 微基准用于定位选层本身开销。

配对定义为每轮的“动态版本时间 − 固定 Top-7 时间”；正值才表示当前版更快。对 30 个配对差 bootstrap 10,000 次得到的中位数 95% 区间：

- 旧版 Top-6 vs 固定 Top-7：`[-0.296, +0.454] ms`。
- 动态 Top-7 vs 固定 Top-7：`[-0.378, +0.541] ms`。

区间均包含 0。这里只是同一输入、同一设备的一次计时实验的重采样不确定性，不是跨任务/硬件的统计保证。不能将 0.02% 量级的差异当作实际加速或退化。

## 分段：究竟省掉什么

诊断运行另做每组 6 次，不混入上面的主计时。只给原函数语句增加计时 scope，不重写数学表达式；诊断的 action、vision latent、mask 与各自无插桩参考逐元素完全一致。

下表为每 chunk 的 **CUDA stream elapsed 中位数**；28 次 attention profile 求和，其他阶段各执行一次，单位 ms：

| 区间 | 每 chunk Top-6 | 每 chunk Top-7 | 固定 Top-7 |
|---|---:|---:|---:|
| 28 层 action-aligned attention profile | 31.356 | 31.450 | 31.521 |
| 全部 28 层 R/H/Q | 0.115 | 0.117 | 0.118 |
| Core 层编号选择/准备 | 0.067 | 0.072 | 0.024 |
| Core 权重与空间分数 | 0.049 | 0.050 | 0.050 |
| 全局 Stable 分数 | 0.069 | 0.069 | 0.072 |
| Token Top-K 与预算校验 | 0.571 | 0.578 | 0.578 |

注意：CUDA events 测的是流上经过时间，含 host launch 空隙，不是 nsys 的 active-kernel 总时长；表中独立阶段中位数也不能严格相加。`stage_scopes.csv` 另保存 host scope 时间，其中 attention profile 的约 108.6 ms、selection plan 的约 2.9 ms 会包含等待此前 Transformer kernel 完成，不能把这些数字全算成可移除的 profile 开销。例：plan 的 stack/validation host scope 约 1.94 ms，但流上只有约 0.06 ms。

再取相同的真实 `[28,8,340]` attention profile，直接反复执行原生、未插桩 `build_core_stable_plan`。每组预热 10 次，200 次正式测量，两端 CUDA synchronize，完整选 mask wall-time：

| 完整选 mask 微基准 | Mean | Median | P90 |
|---|---:|---:|---:|
| 每 chunk Top-6 | 0.7896 ms | 0.7885 ms | 0.8056 ms |
| 每 chunk Top-7 | 0.7913 ms | 0.7902 ms | 0.8073 ms |
| 固定 Top-7 | 0.7560 ms | 0.7555 ms | 0.7710 ms |

同为七层时，节省约 **0.0347 ms/chunk**，只占约 582.8 ms generation 的 **0.006%**。这个微基准不含 attention profile 或 Transformer，不可把约 4.4% 的局部减少称为完整推理加速。

## 原因与边界

源码顺序见 [robolab_version1.py](../cosmos_framework/scripts/robolab_version1.py:177)：

1. 当前 chunk 仍对 B0–B27 采集 attention profile。
2. [R/H/Q 仍全部计算](../cosmos_framework/scripts/robolab_version1.py:201)。
3. [只有 Core 层选择分支变化](../cosmos_framework/scripts/robolab_version1.py:210)：动态路径做 Top-K 并把 CUDA 上的层编号转为 Python 整数；固定路径从缓存的 Python 层编号构造索引张量。
4. 固定七层仍按当前 Q 重新加权；[全局 Stable](../cosmos_framework/scripts/robolab_version1.py:227) 依赖全部 28 层当前 Q 和空间 profile，因此并未省去这些计算。
5. 所有 token-level Top-K、mask 校验和模型 forward 仍照常执行。

如果想获得可见收益，更值得检查 28 层 action attention profile 的实现开销，而不是这次只缓存七个层编号。减少 profile 层数或跨 chunk 复用 Stable/权重会改变当前“Stable 每 chunk 全局选”的定义，需另行确认。本轮没有修改策略或进行 kernel 优化，没有新的质量或成功率结论。

## 复现与文件

脚本：[tools/profile_task_core_layers.py](../tools/profile_task_core_layers.py)。

```bash
cd /root/robolab/worktrees/asi-velocity-cache
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 COSMOS_TRAINING=0 LD_LIBRARY_PATH='' \
  HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH="$PWD:/root/robolab/cosmos-edge-overlay" \
  /root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  -u tools/profile_task_core_layers.py \
  --output experiments/task_core_top7_global_stable_profile_v3 \
  --rounds 30 --warmups 4 --profile-rounds 6
```

输出目录须不存在。脚本内已经配置本地 checkpoint、VAE，不下载环境或权重。

实际保留：

- [最终指标](../experiments/task_core_top7_global_stable_profile_v2/metrics.json)。
- [90 个原生 chunk 计时](../experiments/task_core_top7_global_stable_profile_v2/chunk_timing.csv)。
- [分段原始数据](../experiments/task_core_top7_global_stable_profile_v2/stage_scopes.csv)。
- [完整选 mask 微基准](../experiments/task_core_top7_global_stable_profile_v2/selection_microbench.csv)。
- [参数与源码 SHA256](../experiments/task_core_top7_global_stable_profile_v2/manifest.json)。

执行记录：第一次 `profile_v1` 已完成全部 90 个主计时，但随后 AST 分段脚本将同名区间重复划分，触发断言；未产生诊断结果。修复后 CPU 三组 plan 逐元素验证通过，`profile_v2 --primary-from ...profile_v1` 复用全部既有主计时，CSV 字节一致，再补做诊断和微基准。没有删样本、重测筛选或混入失败计时；两次目录都保留。模型/策略源码未变。

遵循项目 profiling skill 的预热排除和范围限定原则；未收到 nsys 所需的用户指定 NVTX range，因此本轮使用定点 CUDA event/wall-time，不生成 `.nsys-rep`，也不分析初始化或全进程 kernel 百分比。原 baseline/ASI 工作树未修改，无 commit、push 或 merge。
