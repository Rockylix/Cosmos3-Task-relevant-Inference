# 固定 ToCa baseline

2026-09-10 用户指定：后续 ToCa 对照固定使用
`D/C/D/C, r=0.25, spatial_bonus=0, cfg_selection=independent`。

规范配置：[`configs/toca_baseline.json`](../configs/toca_baseline.json)。可直接传给
`cosmos_framework.scripts.action_policy_server_toca_scan --scan-config configs/toca_baseline.json`。
不要使用旧通用服务的 reference/shared 默认值来代替这个配置。

| 参数 | 固定值 |
|---|---|
| full_steps | [0, 2] |
| fresh_ratio / layer_slope | 0.25 / 0.5 |
| period / age_weight | 2 / 0.25 |
| spatial_bonus | 0，关闭 |
| cfg_selection | independent |
| attention_backend | joint |
| num_steps / shift / guidance | 4 / 5 / 3 |
| compile / CUDA graphs | 都关闭 |
| policy seed / deterministic_seed | 0 / False，每任务重置 RNG |
| simulator seed | 0 |

十任务筛选结果：4/10，score 0.4666666687，单 chunk macro median
0.7874686891 s，相同输入配对 Dense 为 0.8709937155 s，加速 1.106067743x。
该时间是十任务各自预热 5 次、测量 30 次后的 median 等权均值。
数据来自 `experiments/toca_hparam_scan10_v1/summary.csv` 和该配置的
`simulator/episode_results.jsonl`；不是多 seed 泛化结论。

用户要求中止剩余扫描并关闭 `toca_scan10`。停止时共保留 169 个 episode，
其中当前未完成配置 `dccc_r0p25_b0_independent` 为 9/10，不属于本固定版本。
SIGINT 的 KeyboardInterrupt 是用户中止，不是策略崩溃；旧结果和错误/重启记录均保留，不删除。

最初冻结时只增加规范配置和说明，未修改扫描算法、旧 manifest 或历史结果；配置中记录原推理源码哈希。

## 提交的配置口径（2026-09-11）

用户确认将本测试版本的源码、配置、测试和 docs 提交到 `experiment/toca-future`，推送至 `project` 远程；不合并 main，不提交 experiments、权重或日志。推理核心文件与上述历史测试 SHA256 一致，本次不修改算法。

JSON 的 `config` 是服务实际读取的 ToCa 参数；新增 `protocol` 固定本轮十任务、seed、sampler 和推理设置。`protocol` 为复现记录，不会自动覆盖服务 CLI，启动时仍需按表传 `--num-steps 4 --shift 5 --guidance 3 --seed 0 --no-deterministic-seed`，并指定 `--scan-config configs/toca_baseline.json` 和新的 `--scan-output`。

本分支保留超参扫描工具以便复现，但所选 baseline **只有本 JSON 对应的一组**，不是 D/C/C/C 或其他扫描点。历史实验原始数据只在本机，GitHub 上的结果摘要以本页为准。
