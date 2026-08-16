# 后续去噪从 B0 开始稀疏的消融实验

## 实验问题

在现有 `C/K80 conditional-only Step0` 策略中，Step 1--3 的 B0--B3
仍为全量计算。本消融只改变这一点：将后续去噪的 G1 稀疏区间从
B4--B11 扩展为 B0--B11，检查速度收益和误差传播。

固定条件：BananaInBowlTask、chunk 3、seed 579362556、guidance 3、
4 个去噪步、shift 5，关闭 Torch compile 和 CUDA graphs。ROI 预算仍为
G1/G2/G3 = 192/160/144 个空间 token/未来帧；mask 仍只由 Step 0
conditional profile 构造，两个 CFG 分支共享 mask。

## 执行路径

| 阶段 | 原 C/K80 conditional-only Step0 | 本消融 |
|---|---|---|
| Step 0 conditional | B0--B27 全量 | B0--B27 全量 |
| Step 0 unconditional | G1 B0--B11；G2 B12--B19；G3 B20--B27 | 不变 |
| Step 1--3，两分支 | B0--B3 全量；G1 B4--B11；G2 B12--B19；G3 B20--B27 | G1 B0--B11；G2 B12--B19；G3 B20--B27 |

本消融在 B0 只 pack 一次 G1 token，B1--B11 连续传递稀疏 hidden；
B12/B20 再切换到 G2/G3。被 G1 排除的 token 保存在 Transformer 输入侧
buffer，最后恢复完整序列。L0、action 和 text token 始终保留。

全程 224 次 block call 中，执行分布从
`52 dense + 60 G1 + 56 G2 + 56 G3` 变为
`28 dense + 84 G1 + 56 G2 + 56 G3`。平均 GEN token 保留比例从约
65.43% 降到 61.33%，平均每个 block call 少算 1196/3093 个 GEN token。

## 稳定单 chunk 结果

计时口径：同一已加载模型、同一冻结输入；每个模式预热 3 次，随后随机
交替测量 20 次；CUDA 同步包围 `generate_samples_from_batch`。

| Mode | Median (s) | P90 (s) | Mean (s) | Speedup vs Dense |
|---|---:|---:|---:|---:|
| Dense | 0.856578 | 0.861701 | 0.857175 | 1.000x |
| 当前 C/K80 | 0.615225 | 0.618697 | 0.616106 | 1.392x |
| 后续从 B0 稀疏 | 0.578816 | 0.582096 | 0.579055 | 1.480x |

相对当前 C/K80，本消融的 median 再降低 5.92%，即吞吐提升约 1.063x。

## 与 Dense 的单 chunk 输出差异

| 策略 | Action MSE | Action rel-L2 | Action cosine | Vision rel-L2 | Vision cosine |
|---|---:|---:|---:|---:|---:|
| 当前 C/K80 | 0.006383 | 0.057475 | 0.998658 | 0.357130 | 0.934076 |
| 后续从 B0 稀疏 | 0.011288 | 0.076431 | 0.997106 | 0.386387 | 0.922680 |

本消融带来明确的额外误差：Action MSE 约为当前策略的 1.77 倍，Vision
cosine 从 0.9341 降至 0.9227。所有输出及中间稀疏 block 均通过 NaN/Inf
检查。

## 闭环检查

BananaInBowlTask 单任务闭环结果为 1/1 成功，167 个环境步完成任务；服务端
共处理 6 个 generation chunk，排除首请求后的 generation wall time median
为 0.590321 s、P90 为 0.592726 s。viewport 视频已保存。

这只能证明该 task/scene seed 下功能闭环可行，不能替代多任务、多场景 seed
成功率评估。当前证据支持“速度继续提高，但输出保真度下降”的结论。

## 复现入口

服务端使用：

```bash
python -m cosmos_framework.scripts.action_policy_server_robolab_v5_3_acd_packed_kernel \
  --checkpoint-path /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID \
  --host 127.0.0.1 --port 8000 --format-prompt-as-json True --no-guardrails \
  --seed 579362556 --deterministic-seed --guidance 3.0 \
  --num-steps 4 --shift 5.0 --ablation-mode c_cond_step0_b0_sparse \
  --intervention-output-dir /root/robolab/experiments/preliminary/sparsity/velocity_cache/<run>/server
```

稳定 benchmark 模式名为 `c_cond_b0_sparse_opt`。
