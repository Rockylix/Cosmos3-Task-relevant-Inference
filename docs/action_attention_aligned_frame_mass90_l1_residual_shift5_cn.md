# Action 对齐逐帧 90% mass 与 L1 residual 补全实验

## 1. 实验流程

本实验使用 Cosmos3 Edge、`shift=5`，每个 Transformer block 执行两次：

1. 完整 probe：保留完整 3093 GEN token，捕获真正进入 attention 的 post-RoPE Q/K，probe 输出丢弃。
2. committed sparse pass：
   - L0、L1、q0、32 个预测 action token 及 UND/text 全量计算；
   - L2--L8 分别使用自己的 action-aligned 空间 mask；
   - 未计算的 L2--L8 token 使用当前 block 的 L1 同空间位置 residual 补全；
   - 每层结束恢复完整 3093 token，下一层重新 probe。

Action query metadata 显式提供 `action_horizon=0..31`。Action 是每视频帧 1 token，vision latent 的时间压缩率为 4，因此使用：

```text
L1 <- action horizon 0--3   （L1 本实验始终全量）
L2 <- action horizon 4--7
L3 <- action horizon 8--11
L4 <- action horizon 12--15
L5 <- action horizon 16--19
L6 <- action horizon 20--23
L7 <- action horizon 24--27
L8 <- action horizon 28--31
```

对于 Lf，对对应的 4 个 action query 和所有 Query heads 求均值：

```text
A_f[p] = mean(a in G_f, h) Attention[h,a,(f,p)]
M_f = MinSet90(A_f)
```

L2--L8 的 `M_f` 彼此独立，不做跨 frame 并集。第二次 block 的 GEN token 数为：

```text
N_sparse = 713 + sum(f=2..8) |M_f|
```

未选位置补全：

```text
R_L1[p] = H_sparse_out[L1,p] - H_in[L1,p]
H_out[Lf,p] = H_in[Lf,p] + R_L1[p], p not in M_f
```

## 2. 固定 chunk 配对结果

配置：BananaInBowlTask、chunk 3、相同输入/seed/noise、4 denoise steps、conditional/unconditional。

| 指标 | Aligned-frame Sparse vs. Baseline |
|---|---:|
| Action MSE | 0.006285 |
| Action relative-L2 | 0.055660 |
| Action cosine | 0.998542 |
| Action max absolute error | 0.235673 |
| Delta-action cosine | 0.077733 |
| Jerk cosine | 0.004605 |
| Vision latent cosine | 0.794277 |
| Vision latent relative-L2 | 0.633000 |
| Decoded RGB cosine | 0.886849 |
| Decoded RGB relative-L2 | 0.473741 |

所有 action、latent、RGB 和中间 block 输出均通过 NaN/Inf 检查。

### Token 统计

| 项目 | 数值 |
|---|---:|
| 平均 L2--L8 空间 token | 223.141 / 340 |
| 最小/最大单帧 mask | 6 / 303 |
| committed GEN 平均保留比例 | 73.55% |
| committed GEN 平均节省 | 818.013 / 3093 |
| committed GEN 平均节省比例 | 26.45% |
| 单 block 最大节省 | 1671 / 3093 |
| 恢复后进入下一 block | 3093 / 3093 |

逐 latent mask 大小：

| Latent | Mean | Min | Max | 对应 action group |
|---|---:|---:|---:|---:|
| L2 | 236.93 | 108 | 302 | 4--7 |
| L3 | 206.57 | 6 | 302 | 8--11 |
| L4 | 236.07 | 104 | 302 | 12--15 |
| L5 | 217.76 | 29 | 302 | 16--19 |
| L6 | 224.08 | 32 | 303 | 20--23 |
| L7 | 234.25 | 81 | 303 | 24--27 |
| L8 | 206.33 | 15 | 303 | 28--31 |

极端收缩集中在后期 step 的 B18--B21。例如 step 2/timestep 833 的 conditional B19、L3 只选择 `6/340` token，但聚合 attention coverage 仍为 0.908。这说明 action 8--11 对 L3 的 attention 在该 block 高度集中；它不能说明 L3 的其他位置对其他 action query、future query 或 MLP 不重要。

产物：

```text
/root/robolab/experiments/preliminary/sparsity/action_attention_l1_residual/
  action_attention_aligned_frame_mass90_l1_residual_shift5_BananaInBowlTask_c3_v1/
```

其中新增 `aligned_frame_selection.csv`，保存每个 branch/step/block/frame 的 action group、raw frame mass、coverage、mask 大小和空间索引。

## 3. 与前两版比较

| 策略 | GEN token 节省 | Action cos | Delta cos | Jerk cos | Latent cos | RGB cos |
|---|---:|---:|---:|---:|---:|---:|
| 224-set shared union | 2.29% | 0.999830 | 0.283916 | 0.164244 | 0.960789 | 0.967197 |
| All-action/all-frame aggregate | 25.76% | 0.998252 | 0.038639 | 0.022400 | 0.795008 | 0.893131 |
| Action-aligned per-frame | 26.45% | 0.998542 | 0.077733 | 0.004605 | 0.794277 | 0.886849 |

逐帧时间对齐没有恢复动作动态。它略微改善 delta-action cosine，但 jerk cosine 进一步下降，latent/RGB 误差与全局聚合版本基本相同。

## 4. 是否运行闭环

本轮没有继续运行 750-step 闭环。理由不是程序错误，而是固定 chunk 已满足预先设置的停止条件：

- Delta-action cosine 仅 0.0777；
- Jerk cosine 仅 0.0046；
- 个别 mask 只有 6 个空间 token；
- 上一版相近的 `delta=0.0386, jerk=0.0224` 已在 BananaInBowlTask 上运行到 750-step timeout，且没有抓取事件。

因此直接闭环大概率只会重复已知失败，并不能增加对当前假设的辨识力。如需观察行为，server 入口和视频输出路径已保留，可以显式运行。

## 5. 结论

1. 每帧使用时间对齐 action group 能避免跨 frame mask 并集，也能获得约 26% 的 committed-pass token 节省。
2. 失败原因不再是 mask 并集过大，而是 action attention 只描述“这些 action query 从该 frame 读取什么”，不能覆盖：
   - 其他 action horizon 对该 frame 的读取；
   - future query 对该 frame 的读取；
   - 被跳过 token 对 attention K/V 和 MLP 输出的贡献。
3. L1 residual 只能补全该 token 的近似 block 更新，无法补回它作为 K/V 对其他 token 输出的影响。
4. 整体 Action cosine 仍接近 0.999，但 delta/jerk 已失真，不能据此判断控制安全。
5. 若继续此方向，应优先增加每帧 `k_min`，或将 aligned group 与跨 horizon 的小比例保护集合结合；不建议继续降低 mass threshold。

## 6. 代码与验证

```text
branch:   experiment/action-attn-aligned-frame-mass90-l1-residual-shift5
worktree: /root/robolab/worktrees/action-attn-aligned-frame-mass90-l1-residual-shift5
ruff:     passed
pytest:   12 passed
```
