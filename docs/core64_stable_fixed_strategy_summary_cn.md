# 当前稀疏策略简述：V6-B Direct Core64 → Stable

## 1. 一句话说明

先用 Step 0 conditional 全量计算得到 Action→Future 的空间关注区域；后续只让选中的
future tokens 进入 Transformer 的 Attention 和 MLP，未选背景在 hidden 输出端由 side
buffer 补齐，并在 UniPC 积分前使用 Step 0 的背景 velocity cache。

当前版本：`v6-b-direct-core64-then-stable-k184-152-136`，固定 `shift=5`、4 个 denoise steps。

## 2. Mask 如何生成

Step 0 conditional branch 全量计算，并从 B4-B27 采集 Action Query 对 L1-L8 future
tokens 的 raw attention probability。每个 future frame 独立生成 `17x20=340` token mask。

Mask 按固定顺序生成：

- **Core（64）**：跨高质量 block 聚合后，每个 future frame 一次性直接 Top-64；
- **Stable（88/72/64）**：跨 future frames 较稳定的环境区域；
- **Adaptive**：某一 future frame 相对共享环境更突出的正变化区域；
- **Fill**：预算仍不足时，按 raw attention mass 补足。

每个 chunk 只生成一次八张 Core64 mask，后续所有 step、block group 和 CFG branch 固定
复用。Core64 固定后，Stable 才从八帧 Core64 空间并集之外选取；不存在 Core48 reference
或 Core48→Core64 的二次扩张。Mask 从 G3 向 G1 构造，严格保证：

`G3 subset G2 subset G1`。

conditional/unconditional 共用同一组 mask；Core、Adaptive 和 Fill 可以随 future frame
变化，Stable 在同一 group 内由 L1-L8 共用。

## 3. Token 预算

| Group | Transformer blocks | 每帧 future tokens | 典型组成 |
|---|---|---:|---|
| G1 | B0-B11 | 184/340 | Core64 + Stable88 + Adaptive/Fill32 |
| G2 | B12-B19 | 152/340 | Core64 + Stable72 + Adaptive/Fill16 |
| G3 | B20-B27 | 136/340 | Core64 + Stable64 + Adaptive/Fill8 |

L0 condition latent、全部 action tokens、文本/条件 tokens 始终保留；只稀疏 L1-L8
future vision tokens。RoPE、attention mask 和 token metadata 使用同一索引裁剪，位置 ID
保持原编号。

完整 GEN 序列为 3093 tokens。稀疏后：

| Group | 实际 GEN tokens | 保留比例 |
|---|---:|---:|
| G1 | 1845 | 59.65% |
| G2 | 1589 | 51.37% |
| G3 | 1461 | 47.24% |

## 4. 四步推理流程

| Denoise step / CFG branch | 执行方式 |
|---|---|
| Step 0 conditional | B0-B27 全量；生成 mask，并缓存 conditional vision prediction |
| Step 0 unconditional | B0-B11 用 G1，B12-B19 用 G2，B20-B27 用 G3 |
| Step 1-3 conditional/unconditional | 两个 CFG branch 都按 G1/G2/G3 稀疏计算 |

每个 sparse group 的真实流程：

1. 保留全部非 future tokens，只 pack 选中的 L1-L8 空间 tokens；
2. packed tokens 真正进入当前 blocks 的 Attention 和 MLP；
3. group 切换时把已计算 token 写回 side buffer，再裁剪到下一组更小的 mask；
4. B27 后恢复完整 hidden：已选位置写入新结果，未选位置沿用 side buffer；
5. 恢复完整序列后，正常进入最终 RMSNorm、vision head 和 action head。

因此它是实际缩短 Transformer 输入，不是仅在全量计算后乘 mask。但未选 hidden token
也没有获得被跳过 block 的新 residual。

## 5. Velocity cache

Step 0 保存完整 guided vision velocity。Step 1-3 在进入 UniPC 更新前组合：

- 最终 G3 ROI 内：使用当前 denoise step 的 velocity；
- ROI 外背景：复用 Step 0 guided velocity；
- L0 condition 与 action 输出：始终使用当前 step 的正常结果。

此外，Step 0 unconditional 在 ROI 外令 `unconditional = conditional`，即背景 CFG delta
为零，避免稀疏 unconditional 背景产生无依据的 CFG 放大。

## 6. 当前状态与边界

- 本次 Direct Core64→Stable 修改后已完成 seed `579362556` 的两个简单任务冒烟测试：
  `BananaInBowlTask` 与 `BananaOnPlateTask` 均成功（`2/2`）；
- 两任务共 11 个 generation chunks，排除首个冷请求后的 generation median
  `0.573228 s`、P90 `0.592286 s`；该数值只是跨两个任务的闭环 warm chunk 冒烟统计，
  不是正式配对性能 benchmark；
- 旧的 `0.566990 s`、`1.508x` 和 `5/9` 属于 Core48→Stable→扩张 Core64 的 V6-A，
  不能作为当前 V6-B 的结果；
- Core 在每个 future frame 独立 Top-64，因此不同帧的 Core 空间位置不保证完全一致；
- Stable 排除八帧 Core64 union，可能改变旧 V6-A 的 Stable 位置，必须重新验证 token
  组成、attention-mass retention、单 chunk 延迟和闭环成功率。

V6-B 两任务报告：
`/root/robolab/experiments/preliminary/sparsity/velocity_cache/direct_core64_stable_shift5_2tasks_seed579362556_v1/report_cn.md`

旧 V6-A 历史实验：[core64_stable_fixed_adaptive_reduced_cn.md](core64_stable_fixed_adaptive_reduced_cn.md)

四任务 mask overlay：
`/root/robolab/experiments/preliminary/sparsity/visualization/core64_mask_overlay_4tasks_seed579362556_v1/README.md`
