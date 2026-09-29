# ToCa-Future 编译与 CUDA Graph 适配

本适配基于冻结 ToCa `f7cf5d6`，于 2026-09-29 归入保留分支
`experiment/toca-future`。默认仍为 eager，编译和 Graph 通过显式开关启用。

## 保持的策略

- D/C/D/C，fresh_ratio=0.25，layer_slope=0.5，age_weight=0.25，period=2。
- spatial_bonus=0、CFG selection=independent，joint attention backend。
- full 步从实际 attention 同时取得 AV 和 incoming future score，不额外重算 QK-softmax。
- cached 步：L0/action 的 Q/O 实算，全部真实 GEN K/V 实算；future attention residual 复用最近 full。
- protected+选中的 future token 重算 MLP，只更新选中的 future MLP cache。
- score、age、稳定排序、token预算、原始RoPE、CFG、UniPC均不变，无跨chunk特征缓存。

## 改的是执行接口

1. `ToCaFutureController(optimized=True)` 不再动态覆盖编译层的 `forward`，由 Transformer 外层路由。
2. full decoder 通过可选 `toca_future_positions` 参数调用原 joint kernel，返回 attention residual、
   MLP residual、score metadata。位置分别为 o_proj 后和 MLP 后、对应 residual add 之前。
3. controller 取出 metadata 并在图外 clone，避免下一次 Graph replay 覆盖缓存；cache只活在当前request。
4. cached tensor函数编译，MLP cache采用函数式更新再clone，不对graph输入做原地持久化修改。
5. 图外仍使用原 `select_fresh()`，没有把选择结果固定到第一轮。
6. actual UND/GEN长度裁掉Graph padding后才做attention和score，score分母是实际token数。
   UND cross K与UND自身causal K仍保留各自原始归一化规则。
7. 输入/输出存储形状恢复padding，不重编号RoPE，也不把padding当token。

`optimized=False` 保留旧hook路径；无ToCa时可选参数为None，Dense调用原attention dispatch。
compiled cached函数可跨request复用代码，但不跨request缓存分数、索引、residual或输出。
使用层级/输出头 `torch.compile(fullgraph=True)` + Inductor `reduce-overhead`，并不是整条UniPC链一张图。
不同block的fresh数量不同，会捕获多种形状，首次编译/捕获不计入稳定延迟。

## 启动仿真策略服务器

已有环境和权重即可，无需新建环境。冻结的 `configs/toca_baseline.json` 保留原始eager协议记录，
本适配通过显式 `--toca-execution` 覆盖执行后端，不改变JSON中的算法参数。

```bash
cd /root/robolab/worktrees/toca-future
export PYTHONPATH="$PWD:/root/robolab/cosmos-edge-overlay"
export LD_LIBRARY_PATH=''
export HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  -m cosmos_framework.scripts.action_policy_server_toca_scan \
  --checkpoint-path /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID \
  --scan-config configs/toca_baseline.json \
  --scan-output experiments/toca_compile_graph_server_new \
  --toca-execution compile-graph \
  --seed 0 --num-steps 4 --guidance 3 --shift 5 --port 8017 --no-guardrails \
  --experiment-overrides \
  model.config.tokenizer.vae_path=/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth \
  model.config.tokenizer.object_store_credential_path_pretrained= \
  model.config.tokenizer.bucket_name=
```

`compile`只编译不启用Graph；`eager`回到旧策略。编译模式拒绝旧dense采集hooks、reference backend和
Dense scan配置，避免静默回退。该scan服务器沿用原固定十任务集合的prompt映射。

## 验证

- `tests/test_toca_future.py`：29项CPU测试；包含r=0/0.25/1、padding、原始/归一化/缓存UND K、CFG隔离、age、更新范围。
- 真实权重旧eager与新接口eager比较：最终action/vision逐元素一致。
- 正式测试额外验证scores和fresh indices、padding路径、改变seed、重复replay，并记录编译数值差异。
- Graph是否实际启用以profiler的`cudaGraphLaunch`为证据，不以配置文件为证据。
- 两轮独立agent review通过；没有改原joint Triton kernel。编译后的浮点/选token变化不等于算法改变，
  也不能以此声称action不变或闭环成功率不变。本轮不重跑仿真任务。

最终五次预热、十五次配对单chunk结果见同目录 `compile_graph_benchmark_cn.md`。
