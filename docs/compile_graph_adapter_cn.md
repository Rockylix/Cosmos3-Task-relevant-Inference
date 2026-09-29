# WorldCache 编译与 CUDA Graph 适配

本适配基于 `763b6d1`，于 2026-09-29 归入保留分支
`experiment/worldcache`；默认 eager，编译和 Graph 通过显式开关启用。

## 算法不变

D/D/D/C、两个CFG branch各自保存step0/1/2输出，step3做原异构外推；每chunk
6次真实network forward+2次缓存预测。stable/chaotic分位数0.30/0.70、n_max=6、
vision与action联合quantile、原始velocity mask/CFG/UniPC不变。
仍缓存真实future video和predicted action；不加入跨chunk缓存、token裁剪或新评分公式。

## 编译接口

旧接口用`llm2vae` forward hook捕获投影。新 `compile_compatible=True` 接口改为：

`llm2vae含bias输出 → _decode_vision output_dict旁路 → denoise透传 → request pop → detach.clone历史`

捕获点仍在unpatchify前，保留真实投影的padding patch；不是从已crop图像重建padding。
不额外计算一次projection，不用动态图hook，下一次forward前clone缓存以避开Graph输出buffer覆盖。
开启标志通过`finally`恢复；`compile_compatible=False`保留旧接口。
仅启用编译且使用新接口时取消旧eager-only限制；不是删除所有安全检查。

native decoder和编码/输出头编译并使用Inductor Graph；分位数/异构预测、CFG和UniPC仍在原路径执行。
不是整个generation共用一张全局Graph。

## 服务器启动

```bash
cd /root/robolab/worktrees/worldcache
export PYTHONPATH="$PWD:/root/robolab/cosmos-edge-overlay"
export LD_LIBRARY_PATH=''
export HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \
  -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path /root/robolab/RoboLab/Cosmos3-Edge-Policy-DROID \
  --worldcache-dddc --worldcache-execution compile-graph \
  --seed 0 --num-steps 4 --guidance 3 --shift 5 --port 8017 --no-guardrails \
  --format-prompt-as-json True \
  --experiment-overrides \
  model.config.tokenizer.vae_path=/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth \
  model.config.tokenizer.object_store_credential_path_pretrained= \
  model.config.tokenizer.bucket_name=
```

`compile`只启用编译；`eager`回原接口。不能同时使用`--eager`或原有dense采集hooks。
两份服务器示例都用8017，仅择一启动；不同时占用GPU。

## 验证

- 20项CPU测试，包含两CFG分支隔离、原CFG/UniPC流程、真实padding projection历史、
  hook与output接口等价、真实denoise旁路传递、history不被buffer覆盖。
- 真实权重新接口eager与原eager的action/vision逐元素一致；两轮独立agent review通过。
- Graph路径实测198次`cudaGraphLaunch`/chunk，即6次forward×(28个decoder+5个编码/输出头)。
- 重复请求、改变seed、NaN/Inf与编译数值差异单独记录。
- 编译改变浮点运算顺序，不声称compiled输出逐位等价或闭环成功率不变。

最终5次预热、15次配对单chunk结果见 `compile_graph_benchmark_cn.md`。
