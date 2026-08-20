# Preliminary experiments

这里保存尚处于现象验证、机制分析、性能探索或小规模闭环验证阶段的实验。

## 分类

- `profiling/`：baseline NVTX、Nsight Systems 与性能剖析。
- `representation/`：hidden state、block residual、RoPE Q/K、stage attribution、空间 token 和离线统计。
- `attention/`：attention mass/enrichment/entropy、V、真实输出以及 action-query 分析。
- `qk_optimization/`：Q/K 相似度 profile、复用干预、真实跳算和 plan sweep。
- `sparsity/`：FIS、frame sparsity、联合模态 sparsity 及其配对质量实验。
- `evaluation/`：多任务闭环成功率和策略对照。
- `_archive/`：已被新版替代的历史产物，不参与默认分析。

## 新实验命名

```text
<topic>_<key-variables>_<task-or-data>_vN
```

禁止默认添加 `cosmos3_edge_`；通常也不要添加 `edge_`。实验名应直接说明研究主题和关键变量。

当前已有的 `edge_*` run ID 属于 legacy 标识，为保持历史引用有效而保留。新工具和新实验应采用上述规则。
