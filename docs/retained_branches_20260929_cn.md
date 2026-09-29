# Edge 保留分支与编译入口

发布远端为 `project`（Rockylix/Cosmos3-Task-relevant-Inference），不是 NVIDIA `origin`。

| 策略 | 分支 | 编译/Graph 入口 |
|---|---|---|
| Dense | baseline | 原始 baseline，不混入策略优化 |
| ASI | version/core80-stable104-action-weighted | 保留 eager、系统优化及 compile/Graph benchmark |
| 真机 | experiment/real-robot-asi-inference | 保留独立真机适配，不用仿真参数替换 |
| ToCa | experiment/toca-future | `action_policy_server_toca_scan --toca-execution eager/compile/compile-graph` |
| WorldCache | experiment/worldcache | `action_policy_server_robolab --worldcache-dddc --worldcache-execution eager/compile/compile-graph` |
| SpecPrune | experiment/specprune-future | `action_policy_server_specprune --specprune-backend eager/compile/graph` |
| C3ache | experiment/c3ache | 配对 benchmark 有编译/Graph 路径；普通 RoboLab 服务仍禁用编译/Graph |

ASI 的 `tools/benchmark_edge_multistrategy.py` 保留相同输入的多策略测试入口。
其 C3ache 路径只编译 decoder/head，chunk 刷新/复用决策留在图外；
不能用普通服务的 `--c3ache` 声称已经启用相同编译路径。
各工具参数以 `--help` 为准；旧 docs 中原始路径及数据属于历史记录。

本次整理保存已有优化，不重新定义各策略的缓存调度、预算或计时口径。
Dense 对照仍 eager；C3ache 必须均摊 refresh/hit 两类 chunk。
Compile 可改变浮点归约次序和 Top-K 边界，不能声称与 eager 数值逐位相等。

CPU 检查不能替代 GPU 复测。本次未产生新的成功率、FLOPs 或延迟数字。
旧 worktree 及实验数据归档在 `/root/robolab/archives/git_cleanup_20260929/`，不上传 Git。
