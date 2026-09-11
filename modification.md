# 修改记录

## 2026-09-11：C³ache 保守首版

基于 baseline `324574a454a989f9b8f5392f7673ace487243e8d`，新增跨 chunk、
同去噪步的全 GEN Transformer 残差缓存。原 baseline 与共享 RoboLab 不改动。

为避免动态跳层被编译图固化，缓存开启时强制关闭 torch.compile 与 CUDA graphs。
**性能影响**：相较 compiled baseline 可能更慢；先使用 period=1 的 eager dense
对照评估缓存收益。后续在验证动态缓存语义后再优化编译边界。

为保守防止布局混用，每次分支前向把少量索引、mask、position IDs、timestep
等元数据复制到 CPU 并精确比较；不复制观测或全量 hidden states 到 CPU。
**性能影响**：引入 GPU→CPU 同步及 Python 比较开销，后续可用受验证的
request-level 签名代替重复检查，但目前保留严格校验。

刷新时克隆 h0，防止 decoder 的原地更新破坏残差边界；残差保留原 dtype 和 device。
**性能影响**：增加一次输入克隆和残差张量显存，默认仅缓存前两步的两个 CFG 分支。
最多保留 4 个 session/episode，LRU 淘汰后下次完整计算。`cached_bytes` 只统计
残差张量字节，不是进程峰值显存。

reset/任务切换时丢弃内存中的旧残差；请求失败、重复或跳号均保守完整计算。
**性能影响**：频繁重连、任务切换或超出 session 容量时命中率下降。
这些操作不删除任何文件。

验证范围（2026-09-11 更新）：18 项 CPU 替代层/协议测试通过；真实 Edge 权重
已在 H800 上完成 RoboLab 闭环批次并确认跨 chunk 缓存命中。1200 episodes
评测进行中；完整模型 dense 数值等价、残差误差和配对性能验证仍待完成。
