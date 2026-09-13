# ASI 真机/仿真公共运行时修复

2026-09-13。基于远端 `experiment/real-robot-asi-inference` 的 `abb8f7d`。
本次仅处理 Edge 公共运行时；没有修改 Piper 客户端或控制机械臂。

## 修复

- `Version1Controller._slice_pack_and_rope` 的 metadata cache 从单一对象改成
  `(实际 UND 长度, GEN 长度, device)` 索引的缓存。conditional 与 unconditional
  的文本长度可以不同，不再因复用首个 sparse stack 的 metadata 而报错。
- 保留逻辑 UND token 数和 padded storage 长度的区别，RoPE 按原 token index 切片。
- compiled decoder 返回的 Profile clone 后保存，避免图的可复用输出存储覆盖之前的层。
- 新增 optimized=True restore/padding 测试、异长 CFG 原始 RoPE 测试、模拟复用
  Profile buffer 测试，以及与原仿真 Core/Stable selector 完全一致的测试。

公共代码与仿真集成分支 `experiment/robolab-asi-optimized` 一致；真机分支不依赖
新增的 RoboLab server 模式开关和 benchmark 工具。

## 计算与验证边界

Core80、Stable104、Top6 block、action 权重 `[1/6,1/3,1/3,1/6]`、one dense/seven
sparse stack、current-stack input restore、CFG/UniPC 公式均未修改。
`optimized` 仍默认 False；已有真机调用者的显式配置保持有效。

RoboLab/DROID 录制输入在项目 Edge 环境做了配对回归：新 optimized-eager 在两组
seed 下 mask、action、vision latent 与旧 ASI 相同。Compile 路径不逐位等价，
会改变 Top-K 边界附近 mask 和输出。真机部署前仍需在实际 Piper 环境用相同
录制输入核对动作；不能将 DROID 的误差或耗时直接当作 Piper14 指标。

这里未移植、重写或验证 Piper 外部的 `reuse_denoise_packing` 和重复图像单帧化。
上述两项仍由真机客户端维护。
