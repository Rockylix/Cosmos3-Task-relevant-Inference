"""Validate and summarize the completed four-strategy Dense-eager comparison."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args = parser.parse_args()
    root=args.results.resolve()
    labels=dict(asi='ASI Core80+Stable104',toca='ToCa D/C/D/C r=0.25',worldcache='WorldCache D/D/D/C',c3ache='C3ache period=2 均摊')
    rows, data = [], {}
    for mode in labels:
        d=json.loads((root/mode/'results.json').read_text())
        assert d['finite'] and d['changed_seed_gate'] and d['dense_matches_recorded_baseline_exact']
        assert d['warmups']==5 and d['repeats']==15
        for relative,digest in d['source_sha256'].items():
            assert hashlib.sha256((Path(d['source'])/relative).read_bytes()).hexdigest()==digest,relative
        for audit in d['runtime_audit'].values():
            assert audit.get('cudaGraphLaunch',0)>0
        if mode in ('worldcache','toca'): assert d['adapter_eager_exact']
        if data:
            assert d['input_sha256']==data['asi']['input_sha256'] and d['kwargs']==data['asi']['kwargs']
        data[mode]=d
        t=d['timing']['amortized_graph' if mode=='c3ache' else 'strategy_graph']
        rows.append(dict(strategy=labels[mode],mean_s=t['mean_s'],median_s=t['median_s'],p90_s=t['p90_s'],
                         paired_dense_eager_median_s=d['timing']['dense_eager']['median_s'],
                         speedup=d['timing']['dense_eager']['median_s']/t['median_s']))
    with (root/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (root/'comparison.json').write_text(json.dumps(dict(rows=rows,source={m:d['source'] for m,d in data.items()},
        input_sha256=data['asi']['input_sha256'],kwargs=data['asi']['kwargs']),indent=2,ensure_ascii=False)+'\n')
    lines=['| 策略 | Mean/s | Median/s | P90/s | 配对 Dense eager median/s | 加速比 |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['strategy']} | {r['mean_s']:.6f} | {r['median_s']:.6f} | {r['p90_s']:.6f} | {r['paired_dense_eager_median_s']:.6f} | {r['speedup']:.3f}× |")
    text='''# 四策略编译适配与 Dense eager 单 chunk 比较（2026-09-13）

本轮完整完成四策略测试。**Dense 始终原生 eager，不启用 Torch Compile、CUDA Graph或Graph padding。**
ASI、ToCa、WorldCache、C3ache启用compile+Graph。所有Dense输出均与同一录制Baseline逐元素一致。

## 稳定单 chunk 时间

'''+ '\n'.join(lines)+'''

GPU为独占RTX4090，Torch2.10.0+cu130、项目原有Edge venv。固定BananaInBowlTask第3个request输入，
seed1097657232、4步UniPC、shift5、guidance3；每模式5次预热后测15次，进程内Dense/策略交替。
不同源码分支分进程顺序运行，不同时占GPU；加速比各自除以同进程配对Dense，未混用历史时间。
各进程Dense随测量时段略有波动，故表中显式列出各自Dense，不能把分母当成同一个常数。

计时：CUDA同步后的完整`generate_samples_from_batch`+controller context。
包含condition VAE encode、prepare/pack、profile、token选择、缓存、CFG、UniPC；
不含CPU输入deepcopy/RNG初始化/controller对象构造、finish统计、CPU输出复制、精度验证、future VAE decode、RPC、仿真。
首次compile/capture、profiler审计均不计入稳定样本。

'''
    c=data['c3ache']['timing']
    text+=f"C3ache刷新median **{c['refresh_graph']['median_s']:.6f}s**，命中median **{c['hit_graph']['median_s']:.6f}s**。\n"
    text+='''每2个chunk刷新一次，刷新D/D/D/D，命中C/C/D/D。
对每个周期先计算`(refresh+hit)/2`，再报告15个周期的mean/median/P90。
不能只报告命中时间。该测试重复同一观测并递增chunk id，仅测刷新/复用成本，不是跨真实观测精度验证。

## 真实Graph执行证据

使用native层级与编码/输出头的torch.compile/Inductor Graph，不是把整条去噪过程捕获成一张图。
Python调度、token选择、WorldCache外推仍在原路径执行；ToCa cached tensor函数额外编译。
profiler在计时完成后单独采样：

'''
    for m,d in data.items():
        text+='- '+labels[m]+': '+', '.join(f"{kind}={a['cudaGraphLaunch']}次cudaGraphLaunch" for kind,a in d['runtime_audit'].items())+'。\n'
    text+='''
ToCa/ASI为224层调用+40编码/输出头；WorldCache为6×(28+5)=198；
C3ache刷新264、命中4×28+40=152。新编译接口不静默回退成eager。

## 等价性与数值边界

ToCa新接口在eager下与旧eager的最终action/vision、score、fresh indices逐元素一致。
WorldCache新projection返回接口在eager下与旧hook接口最终输出逐元素一致。
新接口数学定义、ToCa评分/年龄/缓存范围、WorldCache分位数/外推、CFG和UniPC未改。

但是compiled+padding运行不逐位等价。以下比较各策略compile+Graph与该策略eager，不是与Dense比较：

| 策略 | 原始action MSE | 原始action rel-L2 | vision rel-L2 | vision cosine |
|---|---:|---:|---:|---:|
'''
    for m,d in data.items():
        e=d['graph_vs_eager']
        text+=f"| {labels[m]} | {e['action']['mse']:.8f} | {e['action']['relative_l2']:.6f} | {e['vision']['relative_l2']:.6f} | {e['vision']['cosine']:.6f} |\n"
    s=data['toca']['toca_selected_membership']
    overlap=sum(x['overlap_fraction'] for x in s.values())/len(s)
    text+=f'''
ToCa共{len(s)}组cached-step/block/branch选择，成员重合率均值 **{overlap*100:.3f}%**；
score relative-L2={data['toca']['toca_score_vs_eager']['relative_l2']:.6f}。
浮点顺序和padding路径会造成部分排序变化，不能宣称选token完全相同。
单独padded-eager比较已记录在各results.json的`padding_eager_vs_unpadded`，
因此不把所有数值差异单独归因于compile。
上述raw action包含原始输出全部行/维度，vision包含原始全部latents；不是物理关节误差或future RGB指标。
C3ache该精度行是刷新chunk，不代表跨chunk复用精度。

所有最终输出有限、相同输入多次重放稳定、改变seed得到不同输出。
ToCa/WorldCache另保存seed+1的graph-vs-eager误差。
本轮仅实现和速度验证，不重跑闭环，不声明成功率、action或视觉质量保持。

## 测试、review与分支

- ToCa：29项CPU测试；actual-padding、两个CFG UND长度、normalized/raw/cached UND K、r=0/0.25/1、MLP更新与age/branch隔离。
- WorldCache：20项CPU测试；旧hook与新output接口、真实denoise旁路、padding历史、缓存clone、CFG/UniPC。
- 两个分支均完成两轮独立agent review；统一benchmark也经独立review。
- 原ToCa `f7cf5d6`、WorldCache `763b6d1` worktree未改变。
- 新分支`experiment/toca-compile-graph`：`/root/robolab/worktrees/toca-compile-graph`。
- 新分支`experiment/worldcache-compile-graph`：`/root/robolab/worktrees/worldcache-compile-graph`。
- 未merge、commit或push；修改留在独立worktree供检查。ASI/C3ache生产代码未改。

## 数据与复现

原始目录：`{root}`。
各策略`results.json`保存全部计时、编译审计、原始配置、git HEAD、实际推理源码SHA256和数值验证；
本表生成前逐一核对输入哈希、配置、源码哈希、15样本、实际Graph launch与所有gate。
`comparison.csv`/`comparison.json`为小型汇总，逐策略日志保留。

```bash
cd /root/robolab/worktrees/toca-compile-graph
/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python \\
  tools/benchmark_cache_compile_graph.py --adapted \\
  --modes asi toca worldcache c3ache --warmups 5 --repeats 15 \\
  --output experiments/cache_compile_adapt_repeat
```

首次编译比稳定运行慢；输出必须使用新目录。父进程会继续执行后续策略，即使某分支失败；
最终是否完成需检查所有results.json/验证gate，不能只看父进程退出码。
服务器配置和工作原理见各分支`docs/compile_graph_adapter_cn.md`。
'''
    (root/'report_cn.md').write_text(text)
    for m in ('toca','worldcache'):
        (Path(data[m]['source'])/'docs/compile_graph_benchmark_cn.md').write_text(text)
    (Path(data['asi']['source'])/'docs/cache_compile_adapt_benchmark_cn.md').write_text(text)
    print('\n'.join(lines))


if __name__=='__main__':main()
