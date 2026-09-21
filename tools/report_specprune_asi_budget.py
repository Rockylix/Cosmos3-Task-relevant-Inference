"""Summarize completed fixed ten-task run; no inference, tuning, or filtering."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True)
    a=ap.parse_args();p=a.run
    result=json.loads((p/'results.json').read_text());assert result['complete'] and result['completed']==10
    chunks=[json.loads(s) for s in (p/'requests.jsonl').read_text().splitlines() if s.strip()]
    rows=[]
    for episode in result['episodes']:
        task=episode['task_name'];items=[x for x in chunks if x['task']==task]
        rows.append(dict(task=task,success=bool(episode['success']),score=episode['score'],
            steps=episode.get('episode_step',episode.get('steps')),chunks=len(items),
            mean_future_retention=float(np.mean([x['mean_future_retention'] for x in items])),
            mean_final_tokens_per_frame=float(np.mean([x['selected_future_tokens']/8 for x in items])),
            generation_median_s=float(np.median([x['generation_s'] for x in items if x['chunk']>1])) if len(items)>1 else None))
    with (p/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    last=[x['mean_future_retention'] for x in chunks if x['chunk']>1]
    target=result['asi_future_retention_target'];actual=result['mean_future_retention']
    report=['# SpecPrune 近似匹配 ASI 保留比例：十任务结果','',
        '原十任务，simulator seed=0、policy seed=0、shift5、4steps、guidance3；eager；每任务一次，不筛选、不重试。',
        '预算在运行前冻结：Local58、Global72、动态收缩下限184、keep_ratio0.9。评分与恢复源码未修改。','',
        '| 策略 | 成功率 | 官方平均score | 数据来源 |','|---|---:|---:|---|',
        '| Dense eager | 4/10 | 0.4667 | 之前固定十任务，非本轮重跑 |',
        '| 原 SpecPrune eager | 1/10 | 0.2667 | 之前固定十任务，非本轮重跑 |',
        f"| 本轮 SpecPrune budget | {result['success']}/10 | {result['official_score_mean']:.4f} | 本轮实测 |",'',
        f'- ASI 目标 future 平均保留率：{target:.4%}。',
        '- 旧评分离线代理校准：59.9945%。',
        f'- 新轨迹实测（全部chunk）：{actual:.4%}，与目标差 {(actual-target)*100:+.3f} 个百分点。',
        f'- 不含每任务首chunk：{np.mean(last):.4%}。',
        f"- 包含L0/action的GEN平均保留率：{result['mean_gen_retention']:.4%}。",'',
        '以上不是FLOPs或速度匹配。ASI与SpecPrune对L0/UND的缓存和额外评分不同。',
        '收缩下限不是强制填充；无历史时静态选择可少于184。未因结果偏离目标而重调或重跑。','',
        '## 逐任务','', '| Task | Success | Score | Steps | Future retention |','|---|---:|---:|---:|---:|']
    report += [f"| {r['task']} | {int(r['success'])} | {r['score']:.4f} | {r['steps']} | {r['mean_future_retention']:.2%} |" for r in rows]
    report += ['', '## 描述性 generation 时间','',
               '去掉每任务首chunk；包含在线评分和检查、CUDA同步；不含RPC/仿真/保存。不是固定输入交替测速，不据此报告相对ASI速度。','',
               '```json',json.dumps(result.get('generation_excluding_first_chunk_each_task',{}),indent=2),'```','']
    (p/'report_cn.md').write_text('\n'.join(report))
    print('\n'.join(report[:22]))


if __name__=='__main__':main()
