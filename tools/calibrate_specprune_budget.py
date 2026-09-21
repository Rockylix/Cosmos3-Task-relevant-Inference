"""Budget-only proxy calibration on frozen attention, never on success labels."""
import argparse
import json
from pathlib import Path
import numpy as np


def top(score,k,candidates=None):
    if candidates is None: candidates=np.ones(340,dtype=bool)
    ids=np.flatnonzero(candidates)
    out=np.zeros(340,dtype=bool)
    out[ids[np.argsort(-score[ids],kind='stable')[:k]]]=True
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    requests=[json.loads(s) for s in (args.source/'requests.jsonl').read_text().splitlines() if s.strip()]
    info={(r['task'],r['chunk']):r for r in requests}
    entries=[]
    for task in sorted((args.source/'artifacts').iterdir()):
        previous=None
        for file in sorted(task.glob('chunk_*/raw_scores.npz')):
            scores=dict(np.load(file));c=int(file.parent.name.split('_')[1]);r=info[task.name,c]
            entries.append((task.name,c,scores,previous,int(r['conditional_text_tokens'])+33))
            previous=scores
    assert len(entries)==len(requests)
    target=(340+7*184)/(8*340)
    results=[]
    for local in range(32,129,2):
        glob=round(local*40/32)
        ratios=[]; final=[]; per_task={}
        for task,c,s,prev,nonvisual in entries:
            globmask=np.zeros(340,dtype=bool) if prev is None else top(prev['B13'],glob)|top(prev['B27'],glob)
            dynamic=s['dynamic'].astype(bool)
            first=top(s['B0'],local)
            active=top(s['B0'],2*local)|globmask|dynamic
            b0=int(active.sum())
            active &= top(s['B1'],local,active)|first|globmask|dynamic
            n=int(active.sum());counts=[340,b0]+[n]*9
            # Only counts matter for compute. Original importance scoring selects
            # identities, which does not change the following layer budgets.
            for width in (5,5,5,2):
                n=min(n,max(int(.9*(n+nonvisual)),nonvisual+184)-nonvisual)
                counts.extend([n]*width)
            assert len(counts)==28
            ratio=float(np.mean(counts)/340)
            ratios.append(ratio);final.append(n);per_task.setdefault(task,[]).append(ratio)
        results.append(dict(local_k=local,global_k=glob,min_observation_tokens=184,keep_ratio=.9,
            mean_future_retention=float(np.mean(ratios)),mean_per_task_retention=float(np.mean([np.mean(v) for v in per_task.values()])),
            error_to_asi=abs(float(np.mean(ratios))-target),min_chunk_retention=min(ratios),max_chunk_retention=max(ratios),
            mean_final_tokens=float(np.mean(final)),min_final_tokens=min(final),max_final_tokens=max(final)))
    results.sort(key=lambda r:r['error_to_asi'])
    best=results[0]
    report=dict(asi_target=target,samples=len(entries),source=str(args.source),selected=best,candidates=results,
                limitation='Frozen-score budget proxy, not actual new-policy masks. No success labels accessed. Validate actual retention during rollout; freeze before evaluation.')
    (args.output/'calibration.json').write_text(json.dumps(report,indent=2))
    config={k:best[k] for k in ('local_k','global_k','min_observation_tokens','keep_ratio')}
    (args.output/'budget.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(dict(target=target,best=best,top5=results[:5]),indent=2))


if __name__=='__main__':main()
