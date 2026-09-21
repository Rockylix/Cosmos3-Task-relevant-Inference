"""Frozen budget calibration -> correctness gate -> ten fixed RoboLab tasks."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
ROBOLAB=Path('/root/robolab/RoboLab')
PY='/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/.venv/bin/python'
VAE='/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth'
CAPTURE=Path('/root/robolab/cosmos-framework-edge-core80-stable104-action-weighted/experiments/asi_smoke10_s0_p0_v1/_temporary_inputs/BananaInBowlTask.pt')
TASKS=json.loads((ROOT/'configs/specprune_baseline.json').read_text())['protocol']['tasks']
BUDGET=json.loads((ROOT/'configs/specprune_asi_budget.json').read_text())


def save(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))


def settings(out):
    from cosmos_framework.scripts.action_policy_server_specprune import SpecPruneServerArgs
    return SpecPruneServerArgs(checkpoint_path=str(ROBOLAB/'Cosmos3-Edge-Policy-DROID'),
        specprune=True,specprune_backend='eager',guardrails=False,format_prompt_as_json=True,
        seed=0,deterministic_seed=False,num_steps=4,shift=5,guidance=3,decode_video=False,
        specprune_local_k=BUDGET['local_k'],specprune_global_k=BUDGET['global_k'],
        specprune_min_observation_tokens=BUDGET['min_observation_tokens'],specprune_keep_ratio=BUDGET['keep_ratio'],
        output_dir=out/'model_output',experiment_overrides=[f'model.config.tokenizer.vae_path={VAE}',
        'model.config.tokenizer.object_store_credential_path_pretrained=','model.config.tokenizer.bucket_name='])


def worker(out,port):
    import torch
    import numpy as np
    from cosmos_framework.inference.common.init import init_script
    init_script();torch.set_num_threads(4)
    from cosmos_framework.scripts.action_policy_server_specprune import SpecPrunePolicyService
    from cosmos_framework.scripts.action_policy_server_robolab import _load_openpi_websocket_policy_server
    from cosmos_framework.inference.specprune_future import tensor_metrics
    svc=SpecPrunePolicyService(settings(out))
    # Full keep uses the adapter's native generation path. Disable the wrapper
    # briefly so this guard cannot recurse through adapter.generate.
    native=type(svc.model).generate_samples_from_batch.__get__(svc.model,type(svc.model))
    svc.model.generate_samples_from_batch=native
    sample=torch.load(CAPTURE,map_location='cpu',weights_only=False)
    with torch.inference_mode():
        dense=native(*copy.deepcopy(sample['args']),**sample['kwargs'])
        full=svc.adapter.generate(*copy.deepcopy(sample['args']),**sample['kwargs'],force_full=True)
        for key in ('action','vision'):
            assert dense[key][0].shape==full[key][0].shape and torch.equal(dense[key][0],full[key][0]),key
        svc.adapter.reset()
        sparse=svc.adapter.generate(*copy.deepcopy(sample['args']),**sample['kwargs'])
        first=copy.deepcopy(svc.adapter.last_info)
        second=svc.adapter.generate(*copy.deepcopy(sample['args']),**sample['kwargs'])
        for output in (sparse,second):
            assert all(torch.isfinite(output[k][0]).all() for k in ('vision','action'))
        assert all(r['patch_tokens']==3060 for r in svc.adapter.solver_rows)
        gate=dict(full_keep_exact=True,first=first,second=svc.adapter.last_info,
                  budget=BUDGET,action_error=tensor_metrics(sparse['action'][0],dense['action'][0]),
                  history='gate only: repeated same input; NOT used in rollout')
        save(out/'gate.json',gate)
    # The closed loop always starts with fresh history and original policy RNG.
    svc.model.generate_samples_from_batch=svc.adapter.generate
    svc.reset()
    del dense,full,sparse,second
    torch.cuda.empty_cache()
    metadata=json.loads((ROBOLAB/'robolab/tasks/_metadata/task_metadata.json').read_text())
    names={r['instruction']:r['task_name'] for r in metadata if r['task_name'] in TASKS}
    original=svc.infer
    def infer(obs):
        if obs['prompt'] not in names: raise ValueError('Unexpected task')
        result=original(obs)
        info=svc.adapter.last_info
        tokens=svc.adapter.last_token_rows
        mean_gen=float(np.mean([r['gen_tokens'] for r in tokens]))
        mean_future=(mean_gen-373)/2720
        row=dict(task=names[obs['prompt']],chunk=svc.request_chunk,seed=info['seed'],
            generation_s=info['generation_with_scoring_s'],selected_future_tokens=info['selected_future_tokens'],
            mean_gen_tokens=mean_gen,mean_future_retention=mean_future,gen_retention=mean_gen/3093,
            conditional_text_tokens=info['conditional_text_tokens'],layers=info['selection']['layers'],
            dynamic_count=info['selection']['dynamic_count'],global_count=info['selection']['global_count'],
            attention_validation_max_rel_l2=max(v['relative_l2'] for v in info['attention_validation']))
        with (out/'requests.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(f"CHUNK {row['task']} c{row['chunk']} future_keep={mean_future:.2%} final={row['selected_future_tokens']}/2720",flush=True)
        return result
    svc.infer=infer
    print('ASI_BUDGET_SERVER_READY',flush=True)
    _load_openpi_websocket_policy_server()(policy=svc,host='127.0.0.1',port=port,metadata={}).serve_forever()


def read_rows(path):
    if not path.exists():return []
    return [json.loads(s) for s in path.read_text().splitlines(keepends=True) if s.endswith('\n') and s.strip()]


def summarize(out):
    import numpy as np
    episodes=read_rows(out/'simulator/episode_results.jsonl')
    requests=read_rows(out/'requests.jsonl')
    assert len({r['task_name'] for r in episodes})==len(episodes)<=10
    assert {r['task_name'] for r in episodes}<=set(TASKS)
    success=sum(bool(r['success']) for r in episodes)
    scores=[1. if r['success'] else float(r['score']) for r in episodes]
    stats=dict(completed=len(episodes),total=10,success=success,success_rate=success/len(episodes) if episodes else None,
        official_score_mean=float(np.mean(scores)) if scores else None,
        raw_score_mean=float(np.mean([r['score'] for r in episodes])) if episodes else None,
        chunks=len(requests),mean_future_retention=float(np.mean([r['mean_future_retention'] for r in requests])) if requests else None,
        mean_gen_retention=float(np.mean([r['gen_retention'] for r in requests])) if requests else None,
        asi_future_retention_target=(340+7*184)/(8*340),episodes=episodes)
    warm=[r['generation_s'] for r in requests if r['chunk']>1]
    if warm:stats['generation_excluding_first_chunk_each_task']=dict(mean=float(np.mean(warm)),median=float(np.median(warm)),p90=float(np.percentile(warm,90)),n=len(warm),
        boundary='adapter generation with scoring/checks; CUDA synchronized; excludes RPC/simulation; not same-input alternating benchmark')
    save(out/'status.json',stats)
    return stats


def run(out,port):
    with socket.socket() as s:s.bind(('127.0.0.1',port))
    out.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'LD_LIBRARY_PATH':'','CUDA_VISIBLE_DEVICES':'0','COSMOS_TRAINING':'0',
         'PYTHONPATH':str(ROOT)+':/root/robolab/cosmos-edge-overlay','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'PYTHONUNBUFFERED':'1','NO_PROXY':'localhost,127.0.0.1','no_proxy':'localhost,127.0.0.1',
         'OMNI_KIT_ACCEPT_EULA':'Y','ACCEPT_EULA':'Y','PRIVACY_CONSENT':'Y'}
    env.pop('LD_PRELOAD',None)
    server_cmd=[PY,str(Path(__file__).resolve()),'--mode','server','--output',str(out),'--port',str(port)]
    sim_cmd=[str(ROBOLAB/'.venv/bin/python'),'policies/cosmos3/run.py','--remote-host','127.0.0.1','--remote-port',str(port),
        '--task',*TASKS,'--num-envs','1','--num-runs','1','--headless','--video-mode','none','--output-folder-name',str(out/'simulator')]
    save(out/'manifest.json',dict(budget=BUDGET,tasks=TASKS,policy_seed=0,simulator_seed_expected=0,num_steps=4,shift=5,guidance=3,
        compile=False,cuda_graphs=False,rollouts_per_task=1,task_step_limits='official',retry=False,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            ROOT/'cosmos_framework/inference/specprune_exit_plan.py',ROOT/'cosmos_framework/inference/specprune_observation_exit.py',
            ROOT/'cosmos_framework/scripts/action_policy_server_specprune.py',Path(__file__)]},
        server=server_cmd,simulator=sim_cmd))
    owned=[]
    try:
        with (out/'server.log').open('w') as f:p=subprocess.Popen(server_cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        owned.append(p);deadline=time.monotonic()+600
        while 'ASI_BUDGET_SERVER_READY' not in (out/'server.log').read_text(errors='replace'):
            if p.poll() is not None or time.monotonic()>deadline:raise RuntimeError('GPU gate/server failed; inspect server.log')
            time.sleep(2)
        with (out/'simulator.log').open('w') as f:
            sim=subprocess.Popen(sim_cmd,cwd=ROBOLAB,env={**env,'PYTHONPATH':str(ROBOLAB)+':/root/robolab/cosmos-edge-overlay'},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        owned.append(sim);last=None
        while sim.poll() is None:
            if p.poll() is not None:raise RuntimeError('Policy server exited')
            s=summarize(out);key=(s['completed'],s['chunks'])
            if key!=last:
                print(f"PROGRESS {s['completed']}/10 success={s['success']} rate={s['success_rate']} score={s['official_score_mean']} chunks={s['chunks']} future_keep={s['mean_future_retention']}",flush=True)
                last=key
            time.sleep(5)
        s=summarize(out)
        if sim.returncode or s['completed']!=10:raise RuntimeError('Incomplete rollout; no automatic retry')
        for f in (out/'simulator').glob('*/env_cfg.json'):
            assert json.loads(f.read_text())['seed']==0
        s['complete']=True;save(out/'results.json',s)
        print('TEN_TASKS_COMPLETE '+json.dumps(s,ensure_ascii=False),flush=True)
    finally:
        for p in reversed(owned):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['run','server','summary'],default='run')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--port',type=int,default=8053)
    a=ap.parse_args()
    if a.mode=='server':worker(a.output.resolve(),a.port)
    elif a.mode=='summary':print(json.dumps(summarize(a.output.resolve()),indent=2))
    else:run(a.output.resolve(),a.port)
