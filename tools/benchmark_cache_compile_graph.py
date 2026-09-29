"""Branch-native cache/ASI compile+graph benchmark, without changing algorithms."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from contextlib import nullcontext
from pathlib import Path

ROOT = Path('/root/robolab')
ASI = ROOT / 'cosmos-framework-edge-core80-stable104-action-weighted'
TREES = {'asi': ASI, **{k: ROOT / 'worktrees' / v for k, v in
         [('toca', 'toca-future'), ('worldcache', 'worldcache'), ('c3ache', 'c3ache')]}}
CAPTURE = TREES['toca'] / 'experiments/dense_asi_toca_smoke10_fidelity_s0_p0_v2/dense/server/captures/request_000002/sample.pt'
VAE = '/root/cosmos3/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth'


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def worker(args):
    from cosmos_framework.inference.common.init import init_script
    init_script()
    import random
    import numpy as np
    import torch
    import cosmos_framework
    from cosmos_framework.configs.base.defaults.compile import CompileConfig
    from cosmos_framework.model.generator.mot.parallelize_vfm_network import apply_compile
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs

    assert Path(cosmos_framework.__path__[0]).resolve() == TREES[args.worker] / 'cosmos_framework'
    torch.set_num_threads(4)
    out = args.output / args.worker
    out.mkdir(parents=True, exist_ok=False)

    class Service(RobolabPolicyService):
        def _build_setup_args(self, settings):
            return super()._build_setup_args(settings).model_copy(
                update={'use_torch_compile': False, 'use_cuda_graphs': False})

    service = Service(RobolabServerArgs(
        checkpoint_path=str(ROOT / 'RoboLab/Cosmos3-Edge-Policy-DROID'),
        seed=0, deterministic_seed=False, guidance=3, num_steps=4, shift=5,
        format_prompt_as_json=True, guardrails=False, output_dir=out / 'model_output',
        experiment_overrides=[f'model.config.tokenizer.vae_path={VAE}',
                              'model.config.tokenizer.object_store_credential_path_pretrained=',
                              'model.config.tokenizer.bucket_name=']))
    model, net = service.model, service.model.net
    capture = torch.load(CAPTURE, map_location='cpu', weights_only=False)
    assert capture['kwargs'] == dict(guidance=3.0, seed=[1097657232], num_steps=4, shift=5.0)
    config, cache = None, None
    if args.worker == 'asi':
        from cosmos_framework.inference.edge_core_stable import Version1Controller
    elif args.worker == 'toca':
        from cosmos_framework.inference.toca_future import ToCaFutureConfig, ToCaFutureController
        config = json.loads((TREES['toca'] / 'configs/toca_baseline.json').read_text())['config']
        native_config = ToCaFutureConfig(**dict(config, full_steps=tuple(config['full_steps'])))
    elif args.worker == 'worldcache':
        from cosmos_framework.inference.worldcache import WorldCacheConfig
        config = json.loads((TREES['worldcache'] / 'configs/worldcache_baseline.json').read_text())['config']
        native_config = WorldCacheConfig(**config)
    else:
        from cosmos_framework.inference.c3ache import C3acheCache, C3acheConfig
        config = json.loads((TREES['c3ache'] / 'configs/c3ache_baseline.json').read_text())['config']
        cache = C3acheCache(C3acheConfig(**config))

    heads = ('_encode_text', '_encode_vision', '_encode_action', '_decode_vision', '_decode_action')
    eager_layers = list(net.language_model.model.layers)
    eager_heads = {k: getattr(net, k) for k in heads}
    cfg = CompileConfig(enabled=True, compiled_region='all', use_cuda_graphs=True,
                        compile_dynamic=model.config.compile.compile_dynamic)
    graph_layers = [torch.compile(layer, fullgraph=True, dynamic=cfg.compile_dynamic,
                                  mode='reduce-overhead') for layer in eager_layers]
    apply_compile(net, cfg)
    graph_heads = {k: getattr(net, k) for k in heads}
    selection_audit = {}

    def metric(a, b):
        a, b = a.double().flatten(), b.double().flatten()
        return dict(mse=(a-b).square().mean().item(),
                    relative_l2=((a-b).norm()/b.norm().clamp_min(1e-24)).item(),
                    cosine=((a@b)/(a.norm()*b.norm()).clamp_min(1e-24)).item())

    def run(mode, chunk=0, seed_offset=0):
        graph = mode.endswith('graph')
        layers, hds = (graph_layers, graph_heads) if graph else (eager_layers, eager_heads)
        for i, layer in enumerate(layers):
            net.language_model.model.layers[i] = layer
        for k, v in hds.items():
            setattr(net, k, v)
        net.pad_for_cuda_graphs = graph or mode == 'strategy_padded_eager'
        # Keep public configuration truthful as well as installing compiled modules;
        # in particular do not bypass branch-native eager-only safety checks.
        model.config.compile.enabled = graph
        model.config.compile.use_cuda_graphs = graph
        positional, kwargs = copy.deepcopy(capture['args']), copy.deepcopy(capture['kwargs'])
        kwargs['seed'] = [kwargs['seed'][0] + seed_offset]
        random.seed(kwargs['seed'][0]); np.random.seed(kwargs['seed'][0])
        torch.manual_seed(kwargs['seed'][0]); torch.cuda.manual_seed_all(kwargs['seed'][0])
        ctrl, summary = None, {}
        strategy = not mode.startswith('dense')
        if strategy and args.worker == 'asi':
            ctrl = Version1Controller(torch=torch, net=net, num_steps=4, guidance=3,
                optimized=True, cache_layout=True, compile_profile_decoder=True,
                compile_profile_kernel=False, cuda_graphs=graph)
        elif strategy and args.worker == 'toca':
            extra = dict(optimized=True, use_compile=graph, cuda_graphs=graph) if args.adapted and mode != 'strategy_eager' else {}
            ctrl = ToCaFutureController(net, native_config, **extra)
        elif strategy and args.worker == 'worldcache':
            kwargs['worldcache_config'] = (replace(native_config, compile_compatible=True)
                if args.adapted and mode != 'strategy_eager' else native_config)
        # Construction and CPU input copies outside generation; context, cache lookup,
        # profiling, selection and all GPU work required per chunk remain inside.
        torch.cuda.synchronize()
        start = time.perf_counter()
        if strategy and cache is not None:
            with cache.request(dict(session_id='paired', episode_id='fixed-input', chunk_id=chunk),
                    signature=dict(prompt=capture['metadata']['prompt'], steps=4, guidance=3, shift=5),
                    transformer=net.language_model.model, net=net,
                    branches=('conditional', 'unconditional')) as req:
                result = model.generate_samples_from_batch(*positional, **kwargs, c3ache_request=req)
            summary = dict(req.stats, reason=req.reason)
        else:
            with ctrl if ctrl else nullcontext():
                result = model.generate_samples_from_batch(*positional, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if ctrl:
            summary = ctrl.finish()
            if args.worker == 'toca':
                selection_audit[mode] = dict(
                    scores={str(k): v.detach().cpu().clone() for k,v in ctrl.scores.items()},
                    indices={str(k): v.detach().cpu().clone() for k,v in ctrl.indices.items()})
        if strategy and args.worker == 'worldcache':
            summary = model._last_worldcache_report
            assert (summary['full_forwards'], summary['cache_forwards']) == (6, 2)
        if strategy and cache is not None:
            expected = (8, 0) if chunk % 2 == 0 else (4, 4)
            assert (summary['dense_forwards'], summary['cache_hits']) == expected, summary
        values = {k: result[k][0].detach().cpu().clone() for k in ('action', 'vision')}
        assert all(torch.isfinite(v).all() for v in values.values()), 'NaN/Inf'
        return values, elapsed, summary

    report = dict(source=str(TREES[args.worker]),
        head=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        python=sys.executable, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        input=str(CAPTURE), input_sha256=hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
        kwargs=capture['kwargs'], config=config, compile_dynamic=cfg.compile_dynamic,
        requested_compile=True, requested_cuda_graph=True,
        boundary='CUDA synchronized generation plus controller context; excludes summary, validation, CPU copies, VAE decode, RPC',
        warmups=args.warmups, repeats=args.repeats)
    report['source_sha256'] = {str(p.relative_to(TREES[args.worker])):hashlib.sha256(p.read_bytes()).hexdigest()
        for name in ('inference/toca_future.py','inference/toca_compiled.py','inference/toca_joint_attention.py',
                     'inference/worldcache.py','inference/c3ache.py','inference/edge_core_stable.py',
                     'model/generator/omni_mot_model.py','model/generator/mot/unified_mot.py',
                     'model/generator/mot/cosmos3_vfm_network.py')
        if (p:=TREES[args.worker]/'cosmos_framework'/name).exists()}
    save(out / 'manifest.json', report)
    try:
        with torch.inference_mode():
            dense, _, _ = run('dense_eager')
            for k in dense:
                torch.testing.assert_close(dense[k],capture['outputs'][k],rtol=0,atol=0)
            report['dense_matches_recorded_baseline_exact'] = True
            eager, _, eager_summary = run('strategy_eager')
            if args.adapted and args.worker in ('toca', 'worldcache'):
                adapted, _, _ = run('strategy_adapted_eager')
                for k in adapted:
                    torch.testing.assert_close(adapted[k], eager[k], rtol=0, atol=0)
                if args.worker == 'toca':
                    for field in ('scores','indices'):
                        for key, value in selection_audit['strategy_eager'][field].items():
                            torch.testing.assert_close(value,selection_audit['strategy_adapted_eager'][field][key],rtol=0,atol=0)
                report['adapter_eager_exact'] = True
                padded, _, _ = run('strategy_padded_eager')
                report['padding_eager_vs_unpadded'] = {k:metric(padded[k],eager[k]) for k in eager}
            if cache: cache.episodes.clear()
            modes = ['dense_eager', 'refresh_graph', 'hit_graph'] if cache else ['dense_eager', 'strategy_graph']
            rows, refs, last = [], {}, {}
            for i in range(args.warmups + args.repeats):
                # Keep refresh immediately followed by hit; alternate pair order vs dense.
                order = modes if i % 2 == 0 else (modes[1:] + modes[:1])
                for mode in order:
                    chunk = 2*i + (mode == 'hit_graph')
                    values, elapsed, summary = run(mode, chunk)
                    if i >= args.warmups:
                        for k in values:
                            torch.testing.assert_close(values[k], refs[mode][k], rtol=1e-5, atol=1e-5)
                        rows.append(dict(repeat=i-args.warmups, mode=mode, seconds=elapsed))
                    refs[mode], last[mode] = values, summary
                    print(f'[chunk] {args.worker} {i+1}/{args.warmups+args.repeats} {mode} {elapsed:.6f}s', flush=True)
            report['samples'] = rows
            report['timing'] = {}
            for mode in modes:
                x = [r['seconds'] for r in rows if r['mode'] == mode]
                report['timing'][mode] = dict(mean_s=float(np.mean(x)), median_s=float(np.median(x)), p90_s=float(np.quantile(x,.9)), n=len(x))
            if cache:
                x = [sum(r['seconds'] for r in rows if r['repeat']==i and r['mode'] in ('refresh_graph','hit_graph'))/2 for i in range(args.repeats)]
                report['timing']['amortized_graph'] = dict(mean_s=float(np.mean(x)), median_s=float(np.median(x)), p90_s=float(np.quantile(x,.9)), n=len(x))
            report['last_controller'] = last
            report['graph_vs_eager'] = {k: metric(refs['refresh_graph' if cache else 'strategy_graph'][k], eager[k]) for k in eager}
            if args.worker == 'toca':
                original, compiled = selection_audit['strategy_eager'], selection_audit['strategy_graph']
                report['toca_score_vs_eager'] = metric(torch.cat(list(compiled['scores'].values())),torch.cat(list(original['scores'].values())))
                report['toca_selected_membership'] = {key:dict(count=len(value),same_order=bool(torch.equal(value,original['indices'][key])),
                    overlap_fraction=float(torch.isin(value,original['indices'][key]).float().mean()))
                    for key,value in compiled['indices'].items()}
            report['finite'] = True
            save(out / 'results.json', report)
            # Separate untimed runtime audit: a requested graph flag is not evidence of replay.
            audit = {}
            for j, mode in enumerate(modes[1:]):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                    run(mode, 2*(args.warmups+args.repeats)+(mode=='hit_graph'))
                events = {e.key: e.count for e in prof.key_averages() if 'graph' in e.key.lower() or 'compiled' in e.key.lower()}
                audit[mode] = events
            report['runtime_audit'] = audit
            if cache: cache.episodes.clear()
            changed, _, _ = run('refresh_graph' if cache else 'strategy_graph', 0, seed_offset=1)
            assert not torch.equal(changed['vision'], refs['refresh_graph' if cache else 'strategy_graph']['vision']), 'Stale output'
            report['changed_seed_gate'] = True
            if args.worker in ('toca','worldcache'):
                changed_eager, _, _ = run('strategy_eager', 0, seed_offset=1)
                report['changed_seed_graph_vs_eager'] = {k:metric(changed[k],changed_eager[k]) for k in changed}
            save(out / 'results.json', report)
            print('[result] '+json.dumps(report['timing']), flush=True)
    except Exception:
        report['error'] = traceback.format_exc()
        save(out / 'failure.json', report)
        raise
    finally:
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker', choices=list(TREES))
    p.add_argument('--modes',nargs='+',default=list(TREES),choices=list(TREES))
    p.add_argument('--warmups',type=int,default=5)
    p.add_argument('--repeats',type=int,default=15)
    p.add_argument('--adapted',action='store_true')
    args=p.parse_args(); args.output=args.output.resolve()
    if args.adapted:
        TREES.update(toca=ROOT/'worktrees/toca-compile-graph', worldcache=ROOT/'worktrees/worldcache-compile-graph')
    if args.worker: return worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    for mode in args.modes:
        env=dict(os.environ, PYTHONPATH=f'{TREES[mode]}:{ROOT}/cosmos-edge-overlay', COSMOS_TRAINING='0',
            LD_LIBRARY_PATH='', HF_HOME='/root/cosmos3/cosmos/checkpoints/hf_home', HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
            TORCHINDUCTOR_CACHE_DIR=str(ROOT/'runtime/inductor_cache'))
        cmd=[str(ASI/'.venv/bin/python'),str(Path(__file__).resolve()),'--worker',mode,'--output',str(args.output),
             '--warmups',str(args.warmups),'--repeats',str(args.repeats)]
        if args.adapted: cmd.append('--adapted')
        print('[start] '+mode,flush=True)
        with (args.output/f'{mode}.log').open('x') as f:
            result=subprocess.run(cmd,cwd=TREES[mode],env=env,stdout=f,stderr=subprocess.STDOUT)
        print(f'[end] {mode} exit={result.returncode}',flush=True)


if __name__ == '__main__': main()
