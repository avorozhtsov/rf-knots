"""Continuous native-MPS R2 with hourly full checkpoints and minute status."""
import argparse
import fcntl
import json
import math
import os
import shutil
import signal
import time
from pathlib import Path

import torch
import train_continuous_additive_r2_v11 as generic
from hourly_checkpoint_policy_v1 import CheckpointPolicy

base = generic.base
prior = generic.prior
engine = generic.engine
MIN_FREE = 12 * 1024**3
PROTOCOL = Path('research/399-hourly-full-checkpoint-cadence-v15.md')


def setup(spec_path):
    generic.SPEC = spec_path
    generic.SETTINGS = s = base.read(spec_path)
    prior.BANK = Path(s['bank'])
    prior.BANK_AUDIT = Path(s['audit'])
    prior.PARENT = Path(s['parent'])
    prior.START_STEP = s['sampler_origin_step']
    prior.bank_order = generic.bank_order
    prior.initialize_order = generic.initialize_order
    prior.selected_group = generic.selected_group
    # Keep EXACT previous sampler phase; parent step changes only persistence.
    order = generic.initialize_order()
    base.BANK = prior.BANK
    base.bank_order = generic.bank_order
    base.selected_group = generic.selected_group
    generic.setup()
    base.CONFIG.update(checkpoint_interval=None, checkpoint_interval_seconds=3600,
                       progress_interval_seconds=60, retained_root_draw_fraction=0.9,
                       new_root_only_F_no_aux_rehearsal=True)
    if s['checkpoint_interval_seconds'] != 3600:
        raise ValueError('frozen hourly interval differs')
    paths = [spec_path, prior.BANK/'manifest.json', prior.BANK_AUDIT,
             prior.PARENT, prior.PARENT.parent/'manifest.json', PROTOCOL,
             Path(__file__), Path('scripts/hourly_checkpoint_policy_v1.py'),
             Path('scripts/train_continuous_additive_r2_v11.py'),
             Path('scripts/train_additive_root_only_r2_v3.py'),
             Path('scripts/train_joint_f_r2_continuation_v1.py'),
             Path('scripts/train_joint_f_r2_r4_v1.py'),
             Path('scripts/train_joint_f_r2_r4_bounded_v2.py'),
             Path('scripts/train_joint_f_r2_r4_smoothed_v3.py'),
             Path('scripts/mcts_tournament_signals_v1.py'),
             Path('scripts/tournament_term_sampler_v1.py'),
             Path('scripts/pretrained_raster_policy.py'),
             Path('scripts/mcts_training_supervision.py')]
    frozen = {str(p): base.sha(p) for p in paths}
    for p, h in [(prior.BANK/'manifest.json', s['bank_sha256']),
                 (prior.BANK_AUDIT, s['audit_sha256']),
                 (prior.PARENT, s['parent_sha256']),
                 (prior.PARENT.parent/'manifest.json', s['parent_manifest_sha256'])]:
        if frozen[str(p)] != h:
            raise ValueError('frozen bank/parent identity differs')
    parent = base.read(prior.PARENT.parent/'manifest.json')
    audit = base.read(prior.BANK_AUDIT)
    if (parent['status'] != 'checkpointed' or parent['optimizer_steps'] != s['parent_step']
            or parent['checkpoints'][-1]['sha256'] != s['parent_sha256']
            or audit['status'] != 'passed' or audit['distinct_F_roots'] != s['F_roots']):
        raise ValueError('parent/audit identity differs')
    return s, order, frozen


def numerical_gate(metrics):
    f = metrics['F_encoder_grad_norm']
    if not math.isfinite(metrics['F_loss']) or metrics['F_loss'] < 0 or not math.isfinite(f) or f <= 1e-6:
        raise ValueError('F numerical gate failed')
    for key, cap in [('aux', .100001), ('rehearsal', .200001)]:
        if metrics[key+'_scale'] * metrics[key+'_encoder_grad_norm'] > cap * f:
            raise ValueError('effective gradient ratio gate failed')


def exact_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(exact_tree(x,y) for x,y in zip(a,b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, default=0, help='smoke only; zero means continuous')
    args = parser.parse_args()
    s = base.read(args.spec)
    out = Path(s['out'])
    if args.stop_after < 0 or out.exists() != args.resume:
        raise ValueError('output/resume/smoke arguments differ')
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise RuntimeError('native MPS without fallback required')
    if shutil.disk_usage(out.parent).free < MIN_FREE:
        raise RuntimeError('storage gate before compute')
    torch.set_num_threads(1)
    s, order, frozen = setup(args.spec)
    contract = dict(schema='rf-knots-hourly-R2-F-v15', arm='R2', parent_step=s['parent_step'],
                    config=base.CONFIG.copy(), inputs=frozen, eligible_roots=order,
                    sampler_origin_step=s['sampler_origin_step'])
    digest = base.digest(contract)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mp = out/'manifest.json'
        m = base.read(mp) if args.resume else dict(contract=contract, contract_sha256=digest,
            status='preparing', optimizer_steps=s['parent_step'], checkpoints=[],
            new_active_seconds=0., new_cpu_seconds=0., stage_draws=None)
        if m['contract'] != json.loads(json.dumps(contract)) or m['contract_sha256'] != digest:
            raise ValueError('resume contract differs')
        for e in m['checkpoints']:
            if base.sha(out/e['file']) != e['sha256']:
                raise ValueError('saved checkpoint SHA differs')
        e = m['checkpoints'][-1] if m['checkpoints'] else None
        saved = torch.load(out/e['file'] if e else prior.PARENT, map_location='cpu', weights_only=False)
        expected_contract = digest if e else base.read(prior.PARENT.parent/'manifest.json')['contract_sha256']
        if saved['step'] != m['optimizer_steps'] or saved['contract_sha256'] != expected_contract:
            raise ValueError('loaded full checkpoint identity differs')
        if m['stage_draws'] is None:
            m['stage_draws'] = saved['stage_draws'].copy()
        if saved['stage_draws'] != m['stage_draws']:
            raise ValueError('stage draw state differs')
        net, objective, head, optimizer, rng = engine.make_objects(saved)
        for key, value in [('model', net.state_dict()), ('objective', objective.state_dict()),
                           ('auxiliary_head', head.state_dict()), ('optimizer', optimizer.state_dict()),
                           ('rng', rng.getstate()), ('torch_rng', torch.get_rng_state()),
                           ('mps_rng', torch.mps.get_rng_state())]:
            if not exact_tree(saved[key], value):
                raise ValueError('restore is not bitwise exact: '+key)
        stop = []
        def request_stop(signum, frame):
            stop.append('signal_'+str(signum))
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        step = start = m['optimizer_steps']
        wall = cpu = 0.
        policy = CheckpointPolicy(time.monotonic())
        last_progress = 0.
        m.update(status='training', pid=os.getpid())
        base.atomic(mp, m)
        while True:
            reason = (stop[-1] if stop else 'user_stop_request' if (out/'STOP_REQUEST').exists()
                      else 'storage_gate' if shutil.disk_usage(out).free < MIN_FREE
                      else 'smoke_complete' if args.stop_after and step-start >= args.stop_after else None)
            pending = step-m['optimizer_steps']
            now = time.monotonic()
            if policy.due(now, pending, stopping=bool(reason)):
                numerical_gate(metrics)
                checkpoint = dict(contract_sha256=digest, step=step,
                    model={k:v.detach().cpu() for k,v in net.state_dict().items()},
                    objective={k:v.detach().cpu() for k,v in objective.state_dict().items()},
                    auxiliary_head={k:v.detach().cpu() for k,v in head.state_dict().items()},
                    optimizer=optimizer.state_dict(), rng=rng.getstate(), torch_rng=torch.get_rng_state(),
                    mps_rng=torch.mps.get_rng_state(), stage_draws=m['stage_draws'].copy(), metrics=metrics,
                    root_index=root['root_index'], group_index=group_index)
                path = out/f'step-{step:06d}.pt'
                partial = path.with_suffix('.partial')
                torch.save(checkpoint, partial)
                partial.replace(path)
                m['checkpoints'].append(dict(file=path.name, sha256=base.sha(path), step=step))
                m.update(optimizer_steps=step, new_active_seconds=m['new_active_seconds']+wall,
                         new_cpu_seconds=m['new_cpu_seconds']+cpu, metrics=metrics)
                wall = cpu = 0.
                policy.committed(time.monotonic())
                base.atomic(mp, m)
            if reason:
                m.update(status='checkpointed', pid=None, stop_reason=reason)
                base.atomic(mp, m)
                return
            if now-last_progress >= 60:
                base.atomic(out/'progress.json', dict(pid=os.getpid(), timestamp=time.time(),
                    live_optimizer_steps=step, checkpointed_optimizer_steps=m['optimizer_steps'],
                    unsaved_steps=step-m['optimizer_steps'], unsaved_active_seconds=wall,
                    checkpointed_new_active_seconds=m['new_active_seconds'],
                    metrics=metrics if step > start else m.get('metrics'),
                    free_bytes=shutil.disk_usage(out).free))
                last_progress = now
            if any(base.sha(Path(p)) != h for p,h in frozen.items()):
                raise ValueError('pinned training input changed')
            t,c = time.monotonic(), time.process_time()
            net.train(); objective.train(); head.train()
            root,group,group_index = base.selected_group(order, step)
            metrics = base.train_step(net,objective,head,optimizer,group,step)
            torch.mps.synchronize()
            numerical_gate(metrics)
            wall += time.monotonic()-t
            cpu += time.process_time()-c
            m['stage_draws'][root['stage']] += 1
            step += 1


if __name__ == '__main__':
    main()
