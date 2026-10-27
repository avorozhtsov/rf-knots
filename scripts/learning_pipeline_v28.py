"""Versioned v28 supply and additive audited intake. Never edits active v27.

Reference generation is supply/proof only, NEVER a positive training label.
The collector is bounded and resumable by explicit batch cursors, not new seeds.
"""
import argparse
import collections
import fcntl
import hashlib
import json
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from build_mcts_training_supervision import atomic, sha
from run_continuous_train_mcts_v1 import read
from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness
from rf_knots.embedding_normalization import fast_normalize_global_view as norm
from rf_knots.embedding_replay_corpus import invariants, owner
from export_mcts_root_positives_v1 import external_groups, semantic_id

PROTOCOL = Path('research/460-v28-connected-learning-protocol.md')
SEED = 20261023460
MIN_FREE = 12 * 1024**3


def code_pins():
    files = [PROTOCOL, *Path('src/rf_knots').glob('*.py'),
             *[Path('scripts')/n for n in ['learning_pipeline_v28.py',
                'verified_reduction_search_v28.py', 'positive_sampler_v28.py',
                'train_continuous_random_view_r2_v28.py', 'smoke_connected_learning_cpu_v28.py']],
             Path('tests/test_connected_learning_v28.py')]
    return {str(p): sha(p) for p in files}


def disk_guard(path=Path('artifacts')):
    if shutil.disk_usage(path).free < MIN_FREE:
        raise ValueError('disk guard:12GiB, no automatic deletion')


def pins_ok(pins):
    for p, h in pins.items():
        if sha(Path(p)) != h:
            raise ValueError('frozen input changed: ' + p)


def instance(data):
    return M.Instance(**{**data, 'word': tuple(data['word'])})


def protected_registry():
    """Separate holdout protection from TRAIN novelty; never exclude TRAIN as heldout."""
    from build_continuous_train_roster_v1 import exclusions
    _, held = exclusions()
    groups = set(external_groups())
    states = {norm(*s) for s in held}
    known = set(states)
    pins = {}
    # All existing panels, including the completed A/B panel, remain protected.
    for p in sorted(Path('artifacts').glob('*/panel.json')):
        pins[str(p)] = sha(p)
        for row in read(p).get('rows', []):
            if 'query_group' in row:
                groups.add(tuple(row['query_group']))
            if 'witness' in row:
                w = UnknotWitness.from_dict(row['witness'])
                states.add(norm(w.start.word, w.start.strands))
                for s in [w.start, *[x.after for x in w.steps]]:
                    k = norm(s.word, s.strands)
                    if len(k[0]) >= 20:
                        states.add(k)
    # Never recycle unsolved/partial supply rosters as new opportunities.
    roster_paths = [*Path('artifacts/continuous-random-TRAIN-stream-v1').glob('batch-*/supply.json'),
                    *Path('artifacts/continuous-random-TRAIN-stream-v2').glob('batch-*/supply.json'),
                    Path('artifacts/random-horizon48-feasibility-v3/supply.json')]
    for p in roster_paths:
        if not p.exists():
            continue
        pins[str(p)] = sha(p)
        for row in read(p).get('rows', []):
            w = UnknotWitness.from_dict(row['witness'])
            known.add(norm(w.start.word, w.start.strands))
    ref = read(Path('artifacts/continuous-r2-current-training.json'))
    spec = read(Path(ref['spec']))
    for key in ['bank', 'positive_bank', 'random_positive_bank']:
        bank = Path(spec[key]); m = read(bank / 'manifest.json')
        pins[str(bank / 'manifest.json')] = sha(bank / 'manifest.json')
        for e in m.get('paths', m.get('roots', [])):
            p = bank / e['file']; assert sha(p) == e['sha256']
            r = read(p)
            for d in ([r['instance']] if 'instance' in r else [g['instance'] for g in r['groups']]):
                known.add(norm(tuple(d['word']), d['strands']))
    known.update(states)
    return groups, states, known, pins


def admit(w, inst, groups, states):
    w.verify()
    fp = invariants(w.start.word, w.start.strands)
    group = (fp[0], abs(fp[1]))
    if owner(fp) != 'train' or group in groups or norm(w.start.word, w.start.strands) in states:
        raise ValueError('heldout/ownership violation')
    if (inst.word, inst.strands) != (w.start.word, w.start.strands):
        raise ValueError('raw root differs')
    if w.crossing_changes > inst.cap_crossing_changes or w.moves > inst.cap_moves:
        raise ValueError('resource caps differ')
    seen = {(w.start.word, w.start.strands)}
    for step in w.steps:
        k = (step.after.word, step.after.strands)
        if k in seen:
            return group, False  # a cycle is not a negative action label
        seen.add(k)
    return group, bool(w.steps)


def positive_row(w, inst_data, observation, *, random_source=False):
    return dict(path_id=semantic_id(w), query_group=list(observation['query_group']),
        gradient_scope='exact_zero_spent_TRAIN_root_only',
        origin='MCTS_best_success_not_reference',  # compatibility: incumbent of this condition
        sampling_source='normalized_random_short_horizon' if random_source else 'audited_main_MCTS_v28',
        instance=inst_data, witness=w.to_dict(), root_action=w.steps[0].action.to_dict(),
        achieved=dict(cc=w.crossing_changes, moves=w.moves,
            L10=10*w.crossing_changes+w.moves, L1000=1000*w.crossing_changes+w.moves,
            semantics='verified_upper_bound_not_optimal_cost'), observations=[observation])


def write_bank(out, parent, parent_audit, candidates, inputs, **metadata):
    disk_guard()
    if out.exists():
        raise ValueError('immutable bank already exists')
    old = read(parent / 'manifest.json'); audit = read(parent_audit)
    assert audit['status'] == 'passed_data_only'
    assert audit['manifest_sha256'] == sha(parent / 'manifest.json')
    out.mkdir(parents=True)
    entries, ids = [], set()
    for e in old['paths']:
        assert sha(parent / e['file']) == e['sha256']
        shutil.copyfile(parent / e['file'], out / e['file'])
        entries.append(e); ids.add(read(parent / e['file'])['path_id'])
    added = []
    for row in candidates:
        if row['path_id'] in ids:
            continue
        ids.add(row['path_id']); added.append(row['path_id'])
        f = out / f'v28-path-{len(entries):07d}.json'
        atomic(f, row); entries.append(dict(file=f.name, sha256=sha(f)))
    inputs = {**inputs, str(parent / 'manifest.json'): sha(parent / 'manifest.json'),
              str(parent_audit): sha(parent_audit), str(PROTOCOL): sha(PROTOCOL),
              str(Path(__file__)): sha(Path(__file__))}
    atomic(out / 'manifest.json', dict(schema='rf-additive-positive-v28', status='exported_not_training',
        parent_bank=str(parent), parent_audit=str(parent_audit), paths=entries,
        new_path_ids=added, inputs=inputs, optimizer_connected=False, **metadata))


def audit_bank(bank, out):
    """Independent consumption-side audit; no exporter assertions trusted."""
    from audit_mcts_root_positives_v1 import key
    if out.exists():
        raise ValueError('immutable audit exists')
    m = read(bank / 'manifest.json'); pins_ok(m['inputs'])
    parent = Path(m['parent_bank']); pm = read(parent / 'manifest.json')
    pa = read(Path(m['parent_audit']))
    assert pa['status'] == 'passed_data_only' and pa['manifest_sha256'] == sha(parent / 'manifest.json')
    assert m['paths'][:len(pm['paths'])] == pm['paths']
    for e in pm['paths']:
        assert (bank / e['file']).read_bytes() == (parent / e['file']).read_bytes()
    groups, states, _, _ = protected_registry()
    seen, views, gs, new_views = set(), set(), set(), set()
    for e in m['paths']:
        p = bank / e['file']; assert sha(p) == e['sha256']
        r = read(p); w = UnknotWitness.from_dict(r['witness']); ident = key(w)
        assert ident == r['path_id'] and ident not in seen
        seen.add(ident); inst = instance(r['instance'])
        g, cycle_free = admit(w, inst, groups, states)
        assert cycle_free and g == tuple(r['query_group'])
        assert r['root_action'] == w.steps[0].action.to_dict()
        assert r['gradient_scope'] == 'exact_zero_spent_TRAIN_root_only'
        assert r['origin'] == 'MCTS_best_success_not_reference'
        assert r['achieved'] == dict(cc=w.crossing_changes, moves=w.moves,
            L10=10*w.crossing_changes+w.moves, L1000=1000*w.crossing_changes+w.moves,
            semantics='verified_upper_bound_not_optimal_cost')
        v = norm(w.start.word, w.start.strands); views.add(v); gs.add(g)
        if ident not in m['new_path_ids']:
            continue
        new_views.add(v)
        for o in r['observations']:
            sp = Path(o['shard']); assert sha(sp) == o['shard_sha256']
            assert sha(Path(o['actor']['file'])) == o['actor']['sha256']
            source = read(sp)
            original = source['row']['witness'] if o['source_kind'] == 'main' else source['result']['witness']
            assert key(UnknotWitness.from_dict(original)) == ident
            roster = Path(o['roster']); assert sha(roster) == o['roster_sha256']
            root = read(roster)['rows'][o['root_index']]
            d = root['instances'][0 if o['weight'] == 10 else 1]
            assert d == r['instance']
            if o['source_kind'] == 'main':
                report = Path(o['batch_report']); assert sha(report) == o['batch_report_sha256']
                ar = read(report); assert not ar['partial'] and ar['conditions'] == 32
                assert ar['actor'] == o['actor']
            assert source['seed'] == o['seed']
    out.mkdir(parents=True)
    atomic(out / 'report.json', dict(status='passed_data_only', manifest_sha256=sha(bank/'manifest.json'),
        full_paths_replayed=len(seen), retained_paths=len(pm['paths']), new_paths=len(m['new_path_ids']),
        normalized_views=len(views), TRAIN_query_groups=len(gs), new_normalized_views=len(new_views),
        yield_gate=bool(m['new_path_ids']), optimizer_connected=False, auditor_sha256=sha(Path(__file__))))


def intake_main(out, parent, audit, first_batch, batches):
    """Consume FULL audited batches, not requested simulations or partial rows."""
    from audit_cost_mcts_continuations import audit_episode
    if batches < 1 or batches > 16:
        raise ValueError('bounded intake1..16batches required')
    pm = read(parent/'manifest.json')
    expected = pm.get('main_next_batch', pm.get('cohort_full_batches'))
    if first_batch != expected:
        raise ValueError(f'contiguous intake cursor required: {expected}')
    disk_guard(); source = Path('artifacts/continuous-train-mcts-v2')
    snapshot = read(source / 'manifest.json')
    roster = Path('artifacts/continuous-train-roster-v1/roster.json')
    roots = read(roster)['rows']; reports = {e['batch']: e for e in snapshot['audits']}
    pins_ok(snapshot['contract']['inputs'])
    groups, states, _, protected_pins = protected_registry()
    rows, pins, episodes, conditions, rejected = [], {str(roster): sha(roster), **protected_pins, **code_pins()}, 0, 0, 0
    for bi in range(first_batch, first_batch + batches):
        disk_guard()
        if bi not in reports:
            raise ValueError('batch not FULL independently audited')
        report = Path(reports[bi]['report']); assert sha(report) == reports[bi]['sha256']
        ar = read(report); bp = source / f'batch-{bi:06d}' / 'manifest.json'
        bm = read(bp); assert not ar['partial'] and ar['conditions'] == 32
        assert sha(bp) == ar['batch_manifest_sha256'] and bm['actor'] == ar['actor']
        assert bm['shards'] == snapshot['shards'][bi*32:(bi+1)*32]
        actor = bm['actor']; assert sha(Path(actor['file'])) == actor['sha256']
        pins.update({str(report): sha(report), str(bp): sha(bp), actor['file']: actor['sha256']})
        batch_episodes = 0
        for e in bm['shards']:
            sp = source / e['file']; assert sha(sp) == e['sha256']
            saved = read(sp); row = saved['row']; conditions += 1
            if not row:
                continue
            d = roots[saved['root_index']]['instances'][0 if saved['weight'] == 10 else 1]
            inst = instance(d)
            for episode in row['episodes']:
                audit_episode(inst, episode, dict(horizon=inst.cap_moves, insertion_candidates=4))
                batch_episodes += 1
            if not row['witness']:
                continue  # censored/non-solved is never a negative gradient
            w = UnknotWitness.from_dict(row['witness']); g, ok = admit(w, inst, groups, states)
            if not ok:
                rejected += 1; continue
            obs = dict(source_kind='main', query_group=list(g), shard=str(sp), shard_sha256=sha(sp),
                batch_report=str(report), batch_report_sha256=sha(report), actor=actor,
                roster=str(roster), roster_sha256=sha(roster), root_index=saved['root_index'],
                seed=saved['seed'], weight=saved['weight'])
            rows.append(positive_row(w, d, obs))
        assert batch_episodes == ar['episodes_replayed']
        episodes += batch_episodes
    write_bank(out/'bank', parent, audit, rows, pins, main_first_batch=first_batch,
        main_next_batch=first_batch+batches, replayed_episodes=episodes, conditions=conditions,
        cycle_rejected=rejected, candidate_successes=len(rows))
    audit_bank(out/'bank', out/'audit')
    atomic(out/'READY.json', dict(bank=str(out/'bank'), audit=str(out/'audit/report.json'),
        manifest_sha256=sha(out/'bank/manifest.json'), audit_sha256=sha(out/'audit/report.json'),
        next_main_batch=first_batch+batches, optimizer_connected=False))


def generate(out, batch, prior_paths):
    """Fresh constructive TRAIN roots, not finite suffixes of a256-root roster."""
    from rf_knots.conjugate_witnesses import conjugate_witness
    from evaluate_embedding_cost_search import query_instance
    groups, protected, known, pins = protected_registry()
    for p in prior_paths:
        pins[str(p)] = sha(p)
        for r in read(p)['rows']:
            w = UnknotWitness.from_dict(r['witness']); known.add(norm(w.start.word, w.start.strands))
    pins.update(code_pins())
    if out.exists():
        raise ValueError('immutable supply attempt; no seed retry')
    disk_guard(); out.mkdir(parents=True)
    atomic(out/'contract.json', dict(seed=SEED, batch=batch, opportunities=256, supply_CPU=30,
        requested_views=4, minimum_groups=2, inputs=pins, reference_is_training_label=False))
    rows, seen_groups, rejected = [], set(), collections.Counter()
    start = time.process_time(); op = 0; status = 'supply_exhausted'

    def timeout(*_):
        raise TimeoutError('supply CPU ceiling')

    previous = signal.signal(signal.SIGPROF, timeout); signal.setitimer(signal.ITIMER_PROF, 30)
    try:
        for op in range(256):
            w = conjugate_witness(SEED, batch*256+op, max_strands=12, min_depth=8, max_depth=48)
            k = norm(w.start.word, w.start.strands)
            if len(w.start.word) > 192 or w.moves > 192 or len(k[0]) < 8:
                rejected['size_or_trivial'] += 1; continue
            fp = invariants(w.start.word, w.start.strands); g = (fp[0], abs(fp[1]))
            if owner(fp) != 'train' or g in groups:
                rejected['not_admissible_TRAIN_group'] += 1; continue
            if k in known or g in seen_groups:
                rejected['normalized_known_or_batch_group_duplicate'] += 1; continue
            inst, fits = query_instance(op, w, witness_caps=True)
            if not fits:
                rejected['caps'] += 1; continue
            w.verify(); known.add(k); seen_groups.add(g)
            rows.append(dict(query_group=list(g), generator_index=batch*256+op,
                witness=w.to_dict(), instances=[replace(inst, cc_weight=v).to_dict() for v in [10,1000]]))
            if len(rows) == 4:
                status = 'completed'; break
    except TimeoutError:
        status = 'cpu_ceiling'
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0); signal.signal(signal.SIGPROF, previous)
    atomic(out/'supply.json', dict(status=status, rows=rows, opportunities=op+1,
        rejections=dict(rejected), cpu_seconds=time.process_time()-start,
        normalized_views=len(rows), query_groups=len(seen_groups), training_labels=0))
    return read(out/'supply.json')


def collect(out, parent, audit, batches):
    """A bounded prospective loop: generate -> search -> full replay -> READY.

No trainer stop, pointer promotion, or unbounded retry is performed here.
"""
    from run_continuous_train_mcts_v2 import snapshot
    from run_continuous_train_mcts_v1 import policy
    from verified_reduction_search_v28 import search
    if batches < 1 or batches > 32:
        raise ValueError('explicit batch bound1..32 required')
    disk_guard(); out.mkdir(exist_ok=True)
    with (out/'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mp = out/'manifest.json'
        if mp.exists():
            m = read(mp)
            if m['status'] != 'ready_for_next_batch':
                raise ValueError('partial/stopped collection requires review, no blind retry')
        else:
            m = dict(status='ready_for_next_batch', next_batch=0, current_bank=str(parent),
                     current_audit=str(audit), completed=[])
        for _ in range(batches):
            disk_guard(out)
            if (out/'STOP_REQUEST').exists():
                break
            bi = m['next_batch']; d = out/f'batch-{bi:06d}'
            m.update(status='collecting'); atomic(mp, m)
            supply = generate(d, bi, sorted(out.glob('batch-*/supply.json')))
            if supply['status'] != 'completed':
                m['status'] = supply['status']; atomic(mp, m); return
            # Separate full supply proof/ownership audit before any neural search.
            groups, states, _, _ = protected_registry()
            for r in supply['rows']:
                w = UnknotWitness.from_dict(r['witness'])
                g, ok = admit(w, instance(r['instances'][0]), groups, states)
                assert ok and g == tuple(r['query_group'])
            actor = snapshot(); atomic(d/'actor.json', actor)
            net, objective = policy(actor); items, cpu = [], 0.
            for root, row in enumerate(supply['rows']):
                for wi, weight in enumerate([10,1000]):
                    for seed in [20261023461,20261023462]:
                        disk_guard(out); inst = instance(row['instances'][wi])
                        result = search(inst, net, objective, seed, cpu_seconds=20, simulations=16)
                        sp = d/f'search-{root}-{weight}-{seed}.json'
                        atomic(sp, dict(root=root, weight=weight, seed=seed, actor=actor, result=result))
                        cpu += result['cpu_seconds']
                        if result['witness']:
                            w = UnknotWitness.from_dict(result['witness']); g, ok = admit(w, inst, groups, states)
                            if ok:
                                obs = dict(source_kind='v28', query_group=list(g), shard=str(sp),
                                    shard_sha256=sha(sp), actor=actor, roster=str(d/'supply.json'),
                                    roster_sha256=sha(d/'supply.json'), root_index=root, seed=seed, weight=weight)
                                items.append(positive_row(w, row['instances'][wi], obs, random_source=True))
            solved_views = {(tuple(r['instance']['word']),r['instance']['strands']) for r in items}
            # Consumption-side replay of every persisted condition, including failures.
            from audit_cost_mcts_continuations import audit_episode
            replayed = 0
            for root, row in enumerate(supply['rows']):
                for wi, weight in enumerate([10,1000]):
                    for seed in [20261023461,20261023462]:
                        saved = read(d/f'search-{root}-{weight}-{seed}.json')
                        assert saved['actor'] == actor and saved['root'] == root
                        assert saved['weight'] == weight and saved['seed'] == seed
                        inst = instance(row['instances'][wi])
                        for ep in saved['result']['episodes']:
                            audit_episode(inst, ep, dict(horizon=inst.cap_moves, insertion_candidates=4))
                            replayed += 1
                        if saved['result']['witness']:
                            UnknotWitness.from_dict(saved['result']['witness']).verify()
            atomic(d/'collection.json', dict(conditions=16, search_cpu_seconds=cpu,
                solved_normalized_views=len(solved_views), reference_labels=0,
                independently_replayed_episodes=replayed))
            if len(solved_views) < 2 or cpu/len(solved_views) > 300:
                m['status'] = 'yield_stop'; atomic(mp,m); return
            inputs = {str(f):sha(f) for f in [d/'contract.json', d/'supply.json', d/'actor.json',
                                            Path('scripts/verified_reduction_search_v28.py')]}
            pins_ok(read(d/'contract.json')['inputs'])
            write_bank(d/'bank', Path(m['current_bank']), Path(m['current_audit']), items, inputs,
                       search_cpu_seconds=cpu, source_conditions=16)
            audit_bank(d/'bank', d/'bank-audit')
            m.update(current_bank=str(d/'bank'), current_audit=str(d/'bank-audit/report.json'),
                     next_batch=bi+1, status='ready_for_next_batch')
            m['completed'].append(dict(batch=bi,bank_sha256=sha(d/'bank/manifest.json')))
            atomic(mp,m)
            atomic(out/'READY.json',dict(bank=m['current_bank'],audit=m['current_audit'],
                manifest_sha256=sha(d/'bank/manifest.json'),
                audit_sha256=sha(d/'bank-audit/report.json'),optimizer_connected=False))


def prepare_spec(out, main_ready, random_ready, cpu_report):
    """Explicit full-state handoff only; NEVER requests the v27 stop itself."""
    if out.exists():
        raise ValueError('new spec already exists')
    ref = read(Path('artifacts/continuous-r2-current-training.json')); s = read(Path(ref['spec']))
    parent = Path(ref['out']); m = read(parent/'manifest.json')
    if m['status'] != 'checkpointed' or m['stop_reason'] != 'user_stop_request':
        raise ValueError('activation requires a separately authorized graceful parent handoff')
    report = read(cpu_report)
    assert report['status'] == 'passed_isolated_CPU_v28' and report['updates'] == 2
    e = m['checkpoints'][-1]; assert sha(parent/e['file']) == e['sha256']
    assert report['source_sha256'] == e['sha256']
    for prefix, ready_path in [('positive',main_ready),('random_positive',random_ready)]:
        ready = read(ready_path); bank, audit = Path(ready['bank']), Path(ready['audit'])
        assert sha(bank/'manifest.json') == ready['manifest_sha256']
        assert sha(audit) == ready['audit_sha256']
        ar = read(audit); assert ar['status'] == 'passed_data_only' and ar['manifest_sha256'] == sha(bank/'manifest.json')
        assert report[prefix+'_bank_sha256'] == sha(bank/'manifest.json')
        old = Path(s[prefix+'_bank']); oldentries = read(old/'manifest.json')['paths']
        assert read(bank/'manifest.json')['paths'][:len(oldentries)] == oldentries
        for entry in oldentries:
            assert (bank/entry['file']).read_bytes() == (old/entry['file']).read_bytes()
        s.update({prefix+'_bank':str(bank),prefix+'_bank_sha256':sha(bank/'manifest.json'),
                  prefix+'_audit':str(audit),prefix+'_audit_sha256':sha(audit)})
    target = out.parent/out.stem.replace('start-spec-','')
    assert not target.exists()
    s.update(parent=str(parent/e['file']),parent_step=e['step'],parent_sha256=e['sha256'],
             parent_manifest_sha256=sha(parent/'manifest.json'),out=str(target),
             version='v28_prepared_not_activated', cpu_report=str(cpu_report),cpu_report_sha256=sha(cpu_report))
    atomic(out,s)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    for command in ['intake-main','collect']:
        a=sub.add_parser(command);a.add_argument('--out',type=Path,required=True)
        a.add_argument('--parent-bank',type=Path,required=True);a.add_argument('--parent-audit',type=Path,required=True)
        a.add_argument('--batches',type=int,default=1)
        if command=='intake-main':a.add_argument('--first-batch',type=int,required=True)
    a=sub.add_parser('audit');a.add_argument('--bank',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('supply');a.add_argument('--out',type=Path,required=True);a.add_argument('--batch',type=int,default=0)
    a=sub.add_parser('prepare-spec');a.add_argument('--out',type=Path,required=True)
    a.add_argument('--main-ready',type=Path,required=True);a.add_argument('--random-ready',type=Path,required=True)
    a.add_argument('--cpu-report',type=Path,required=True)
    a=p.parse_args()
    if a.command=='intake-main':intake_main(a.out,a.parent_bank,a.parent_audit,a.first_batch,a.batches)
    elif a.command=='collect':collect(a.out,a.parent_bank,a.parent_audit,a.batches)
    elif a.command=='audit':audit_bank(a.bank,a.out)
    elif a.command=='supply':generate(a.out,a.batch,[])
    else:prepare_spec(a.out,a.main_ready,a.random_ready,a.cpu_report)


if __name__=='__main__':main()
