"""One reviewed storage recovery, through the opt-in compact v31 writer.

Preserve seven existing 305 searches. Run nine missing conditions and the three
never-started prospective batches in the ORIGINAL four-batch window, not a retry.
"""
from pathlib import Path
import fcntl
import os
import shutil
import learning_pipeline_v30 as p
import collect_connected_learning_v30 as collector
import eligible_positive_bank_v28a as eligible
from recover_random_window_277_280_v30a import condition_names, missing_conditions

ROOT = Path('artifacts/random-TRAIN-stream-v30')
SUP = Path('artifacts/connected-learning-live-v30')
WINDOW = ROOT / 'window-000305-000308'
PLAN = ROOT / 'storage-recovery-305-308-v31.json'
COMPLETE = WINDOW / 'storage-recovery-completed-v31.json'
FULL_SHA = 'e9cf0673d3c615520a42edf88d065c8ed288a0189a1a31396bc7f79071793deb'
A_SHA = 'a2bf6f7b975a134352176e9d8c36059f9e9584bfd4bf47a0614747bf90541d1c'
ACTOR305_SHA = 'b7cf15d24e967bd3019ac67c939fe432f2c0a2903c9252a659f6148f2e033b76'


def preflight_layout():
    assert len(missing_conditions(ROOT/'batch-000305')) == 9
    assert not (ROOT/'batch-000305/collection.json').exists()
    for batch in (306, 307, 308):
        assert not (ROOT/f'batch-{batch:06d}').exists(), 'never retry a prospective attempt'
    for name in ('bank', 'bank-audit', 'yield-audit.json'):
        assert not (WINDOW/name).exists(), 'immutable publication already exists'


def replay(batch, groups, states, inputs, *, partial=False):
    from audit_cost_mcts_continuations import audit_episode
    folder = ROOT/f'batch-{batch:06d}'
    missing = missing_conditions(folder)
    if not partial:
        assert not missing
    supply, actor = p.read(folder/'supply.json'), p.read(folder/'actor.json')
    assert supply['status'] == 'completed' and len(supply['rows']) == 4
    contract = p.read(folder/'contract.json')
    assert (contract['seed'], contract['batch'], contract['opportunities'], contract['supply_CPU'],
            contract['min_depth'], contract['max_depth'], contract['max_strands']) == (20261023460, batch, 256, 30, 4, 8, 6)
    p.pins_ok(contract['inputs'])
    assert p.sha(Path(actor['file'])) == actor['sha256']
    assert p.sha(Path(actor['training_spec'])) == actor['training_spec_sha256']
    items, cpu, episodes = [], 0., 0
    for ri, row in enumerate(supply['rows']):
        g, ok = p.admit(p.UnknotWitness.from_dict(row['witness']), p.instance(row['instances'][0]), groups, states)
        assert ok and g == tuple(row['query_group'])
        for wi, weight in enumerate((10, 1000)):
            inst = p.instance(row['instances'][wi])
            for seed in (20261023461, 20261023462):
                path = folder/f'search-{ri}-{weight}-{seed}.json'
                if partial and not path.exists():
                    continue
                saved = p.read(path)
                assert (saved['root'], saved['weight'], saved['seed'], saved['actor']) == (ri, weight, seed, actor)
                result = saved['result']; cpu += result['cpu_seconds']
                for ep in result['episodes']:
                    audit_episode(inst, ep, dict(horizon=inst.cap_moves, insertion_candidates=4)); episodes += 1
                inputs[str(path)] = p.sha(path)
                if result['witness']:
                    witness = p.UnknotWitness.from_dict(result['witness'])
                    g, ok = p.admit(witness, inst, groups, states)
                    if ok:
                        obs = dict(source_kind='v28', query_group=list(g), shard=str(path), shard_sha256=p.sha(path),
                            actor=actor, roster=str(folder/'supply.json'), roster_sha256=p.sha(folder/'supply.json'),
                            root_index=ri, seed=seed, weight=weight)
                        items.append(p.positive_row(witness, row['instances'][wi], obs, random_source=True))
    views = {(tuple(r['instance']['word']), r['instance']['strands']) for r in items}
    report = dict(conditions=16-len(missing), search_cpu_seconds=cpu, solved_normalized_views=len(views),
        reference_labels=0, independently_replayed_episodes=episodes)
    for name in ('contract.json', 'supply.json', 'actor.json'):
        inputs[str(folder/name)] = p.sha(folder/name)
    if not partial:
        assert not (folder/'collection.json').exists()
        p.atomic(folder/'collection.json', report)
        inputs[str(folder/'collection.json')] = p.sha(folder/'collection.json')
    return items, views, report


def repaired_state(state, passed):
    assert passed, 'never clear a failure before the independent full-window audit'
    assert state['status'] == 'STOPPED_LEARNER' and state['exit_code'] == 0
    assert state['stop_reason'] == 'storage_gate' and state['supply_stopped']
    assert not state['main_intake_stopped']
    return {**state, 'supply_stopped': False}


def main():
    from run_continuous_train_mcts_v2 import snapshot
    from run_continuous_train_mcts_v1 import policy
    from verified_reduction_search_v28 import search
    import connected_learning_supervisor_v30 as supervisor
    assert os.environ.get('RF_KNOTS_COMPACT_V31') == '1', 'use compact v31 writer launcher'
    assert shutil.disk_usage('.').free >= 16*1024**3
    for fragment in ('scripts/connected_learning_supervisor_', 'scripts/train_connected_learning_live_',
                     'scripts/collect_connected_learning_', 'scripts/run_continuous_train_mcts_v2.py'):
        assert not supervisor.has_process(fragment), 'another live writer exists'
    with (SUP/'writer.lock').open('a') as super_lock, (ROOT/'writer.lock').open('a') as lock:
        for handle in (super_lock, lock):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not PLAN.exists(), 'one-shot plan already exists; review partial work, do not retry'
        assert not (SUP/'failure.json').exists() and not (SUP/'STOP_REQUEST').exists()
        state = p.read(SUP/'state.json'); repaired_state(state, True)
        spec, out, manifest, full, saved = supervisor.validate_parent()
        assert full['step'] == saved['step'] == 2056741 and full['sha256'] == FULL_SHA
        assert p.sha(Path('artifacts/paired-learning-48h100k-v3/baseline.pt')) == A_SHA
        preflight_layout()
        m = p.read(ROOT/'manifest.json')
        assert m['status'] == 'collecting' and m['next_batch'] == 305
        assert m['current_bank'] == str(ROOT/'window-000301-000304/bank')
        assert p.read(WINDOW/'contract.json') == dict(first_batch=305, batches=4, minimum_solved_views=8,
            maximum_CPU_per_solved_view=300, all_batches_count=True,
            parent_manifest_sha256=p.sha(Path(m['current_bank'])/'manifest.json'))
        actor305 = p.read(ROOT/'batch-000305/actor.json')
        assert actor305['step'] == 2056598 and actor305['sha256'] == ACTOR305_SHA
        future_actor = snapshot()  # frozen before observing any new search result
        assert future_actor['step'] == 2056741 and future_actor['sha256'] == FULL_SHA
        frozen = {str(f):p.sha(f) for f in (ROOT/'batch-000305').glob('*.json')}
        frozen[str(WINDOW/'contract.json')] = p.sha(WINDOW/'contract.json')
        for f in (Path(__file__), Path('tests/test_random_recovery_305_308_v31.py'),
                  Path('research/467-compact-v31-window305-308-recovery.md')):
            frozen[str(f)] = p.sha(f)
        groups, states, _, protected = p.protected_registry()
        inputs = dict(protected)
        _, _, partial_report = replay(305, groups, states, inputs, partial=True)
        assert partial_report['conditions'] == 7
        p.atomic(PLAN, dict(status='reviewed_recovery_started', authorization='USER2026_10_27',
            original_manifest=m, original_supervisor_state=state, existing_files=frozen,
            original_actor305=actor305, prospective_actor306_308=future_actor,
            missing_305_conditions=missing_conditions(ROOT/'batch-000305'), never_started_batches=[306,307,308],
            preserved_conditions=7, new_search_conditions=57, partial_replay=partial_report,
            saved_full=full, A_sha256=A_SHA, no_clock_reset=True, no_deletion=True))
        parent = collector.parent; parent.pipeline = p; parent.FIRST_BATCH = 61
        parent.prior_rosters = collector.rosters; parent.code_pins = collector.pins
        items, views, reports = [], set(), []
        for batch in (305, 306, 307, 308):
            folder = ROOT/f'batch-{batch:06d}'
            if batch != 305:
                supply = parent.generate(folder, batch, sorted(ROOT.glob('batch-*/supply.json')))
                assert supply['status'] == 'completed', 'supply failure: stop, no replacement'
                p.atomic(folder/'actor.json', future_actor)
            supply, actor = p.read(folder/'supply.json'), p.read(folder/'actor.json')
            for row in supply['rows']:
                g, ok = p.admit(p.UnknotWitness.from_dict(row['witness']), p.instance(row['instances'][0]), groups, states)
                assert ok and g == tuple(row['query_group'])
            net, objective = policy(actor)
            absent = set(missing_conditions(folder))
            assert len(absent) == (9 if batch == 305 else 16)
            for ri, row in enumerate(supply['rows']):
                for wi, weight in enumerate((10, 1000)):
                    for seed in (20261023461, 20261023462):
                        path = folder/f'search-{ri}-{weight}-{seed}.json'
                        if path.name not in absent:
                            continue
                        assert not path.exists(); p.disk_guard()
                        result = search(p.instance(row['instances'][wi]), net, objective, seed, cpu_seconds=20, simulations=16)
                        p.atomic(path, dict(root=ri, weight=weight, seed=seed, actor=actor, result=result))
            it, vs, report = replay(batch, groups, states, inputs)
            items.extend(it); views.update(vs)
            reports.append(dict(batch=batch, solved_views=len(vs), cpu_seconds=report['search_cpu_seconds'],
                replayed_episodes=report['independently_replayed_episodes']))
            p.pins_ok(frozen)
            print(dict(batch=batch, full_report=report), flush=True)
        cpu = sum(r['cpu_seconds'] for r in reports)
        report = dict(batches=reports, solved_normalized_views=len(views), search_cpu_seconds=cpu,
            passed=p.window_gate(len(views), cpu), reference_labels=0, conditions=64, all_batches_counted=True)
        p.atomic(WINDOW/'yield-audit.json', report)
        assert report['passed'], 'original fixed-window gate failed: preserve and stop, no extension'
        inputs.update(frozen)
        for f in (PLAN, WINDOW/'yield-audit.json', Path('scripts/verified_reduction_search_v28.py')):
            inputs[str(f)] = p.sha(f)
        parent.write_bank(WINDOW/'bank', Path(m['current_bank']), Path(m['current_audit']), items, inputs,
            search_cpu_seconds=cpu, source_conditions=64)
        eligible.audit(WINDOW/'bank', WINDOW/'bank-audit')
        p.pins_ok(frozen)
        bank_manifest = p.read(WINDOW/'bank/manifest.json')
        m.update(status='ready_for_next_batch', next_batch=309, current_bank=str(WINDOW/'bank'),
            current_audit=str(WINDOW/'bank-audit/report.json'), window_batches=reports)
        m['completed'].append(dict(first_batch=305, last_batch=308, bank_sha256=p.sha(WINDOW/'bank/manifest.json')))
        p.atomic(ROOT/'manifest.json', m)
        p.atomic(ROOT/'READY.json', dict(bank=m['current_bank'], audit=m['current_audit'],
            manifest_sha256=p.sha(WINDOW/'bank/manifest.json'), audit_sha256=p.sha(Path(m['current_audit'])),
            optimizer_connected=False))
        assert p.read(SUP/'state.json') == state, 'concurrent state change'
        p.atomic(COMPLETE, dict(status='passed', old_files_byte_identical=True, preserved_search_conditions=7,
            new_search_conditions=57, original_actor305=actor305, prospective_actor306_308=future_actor,
            recovery_plan_sha256=p.sha(PLAN), yield_report=report, new_path_ids=bank_manifest['new_path_ids'],
            READY_sha256=p.sha(ROOT/'READY.json'), independent_audit=p.read(Path(m['current_audit'])),
            saved_resume_step=2056741, optimizer_connected=False))
        backup = SUP/'supply-recovery-305-308-before.json'
        assert not backup.exists()
        backup.write_bytes((SUP/'state.json').read_bytes())
        assert p.read(backup) == state
        p.atomic(SUP/'state.json', repaired_state(state, p.read(COMPLETE)['status'] == 'passed'))
        print(dict(status='RECOVERED_FIXED_WINDOW_V31', report=report, new_paths=len(bank_manifest['new_path_ids'])), flush=True)


if __name__ == '__main__':
    main()
