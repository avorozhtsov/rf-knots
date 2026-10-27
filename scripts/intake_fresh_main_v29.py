"""Post-A MAIN intake cursor separate from preserved historical backlog."""
import argparse
from learning_pipeline_v28 import *


def validate_fresh(bank):
    m=read(bank/'manifest.json');ids=set(m['fresh_main_path_ids'])
    rows={read(bank/e['file'])['path_id']:read(bank/e['file']) for e in m['paths']}
    seen=set()
    for e in m['fresh_main_evidence']:
        ident=e['path_id'];o=e['observation'];sp=Path(o['shard'])
        assert int(sp.stem.split('-')[-1])>=39168,'historical MCTS is not fresh'
        assert sha(sp)==o['shard_sha256']
        saved=read(sp);w=UnknotWitness.from_dict(saved['row']['witness']);w.verify()
        assert semantic_id(w)==ident and ident in rows
        report=Path(o['batch_report']);assert sha(report)==o['batch_report_sha256']
        r=read(report);assert not r['partial'] and r['conditions']==32
        assert sha(Path(o['actor']['file']))==o['actor']['sha256']
        assert saved['seed']==o['seed'] and saved['weight']==o['weight']
        observed=read(Path(o['roster']))['rows'][o['root_index']]['instances'][0 if o['weight']==10 else 1]
        assert {k:v for k,v in rows[ident]['instance'].items() if k!='cc_weight'}=={k:v for k,v in observed.items() if k!='cc_weight'}
        seen.add(ident)
    assert ids==seen and ids,'fresh MAIN evidence is absent'
    return ids


def intake_main(out, parent, audit, first_batch, batches):
    """Consume FULL audited batches, not requested simulations or partial rows."""
    from audit_cost_mcts_continuations import audit_episode
    if batches < 1 or batches > 16:
        raise ValueError('bounded intake1..16batches required')
    pm = read(parent/'manifest.json')
    expected = pm.get('fresh_main_next_batch',1224)
    if first_batch != expected:
        raise ValueError(f'contiguous intake cursor required: {expected}')
    disk_guard(); source = Path('artifacts/continuous-train-mcts-v2')
    snapshot = read(source / 'manifest.json')
    roster = Path('artifacts/continuous-train-roster-v1/roster.json')
    roots = read(roster)['rows']; reports = {e['batch']: e for e in snapshot['audits']}
    pins_ok(snapshot['contract']['inputs'])
    groups, states, _, protected_pins = protected_registry()
    rows, pins, episodes, conditions, rejected = [], {str(roster): sha(roster), **protected_pins, **code_pins(),str(Path(__file__)):sha(Path(__file__)),'artifacts/connected-learning-live-v29-protocol.json':sha(Path('artifacts/connected-learning-live-v29-protocol.json'))}, 0, 0, 0
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
    write_bank(out/'bank', parent, audit, rows, pins,
        fresh_main_next_batch=first_batch+batches,
        historical_backfill_next_batch=pm.get('historical_backfill_next_batch',pm.get('main_next_batch')),
        fresh_main_path_ids=sorted(set(pm.get('fresh_main_path_ids',[]))|{r['path_id'] for r in rows}),
        fresh_main_evidence=pm.get('fresh_main_evidence',[])+[dict(path_id=r['path_id'],observation=r['observations'][0]) for r in rows],
        main_first_batch=first_batch,
        main_next_batch=first_batch+batches, replayed_episodes=episodes, conditions=conditions,
        cycle_rejected=rejected, candidate_successes=len(rows))
    audit_bank(out/'bank', out/'audit')
    atomic(out/'READY.json', dict(bank=str(out/'bank'), audit=str(out/'audit/report.json'),
        manifest_sha256=sha(out/'bank/manifest.json'), audit_sha256=sha(out/'audit/report.json'),
        next_main_batch=first_batch+batches, optimizer_connected=False))



if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['out','parent-bank','parent-audit']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--first-batch',type=int,required=True);p.add_argument('--batches',type=int,required=True)
    a=p.parse_args();assert a.first_batch>=1224
    intake_main(a.out,a.parent_bank,a.parent_audit,a.first_batch,a.batches)
    print(dict(status='fresh_MAIN_audited',eligible=len(validate_fresh(a.out/'bank'))))
