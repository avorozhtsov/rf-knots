"""Bounded read-only inventory of routed graph TRAIN query groups."""

import hashlib
import importlib.util
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

from rf_knots.embedding_replay_corpus import invariants, owner

PROOF = Path('/Users/artemvorozhtsov/projects/unknotdb/outputs/unknotdb-dgkt-witness-coverage-v1-final.sqlite')
ROUTING = Path('/Users/artemvorozhtsov/projects/unknotdb/outputs/unknotdb-dgkt-routing-v1.sqlite')
DECODER = Path('/Users/artemvorozhtsov/projects/unknotdb/tools/build_graph_identification_sidecar.py')
RF = Path('src/rf_knots/embedding_replay_corpus.py')
BANK = Path('artifacts/additive-root-only-f-bank-v3/manifest.json')
PROTOCOL = Path('research/356-routed-graph-group-inventory-v1.md')
EXPECTED = {
    PROOF: 'e31028def491caab7317ab63eb9829a1e63e1fa9a512f2ccf4daac6d178ffd20',
    ROUTING: '673e0fcbae9c7b582e44fed8e92e09ec0716331e80a1c82a81c48bd7c9b77001',
    DECODER: 'ad6f6cb143237c0cf1de489a3c8aa9b95eaf19a2bc65a2d8f09541e85cb26b9c',
    RF: '1a34624266881104ec35f7de195a84b5052458c5d04be3d8b07428feec4ffa94',
    BANK: 'e1129f31cacb538a9b3a8824f8f552dabdd80803d7fd95c5feb0fb1aaa449c33',
}
OLD_VIEWED = {106255, 6132, 9009, 51535, 7151, 65707, 82922, 61145,
              54793, 79520, 42861, 71484, 58409, 88914, 11394, 63897}
BANDS = ((1000, 3000), (3000, 5000), (5000, 7000), (7000, None))
LIMIT_PER_BAND = 32
CPU_CEILING = 120.0


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def group(row):
    raw = row.get('query_group')
    return tuple(raw) if isinstance(raw, list) and len(raw) == 2 else None


def exclusions():
    bank = json.loads(BANK.read_text())
    old = set()
    for entry in bank['roots']:
        path = BANK.parent / entry['file']
        if sha(path) != entry['sha256']:
            raise ValueError(f'bank root hash mismatch: {path}')
        data = json.loads(path.read_text())
        if group(data):
            old.add(group(data))
        for row in data.get('rows', []):
            if group(row):
                old.add(group(row))
    rosters = sorted(Path('artifacts').glob('f-*/roster.json'))
    for path in rosters:
        data = json.loads(path.read_text())
        for stage in data.get('stages', []):
            for row in stage:
                if group(row):
                    old.add(group(row))
        for row in data.get('rows', []):
            if group(row):
                old.add(group(row))
    viewed = set()
    panels = sorted(Path('artifacts').glob('**/panel.json'))
    for path in panels:
        data = json.loads(path.read_text())
        for row in data.get('rows', data.get('validation', [])):
            if group(row):
                viewed.add(group(row))
    return old, viewed, rosters, panels


def main():
    start_cpu = time.process_time()
    pins = {str(path): sha(path) for path in (*EXPECTED, PROTOCOL, Path(__file__))}
    for path, expected in EXPECTED.items():
        if pins[str(path)] != expected:
            raise ValueError(f'pinned input changed: {path}')
    spec = importlib.util.spec_from_file_location('unknotdb_readonly_decoder', DECODER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old, viewed, rosters, panels = exclusions()
    pins.update({str(p): sha(p) for p in (*rosters, *panels)})
    proof = sqlite3.connect(f'file:{PROOF}?mode=ro', uri=True)
    routing = sqlite3.connect(f'file:{ROUTING}?mode=ro', uri=True)
    proof.execute('PRAGMA query_only=ON')
    routing.execute('PRAGMA query_only=ON')
    proof_meta = dict(proof.execute('SELECT key,value FROM meta'))
    route_meta = dict(routing.execute('SELECT key,value FROM meta'))
    if route_meta.get('proof_sha256') != pins[str(PROOF)]:
        raise ValueError('routing proof pin mismatch')
    proof.execute("ATTACH DATABASE ? AS route", (f'file:{ROUTING}?mode=ro',))
    rows = proof.execute('''WITH e AS (
      SELECT a.source_node,a.edge_id,a.first_action,a.target_node,
             k.l1000_cost+r.l1000_score AS cost
      FROM edges a JOIN route.edge_costs k USING(edge_id)
      JOIN route.routes r ON r.node_id=a.target_node),
    b AS (SELECT DISTINCT x.source_node FROM e x JOIN e y
      ON x.source_node=y.source_node AND x.edge_id<y.edge_id
      WHERE x.first_action<>y.first_action AND x.target_node<>y.target_node
      AND x.cost<>y.cost)
    SELECT b.source_node,r.l1000_score FROM b JOIN route.routes r
      ON r.node_id=b.source_node WHERE r.l1000_score>0''').fetchall()
    selected = []
    eligible = []
    for lo, hi in BANDS:
        band = [(int(node), int(score)) for node, score in rows
                if score >= lo and (hi is None or score < hi) and node not in OLD_VIEWED]
        eligible.append(len(band))
        band.sort(key=lambda row: hashlib.sha256(f'20261014:{row[0]}'.encode()).hexdigest())
        selected.extend((len(eligible)-1, node, score) for node, score in band[:LIMIT_PER_BAND])
    stats = [Counter() for _ in BANDS]
    novel = [set() for _ in BANDS]
    observations = []
    for band, node, score in selected:
        if time.process_time() - start_cpu >= CPU_CEILING:
            break
        encoding = proof.execute('SELECT encoding FROM representations WHERE node_id=?', (node,)).fetchone()
        item = {'band': band, 'node_id': node, 'graph_score': score}
        try:
            if encoding is None:
                raise ValueError('missing representation')
            strands, cyclic, word = mod.decode_representation(bytes(encoding[0]))
            det, sig = invariants(tuple(word), strands)
            key = (det, abs(sig))
            own = owner((det, sig))
            item.update(group=list(key), owner=own, strands=strands, word_length=len(word), cyclic=cyclic)
            reason = ('not_train' if own != 'train' else 'old_group' if key in old
                      else 'viewed_group' if key in viewed else 'novel_train_group')
            item['reason'] = reason
            if reason == 'novel_train_group':
                novel[band].add(key)
        except (ValueError, TypeError, KeyError) as exc:
            item['reason'] = 'decode_or_invariant_error'
            item['error'] = str(exc)
        stats[band][item['reason']] += 1
        observations.append(item)
    complete = len(observations) == len(selected)
    union = set().union(*novel)
    report = {
        'schema': 'rf-knots-routed-graph-group-inventory-v1',
        'role': 'read-only candidate supply; zero F labels',
        'complete': complete,
        'cpu_seconds': time.process_time() - start_cpu,
        'cpu_ceiling_seconds': CPU_CEILING,
        'eligible_by_band': eligible,
        'selected_by_band': [sum(x[0] == i for x in selected) for i in range(4)],
        'processed_by_band': [sum(x['band'] == i for x in observations) for i in range(4)],
        'reasons_by_band': [dict(s) for s in stats],
        'novel_train_groups_by_band': [len(s) for s in novel],
        'novel_train_groups_total': len(union),
        'old_excluded_group_count': len(old),
        'viewed_excluded_group_count': len(viewed),
        'stop_go': 'dual_route_replay_sample_permitted' if complete and len(union) >= 16
                   and sum(len(s) >= 4 for s in novel) >= 2 else 'no_scale_or_censored',
        'pins': pins,
        'observations': observations,
    }
    output = Path('artifacts/unknotdb-routed-group-inventory-v1/report.json')
    if output.exists():
        raise ValueError('inventory report already exists')
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('observations','pins')}, indent=2))
    print('report_sha256', sha(output))


if __name__ == '__main__':
    main()
