"""Read-only bounded structural screen, not canonical path/F admission."""

import hashlib
import heapq
import importlib.util
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import inventory_unknotdb_routed_groups_v1 as old_exclusions
from rf_knots.embedding_replay_corpus import invariants, owner

SOURCE = Path('/Users/artemvorozhtsov/projects/unknotdb/outputs/unknotdb-embedding-pairs-v0.sqlite')
DECODER = Path('/Users/artemvorozhtsov/projects/unknotdb/tools/build_graph_identification_sidecar.py')
RF = Path('src/rf_knots/embedding_replay_corpus.py')
BANK = Path('artifacts/additive-root-only-f-bank-v3/manifest.json')
PROTOCOL = Path('research/362-embedding-pair-directed-route-screen-v1.md')
EXCLUSION_SOURCE = Path('scripts/inventory_unknotdb_routed_groups_v1.py')
EXPECTED = {
    SOURCE: '8dd06e378a3794cb3df1e2314c06b425793e4ce8751919a554b48d342a412848',
    DECODER: 'ad6f6cb143237c0cf1de489a3c8aa9b95eaf19a2bc65a2d8f09541e85cb26b9c',
    RF: '1a34624266881104ec35f7de195a84b5052458c5d04be3d8b07428feec4ffa94',
    BANK: 'e1129f31cacb538a9b3a8824f8f552dabdd80803d7fd95c5feb0fb1aaa449c33',
}
SEED = '20261014'
PER_STRATUM = 32
CPU_CEILING = 120.0
ALLOWED = ('replayed_zero_cc_action', 'replayed_cc_upper_bound',
           'replayed_cc_distinct_attested_knots')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check_time(start):
    if time.process_time() - start >= CPU_CEILING:
        raise TimeoutError('prospectively bounded CPU ceiling')


def best_paths(reverse, sinks, cost_multiplier, start):
    distance = {node: 0 for node in sinks}
    next_arc = {}
    queue = [(0, node) for node in sinks]
    heapq.heapify(queue)
    seen = 0
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distance[node]:
            continue
        for pred, arc in reverse.get(node, ()):
            candidate = cost + 1 + cost_multiplier * arc[4]
            if candidate < distance.get(pred, math.inf):
                distance[pred] = candidate
                next_arc[pred] = arc
                heapq.heappush(queue, (candidate, pred))
        seen += 1
        if seen % 4096 == 0:
            check_time(start)
    return distance, next_arc


def unwind(root, first, next_arc, sinks):
    route = [first]
    seen = {root}
    node = first[1]
    while node not in sinks:
        if node in seen or node not in next_arc or len(route) > 256:
            return None
        seen.add(node)
        arc = next_arc[node]
        route.append(arc)
        node = arc[1]
    return route


def main():
    start = time.process_time()
    pins = {str(p): sha(p) for p in (*EXPECTED, PROTOCOL, EXCLUSION_SOURCE, Path(__file__))}
    for path, expected in EXPECTED.items():
        if pins[str(path)] != expected:
            raise ValueError(f'pinned input changed: {path}')
    old, viewed, rosters, panels = old_exclusions.exclusions()
    pins.update({str(p): sha(p) for p in (*rosters, *panels)})
    spec = importlib.util.spec_from_file_location('unknotdb_readonly_decoder', DECODER)
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    db = sqlite3.connect(f'file:{SOURCE}?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    meta = dict(db.execute('SELECT key,value FROM meta'))
    if (meta.get('schema') != 'unknotdb-embedding-pairs-v0'
            or meta.get('proof_sha256') !=
            'e31028def491caab7317ab63eb9829a1e63e1fa9a512f2ccf4daac6d178ffd20'):
        raise ValueError('source schema/proof pin differs')
    provenance = ','.join('?' for _ in ALLOWED)
    rows = db.execute(f'''SELECT left_representation,right_representation,
        edge_id,instruction_index,provenance FROM pairs
        WHERE split='train' AND metric='cc' AND provenance IN ({provenance})
        AND left_representation<>right_representation''', ALLOWED)
    forward = defaultdict(list)
    reverse = defaultdict(list)
    arcs = 0
    for left, right, edge, instruction, kind in rows:
        cc = int(kind != 'replayed_zero_cc_action')
        arc = (int(left), int(right), int(edge), int(instruction), cc)
        forward[arc[0]].append(arc)
        reverse[arc[1]].append((arc[0], arc))
        arcs += 1
        if arcs % 4096 == 0:
            check_time(start)
    candidates = [[], []]
    for root, outgoing in forward.items():
        if len({arc[1] for arc in outgoing}) < 2:
            continue
        split, knot_id = db.execute('SELECT split,knot_id FROM representations WHERE representation_id=?',
                                   (root,)).fetchone()
        if split == 'train' and knot_id != 'knot:0_1':
            candidates[0 if knot_id else 1].append(root)
    chosen = []
    for stratum, roots in enumerate(candidates):
        roots.sort(key=lambda root: hashlib.sha256(f'{SEED}:{root}'.encode()).hexdigest())
        chosen.extend((stratum, root) for root in roots[:PER_STRATUM])
    check_time(start)
    sinks = {row[0] for row in db.execute(
        "SELECT representation_id FROM representations WHERE split='train' AND knot_id='knot:0_1'")}
    distances = {}
    next_arcs = {}
    for objective in (10, 1000):
        distances[objective], next_arcs[objective] = best_paths(reverse, sinks, objective, start)
    observations = []
    reason_counts = Counter()
    route_groups = set()
    for stratum, root in chosen:
        check_time(start)
        item = {'stratum': 'known' if stratum == 0 else 'anonymous', 'root': root}
        try:
            encoding = db.execute('SELECT encoding FROM representations WHERE representation_id=?',
                                  (root,)).fetchone()[0]
            strands, cyclic, word = decoder.decode_representation(bytes(encoding))
            det, signed = invariants(tuple(word), strands)
            group = (det, abs(signed))
            own = owner((det, signed))
            item.update(group=list(group), owner=own, strands=strands,
                        word_length=len(word), cyclic=cyclic)
            if own != 'train':
                item['reason'] = 'not_RF_train'
            elif group in old:
                item['reason'] = 'old_group'
            elif group in viewed:
                item['reason'] = 'viewed_group'
            else:
                alternatives = []
                for objective in (10, 1000):
                    for arc in forward[root]:
                        rest = distances[objective].get(arc[1])
                        if rest is None:
                            continue
                        route = unwind(root, arc, next_arcs[objective], sinks)
                        if route is None:
                            continue
                        alternatives.append((objective, 1 + objective * arc[4] + rest,
                                             arc[1], route))
                pairs = []
                for i, a in enumerate(alternatives):
                    for b in alternatives[i + 1:]:
                        if a[0] == b[0] and a[2] != b[2] and a[1] != b[1]:
                            pairs.append((a, b))
                if pairs:
                    a, b = min(pairs, key=lambda pair: (pair[0][0], pair[0][1] + pair[1][1],
                                                          pair[0][2], pair[1][2]))
                    item['reason'] = 'two_structural_routes'
                    item['objective'] = a[0]
                    item['graph_costs'] = [a[1], b[1]]
                    item['routes'] = [x[3] for x in (a, b)]
                    route_groups.add(group)
                else:
                    item['reason'] = 'no_two_structural_routes'
        except (ValueError, TypeError, KeyError) as exc:
            item['reason'] = 'decode_or_invariant_error'
            item['error'] = str(exc)
        reason_counts[item['reason']] += 1
        observations.append(item)
    report = {
        'schema': 'rf-knots-embedding-pair-directed-route-screen-v1',
        'role': 'structural candidate upper bound; zero full paths or F labels',
        'complete': len(observations) == len(chosen),
        'cpu_seconds': time.process_time() - start,
        'cpu_ceiling_seconds': CPU_CEILING,
        'eligible_by_stratum': list(map(len, candidates)),
        'selected_by_stratum': [sum(s == i for s, _ in chosen) for i in (0, 1)],
        'primitive_arcs': arcs,
        'attested_unknot_sinks': len(sinks),
        'reasons': dict(reason_counts),
        'distinct_structural_train_groups': len(route_groups),
        'stop_go': 'full_program_replay_sample_permitted' if len(route_groups) >= 4
                   else 'stop_this_sample',
        'pins': pins,
        'observations': observations,
    }
    out = Path('artifacts/embedding-pair-directed-route-screen-v1/report.json')
    if out.exists():
        raise ValueError('report already exists')
    out.parent.mkdir(parents=True, exist_ok=False)
    out.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('pins', 'observations')}, indent=2))
    print('report_sha256', sha(out))


if __name__ == '__main__':
    main()
