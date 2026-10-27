"""50:50 sources; uniform group -> exact view -> semantic completed path.

Unlike v27 best-path-only selection, each admitted semantic path is reachable.
Repeated search observations never increase a path's sampling weight.
"""
from pathlib import Path
from run_continuous_train_mcts_v1 import read
from build_mcts_training_supervision import sha
from random_positive_sampler_v1 import source_ordinal
from rf_knots import mcts_loop as M

SOURCES = {}
LAST = None


def load_bank(bank):
    groups, seen = {}, set()
    for entry in read(bank / 'manifest.json')['paths']:
        p = bank / entry['file']
        if sha(p) != entry['sha256']:
            raise ValueError('path SHA differs')
        row = read(p)
        if row['path_id'] in seen:
            raise ValueError('semantic duplicate changes sampler weight')
        seen.add(row['path_id'])
        view = (tuple(row['instance']['word']), row['instance']['strands'])
        groups.setdefault(tuple(row['query_group']), {}).setdefault(view, []).append(row)
    if not groups:
        raise ValueError('empty bank')
    return [(g, [(v, sorted(rows, key=lambda r: r['path_id']))
                  for v, rows in sorted(views.items())]) for g, views in sorted(groups.items())]


def initialize(main_bank, random_bank):
    global SOURCES
    SOURCES = {'retained': load_bank(Path(main_bank)), 'random': load_bank(Path(random_bank))}


def select(step):
    global LAST
    source, n = source_ordinal(step)
    groups = SOURCES[source]
    group, views = groups[n % len(groups)]
    visits = n // len(groups)
    view, rows = views[visits % len(views)]
    visits //= len(views)
    row = rows[visits % len(rows)]
    weight = (10, 1000)[(visits // len(rows)) % 2]
    LAST = dict(source=source, path_id=row['path_id'], query_group=list(group),
                word=list(view[0]), strands=view[1], shard=row['observations'][0]['shard'])
    d = {**row['instance'], 'cc_weight': weight}
    return row, M.Instance(**{**d, 'word': tuple(d['word'])})
