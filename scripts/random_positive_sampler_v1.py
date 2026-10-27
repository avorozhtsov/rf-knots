"""Deterministic 50:50 positive-source draws, unchanged gradient objective."""
from pathlib import Path
from run_continuous_train_mcts_v1 import read
from build_mcts_training_supervision import sha
from rf_knots import mcts_loop as M
OLD_SELECT=None
RANDOM_GROUPS=None
LAST_SOURCE=None

def source_ordinal(step):
    n=step-351232
    if n<0:raise ValueError('sampler before origin')
    return ('random' if n%2 else 'retained'),n//2

def initialize(bank):
    global RANDOM_GROUPS
    groups={}
    for e in read(bank/'manifest.json')['paths']:
        p=bank/e['file'];assert sha(p)==e['sha256'];r=read(p)
        assert r['sampling_source']=='normalized_random_short_horizon'
        groups.setdefault(tuple(r['query_group']),[]).append(r)
    if len(groups)<2:raise ValueError('too few audited random groups')
    RANDOM_GROUPS=[(g,groups[g]) for g in sorted(groups)]

def select(step):
    global LAST_SOURCE
    source,n=source_ordinal(step);LAST_SOURCE=source
    if source=='retained':return OLD_SELECT(351232+n)
    _,rows=RANDOM_GROUPS[n%len(RANDOM_GROUPS)];weight=(10,1000)[(n//len(RANDOM_GROUPS))%2]
    row=min(rows,key=lambda r:(weight*r['achieved']['cc']+r['achieved']['moves'],r['path_id']))
    d={**row['instance'],'cc_weight':weight};return row,M.Instance(**{**d,'word':tuple(d['word'])})
