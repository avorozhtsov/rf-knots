"""Source-balanced, group-balanced, view-balanced deterministic sampler."""
from pathlib import Path
from run_continuous_train_mcts_v1 import read
from build_mcts_training_supervision import sha
from rf_knots import mcts_loop as M
from random_positive_sampler_v1 import source_ordinal
OLD_SELECT=None
GROUPS=None
LAST_SOURCE=None
LAST_VIEW=None
LAST_SHARD=None

def view_slot(n,group_count,view_count):
    visits=n//group_count
    return visits%view_count,(10,1000)[(visits//view_count)%2]

def initialize(bank):
    global GROUPS
    groups={}
    for e in read(bank/'manifest.json')['paths']:
        p=bank/e['file'];assert sha(p)==e['sha256'];r=read(p);d=r['instance'];view=(tuple(d['word']),d['strands'])
        assert r['sampling_source']=='normalized_random_short_horizon'
        groups.setdefault(tuple(r['query_group']),{}).setdefault(view,[]).append(r)
    assert len(groups)>=2
    GROUPS=[(g,[(v,groups[g][v]) for v in sorted(groups[g])]) for g in sorted(groups)]

def select(step):
    global LAST_SOURCE,LAST_VIEW,LAST_SHARD
    source,n=source_ordinal(step);LAST_SOURCE=source
    if source=='retained':
        r,i=OLD_SELECT(351232+n);LAST_VIEW=(tuple(i.word),i.strands);LAST_SHARD=r['observations'][0]['shard'];return r,i
    _,views=GROUPS[n%len(GROUPS)];slot,weight=view_slot(n,len(GROUPS),len(views));LAST_VIEW,rows=views[slot]
    r=min(rows,key=lambda r:(weight*r['achieved']['cc']+r['achieved']['moves'],r['path_id']));LAST_SHARD=r['observations'][0]['shard'];d={**r['instance'],'cc_weight':weight}
    return r,M.Instance(**{**d,'word':tuple(d['word'])})
