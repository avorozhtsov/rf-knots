"""Explicit legacy quarantine, with no removal or alteration of evidence rows."""
from pathlib import Path
import positive_sampler_v28 as base
from run_continuous_train_mcts_v1 import read

INITIALIZE=base.initialize


def initialize(main_bank,random_bank):
    INITIALIZE(main_bank,random_bank)
    for source,bank in [('retained',main_bank),('random',random_bank)]:
        excluded=set(read(Path(bank)/'manifest.json').get('quarantined_path_ids',[]))
        filtered=[];found=set()
        for group,views in base.SOURCES[source]:
            kept=[]
            for view,rows in views:
                found.update(r['path_id'] for r in rows if r['path_id'] in excluded)
                rows=[r for r in rows if r['path_id'] not in excluded]
                if rows:kept.append((view,rows))
            if kept:filtered.append((group,kept))
        assert found==excluded and filtered
        base.SOURCES[source]=filtered
