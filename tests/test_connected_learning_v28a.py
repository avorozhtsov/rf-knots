from unittest.mock import patch
import positive_sampler_v28a as eligible


def test_quarantined_legacy_paths_are_retained_but_never_sampled():
    rows=[dict(path_id='legacy-protected'),dict(path_id='new-MCTS')]
    groups=[((3,2),[(((1,1,1),2),rows)])]
    def init(*_):eligible.base.SOURCES={'retained':groups,'random':groups}
    with patch.object(eligible,'INITIALIZE',init),patch.object(eligible,'read',return_value={'quarantined_path_ids':['legacy-protected']}):
        eligible.initialize('main','random')
        for source in eligible.base.SOURCES.values():
            assert [r['path_id'] for _,vs in source for _,rs in vs for r in rs]==['new-MCTS']
    assert len(rows)==2 # original evidence rows were not mutated or deleted
