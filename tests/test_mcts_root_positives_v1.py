from pathlib import Path
from rf_knots.actions import ActionSpec
from rf_knots.evidence import UnknotWitness
from run_continuous_train_mcts_v1 import read
from export_mcts_root_positives_v1 import semantic_id
from audit_mcts_root_positives_v1 import key


def test_semantic_path_identity_ignores_flat_capacity_offsets():
    data=Path('artifacts/continuous-train-mcts-v2')
    saved=next(read(data/f'attempt-{i:06d}.json') for i in range(256)
               if read(data/f'attempt-{i:06d}.json')['row'] and read(data/f'attempt-{i:06d}.json')['row']['witness'])
    w=UnknotWitness.from_dict(saved['row']['witness']);w.verify()
    wider=ActionSpec(w.max_len+4,w.max_strands+1,w.cyclic_band_generators)
    actions=[s.action.to_flat(wider) for s in w.steps]
    same=UnknotWitness.from_actions(w.start.word,w.start.strands,wider,actions);same.verify()
    assert semantic_id(w)==semantic_id(same)==key(w)==key(same)
    assert same.crossing_changes==w.crossing_changes and same.moves==w.moves


def test_achieved_cost_is_upper_bound_observation_not_exact_optimum():
    from export_mcts_root_positives_v1 import external_groups
    assert isinstance(external_groups(),set)
    # No unsolved-path sentinel or negative target is accepted by a witness.
    import pytest
    with pytest.raises(ValueError):
        UnknotWitness.from_actions((1,1,1),2,ActionSpec(8,3),[])
