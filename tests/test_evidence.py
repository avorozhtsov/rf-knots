import dataclasses
import json

import numpy as np
import pytest

from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, REDUCE, ActionSpec
from rf_knots.evidence import (
    BraidState,
    EvidenceRecord,
    EvidenceStore,
    LowerBoundClaim,
    UnknotWitness,
    braid_instance_id,
    braid_knot_key,
    shared_witness_targets,
    upper_bound_hinge,
)


def trefoil_witness() -> UnknotWitness:
    spec = ActionSpec(max_len=12, max_strands=4)
    actions = [
        spec.encode(CROSSING_CHANGE, position=0),
        spec.encode(REDUCE, position=0),
        spec.encode(DESTABILIZE),
    ]
    return UnknotWitness.from_actions((1, 1, 1), 2, spec, actions)


def test_witness_replays_and_counts_crossing_changes():
    witness = trefoil_witness()
    assert witness.verify() == BraidState((), 1)
    assert witness.crossing_changes == 1
    assert witness.moves == 3
    assert [
        target.remaining_crossing_changes for target in shared_witness_targets(witness)
    ] == [1, 0, 0]
    assert [target.remaining_moves for target in shared_witness_targets(witness)] == [3, 2, 1]


def test_state_trajectory_translation_omits_controller_only_steps():
    witness = trefoil_witness()
    states = [witness.start, witness.start]
    for step in witness.steps:
        states.extend([step.after, step.after])
    translated = UnknotWitness.from_states(states, ActionSpec(12, 4))
    assert translated.to_dict() == witness.to_dict()


def test_cyclic_band_witness_round_trips_and_verifies() -> None:
    spec = ActionSpec(max_len=16, max_strands=4, cyclic_band_generators=True)
    actions = [
        spec.encode(REDUCE, position=0),
        spec.encode(DESTABILIZE),
        spec.encode(DESTABILIZE),
    ]

    witness = UnknotWitness.from_actions((3, -3, 1, 2), 3, spec, actions)
    restored = UnknotWitness.from_dict(witness.to_dict())

    assert restored.cyclic_band_generators is True
    assert restored.verify() == BraidState((), 1, True)


def test_witness_rejects_a_tampered_intermediate_state():
    witness = trefoil_witness()
    bad_step = dataclasses.replace(witness.steps[0], after=BraidState((1, 1, 1), 2))
    bad = dataclasses.replace(witness, steps=(bad_step,) + witness.steps[1:])
    with pytest.raises(ValueError, match="replay mismatch"):
        bad.verify()


def test_evidence_store_hashes_replays_and_folds_best(tmp_path):
    witness = trefoil_witness()
    instance_id = braid_instance_id((1, 1, 1), 2)
    record = EvidenceRecord(
        instance_id=instance_id,
        knot_key=braid_knot_key((1, 1, 1), 2),
        solver="unit-test",
        search_budget={"simulations": 32},
        outcome="solved",
        witness=witness,
        lower_bounds=(LowerBoundClaim(1, "signature", "spherogram", details={"sigma": -2}),),
    )
    assert record.exact_unknotting_number == 1
    store = EvidenceStore(tmp_path / "evidence.jsonl")
    record_id = store.append(record)
    assert len(record_id) == 64
    assert store.records() == [record]
    assert store.best_witnesses()[instance_id] == record

    row = json.loads((tmp_path / "evidence.jsonl").read_text())
    row["solver"] = "tampered"
    (tmp_path / "bad.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="hash"):
        EvidenceStore(tmp_path / "bad.jsonl").records()


def test_upper_bound_hinge_is_one_sided():
    actual = upper_bound_hinge(np.array([1.0, 3.0, 7.0]), np.array([2.0, 3.0, 5.0]))
    np.testing.assert_array_equal(actual, np.array([0.0, 0.0, 2.0]))
