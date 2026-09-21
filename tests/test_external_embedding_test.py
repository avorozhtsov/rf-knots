import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, REDUCE, ActionSpec
from rf_knots.evidence import UnknotWitness


@pytest.fixture
def builder():
    path = Path(__file__).parents[1] / "scripts/build_external_embedding_test.py"
    spec = importlib.util.spec_from_file_location("external_embedding_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_rows():
    spec = ActionSpec(16, 5)
    witness = UnknotWitness.from_actions(
        (1, 1, 1),
        2,
        spec,
        [spec.encode(CROSSING_CHANGE, 0), spec.encode(REDUCE, 0), spec.encode(DESTABILIZE)],
    )
    rows = []
    for i, step in enumerate(witness.steps):
        rows.append(
            {
                "step_index": str(i),
                "start_word": "1,1,1",
                "start_strands": "2",
                "kind": step.action.kind,
                "position": str(step.action.position),
                "generator": str(step.action.generator),
                "sign": str(step.action.sign),
                "after_word": ",".join(map(str, step.after.word)),
                "after_strands": str(step.after.strands),
                "declared_cc": "1",
                "declared_moves": "3",
                "l10": "13",
                "l1000": "1003",
                "pool_sha256": "frozen-pool",
                "evidence_id": "proof",
                "representation_id": "root",
            }
        )
    return witness, rows


def test_external_steps_replay_and_order(builder):
    expected, rows = source_rows()
    actual = builder.replay_rows(list(reversed(rows)))
    assert actual.start == expected.start
    assert actual.steps == expected.steps
    actual.verify()


@pytest.mark.parametrize(
    "key,value",
    [
        ("step_index", "7"),
        ("start_word", "-1,-1,-1"),
        ("after_word", "1,1,1"),
        ("declared_cc", "0"),
        ("declared_moves", "2"),
        ("l10", "12"),
        ("l1000", "1002"),
        ("pool_sha256", "different-pool"),
        ("evidence_id", "different-proof"),
        ("representation_id", "different-root"),
    ],
)
def test_external_corruption_rejected(builder, key, value):
    _, rows = source_rows()
    rows = deepcopy(rows)
    rows[1][key] = value
    with pytest.raises(ValueError):
        builder.replay_rows(rows)
