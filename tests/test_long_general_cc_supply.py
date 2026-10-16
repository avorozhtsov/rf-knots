import importlib
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("build_long_general_cc_supply")


@pytest.mark.parametrize("index", range(9))
def test_construction_length_components_and_repeatability(module, index):
    source, states, positions = module.construct(index)
    assert (source, states, positions) == module.construct(index)
    assert len(positions) == index % 3
    lo, hi = module.CONFIG["length_bands"][(index // 3) % 3]
    assert all(lo <= len(s["word"]) <= hi for s in states)
    assert all(module.reference.num_components(tuple(s["word"]), s["strands"]) == 1 for s in states)


@pytest.mark.parametrize("index", [0, 1, 2, 279])
def test_real_certificate(module, index):
    row = module.make_row(index)
    module.audit_row(row, index)
    assert 0 <= row["lower"] <= row["upper"] <= 2
    assert all(len(s["word"]) > 48 for s in row["states"])


@pytest.mark.parametrize("field", ["source", "fingerprints", "upper", "cyclic_rotation", "index"])
def test_reject_changed_evidence(module, field):
    row = deepcopy(module.make_row(1))
    if field == "source":
        row[field]["selected_blocks"] = []
    elif field == "fingerprints":
        row[field][0][0] += 2
    elif field == "upper":
        row[field] = 1.0
    elif field == "cyclic_rotation":
        row[field] = 0
    else:
        row[field] = True
    with pytest.raises(ValueError):
        module.audit_row(row, 1)


def fake_row(module, index, fingerprints, lower=1, upper=1):
    """Selection-only fixture, not a mathematical certificate."""
    states = [dict(word=[1, 2, 1 + index], strands=5), dict(word=[1, -2, 1 + index], strands=5)]
    return dict(
        index=index,
        source=dict(length_band=0),
        states=states,
        fingerprints=fingerprints,
        cc_positions=[1],
        lower=lower,
        upper=upper,
    )


def test_all_train_path_states_must_be_train_and_unexposed(module, monkeypatch):
    monkeypatch.setattr(module, "owner", lambda fp: {3: "train", 5: "validation", 7: "test"}[fp[0]])
    row = fake_row(module, 0, [[3, 2], [5, 0], [3, 2]])
    records, decisions = module.select([row], set(), set())
    assert decisions == ["non_train_path"] and not any(records.values())
    row["fingerprints"] = [[3, 2], [3, -2]]
    assert module.select([row], set(), {(3, 2)})[1] == ["excluded_train_path"]
    assert module.select([row], set(), set())[1] == ["accepted"]


def test_fresh_query_exclusion_and_known_reference(module, monkeypatch):
    monkeypatch.setattr(module, "owner", lambda fp: "test" if fp[0] == 7 else "train")
    row = fake_row(module, 0, [[7, 0], [3, 2]])
    assert module.select([row], {(7, 0)}, set())[1] == ["known_evaluation_query"]
    assert module.select([row], {(3, 2)}, {(7, 0)})[1] == ["known_evaluation_query"]
    assert module.select([row], {(3, 2)}, set())[1] == ["accepted"]


def test_quarantine_intermediate_and_reference(module, monkeypatch):
    monkeypatch.setattr(module, "owner", lambda fp: "quarantine" if fp[0] == 49 else "train")
    for fps in ([[3, 2], [49, 0], [3, 2]], [[3, 2], [49, 0]], [[49, 0], [3, 2]]):
        assert module.select([fake_row(module, 0, fps)], set(), set())[1] == ["quarantine"]


def test_pair_dedup_and_group_cap(module, monkeypatch):
    monkeypatch.setattr(module, "owner", lambda fp: "test")
    rows = [fake_row(module, i, [[7, 0], [3, 2]]) for i in range(3)]
    records, reasons = module.select([rows[0], rows[0], *rows[1:]], set(), set())
    assert reasons == ["accepted", "duplicate_pair", "accepted", "group_cap"]
    assert len(records["test"]) == 2


def test_interval_cannot_fill_exact_two_support(module):
    row = fake_row(module, 0, [[7, 0], [3, 2]], lower=1, upper=2)
    records = dict(train=[row] * 100, validation=[], test=[])
    summary, gaps = module.supply_summary(records)
    assert summary["train"]["classes"] == {"interval": 100}
    assert summary["train"]["unique_canonical_pairs"] == 1
    assert (
        next(g for g in gaps if g["split"] == "train" and g["band"] == 0 and g["category"] == "2")[
            "pairs"
        ]
        == 0
    )


def test_all_band_and_class_gates_required(module):
    summary, gaps = module.supply_summary(dict(train=[], validation=[], test=[]))
    assert len(gaps) == 27
    assert all(summary[s]["lengths"] is None for s in summary)


def test_shard_coverage_hash_and_orphan_guard(module, tmp_path, monkeypatch):
    monkeypatch.setitem(module.CONFIG, "shard_rows", 2)
    rows = [dict(index=i) for i in range(2)]
    path = tmp_path / "attempts-000000.json"
    module.atomic(path, rows)
    progress = dict(shards=[dict(file=path.name, sha256=module.sha(path))], next_index=2)
    assert module.load_shards(tmp_path, progress, replay=False) == rows
    progress["next_index"] = 1
    with pytest.raises(ValueError, match="coverage"):
        module.load_shards(tmp_path, progress, replay=False)
    progress["next_index"] = 2
    module.atomic(tmp_path / "attempts-000002.json", rows)
    with pytest.raises(ValueError, match="uncommitted"):
        module.load_shards(tmp_path, progress, replay=False)


def test_old_configuration_unchanged(module):
    old = importlib.import_module("build_resource_general_test")
    assert old.CONFIG["seed"] == 20261073
    assert module.CONFIG["seed"] == 20261075
