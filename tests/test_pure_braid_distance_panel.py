import importlib
from copy import deepcopy
from pathlib import Path

import pytest

from rf_knots import reference


@pytest.fixture
def builder(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("build_pure_braid_distance_panel")


def permutation(word, n):
    labels = list(range(n))
    for g in word:
        p = abs(g) - 1
        labels[p], labels[p + 1] = labels[p + 1], labels[p]
    return labels


def test_every_block_is_pure_and_all_sizes_remain_knots(builder):
    for i in range(186):
        meta, states, positions = builder.construct(i)
        n = meta["reference_strands"]
        assert n == 3 + (i // 3) % 31
        assert len(positions) == i % 3
        assert 8 <= len(meta["blocks"]) <= 32
        for b in meta["blocks"]:
            c = b["conjugator"]
            g = b["generator"] * b["sign"]
            block = c + [g, g] + [-x for x in reversed(c)]
            assert permutation(block, n) == list(range(n))
            if meta["positive_centers"]:
                assert b["sign"] == 1
        for s in states:
            assert reference.num_components(tuple(s["word"]), n) == 1
            assert all(0 < abs(g) < n for g in s["word"])
        assert (meta, states, positions) == builder.construct(i)


def test_signed_path_and_zero_certificate_have_distinct_transcripts(builder):
    for i in (0, 1, 2, 90, 91, 92, 93, 94, 95):
        meta, states, positions = builder.construct(i)
        if not positions:
            assert len(states) == 2 and meta["rotation_from_reference"] is not None
            k = builder.cyclic_rotation(states[0], states[-1])
            w = states[0]["word"]
            assert w[k:] + w[:k] == states[-1]["word"]
        else:
            assert meta["rotation_from_reference"] is None
            for a, b, p in zip(states[:-1], states[1:], positions, strict=True):
                w = a["word"].copy()
                w[p] *= -1
                assert b["word"] == w


def synthetic_rows(builder, count=36):
    rows = []
    for i in range(count):
        source, states, positions = builder.construct(i)
        rows.append(
            {
                "index": i,
                "source": source,
                "states": states,
                "cc_positions": positions,
                "fingerprints": [[101 + 2 * i, 0], [3, 2]],
                "lower": i % 3,
                "upper": i % 3,
                "cyclic_rotation": None,
            }
        )
    return rows


def test_fixed_class_supply_gate_and_caps(builder, monkeypatch):
    monkeypatch.setattr(builder, "owner", lambda f: "test")
    rows = synthetic_rows(builder)
    records, decisions, classes, groups, adequate = builder.select(rows, set(), set())
    assert len(records) == groups == 36 and adequate
    assert classes == {"0": 12, "1": 12, "2": 12} and set(decisions) == {"accepted"}
    _, decisions, classes, _, adequate = builder.select(
        rows, set(), set(), {**builder.CONFIG, "per_category": 1}
    )
    assert classes == {"0": 1, "1": 1, "2": 1} and not adequate
    assert decisions.count("class_cap") == 33


def test_query_group_cap_is_global_and_uncertified_reference_rejected(builder, monkeypatch):
    monkeypatch.setattr(builder, "owner", lambda f: "test")
    rows = synthetic_rows(builder, 6)
    for r in rows:
        r["fingerprints"][0] = [101, 0]
    _, decisions, _, _, _ = builder.select(rows, set(), set())
    assert decisions == ["accepted"] * 4 + ["group_cap"] * 2
    rows[0]["fingerprints"][-1] = [1, 0]
    assert builder.select(rows, set(), set())[1][0] == "uncertified_nontrivial_reference"
    assert builder.select(rows, {(101, 0)}, set())[1][0] == "optimizer_collision"
    assert builder.select(rows, set(), {(101, 0)})[1][0] == "excluded_query"


def test_unordered_endpoint_dedup_does_not_quotient_mirrors(builder, monkeypatch):
    monkeypatch.setattr(builder, "owner", lambda f: "test")
    a = synthetic_rows(builder, 2)[1]
    b = deepcopy(a)
    b["states"].reverse()
    b["fingerprints"].reverse()
    c = deepcopy(a)
    c["states"] = [{**s, "word": [-g for g in s["word"]]} for s in c["states"]]
    assert builder.select([a, b, c], set(), set())[1] == ["accepted", "duplicate_pair", "accepted"]


def test_independent_constructor_matches_declared_rng_across_all_strands(builder):
    audit = importlib.import_module("audit_pure_braid_distance_panel")
    for i in range(186):
        assert audit.reconstruct(i) == builder.construct(i)


def actual_row(builder, index):
    source, states, positions = builder.construct(index)
    fps = [list(builder.invariants(tuple(s["word"]), s["strands"])) for s in states]
    rotation = builder.cyclic_rotation(states[0], states[-1])
    return {
        "index": index,
        "source": source,
        "states": states,
        "cc_positions": positions,
        "fingerprints": fps,
        "cyclic_rotation": rotation,
        "lower": max(abs(fps[0][1] - fps[-1][1]) // 2, int(fps[0][0] != fps[-1][0])),
        "upper": 0 if rotation is not None else len(positions),
    }


@pytest.mark.parametrize("index", [0, 1, 2, 92])
def test_independent_replay_and_bounds_for_zero_and_cc_paths(builder, index):
    audit = importlib.import_module("audit_pure_braid_distance_panel")
    audit.check_row(actual_row(builder, index), index)


@pytest.mark.parametrize(
    "change", ["source", "position", "rotation", "fingerprint", "bound", "float_bound"]
)
def test_independent_audit_rejects_corruption(builder, change):
    audit = importlib.import_module("audit_pure_braid_distance_panel")
    row = actual_row(builder, 1)
    if change == "source":
        row["source"]["blocks"][0]["sign"] *= -1
    elif change == "position":
        row["cc_positions"][0] += 1
    elif change == "rotation":
        row["cyclic_rotation"] = 0
    elif change == "fingerprint":
        row["fingerprints"][0][0] += 2
    elif change == "bound":
        row["upper"] = 2
    else:
        row["lower"] = float(row["lower"])
    with pytest.raises(ValueError):
        audit.check_row(row, 1)


def test_independent_first_eligible_selection_matches_without_shared_helper(builder, monkeypatch):
    audit = importlib.import_module("audit_pure_braid_distance_panel")
    monkeypatch.setattr(builder, "owner", lambda fp: "test")
    monkeypatch.setattr(audit, "owner", lambda fp: "test")
    rows = synthetic_rows(builder, 36)
    rows.append(deepcopy(rows[0]))
    records, decisions, classes, count, adequate = builder.select(rows, {(101, 0)}, {(105, 0)})
    a, b, c, groups, d = audit.selection(rows, {(101, 0)}, {(105, 0)})
    assert (a, b, c, len(groups), d) == (records, decisions, classes, count, adequate)
