import importlib
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return (
        importlib.import_module("build_long_general_cc_supply"),
        importlib.import_module("audit_long_general_cc_supply"),
    )


def selected(module, index):
    row = module.make_row(index)
    fps = row["fingerprints"]
    row.update(
        split=module.owner(fps[0]),
        pair_key=module.pair_key(row),
        left_fingerprint=fps[0],
        right_fingerprint=fps[-1],
        interval=dict(
            metric="cc",
            left=row["states"][0],
            right=row["states"][-1],
            lower=row["lower"],
            upper=row["upper"],
            witness_semantic_length=len(row["cc_positions"]),
        ),
    )
    return row


@pytest.mark.parametrize("index", [0, 1, 2])
def test_real_evidence_separately_consumed(modules, index, monkeypatch):
    builder, consumer = modules
    # Ownership is tested separately: all real generated certificate shapes,
    # including attempts rejected for their split, exercise this replay checker.
    monkeypatch.setattr(consumer, "owner", lambda fp: "train")
    row = selected(builder, index)
    row["split"] = "train"
    assert consumer.verify_pair(row, "train", set(), set()) == builder.group(row["fingerprints"][0])


@pytest.mark.parametrize("field", ["fingerprint", "interval", "position", "split", "key", "size"])
def test_reject_mutated_certificate(modules, field, monkeypatch):
    builder, consumer = modules
    monkeypatch.setattr(consumer, "owner", lambda fp: "train")
    row = deepcopy(selected(builder, 1))
    row["split"] = "train"
    if field == "fingerprint":
        row["fingerprints"][0][0] += 2
    elif field == "interval":
        row["interval"]["upper"] += 1
    elif field == "position":
        row["cc_positions"][0] = -1
    elif field == "split":
        row["split"] = "test"
    elif field == "key":
        row["pair_key"] = "wrong"
    else:
        row["source"]["length_band"] = 2
    with pytest.raises(ValueError):
        consumer.verify_pair(row, "train", set(), set())


def test_heldout_query_cannot_be_optimizer_group(modules, monkeypatch):
    builder, consumer = modules
    row = selected(builder, 1)
    monkeypatch.setattr(consumer, "owner", lambda fp: "test")
    row["split"] = "test"
    g = builder.group(row["fingerprints"][0])
    with pytest.raises(ValueError, match="exposed"):
        consumer.verify_pair(row, "test", {g}, set())


def test_non_train_intermediate_rejected(modules, monkeypatch):
    builder, consumer = modules
    row = selected(builder, 2)
    row["split"] = "train"
    middle = row["fingerprints"][1]
    # Use call order to isolate the all-state ownership requirement even if
    # actual endpoint invariants happen to collide in this deterministic row.
    calls = iter(["train"] * 4 + ["train", "validation", "train"])
    monkeypatch.setattr(consumer, "owner", lambda fp: next(calls))
    assert isinstance(middle, list)
    with pytest.raises(ValueError, match="non-TRAIN"):
        consumer.verify_pair(row, "train", set(), set())
