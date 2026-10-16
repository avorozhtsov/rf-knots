import importlib
from dataclasses import replace
from pathlib import Path

import pytest

from rf_knots.embedding_augmentation import (
    AugmentedRepresentation,
    ZeroCCOperation,
    augment_zero_cc,
)
from rf_knots.embedding_supervision import EmbeddingRepresentation


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("verified_zero_cc_views")


@pytest.mark.parametrize("word,n", [((), 1), ((1,), 2), ((1, 1, 1), 2), ((1, 2), 3)])
def test_valid_deterministic_round_trip(module, word, n):
    source = EmbeddingRepresentation(-1, word, n, None, "train")
    view = module.make_view(source, 20261061, 2)
    assert view == module.make_view(source, 20261061, 2)
    assert module.verify_view(source, view)["crossing_changes"] == 0
    assert view.strands > n and len(view.word) > len(word)
    if n == 1:
        assert view.operations[0] == ZeroCCOperation("stabilize", 0, 1, 1)


def test_existing_single_strand_insertion_needs_wrapper(module):
    source = EmbeddingRepresentation(-1, (), 1, None, "train")
    view = augment_zero_cc(source, 1, minimum_insertions=1, maximum_insertions=1)
    with pytest.raises(ValueError, match="generator"):
        module.verify_view(source, view)


@pytest.mark.parametrize(
    "operation",
    [
        ZeroCCOperation("stabilize", 1, 2, 1),
        ZeroCCOperation("stabilize", 3, 1, 1),
        ZeroCCOperation("insert_inverse_pair", 0, 2, 1),
        ZeroCCOperation("conjugate", 1, 1, 1),
        ZeroCCOperation("cyclic_shift", 0, amount=4),
        ZeroCCOperation("mirror", 0),
    ],
)
def test_tampered_operations_fail(module, operation):
    with pytest.raises(ValueError):
        module.forward_replay((1, 1, 1), 2, [operation])


def test_changed_endpoint_and_non_knot_fail(module):
    source = EmbeddingRepresentation(-1, (1, 1, 1), 2, None, "train")
    view = module.make_view(source, 19, 1)
    with pytest.raises(ValueError):
        module.verify_view(source, replace(view, strands=view.strands + 1))
    with pytest.raises(ValueError):
        module.verify_view(
            source, AugmentedRepresentation(view.word[::-1], view.strands, view.operations)
        )
    with pytest.raises(ValueError):
        module.validate_knot((), 2)
    with pytest.raises(ValueError):
        module.make_view(replace(source, split="validation"), 19, 1)
