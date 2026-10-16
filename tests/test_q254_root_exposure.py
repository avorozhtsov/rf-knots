import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("check_q254_registered_root_exposure")


def test_frozen_prefix_uses_native_compact_order_hash(audit):
    prefix = ["first", "second"]
    frozen = {
        "processed_count": 2,
        "processed_order_sha256": hashlib.sha256(
            json.dumps(prefix, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    assert audit.processed_prefix(frozen, {"processed": [*prefix, "later"]}) == prefix
    with pytest.raises(ValueError, match="prefix"):
        audit.processed_prefix(frozen, {"processed": ["second", "first", "later"]})


def test_bank_hash_is_not_compact_json_hash(audit):
    payload = {"b": 2, "a": 1}
    assert (
        audit.bank_digest(payload)
        == hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    )
    assert (
        audit.bank_digest(payload)
        != hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_fingerprint_collision_and_exact_word_match_are_distinct(audit):
    panel = {
        "validation": [
            {
                "interval": {
                    "left": {"word": [1, -2, 1], "strands": 3},
                    "right": {"word": [], "strands": 1},
                },
                "left_fingerprint": [5, 2],
                "right_fingerprint": [1, 0],
            }
        ]
    }
    roots = [
        {"id": "collision-only", "word": [2, 2, 2], "strands": 3, "fingerprint": [5, -2]},
        {"id": "cyclic", "word": [-2, 1, 1], "strands": 3, "fingerprint": [5, 2]},
    ]
    result = audit.compare_queries(panel, roots)
    left = result["validation/left"]["rows"][0]
    assert left["fingerprint_matches"] == ["collision-only", "cyclic"]
    assert left["cyclic_word_matches"] == ["cyclic"]
    assert result["validation/right"]["explicit_unknot_anchors"] == 1
