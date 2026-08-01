import json

import pytest

from rf_knots.benchmarks import BenchmarkInstance, BenchmarkManifest, split_for


def test_split_is_stable_and_manifest_round_trips(tmp_path):
    instance = BenchmarkInstance.braid((1, 1, 1), 2, source_id="3_1", known_unknotting_number=1)
    assert instance.split == split_for(instance.instance_id)
    manifest = BenchmarkManifest("tiny", "1", {"fixture": True}, (instance,))
    path = tmp_path / "tiny.json"
    manifest.write(path)
    assert BenchmarkManifest.read(path) == manifest


def test_manifest_rejects_duplicate_instances(tmp_path):
    instance = BenchmarkInstance.braid((1, 1, 1), 2)
    manifest = BenchmarkManifest("bad", "1", {}, (instance, instance))
    path = tmp_path / "bad.json"
    manifest.write(path)
    with pytest.raises(ValueError, match="duplicate"):
        BenchmarkManifest.read(path)


def test_manifest_rejects_unknown_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "future"}))
    with pytest.raises(ValueError, match="schema"):
        BenchmarkManifest.read(path)
