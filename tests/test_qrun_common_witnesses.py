import importlib
from pathlib import Path


def test_fixed_panel_is_replayable_and_proposals_cover_witnesses(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = importlib.import_module("check_qrun_common_witnesses")
    config = {**module.CONFIG, "per_stratum": 1}
    panel = module.build_panel(config)
    assert panel["complete"]
    assert len(panel["rows"]) == 4
    assert set(panel["strata"].values()) == {1}
    for row in panel["rows"]:
        witness = module.UnknotWitness.from_dict(row["witness"])
        witness.verify()
        assert witness.max_len == 48 and witness.max_strands == 12
        assert len(witness.start.word) <= 43
        assert all(module.owner(f) == "train" for f in row["fingerprints"])
        assert row["fingerprints"][0][0] > 1
        assert module.rf_coverage(witness)
    assert module.build_panel(config) == panel


def test_insufficient_supply_is_not_relaxed(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = importlib.import_module("check_qrun_common_witnesses")
    panel = module.build_panel({**module.CONFIG, "attempts": 0})
    assert not panel["complete"] and panel["rows"] == []
