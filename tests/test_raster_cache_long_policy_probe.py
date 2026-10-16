import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize("family", ["structured", "random-inserted-trefoil"])
@pytest.mark.parametrize("length,strands", [(64, 5), (256, 17), (1024, 65)])
def test_edit_chain_is_deterministic_and_replayable(monkeypatch, family, length, strands):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    p = importlib.import_module("probe_raster_cache_long_policy")
    batches, row = p.edit_chain(family, length, strands)
    assert row == p.edit_chain(family, length, strands)[1]
    assert len(batches) == len(row["states"]) == 5
    instance = p.M.Instance(**{**row["instance"], "word": tuple(row["instance"]["word"])})
    current = (instance.word, instance.strands)
    assert len(current[0]) == length and p.reference.num_components(*current) == 1
    for step, state in enumerate(row["states"]):
        assert current == (tuple(state["word"]), state["strands"])
        assert state["used_moves"] == step
        assert state["used_cc"] == min(step, 2)
        assert len(state["actions"]) == 6
        assert batches[step][0]["site_mask"].sum().item() == 6
        if step < 4:
            action = row["applied_actions"][step]
            assert p.reference.is_legal(instance.spec, *current, action, True)
            current = p.reference.apply(instance.spec, *current, action)
    assert len(current[0]) == length + 3 and current[1] == strands + 1
