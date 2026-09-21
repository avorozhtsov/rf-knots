import importlib.util
from pathlib import Path


def builder():
    path = Path(__file__).parents[1] / "scripts/build_fresh_group_panel.py"
    spec = importlib.util.spec_from_file_location("fresh_panel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exclusions_include_both_endpoints_and_mirrors():
    module = builder()
    panel = {"validation": [{"left_fingerprint": [7, -2], "right_fingerprint": [3, 2]}],
             "size_shift": [{"left_fingerprint": [11, 0], "right_fingerprint": [1, 0]}]}
    assert module.excluded_groups([panel]) == {(7, 2), (3, 2), (11, 0), (1, 0)}


def test_adequacy_requires_groups_and_representations():
    module = builder()
    rows = [{"left_fingerprint": [2*i+1, 0]} for i in range(8)]
    assert not module.adequate(rows)
    assert module.adequate(rows * 2)
    assert not module.adequate(rows[:7] * 4)
