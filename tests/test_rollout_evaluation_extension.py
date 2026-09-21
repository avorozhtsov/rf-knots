import importlib.util
from pathlib import Path

import pytest


def test_empty_or_overlapping_query_sets_rejected():
    path = Path(__file__).parents[1] / "scripts/verify_rollout_evaluation_extension.py"
    spec = importlib.util.spec_from_file_location("rollout_extension", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.check_sets({(5, 0)}, {(3, 2)}, {(9, 4)})
    for q, r, m in [(set(), set(), set()), ({(5, 0)}, {(5, 0)}, set()),
                    ({(5, 0)}, set(), {(5, 0)})]:
        with pytest.raises(ValueError):
            module.check_sets(q, r, m)
