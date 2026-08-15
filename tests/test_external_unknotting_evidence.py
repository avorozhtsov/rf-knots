import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "import_external_unknotting_evidence.py"
    spec = importlib.util.spec_from_file_location("import_external_unknotting_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pd_validation_requires_every_arc_label_exactly_twice():
    valid_pd = _module().valid_pd
    assert valid_pd([[0, 2, 1, 3], [2, 0, 3, 1]])
    assert not valid_pd([[0, 2, 1, 3], [2, 0, 3, 4]])
    assert not valid_pd([])


def test_pd_validation_rejects_non_crossing_rows():
    valid_pd = _module().valid_pd
    assert not valid_pd([[0, 1, 0]])
    assert not valid_pd("[[0, 1, 2, 3]]")
