import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def audit():
    spec = importlib.util.spec_from_file_location(
        'route_audit', Path(__file__).parents[1] / 'scripts/audit_embedding_route_report.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case(edge, rank):
    return dict(edge_id=edge, source_node=1, target_node=edge + 10,
                embedding_rank=rank, structural_rank=2, candidate_pool=100,
                cosine=0.8, provenance='graph')


def test_provenance_duplication_does_not_inflate_recall(audit):
    first, second = case(1, 1), case(2, 50)
    rows = audit.unique_cases([first, {**first, 'provenance': 'replay'}, second])
    assert len(rows) == 2
    assert audit.rank_summary(rows, 'embedding_rank')['recall_at_10'] == 0.5
    # Use the conventional median, including when sample size is even.
    assert audit.rank_summary(rows, 'embedding_rank')['median_rank'] == 25.5


def test_conflicting_duplicate_is_not_silently_selected(audit):
    first = case(1, 1)
    with pytest.raises(ValueError, match='conflicting duplicate'):
        audit.unique_cases([first, {**first, 'embedding_rank': 99}])
