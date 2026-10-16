import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def profile(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("profile_pretrained_raster_training")


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.project = torch.nn.Linear(3, 8)

    def forward(self, word, lengths, strands):
        features = torch.stack((word.float().sum(1), lengths.float(), strands.float()), dim=1)
        value = torch.nn.functional.normalize(self.project(features), dim=-1)
        return {key: value for key in ("knot", "layout", "isotopy")}


@pytest.mark.parametrize("microbatch", [1, 2, 4])
def test_accumulation_matches_weighted_full_batch(profile, microbatch):
    from run_strand_graph_pilot import pair_row

    torch.manual_seed(90)
    net = torch.nn.Module()
    net.encoder = TinyEncoder()
    objective = profile.MetricEmbeddingObjective()
    rows = [
        pair_row(i, "probe", ((1,) * (i + 3), 2), ((-1, -1, -1), 2), i % 3, i % 3, length)
        for i, length in enumerate((21, 12, 3, 8))
    ]
    clone, other = copy.deepcopy(net), copy.deepcopy(objective)
    expected, _ = profile.metric_loss(net, objective, rows, "cpu")
    expected.backward()
    actual = profile.backward_metric(clone, other, rows, microbatch)
    assert actual == pytest.approx(float(expected.detach()), rel=1e-6, abs=1e-7)
    for left, right in zip(
        [*net.parameters(), *objective.parameters()],
        [*clone.parameters(), *other.parameters()],
        strict=True,
    ):
        if left.grad is None:
            assert right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)


def test_selection_is_score_blind_and_maximum_is_included(profile):
    from run_strand_graph_pilot import pair_row

    rows = [pair_row(i, "probe", ((1,) * i, 2), ((1,), 2), 1, 1, 1) for i in (3, 5, 4, 1, 2)]
    sampler = SimpleNamespace(bins=[[[rows[0], rows[1]], rows[2:]]])
    chosen, count = profile.selected_rows(sampler, 1, 2)
    assert count == 5
    assert [row.spec.index for row in chosen] == [4, 5]
    with pytest.raises(ValueError):
        profile.selected_rows(sampler, -1, 2)


def test_rank_loss_cannot_silently_change_under_accumulation(profile):
    objective = profile.MetricEmbeddingObjective(cc_rank_weight=1)
    with pytest.raises(ValueError, match="rank-free"):
        profile.backward_metric(None, objective, [], 1)
