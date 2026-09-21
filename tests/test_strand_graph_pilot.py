import importlib.util
import json
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "strand_graph_pilot", Path(__file__).parents[1] / "scripts/run_strand_graph_pilot.py"
)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_mirror_trefoils_are_not_a_zero_distance_target():
    pytest.importorskip("spherogram")
    assert pilot.seifert.signature((1, 1, 1), 2) == -2
    assert pilot.seifert.signature((-1, -1, -1), 2) == 2


def test_generated_intervals_and_representation_exclusion():
    pytest.importorskip("spherogram")
    rows, witnesses, _ = pilot.make_corpus(4, 37)
    forbidden = {pilot.canonical(w.start.word, w.start.strands) for w in witnesses}
    other, _, _ = pilot.make_corpus(4, 38, forbidden)
    assert all(r.spec.distance_lower <= r.spec.distance_upper for r in rows)
    assert all(pilot.canonical(r.left_word, r.left_strands) not in forbidden for r in other)
    for witness in witnesses:
        witness.verify()


def test_loss_trains_actual_pair_geometry():
    net = pilot.M.SitePolicyValueNet(
        width=16, depth=2, embedding_dim=8, encoder_kind="strand_graph"
    )
    objective = pilot.MetricEmbeddingObjective()
    row = pilot.pair_row(0, "test", ((1, 1, 1), 2), ((-1, -1, -1), 2), 2, 2, 6)
    loss, pred = pilot.metric_loss(net, objective, [row], "cpu")
    assert torch.isfinite(pred).all()
    loss.backward()
    assert net.encoder.tokens.weight.grad.abs().sum() > 0


def test_actual_mcts_adapter_and_witness_costs():
    net = pilot.M.SitePolicyValueNet(
        width=16, depth=2, embedding_dim=8, encoder_kind="strand_graph"
    )
    instances = [pilot.M.Instance("trefoil", (1, 1, 1), 2, 1, cap_moves=8, uid="t")]
    report = pilot.search_report(net, instances, 20, 1, "cpu")
    assert report["trees"] == 1
    for row in report["rows"]:
        if row["solved"]:
            assert row["L1000"] - row["L10"] == 990 * row["cc"]


def test_search_report_accounts_for_site_proposal_work():
    from rf_knots.site_proposals import SiteProposalProvider

    torch.set_num_threads(1)
    net=pilot.M.SitePolicyValueNet(width=16,depth=2,embedding_dim=8,encoder_kind="strand_graph")
    instances=[pilot.M.Instance("trefoil",(1,1,1),2,1,cap_moves=8,uid="t")]
    provider=SiteProposalProvider(net)
    # Reuse a provider: each report must charge this run, not lifetime totals.
    for _ in range(2):
        report=pilot.search_report(net,instances,8,2,"cpu",widening_batch=2,proposal_provider=provider)
        proposal=report["proposal_compute"]
        assert report["proposal_kind"]=="SiteProposalProvider"
        assert report["factorized_proposals"] is False
        assert proposal["state_evaluations"]>0 and proposal["action_scores"]>0
        assert report["total_neural_state_evaluations"]==report["evaluator_requests"]+proposal["state_evaluations"]
        assert proposal["logical_requests"]<=16


def test_explicit_short_u_anchor_is_not_double_counted(tmp_path):
    from rf_knots.embedding_replay_corpus import invariants

    config = pilot.InterleavedConfig(crossing_changes=0, walk_length=2)
    root = pilot.torus_seed(1, config.spec())
    witness = pilot.generate(config, 3, 0, root).witness
    fp = invariants(witness.start.word, witness.start.strands)
    row = {
        "split": "train",
        "pair_prefix_length": witness.moves,
        "left_fingerprint": list(fp),
        "right_fingerprint": [1, 0],
        "interval": {
            "left": witness.start.to_dict(),
            "right": witness.steps[-1].after.to_dict(),
            "lower": 1,
            "upper": 1,
            "witness_semantic_length": witness.moves,
        },
        "witness": witness.to_dict(),
    }
    path = tmp_path / "anchors.json"
    path.write_text(
        json.dumps(
            {
                "schema": "rf-knots-replay-fingerprint-corpus-v1",
                "train": [row],
                "validation": [],
                "size_shift": [],
            }
        )
    )
    _, result = pilot.load_frozen_corpus(path)
    rows, witnesses, derived_u = result["train"]
    assert len(rows) == len(witnesses) == 1
    assert derived_u == []
