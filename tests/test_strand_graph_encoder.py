import pytest
import torch

from rf_knots.mcts_loop import SitePolicyValueNet
from rf_knots.site_head import FactorizedSiteHead
from rf_knots.strand_graph_encoder import StrandGraphBraidEncoder, crossing_adjacency


def model():
    torch.manual_seed(41)
    return StrandGraphBraidEncoder(width=16, depth=2, embedding_dim=8).eval()


def test_adjacency_uses_actual_lanes_and_closure():
    word = torch.tensor([[1, 3, 2, -1]])
    edges = crossing_adjacency(word, torch.tensor([4]))
    assert edges.tolist() == [[[3, 3, 3, 2], [2, 1, 2, 1], [0, 1, 3, 1], [0, 2, 0, 0]]]


@pytest.mark.parametrize("n,length", [(2, 3), (2049, 257), (1000001, 4097)])
def test_same_checkpoint_beyond_old_size_limits(n, length):
    net = model()
    state = net.state_dict()
    fresh = model()
    fresh.load_state_dict(state, strict=True)
    word = torch.arange(length).remainder(n - 1).add(1).unsqueeze(0)
    word[0, -1] = n - 1
    out = fresh.encode_sites(word, torch.tensor([length]), torch.tensor([n]))
    assert out["tokens"].shape == (1, length, 16)
    assert all(torch.isfinite(out[key]).all() for key in ("tokens", "knot", "isotopy", "layout"))
    out["knot"][0, 0].backward()
    assert all(torch.isfinite(p.grad).all() for p in fresh.parameters() if p.grad is not None)
    assert sum(p.numel() for p in fresh.parameters()) == sum(p.numel() for p in net.parameters())


def test_rotation_padding_and_empty_contract():
    net = model().double()
    word = torch.tensor([[1, -2, 3, -1, 2, 1]])
    lengths, strands = torch.tensor([6]), torch.tensor([4])
    a = net.encode_sites(word, lengths, strands)
    b = net.encode_sites(word.roll(2, dims=1), lengths, strands)
    c = net.encode_sites(torch.cat([word, torch.full((1, 5), 9000)], 1), lengths, strands)
    torch.testing.assert_close(a["tokens"].roll(2, dims=1), b["tokens"])
    torch.testing.assert_close(a["tokens"], c["tokens"][:, :6])
    for key in ("knot", "isotopy", "layout"):
        torch.testing.assert_close(a[key], b[key])
        torch.testing.assert_close(a[key], c[key])
    empty = net.encode_sites(
        torch.zeros((1, 1), dtype=torch.long), torch.tensor([0]), torch.tensor([1])
    )
    assert torch.isfinite(empty["knot"]).all()
    assert not empty["tokens"].any()


def test_invalid_active_generator_rejected():
    with pytest.raises(ValueError, match="Artin"):
        model()(torch.tensor([[4]]), torch.tensor([1]), torch.tensor([4]))


def test_graph_reaches_distant_word_positions():
    net = model()
    assert crossing_adjacency(torch.tensor([[1, 3, 3, 3, 3, 1]]), torch.tensor([6]))[0, 0, 2] == 5
    # Port-specific update depends on that actual nonlocal neighbour.
    state = torch.randn(1, 6, 16, requires_grad=True)
    edges = crossing_adjacency(torch.tensor([[1, 3, 3, 3, 3, 1]]), torch.tensor([6]))
    net.strand_message(state, edges, torch.ones(1, 6, dtype=torch.bool))[0, 0, 0].backward()
    assert state.grad[0, 5].abs().sum() > 0


def test_opt_in_policy_network():
    net = SitePolicyValueNet(width=16, depth=2, embedding_dim=8, encoder_kind="strand_graph")
    assert net.encoder.maximum_strands is None
    fresh = SitePolicyValueNet(
        width=16, depth=2, embedding_dim=8, encoder_kind="strand_graph", graph_rounds=5
    )
    fresh.load_state_dict(net.state_dict(), strict=True)


def test_normalized_knot_retains_raw_sites_and_other_globals():
    net = StrandGraphBraidEncoder(width=16, depth=2, embedding_dim=8, normalized_knot=True)
    raw = StrandGraphBraidEncoder(width=16, depth=2, embedding_dim=8)
    raw.load_state_dict(net.state_dict(), strict=True)
    original = torch.tensor([[1, 1, 1]])
    lifted = torch.tensor([[1, 1, 1, 2, 3, -3, 3, 4]])
    a = net.encode_sites(original, torch.tensor([3]), torch.tensor([2]))
    b = net.encode_sites(lifted, torch.tensor([8]), torch.tensor([5]))
    c = raw.encode_sites(lifted, torch.tensor([8]), torch.tensor([5]))
    torch.testing.assert_close(a["knot"], b["knot"], rtol=0, atol=0)
    for name in ("tokens", "mask", "isotopy", "layout"):
        torch.testing.assert_close(b[name], c[name], rtol=0, atol=0)
    b["knot"][0, 0].backward()
    assert net.structural_projection[0].weight.grad.abs().sum() > 0


def test_normalized_knot_rotation_and_padding():
    net = StrandGraphBraidEncoder(
        width=16, depth=2, embedding_dim=8, normalized_knot=True, context_rounds=-1
    ).double()
    word = torch.tensor([[2, 1, 1, 1, -2, 3, -3, 3]])
    lengths, strands = torch.tensor([8]), torch.tensor([4])
    original = net.encode_sites(word, lengths, strands)
    for shift in range(8):
        rotated = net.encode_sites(word.roll(shift, dims=1), lengths, strands)
        torch.testing.assert_close(original["knot"], rotated["knot"])
        torch.testing.assert_close(original["tokens"].roll(shift, dims=1), rotated["tokens"])
    padded = net.encode_sites(torch.cat((word, torch.full((1, 9), 999)), 1), lengths, strands)
    torch.testing.assert_close(original["knot"], padded["knot"])
    torch.testing.assert_close(original["tokens"], padded["tokens"][:, :8])


def test_large_connected_knot_not_only_sparse_multicomponent_braid():
    from rf_knots.reference import num_components

    n = 2049
    word = tuple(range(1, n)) + (1, 1)
    assert num_components(word, n) == 1
    net = model()
    out = net(torch.tensor([word]), torch.tensor([len(word)]), torch.tensor([n]))
    assert torch.isfinite(out["knot"]).all()


def test_interior_insertion_generators_do_not_alias_and_padding_has_no_gradient():
    from rf_knots.actions import INSERT

    net = SitePolicyValueNet(width=16, depth=2, embedding_dim=8, encoder_kind="strand_graph")
    word = torch.tensor([[500, 501, -500]])
    sites = torch.tensor([[[INSERT, 0, 500, 1], [INSERT, 0, 501, 1], [INSERT, 0, 999999, 1]]])
    logits, value, effective = net(
        word,
        torch.tensor([3]),
        torch.tensor([2000]),
        sites,
        torch.tensor([[True, True, False]]),
        torch.tensor([[10.0, 2.0]]),
    )
    assert logits[0, 0] != logits[0, 1]
    assert logits[0, 2].isneginf()
    (logits[0, :2].sum() + value.sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_automatic_multiscale_has_no_batch_length_or_seam_dependence():
    net = StrandGraphBraidEncoder(width=16, depth=2, embedding_dim=8, context_rounds=-1).double()
    word = torch.tensor([[1, -2, 3, -1, 2, 1]])
    lengths, strands = torch.tensor([6]), torch.tensor([4])
    a = net.encode_sites(word, lengths, strands)
    b = net.encode_sites(word.roll(2, dims=1), lengths, strands)
    c = net.encode_sites(torch.nn.functional.pad(word, (0, 37)), lengths, strands)
    torch.testing.assert_close(a["tokens"].roll(2, dims=1), b["tokens"])
    torch.testing.assert_close(a["tokens"], c["tokens"][:, :6])
    torch.testing.assert_close(a["knot"], b["knot"])
    torch.testing.assert_close(a["knot"], c["knot"])
    fixed = StrandGraphBraidEncoder(width=16, depth=2, embedding_dim=8, context_rounds=7)
    fixed.load_state_dict(net.state_dict(), strict=True)


def test_multiscale_transmits_context_across_a_long_word():
    from rf_knots.strand_graph_encoder import CircularMultiscaleContext

    block = CircularMultiscaleContext(8)
    state = torch.randn(1, 257, 8, requires_grad=True)
    output = block(state, torch.ones(1, 257, dtype=torch.bool), torch.tensor([257]), -1)
    output[0, 0, 0].backward()
    assert state.grad[0, 128].abs().sum() > 0
    assert torch.isfinite(state.grad).all()


def test_hierarchical_windows_are_relative_and_padding_independent():
    net = StrandGraphBraidEncoder(
        width=16, depth=2, embedding_dim=8, window_size=3, hierarchy_levels=4
    ).double()
    word = torch.tensor([[1, -2, 3, -1, 2, 1, 3]])
    lengths, strands = torch.tensor([7]), torch.tensor([4])
    original = net.encode_sites(word, lengths, strands)
    rotated = net.encode_sites(word.roll(3, dims=1), lengths, strands)
    padded = net.encode_sites(torch.nn.functional.pad(word, (0, 19), value=9000), lengths, strands)
    torch.testing.assert_close(original["tokens"].roll(3, dims=1), rotated["tokens"])
    torch.testing.assert_close(original["tokens"], padded["tokens"][:, :7])
    for name in ("knot", "isotopy", "layout"):
        torch.testing.assert_close(original[name], rotated[name])
        torch.testing.assert_close(original[name], padded[name])


def test_hierarchical_windows_give_logarithmic_receptive_field():
    from rf_knots.strand_graph_encoder import HierarchicalWindowContext

    block = HierarchicalWindowContext(width=8, window_size=3, levels=5)
    state = torch.randn(1, 257, 8, requires_grad=True)
    output = block(state, torch.ones(1, 257, dtype=torch.bool), torch.tensor([257]))
    output[0, 0, 0].backward()
    assert state.grad[0, 48].abs().sum() > 0
    assert torch.isfinite(state.grad).all()


def test_factorized_site_head_scores_factors_without_l_times_n_tensor():
    torch.manual_seed(17)
    head = FactorizedSiteHead(width=16, embedding_dim=8, budget_dim=5)
    tokens = torch.randn(2, 7, 16)
    mask = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])
    globals_ = {name: torch.randn(2, 8) for name in ("knot", "isotopy", "layout")}
    budgets = torch.tensor([[10.0, 2.0, 20.0, 4.0, 1000.0], [3.0, 1.0, 5.0, 0.0, 10.0]])
    scores = head(tokens, mask, globals_, budgets=budgets, strands=torch.tensor([4, 1]))
    assert scores["kind_scores"].shape == (2, 9)
    assert scores["position_scores"].shape == (2, 9, 7)
    assert scores["generator_scores"].shape == (2, 3)
    assert scores["generator_mask"].tolist() == [[True, True, True], [False, False, False]]
    assert torch.isneginf(scores["position_scores"][1, :, 4:]).all()
    assert torch.isfinite(scores["generator_scores"][0]).all()
    assert torch.isneginf(scores["generator_scores"][1]).all()
    (
        scores["kind_scores"].sum()
        + scores["position_scores"][..., :4].sum()
        + scores["generator_scores"][0].sum()
        + scores["sign_scores"].sum()
    ).backward()


def test_factorized_action_scores_compose_only_supplied_sites():
    torch.manual_seed(18)
    head = FactorizedSiteHead(width=8, embedding_dim=4, budget_dim=2)
    tokens = torch.randn(1, 5, 8)
    mask = torch.ones(1, 5, dtype=torch.bool)
    globals_ = {name: torch.randn(1, 4) for name in ("knot", "isotopy", "layout")}
    factors = head(tokens, mask, globals_, budgets=torch.tensor([[8.0, 2.0]]), strands=torch.tensor([4]))
    sites = torch.tensor([[[3, 2, 2, -1], [7, 0, 1, 1]]])
    scores = head.action_scores(factors, sites)
    expected_insert = (
        factors["position_scores"][0, 3, 2]
        + factors["generator_scores"][0, 1]
        + factors["sign_scores"][0, 1]
    )
    torch.testing.assert_close(scores[0, 0], expected_insert)
    torch.testing.assert_close(scores[0, 1], factors["kind_scores"][0, 7])
