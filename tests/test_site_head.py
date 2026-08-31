"""The per-site head of note 22 §M1, and the set-valued loss of §5.3.

Four properties carry the milestone and are tested here as properties rather than
as regression values: the parameter count does not depend on `(max_len,
max_strands)`, a site row round-trips through the action space it replaces, a
masked site is inert in both directions, and the loss cannot prefer one
demonstrated-good site over another.
"""

from __future__ import annotations

import pytest
import torch

from rf_knots.actions import (
    BRAID,
    COMMUTE,
    CROSSING_CHANGE,
    DESTABILIZE,
    INSERT,
    PASS,
    REDUCE,
    ActionSpec,
)
from rf_knots.site_head import (
    MASKED_LOGIT,
    POSITIONLESS_KINDS,
    SiteHead,
    masked_log_policy,
    set_valued_loss,
    sites_from_actions,
)

WIDTH, EMBEDDING = 16, 8


def make_head(seed: int = 0, **kwargs) -> SiteHead:
    torch.manual_seed(seed)
    return SiteHead(width=WIDTH, embedding_dim=EMBEDDING, **kwargs).double()


def make_batch(
    length: int,
    strands: int,
    sites: list[list[tuple[int, int, int, int]]],
    *,
    lengths: list[int] | None = None,
    seed: int = 1,
):
    """A synthetic encoder output plus a site table, in the §M1 contract's shapes."""
    generator = torch.Generator().manual_seed(seed)
    batch = len(sites)
    lengths = lengths or [length] * batch
    tokens = torch.randn(batch, length, WIDTH, generator=generator, dtype=torch.double)
    mask = torch.arange(length)[None, :] < torch.tensor(lengths)[:, None]
    tokens = tokens * mask.unsqueeze(-1)  # "zeroed on padding", per the contract
    globals_ = {
        name: torch.randn(batch, EMBEDDING, generator=generator, dtype=torch.double)
        for name in ("knot", "isotopy", "layout")
    }
    site_tensor = torch.tensor(sites, dtype=torch.long)
    site_mask = torch.ones(site_tensor.shape[:2], dtype=torch.bool)
    strand_tensor = torch.full((batch,), strands, dtype=torch.long)
    return tokens, mask, globals_, site_tensor, site_mask, strand_tensor


#: One of every kind, with the placeholder columns `decode` actually emits.
SMALL_SITES = [
    [
        (REDUCE, 0, 1, 1),
        (BRAID, 2, 1, 1),
        (INSERT, 1, 2, 1),
        (INSERT, 1, 2, -1),
        (INSERT, 3, 1, 1),
        (CROSSING_CHANGE, 4, 1, 1),
        (DESTABILIZE, 0, 1, 1),
        (PASS, 0, 1, 1),
    ]
]


# -- 1. the milestone: no parameter is sized by (max_len, max_strands) ---------


def test_parameter_shapes_mention_only_the_head_s_own_hyperparameters() -> None:
    head = make_head()
    rows = head.boundary_clip + 2
    allowed = {
        1,
        3,
        WIDTH,
        2 * WIDTH,
        EMBEDDING,
        head.num_kinds,
        head.context_dim,
        rows,
        rows + 1,
    }
    for name, parameter in head.named_parameters():
        for dimension in parameter.shape:
            assert dimension in allowed, f"{name} has a dimension {dimension} from nowhere"


def test_one_instance_scores_two_very_different_configurations() -> None:
    """`(L, N) = (8, 3)` and `(1024, 257)` through the same weights, unchanged."""
    head = make_head()
    before = sum(p.numel() for p in head.parameters())
    shapes_before = {n: tuple(p.shape) for n, p in head.named_parameters()}

    tiny = make_batch(8, 3, [[(REDUCE, 0, 1, 1), (INSERT, 2, 1, -1), (PASS, 0, 1, 1)]])
    small_logits = head(*tiny[:5], strands=tiny[5])
    assert small_logits.shape == (1, 3)

    wide = [[(REDUCE, index, 1, 1) for index in range(0, 1024, 4)] + [(INSERT, 7, 256, 1)]]
    huge = make_batch(1024, 257, wide)
    big_logits = head(*huge[:5], strands=huge[5])
    assert big_logits.shape == (1, 257)

    assert sum(p.numel() for p in head.parameters()) == before
    assert {n: tuple(p.shape) for n, p in head.named_parameters()} == shapes_before
    assert torch.isfinite(small_logits).all() and torch.isfinite(big_logits).all()


def test_a_tier1_state_dict_loads_and_scores_at_tier2() -> None:
    """The transfer M0.4 measured failing for the flat head: `[1156,96]` vs `[25604,96]`."""
    tier1, tier2 = ActionSpec(64, 8), ActionSpec(512, 24)
    assert tier1.num_actions == 1156 and tier2.num_actions == 25604

    trained = make_head(seed=3)
    checkpoint = {k: v.clone() for k, v in trained.state_dict().items()}
    fresh = make_head(seed=99)
    assert not torch.allclose(fresh.kind_embedding.weight, trained.kind_embedding.weight)
    fresh.load_state_dict(checkpoint)  # would raise on any size mismatch

    batch = make_batch(512, 24, [[(INSERT, 400, 23, -1), (REDUCE, 511, 1, 1), (PASS, 0, 1, 1)]])
    torch.testing.assert_close(
        fresh(*batch[:5], strands=batch[5]),
        trained(*batch[:5], strands=batch[5]),
        rtol=0,
        atol=0,
    )


def test_output_width_is_the_number_of_sites_not_the_action_space() -> None:
    head = make_head()
    for count in (1, 5, 137):
        sites = [[(INSERT, index % 64, 1 + index % 7, 1) for index in range(count)]]
        batch = make_batch(64, 8, sites)
        assert head(*batch[:5], strands=batch[5]).shape == (1, count)


# -- 2. a site row is the action, spelled out ---------------------------------


@pytest.mark.parametrize("shape", [(16, 4), (64, 8)])
def test_every_action_round_trips_through_a_site_row(shape) -> None:
    """`encode(*decode(a)) == a` for the whole space, which is what the 4 columns buy."""
    spec = ActionSpec(*shape)
    actions = list(range(spec.num_actions))
    sites = sites_from_actions(actions, spec)
    assert sites.shape == (spec.num_actions, 4)
    for action, row in zip(actions, sites.tolist(), strict=True):
        assert spec.encode(*row) == action


def test_the_two_insert_orientations_are_distinct_rows_and_distinct_scores() -> None:
    """`INSERT(p, g, +1)` and `INSERT(p, g, -1)` are different actions, so different sites."""
    spec = ActionSpec(max_len=8, max_strands=5)
    positive = spec.encode(INSERT, 5, 2, 1)
    negative = spec.encode(INSERT, 5, 2, -1)
    assert positive != negative
    rows = sites_from_actions([positive, negative], spec)
    torch.testing.assert_close(rows, torch.tensor([[INSERT, 5, 2, 1], [INSERT, 5, 2, -1]]))

    head = make_head(seed=7)
    batch = make_batch(8, 5, [rows.tolist()])
    logits = head(*batch[:5], strands=batch[5])[0]
    assert logits[0] != logits[1], "the sign column must reach the score"


def test_insert_distinguishes_its_generator() -> None:
    head = make_head(seed=7)
    batch = make_batch(6, 8, [[(INSERT, 2, 1, 1), (INSERT, 2, 4, 1), (INSERT, 2, 7, 1)]])
    logits = head(*batch[:5], strands=batch[5])[0]
    assert len(set(logits.tolist())) == 3, "same position, three generators, three scores"


def test_the_placeholder_columns_of_a_position_only_kind_are_never_read() -> None:
    """`decode` returns `generator = 1, sign = 1` there, and `1` is a real generator."""
    head = make_head(seed=11)
    kinds = [REDUCE, COMMUTE, BRAID, CROSSING_CHANGE, *POSITIONLESS_KINDS]
    honest = [[(kind, 2 if kind not in POSITIONLESS_KINDS else 0, 1, 1) for kind in kinds]]
    batch = make_batch(6, 8, honest)
    baseline = head(*batch[:5], strands=batch[5])

    for generator, sign in ((5, -1), (7, 1), (-3, 0), (999, -1)):
        poisoned = batch[3].clone()
        poisoned[..., 2] = generator
        poisoned[..., 3] = sign
        torch.testing.assert_close(
            head(batch[0], batch[1], batch[2], poisoned, batch[4], strands=batch[5]),
            baseline,
            rtol=0,
            atol=0,
        )


def test_a_positionless_kind_never_reads_the_position_column() -> None:
    head = make_head(seed=13)
    honest = [[(kind, 0, 1, 1) for kind in POSITIONLESS_KINDS]]
    batch = make_batch(6, 4, honest)
    baseline = head(*batch[:5], strands=batch[5])
    poisoned = batch[3].clone()
    poisoned[..., 1] = torch.tensor([3, 5, -2, 4_000])
    torch.testing.assert_close(
        head(batch[0], batch[1], batch[2], poisoned, batch[4], strands=batch[5]),
        baseline,
        rtol=0,
        atol=0,
    )


# -- 3. the scored form is the one §M1 specifies ------------------------------


def test_logit_equals_the_written_scalar_product() -> None:
    """`<e_t, W h_s> + b_t`, with `h_s` the FiLM-modulated token, computed by hand."""
    head = make_head(seed=5)
    batch = make_batch(6, 4, SMALL_SITES)
    tokens, mask, globals_, sites, site_mask, strands = batch
    budgets = torch.tensor([[4.0, 2.0]], dtype=torch.double)
    logits = head(tokens, mask, globals_, sites, site_mask, budgets=budgets, strands=strands)

    context = head._context_features(globals_, budgets, 1, tokens.device, tokens.dtype)
    gamma_c, beta_c = head.context_film(context).chunk(2, dim=-1)
    table = head._generator_film_table()

    for index in range(sites.shape[1]):
        kind, position, generator, sign = (int(v) for v in sites[0, index])
        if kind in POSITIONLESS_KINDS:
            raw = head.positionless_vector.weight[kind]
        else:
            raw = tokens[0, position]
        present = torch.tensor([[kind == INSERT]])
        rows = head._generator_rows(
            torch.tensor([[generator]]), torch.tensor([[sign]]), present, strands
        )
        scale, shift = table[rows[0, 0]].chunk(2, dim=-1)
        h = head.token_norm(raw)
        h = h * scale + shift  # the table already carries the FiLM's `1 +`
        h = h * (1.0 + gamma_c[0]) + beta_c[0]
        expected = head.kind_embedding.weight[kind] @ head.projection(h) + head.kind_bias[kind]
        torch.testing.assert_close(logits[0, index], expected, rtol=1e-10, atol=1e-10)


def test_the_strand_axis_is_absolute_and_knows_its_far_boundary() -> None:
    """§6.4: distance to strand `n-1` is real structure, and `g` is never scaled by `n`."""
    head = make_head(seed=11)
    sites = [[(INSERT, 2, 3, 1)]]
    near_top = make_batch(6, 5, sites)  # generator 3 of 4: one from the top
    far_from_top = make_batch(6, 21, sites, seed=1)  # same generator, wide braid
    assert near_top[0].equal(far_from_top[0])
    assert head(*near_top[:5], strands=near_top[5]) != head(
        *far_from_top[:5], strands=far_from_top[5]
    )
    # ... and with no strand count supplied the feature reads "unknown", not "1".
    unknown = head(*near_top[:5])
    assert torch.isfinite(unknown).all()


def test_positionless_actions_are_scored_against_the_global_context() -> None:
    head = make_head(seed=13)
    sites = [[(kind, 0, 1, 1) for kind in POSITIONLESS_KINDS]]
    first = make_batch(6, 4, sites, seed=1)
    second = make_batch(6, 4, sites, seed=2)
    a = head(*first[:5], strands=first[5])
    b = head(*second[:5], strands=second[5])
    assert not torch.allclose(a, b), "the four sited-less actions must read the context"
    assert len(set(a[0].tolist())) == 4, "each gets its own learned vector"


def test_budgets_condition_the_score_and_are_optional() -> None:
    head = make_head(seed=17)
    batch = make_batch(6, 4, SMALL_SITES)
    plenty = head(*batch[:5], budgets=torch.tensor([[9.0, 3.0]], dtype=torch.double))
    nearly_none = head(*batch[:5], budgets=torch.tensor([[1.0, 0.0]], dtype=torch.double))
    unconditioned = head(*batch[:5])
    assert not torch.allclose(plenty, nearly_none)
    assert torch.isfinite(unconditioned).all()


# -- 4. masking, in both directions -------------------------------------------


def test_masked_sites_are_negative_infinity_and_carry_no_probability() -> None:
    head = make_head(seed=19)
    batch = make_batch(6, 4, SMALL_SITES)
    tokens, mask, globals_, sites, site_mask, strands = batch
    site_mask = site_mask.clone()
    site_mask[0, 1] = site_mask[0, 5] = False
    logits = head(tokens, mask, globals_, sites, site_mask, strands=strands)
    assert (logits[~site_mask] == MASKED_LOGIT).all()
    probabilities = torch.softmax(logits, dim=-1)
    assert (probabilities[~site_mask] == 0.0).all()
    torch.testing.assert_close(probabilities.sum(), torch.ones((), dtype=torch.double))


def test_junk_in_a_masked_site_cannot_move_any_other_logit() -> None:
    """Adversarial: the padding rows hold out-of-range kinds, positions, generators."""
    head = make_head(seed=23)
    batch = make_batch(6, 4, SMALL_SITES)
    tokens, mask, globals_, sites, site_mask, strands = batch
    site_mask = site_mask.clone()
    site_mask[0, 2] = site_mask[0, 6] = False
    clean = head(tokens, mask, globals_, sites, site_mask, strands=strands)

    poisoned = sites.clone()
    poisoned[0, 2] = torch.tensor([-97, 10_000, -10_000, 0])
    poisoned[0, 6] = torch.tensor([9_999, -5, 123_456, -7])
    dirty = head(tokens, mask, globals_, poisoned, site_mask, strands=strands)
    torch.testing.assert_close(clean[site_mask], dirty[site_mask], rtol=0, atol=0)


def test_appending_masked_sites_changes_nothing() -> None:
    head = make_head(seed=29)
    batch = make_batch(6, 4, SMALL_SITES)
    tokens, mask, globals_, sites, site_mask, strands = batch
    short = head(tokens, mask, globals_, sites, site_mask, strands=strands)

    padding = torch.full((1, 5, 4), -1, dtype=torch.long)
    long_sites = torch.cat([sites, padding], dim=1)
    long_mask = torch.cat([site_mask, torch.zeros(1, 5, dtype=torch.bool)], dim=1)
    long = head(tokens, mask, globals_, long_sites, long_mask, strands=strands)
    torch.testing.assert_close(short, long[:, : sites.shape[1]], rtol=0, atol=0)


def test_a_site_on_word_padding_is_dropped_even_if_called_legal() -> None:
    head = make_head(seed=31)
    sites = [
        [
            (REDUCE, 1, 1, 1),
            (REDUCE, 4, 1, 1),
            (REDUCE, 99, 1, 1),
            (REDUCE, -1, 1, 1),
            (PASS, 0, 1, 1),
        ]
    ]
    batch = make_batch(8, 4, sites, lengths=[3])
    logits = head(*batch[:5], strands=batch[5])[0]
    assert torch.isfinite(logits[0]) and torch.isfinite(logits[4])
    assert (logits[1:4] == MASKED_LOGIT).all(), "positions 4, 99 and -1 are not word positions"
    torch.testing.assert_close(
        head.effective_site_mask(batch[1], batch[3], batch[4])[0],
        torch.tensor([True, False, False, False, True]),
    )


def test_a_masked_site_receives_and_transmits_exactly_zero_gradient() -> None:
    head = make_head(seed=37)
    sites = [[(REDUCE, index, 1, 1) for index in range(8)] + [(PASS, 0, 1, 1)]]
    batch = make_batch(8, 4, sites)
    tokens, mask, globals_, site_tensor, site_mask, strands = batch
    tokens = tokens.detach().requires_grad_(True)
    site_mask = site_mask.clone()
    site_mask[0, 5:8] = False  # word positions 5, 6 and 7 are now referenced only by masks

    logits = head(tokens, mask, globals_, site_tensor, site_mask, strands=strands)
    logits[site_mask].sum().backward()
    assert (tokens.grad[0, 5:8] == 0.0).all(), "a masked site must not reach its own token"
    assert (tokens.grad[0, :5] != 0.0).any()


def test_masked_sites_do_not_perturb_the_parameter_gradients() -> None:
    head = make_head(seed=41)
    batch = make_batch(6, 4, SMALL_SITES)
    tokens, mask, globals_, sites, site_mask, strands = batch

    def gradients(site_tensor, mask_tensor):
        head.zero_grad(set_to_none=True)
        logits = head(tokens, mask, globals_, site_tensor, mask_tensor, strands=strands)
        logits[mask_tensor].sum().backward()
        return {n: p.grad.clone() for n, p in head.named_parameters()}

    plain = gradients(sites, site_mask)
    poisoned = torch.cat([sites, torch.full((1, 4, 4), 7_777, dtype=torch.long)], dim=1)
    wider = torch.cat([site_mask, torch.zeros(1, 4, dtype=torch.bool)], dim=1)
    padded = gradients(poisoned, wider)
    for name in plain:
        torch.testing.assert_close(plain[name], padded[name], rtol=0, atol=0, msg=name)


def test_masked_log_policy_normalises_over_the_legal_sites_only() -> None:
    logits = torch.tensor([[1.0, 2.0, -3.0, 0.5]], dtype=torch.double)
    legal = torch.tensor([[True, True, False, True]])
    log_policy = masked_log_policy(logits, legal)
    assert log_policy[0, 2] == MASKED_LOGIT
    torch.testing.assert_close(log_policy[legal].exp().sum(), torch.ones((), dtype=torch.double))


# -- 5. the set-valued loss of §5.3 -------------------------------------------


def loss_fixture(seed: int = 0):
    torch.manual_seed(seed)
    logits = torch.randn(1, 7, dtype=torch.double, requires_grad=True)
    legal = torch.tensor([[True, True, True, True, True, True, False]])
    positive = torch.tensor([[True, False, True, False, False, False, False]])
    return logits, legal, positive


def test_loss_is_minus_log_of_the_total_mass_on_a_plus() -> None:
    logits, legal, positive = loss_fixture()
    probabilities = torch.softmax(logits.masked_fill(~legal, -torch.inf), dim=-1)
    expected = -torch.log(probabilities[positive].sum())
    torch.testing.assert_close(set_valued_loss(logits, legal, positive), expected)


def test_no_gradient_pressure_between_two_demonstrated_good_sites() -> None:
    """Move mass from one positive to the other at fixed total: the loss does not care."""
    logits, legal, positive = loss_fixture(seed=2)
    set_valued_loss(logits, legal, positive).backward()
    grad = logits.grad[0]
    first, second = 0, 2
    direction = torch.zeros(7, dtype=torch.double)
    direction[first] = torch.exp(-logits[0, first]).item()
    direction[second] = -torch.exp(-logits[0, second]).item()
    direction = direction / direction.norm()
    assert abs(float(grad @ direction)) < 1e-12


def test_the_loss_is_a_function_of_the_positive_set_s_total_mass_only() -> None:
    logits, legal, positive = loss_fixture(seed=3)
    before = set_valued_loss(logits, legal, positive)
    # Redistribute the two positives onto a common value with the same logsumexp.
    shared = torch.logsumexp(logits[0, [0, 2]], dim=0) - torch.log(torch.tensor(2.0).double())
    moved = logits.detach().clone()
    moved[0, 0] = moved[0, 2] = shared
    torch.testing.assert_close(set_valued_loss(moved, legal, positive), before.detach())


def test_positives_are_pushed_up_and_unexplored_sites_only_share_the_normaliser() -> None:
    logits, legal, positive = loss_fixture(seed=5)
    set_valued_loss(logits, legal, positive).backward()
    grad = logits.grad[0]
    policy = torch.softmax(logits.detach().masked_fill(~legal, -torch.inf), dim=-1)[0]
    unexplored = legal[0] & ~positive[0]
    # An unexplored site's gradient is exactly its own share of the normaliser --
    # the same term the positives carry.  Nothing in the loss names it.
    torch.testing.assert_close(grad[unexplored], policy[unexplored])
    assert (grad[positive[0]] < 0).all(), "every demonstrated-good site is pushed up"
    ratio = grad[unexplored] / policy[unexplored]
    torch.testing.assert_close(ratio, torch.ones_like(ratio))


def test_cross_entropy_on_one_witness_would_suppress_the_other_known_good_site() -> None:
    """The reason §5.3 is set-valued, made concrete against the loss it replaces."""
    logits, legal, positive = loss_fixture(seed=7)
    floor = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~legal, floor)
    witness = torch.tensor([0])  # the site this instance's witness happened to take

    (single,) = torch.autograd.grad(torch.nn.functional.cross_entropy(masked, witness), logits)
    assert single[0, 2] > 0, "one-hot supervision pushes the *other* good site down"

    (setwise,) = torch.autograd.grad(set_valued_loss(logits, legal, positive), logits)
    assert setwise[0, 0] < 0 and setwise[0, 2] < 0, "both are pushed up instead"


def test_training_lifts_the_set_as_a_whole() -> None:
    logits, legal, positive = loss_fixture(seed=7)

    def policy(tensor):
        return torch.softmax(tensor.detach().masked_fill(~legal, -torch.inf), dim=-1)[0]

    before = policy(logits)
    for _ in range(400):
        logits.grad = None
        set_valued_loss(logits, legal, positive).backward()
        with torch.no_grad():
            logits -= 1.0 * logits.grad
    after = policy(logits)
    assert after[positive[0]].sum() > 0.99, "the set as a whole is learned"
    assert after[legal[0] & ~positive[0]].max() < before[legal[0] & ~positive[0]].max()
    # Where inside the set the mass ends up is decided by whatever the model
    # already believed -- the two positives keep their order -- because the loss
    # itself says nothing about it.  See the two tests above for that statement
    # in its exact form.
    assert (after[0] > after[2]) == (before[0] > before[2])


def test_the_loss_treats_two_demonstrated_good_sites_interchangeably() -> None:
    logits, legal, positive = loss_fixture(seed=9)
    swapped = logits.detach().clone()
    swapped[0, [0, 2]] = swapped[0, [2, 0]]
    plain = logits.detach().clone().requires_grad_(True)
    exchanged = swapped.requires_grad_(True)

    a = set_valued_loss(plain, legal, positive)
    b = set_valued_loss(exchanged, legal, positive)
    torch.testing.assert_close(a, b)
    a.backward()
    b.backward()
    torch.testing.assert_close(plain.grad[0, 0], exchanged.grad[0, 2])
    torch.testing.assert_close(plain.grad[0, 1], exchanged.grad[0, 1])


def test_illegal_logits_are_inert_in_the_loss() -> None:
    logits, legal, positive = loss_fixture(seed=11)
    baseline = set_valued_loss(logits, legal, positive)
    baseline.backward()
    assert logits.grad[0, 6] == 0.0

    poisoned = logits.detach().clone()
    poisoned[0, 6] = 1e9
    torch.testing.assert_close(
        set_valued_loss(poisoned, legal, positive), baseline.detach(), rtol=0, atol=0
    )
    minus_infinity = logits.detach().clone()
    minus_infinity[0, 6] = MASKED_LOGIT  # what `forward` actually returns
    value = set_valued_loss(minus_infinity, legal, positive)
    assert torch.isfinite(value)
    torch.testing.assert_close(value, baseline.detach(), rtol=0, atol=0)


def test_rows_without_a_positive_are_skipped_rather_than_invented() -> None:
    logits = torch.randn(3, 5, dtype=torch.double, requires_grad=True)
    legal = torch.ones(3, 5, dtype=torch.bool)
    positive = torch.zeros(3, 5, dtype=torch.bool)
    positive[0, 1] = positive[2, 4] = True

    per_row = set_valued_loss(logits, legal, positive, reduction="none")
    assert per_row[1] == 0.0
    torch.testing.assert_close(
        set_valued_loss(logits, legal, positive), (per_row[0] + per_row[2]) / 2
    )
    set_valued_loss(logits, legal, positive).backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[1] == 0.0).all(), "an unlabelled state teaches nothing at all"


def test_a_fully_illegal_row_is_finite_and_silent() -> None:
    logits = torch.zeros(1, 4, dtype=torch.double, requires_grad=True)
    nothing = torch.zeros(1, 4, dtype=torch.bool)
    value = set_valued_loss(logits, nothing, nothing)
    value.backward()
    assert value == 0.0
    assert (logits.grad == 0.0).all()


def test_adding_a_demonstrated_good_site_can_only_lower_the_loss() -> None:
    logits, legal, positive = loss_fixture(seed=13)
    smaller = positive.clone()
    smaller[0, 2] = False
    assert set_valued_loss(logits, legal, positive) < set_valued_loss(logits, legal, smaller)


def test_the_api_refuses_the_two_ways_of_saying_negative() -> None:
    logits, legal, positive = loss_fixture()
    with pytest.raises(TypeError, match="no weight and no negative"):
        set_valued_loss(logits, legal, positive.double())
    outside = positive.clone()
    outside[0, 6] = True  # a "positive" the softmax never sees
    with pytest.raises(ValueError, match="subset of the"):
        set_valued_loss(logits, legal, outside)
    with pytest.raises(ValueError, match="unknown reduction"):
        set_valued_loss(logits, legal, positive, reduction="average")


def test_loss_runs_end_to_end_from_the_head() -> None:
    head = make_head(seed=43)
    batch = make_batch(6, 4, SMALL_SITES)
    tokens, mask, globals_, sites, site_mask, strands = batch
    site_mask = site_mask.clone()
    site_mask[0, 5] = False
    logits = head(tokens, mask, globals_, sites, site_mask, strands=strands)
    effective = head.effective_site_mask(mask, sites, site_mask)
    positive = torch.zeros_like(effective)
    positive[0, 2] = positive[0, 3] = True
    loss = SiteHead.set_valued_loss(logits, effective, positive)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in head.parameters())


# -- 6. the whole legal set, and the shape contract ---------------------------


def test_the_head_scores_a_whole_action_space_built_from_action_ids() -> None:
    spec = ActionSpec(max_len=16, max_strands=4)
    sites = sites_from_actions(list(range(spec.num_actions)), spec)
    head = make_head(seed=47)
    tokens, mask, globals_, _, _, strands = make_batch(16, 4, [[(REDUCE, 0, 1, 1)]])
    site_mask = torch.ones(1, sites.shape[0], dtype=torch.bool)
    logits = head(tokens, mask, globals_, sites[None], site_mask, strands=strands)
    assert logits.shape == (1, spec.num_actions)
    assert torch.isfinite(logits).all()
    assert len(set(logits[0].tolist())) > spec.num_actions // 2, "sites are told apart"


def test_shape_errors_are_raised_rather_than_broadcast() -> None:
    head = make_head()
    tokens, mask, globals_, sites, site_mask, strands = make_batch(6, 4, SMALL_SITES)
    with pytest.raises(ValueError, match=r"sites must be \[B, S, 4\]"):
        head(tokens, mask, globals_, sites[..., :3], site_mask)
    with pytest.raises(TypeError, match="site_mask must be a BoolTensor"):
        head(tokens, mask, globals_, sites, site_mask.long())
    with pytest.raises(ValueError, match="token width"):
        head(tokens[..., :4], mask, globals_, sites, site_mask)
    with pytest.raises(KeyError, match="isotopy"):
        head(tokens, mask, {"knot": globals_["knot"]}, sites, site_mask)
    with pytest.raises(ValueError, match="budgets must be"):
        head(tokens, mask, globals_, sites, site_mask, budgets=torch.zeros(1, 3).double())
    with pytest.raises(ValueError, match=r"strands must be \[B\]"):
        head(tokens, mask, globals_, sites, site_mask, strands=strands[None])


def test_globals_may_arrive_as_one_concatenated_tensor() -> None:
    head = make_head(seed=53)
    tokens, mask, globals_, sites, site_mask, strands = make_batch(6, 4, SMALL_SITES)
    stacked = torch.cat([globals_[n] for n in ("knot", "isotopy", "layout")], dim=-1)
    torch.testing.assert_close(
        head(tokens, mask, stacked, sites, site_mask, strands=strands),
        head(tokens, mask, globals_, sites, site_mask, strands=strands),
        rtol=0,
        atol=0,
    )


def test_batched_states_are_independent() -> None:
    head = make_head(seed=59)
    two = make_batch(8, 5, [SMALL_SITES[0], SMALL_SITES[0]], lengths=[5, 8])
    tokens, mask, globals_, sites, site_mask, strands = two
    together = head(tokens, mask, globals_, sites, site_mask, strands=strands)
    for row in (0, 1):
        alone = head(
            tokens[row : row + 1],
            mask[row : row + 1],
            {k: v[row : row + 1] for k, v in globals_.items()},
            sites[row : row + 1],
            site_mask[row : row + 1],
            strands=strands[row : row + 1],
        )
        torch.testing.assert_close(alone[0], together[row], rtol=1e-12, atol=1e-12)
