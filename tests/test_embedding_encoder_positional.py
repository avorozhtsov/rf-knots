"""Properties of the two positional geometries of the braid encoder.

These assert behaviour, not shapes: that `forward` still computes exactly what
the pre-M1 encoder computed, and that the necklace geometry of note 22 §6.4
really is `Z_L`-equivariant on the word axis and really is *not* cyclic on the
strand axis unless `cyclic_band_generators` says the diagram is a torus.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from torch.nn import functional as F  # noqa: E402

from rf_knots.embedding_encoder import ScalableBraidEmbeddingEncoder  # noqa: E402

WORD = [1, -2, 3, -1, 2, 2, -3, 1]
STRANDS = 5


def encoder(seed: int = 11, **kwargs) -> ScalableBraidEmbeddingEncoder:
    torch.manual_seed(seed)
    return ScalableBraidEmbeddingEncoder(
        width=16, depth=2, embedding_dim=8, **kwargs
    ).eval()


def necklace(seed: int = 11, **kwargs) -> ScalableBraidEmbeddingEncoder:
    return encoder(seed, positional_geometry="necklace", **kwargs)


def batch(word: list[int], padded: int | None = None, strands: int = STRANDS):
    length = len(word)
    padded = length if padded is None else padded
    tensor = torch.zeros((1, padded), dtype=torch.long)
    tensor[0, :length] = torch.tensor(word, dtype=torch.long)
    return tensor, torch.tensor([length]), torch.tensor([strands])


def rotate(word: list[int], shift: int) -> list[int]:
    shift %= len(word)
    return word[-shift:] + word[:-shift] if shift else list(word)


def cosine_gram(tokens: torch.Tensor) -> torch.Tensor:
    unit = F.normalize(tokens, dim=-1)
    return unit @ unit.transpose(-1, -2)


# --------------------------------------------------------------------- legacy


def legacy_reference_forward(
    model: ScalableBraidEmbeddingEncoder,
    word: torch.Tensor,
    lengths: torch.Tensor,
    strands: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """A verbatim copy of the encoder's `forward` as it stood before M1.

    Kept here so that a future edit to the shared trunk cannot silently move the
    legacy embeddings that shipped checkpoints and `embedding_audit_gate` read.
    """
    mask = torch.arange(word.shape[1], device=word.device)[None, :] < lengths[:, None]
    magnitude = word.abs()
    ids = torch.where(word > 0, 2 * magnitude - 1, 2 * magnitude)
    state = model.tokens(ids)
    position = torch.arange(word.shape[1], dtype=torch.float32)[None, :]
    phase = position / lengths.clamp_min(1).to(dtype=torch.float32)[:, None]
    angle = 2 * math.pi * phase
    features = torch.stack(
        [
            torch.sin(angle),
            torch.cos(angle),
            torch.sin(2 * angle),
            torch.cos(2 * angle),
            torch.sin(4 * angle),
            torch.cos(4 * angle),
            phase,
            torch.ones_like(phase),
        ],
        dim=-1,
    )
    state = state + model.position_projection(features)
    state = state + model.strands(strands)[:, None, :]
    state = state * mask.unsqueeze(-1)
    for block in model.blocks:
        update = block.norm(state).transpose(1, 2)
        update = block.depthwise(update)
        update = F.glu(block.pointwise(update), dim=1)
        update = block.output(update).transpose(1, 2)
        state = (state + update) * mask.unsqueeze(-1)
    divisor = lengths.clamp_min(1).unsqueeze(-1)
    mean = state.sum(1) / divisor
    maximum = state.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(1)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    pooled = model.pool_projection(torch.cat((mean, maximum, state[:, 0]), dim=-1))
    return {
        name: F.normalize(head(pooled), dim=-1) for name, head in model.heads.items()
    }


def test_forward_is_bit_identical_to_the_pre_m1_encoder() -> None:
    word = torch.tensor(
        [[1, -2, 3, -1, 2, 0, 0], [2, 2, -3, 1, 1, -2, 3], [-1, 1, 0, 0, 0, 0, 0]],
        dtype=torch.long,
    )
    lengths = torch.tensor([5, 7, 2])
    strands = torch.tensor([4, 5, 3])
    for seed in (0, 7, 1234):
        for width, depth, dim in ((16, 1, 8), (32, 2, 12)):
            torch.manual_seed(seed)
            model = ScalableBraidEmbeddingEncoder(
                width=width, depth=depth, embedding_dim=dim
            ).eval()
            with torch.no_grad():
                produced = model(word, lengths, strands)
                expected = legacy_reference_forward(model, word, lengths, strands)
            assert set(produced) == {"knot", "isotopy", "layout"}
            for name in produced:
                assert torch.equal(produced[name], expected[name]), (seed, width, name)


def test_legacy_parameter_set_and_initialisation_order_are_frozen() -> None:
    """A new parameter drawn before the old ones would move every checkpoint."""
    torch.manual_seed(3)
    model = ScalableBraidEmbeddingEncoder(width=16, depth=1, embedding_dim=8)
    assert list(model.state_dict()) == [
        "tokens.weight",
        "strands.weight",
        "position_projection.weight",
        "blocks.0.norm.weight",
        "blocks.0.norm.bias",
        "blocks.0.depthwise.weight",
        "blocks.0.depthwise.bias",
        "blocks.0.pointwise.weight",
        "blocks.0.pointwise.bias",
        "blocks.0.output.weight",
        "blocks.0.output.bias",
        "pool_projection.0.weight",
        "pool_projection.0.bias",
        "pool_projection.2.weight",
        "pool_projection.2.bias",
        "heads.knot.weight",
        "heads.knot.bias",
        "heads.isotopy.weight",
        "heads.isotopy.bias",
        "heads.layout.weight",
        "heads.layout.bias",
    ]
    torch.manual_seed(3)
    twin = ScalableBraidEmbeddingEncoder(width=16, depth=1, embedding_dim=8)
    assert torch.equal(model.tokens.weight, twin.tokens.weight)
    # A shipped checkpoint must still load without `strict=False`.
    model.load_state_dict(twin.state_dict())


def test_necklace_adds_parameters_without_disturbing_the_legacy_trunk() -> None:
    plain = encoder()
    curved = necklace()
    for name, parameter in plain.state_dict().items():
        assert torch.equal(parameter, curved.state_dict()[name]), name
    extra = set(curved.state_dict()) - set(plain.state_dict())
    assert extra == {
        "strand_relative.weight",
        "strand_low.weight",
        "strand_high.weight",
    }
    missing, unexpected = curved.load_state_dict(plain.state_dict(), strict=False)
    assert set(missing) == extra and not unexpected


# -------------------------------------------------------------- the contract


@pytest.mark.parametrize("geometry", ["legacy", "necklace"])
def test_encode_sites_returns_tokens_alongside_the_forward_globals(geometry) -> None:
    model = encoder(positional_geometry=geometry)
    word, lengths, strands = batch(WORD, padded=len(WORD) + 3)
    with torch.no_grad():
        sites = model.encode_sites(word, lengths, strands)
        globals_ = model(word, lengths, strands)
    assert set(sites) == {"tokens", "mask", "knot", "isotopy", "layout"}
    assert sites["tokens"].shape == (1, len(WORD) + 3, 16)
    assert sites["tokens"].dtype == torch.float32
    assert sites["mask"].dtype == torch.bool
    assert sites["mask"].tolist() == [[True] * len(WORD) + [False] * 3]
    assert torch.equal(
        sites["tokens"][0, len(WORD) :], torch.zeros(3, 16)
    ), "tokens must be zeroed on padding"
    for name in ("knot", "isotopy", "layout"):
        assert sites[name].shape == (1, 8)
        assert torch.equal(sites[name], globals_[name])


def test_cyclic_band_generators_needs_the_necklace_geometry() -> None:
    with pytest.raises(ValueError, match="necklace"):
        ScalableBraidEmbeddingEncoder(width=16, cyclic_band_generators=True)


# ------------------------------------------------------------- the word axis


@pytest.mark.parametrize("shift", [1, 3, 7])
def test_necklace_tokens_are_exactly_cyclic_equivariant(shift) -> None:
    """RoPE's defining property, on the axis where the group is really `Z_L`.

    Conjugation moves the storage seam and nothing else, so the token geometry
    must be carried along by the shift: token inner products are invariant and
    token norms are permuted, exactly.
    """
    model = necklace().double()
    with torch.no_grad():
        plain = model.encode_sites(*batch(WORD))["tokens"][0]
        turned = model.encode_sites(*batch(rotate(WORD, shift)))["tokens"][0]
    order = (torch.arange(len(WORD)) - shift) % len(WORD)
    torch.testing.assert_close(
        turned @ turned.T, (plain @ plain.T)[order][:, order], atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        turned.norm(dim=-1), plain.norm(dim=-1)[order], atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("shift", [1, 3, 7])
def test_necklace_trunk_is_bitwise_cyclic_equivariant(shift) -> None:
    """Strip the rotary and the tokens themselves permute, to the last bit."""
    model = necklace()
    model.word_harmonics.zero_()  # every SO(2) block becomes the identity
    with torch.no_grad():
        plain = model.encode_sites(*batch(WORD))["tokens"][0]
        turned = model.encode_sites(*batch(rotate(WORD, shift)))["tokens"][0]
    order = (torch.arange(len(WORD)) - shift) % len(WORD)
    assert torch.equal(turned, plain[order])


@pytest.mark.parametrize("shift", [1, 3, 7])
def test_necklace_globals_are_invariant_under_conjugation(shift) -> None:
    """§6.1 wants `knot` invariant under every zero-CC move; rotation is one."""
    model = necklace()
    with torch.no_grad():
        plain = model(*batch(WORD))
        turned = model(*batch(rotate(WORD, shift)))
    for name in ("knot", "isotopy", "layout"):
        torch.testing.assert_close(turned[name], plain[name], atol=1e-6, rtol=0)


def test_the_legacy_geometry_is_not_cyclic_equivariant() -> None:
    """The contrast that motivates §6.4 -- a seam that should not exist."""
    model = encoder()
    with torch.no_grad():
        plain = model(*batch(WORD))["knot"]
        turned = model(*batch(rotate(WORD, 1)))["knot"]
    assert (turned - plain).abs().max() > 1e-3


def test_word_frequencies_are_normalised_by_the_braid_length() -> None:
    """One set of weights must score a 20-letter and a 254-letter word.

    Doubling the length and doubling the position must land on the same angle,
    which is exactly what `theta = 2 * pi / L` buys.
    """
    model = necklace()

    def angles(length: int) -> torch.Tensor:
        return model._rotary_angles(
            torch.zeros((1, length), dtype=torch.long),
            torch.tensor([length]),
            torch.tensor([5]),
        )[0]

    short, doubled = angles(20), angles(40)
    for position in range(20):
        torch.testing.assert_close(
            doubled[2 * position], short[position], atol=1e-6, rtol=0
        )
    assert torch.equal(short[0], torch.zeros_like(short[0]))
    # ... and the schedule really is length-dependent, not an absolute ladder.
    assert (angles(254)[1] - short[1]).abs().max() > 1e-3


@pytest.mark.parametrize("padded", [len(WORD), len(WORD) + 1, len(WORD) + 9, 64])
def test_necklace_tokens_are_invariant_to_batch_padding(padded) -> None:
    """The word wraps at its own length, never at an unrelated padding width."""
    model = necklace()
    with torch.no_grad():
        tight = model.encode_sites(*batch(WORD))["tokens"][0]
        loose = model.encode_sites(*batch(WORD, padded=padded))["tokens"][0]
    torch.testing.assert_close(loose[: len(WORD)], tight, atol=1e-6, rtol=0)
    assert torch.equal(loose[len(WORD) :], torch.zeros(padded - len(WORD), 16))


def test_necklace_tokens_do_not_depend_on_the_rest_of_the_batch() -> None:
    model = necklace()
    padded = torch.tensor(
        [WORD + [0, 0, 0, 0], [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]], dtype=torch.long
    )
    with torch.no_grad():
        alone = model.encode_sites(*batch(WORD))["tokens"][0]
        together = model.encode_sites(
            padded, torch.tensor([len(WORD), 12]), torch.tensor([STRANDS, 3])
        )["tokens"][0]
    torch.testing.assert_close(together[: len(WORD)], alone, atol=1e-6, rtol=0)


# ----------------------------------------------------------- the strand axis


def positional_only(model: ScalableBraidEmbeddingEncoder):
    """Silence the per-generator lookup so only the strand encoding speaks."""
    with torch.no_grad():
        model.tokens.weight.zero_()
    return model


def relabel(word: list[int], modulus: int) -> list[int]:
    """Rotate strand indices: `g -> (g mod modulus) + 1`, keeping the sign."""
    return [(1 if letter > 0 else -1) * ((abs(letter) % modulus) + 1) for letter in word]


def test_strand_axis_is_not_cyclic_without_the_band_generator_flag() -> None:
    """Strand 1 and strand `n` are not adjacent; the encoding must say so."""
    model = positional_only(necklace())
    with torch.no_grad():
        plain = model.encode_sites(*batch(WORD))["tokens"][0]
        turned = model.encode_sites(*batch(relabel(WORD, STRANDS - 1)))["tokens"][0]
    assert (cosine_gram(turned) - cosine_gram(plain)).abs().max() > 1e-2


def test_strand_axis_is_cyclic_with_the_band_generator_flag() -> None:
    """With the seam generator the diagram is a torus and `U(1)` is real."""
    model = positional_only(necklace(cyclic_band_generators=True))
    with torch.no_grad():
        plain = model.encode_sites(*batch(WORD))["tokens"][0]
        turned = model.encode_sites(*batch(relabel(WORD, STRANDS)))["tokens"][0]
    torch.testing.assert_close(
        cosine_gram(turned), cosine_gram(plain), atol=1e-5, rtol=0
    )


def test_strand_buckets_are_exact_at_the_braid_relation_threshold() -> None:
    """`|g - h| = 1` braids, `|g - h| >= 2` commutes; the bucket must separate.

    Beyond the clip radius the bucket saturates, so two words that differ only
    in *how far apart* their far-commuting generators are must encode alike.
    """
    model = necklace(strand_bucket_radius=2)
    with torch.no_grad():
        model.strand_low.weight.zero_()
        model.strand_high.weight.zero_()

    def features(word: list[int]) -> torch.Tensor:
        tensor = torch.tensor([word], dtype=torch.long)
        lengths = torch.tensor([len(word)])
        halo = model._necklace_halo(len(word), lengths, model.blocks[0].radius)
        with torch.no_grad():
            return model._strand_features(tensor, torch.tensor([21]), halo)[0]

    braided = features([1, 2, 1, 2, 1, 2])  # |g - h| = 1, the braid relation
    commuting = features([1, 4, 1, 4, 1, 4])  # |g - h| = 3, free commutation
    farther = features([1, 6, 1, 6, 1, 6])  # |g - h| = 5, the same legal move
    assert (commuting - braided).abs().max() > 1e-3, "the threshold must be visible"
    assert torch.equal(farther, commuting), "past the clip the bucket must saturate"


def test_strand_distance_is_absolute_not_normalised_by_the_strand_count() -> None:
    """Two strands apart means the same at `n = 5` and at `n = 21`."""
    model = necklace()
    with torch.no_grad():
        model.strand_low.weight.zero_()
        model.strand_high.weight.zero_()
        model.strands.weight.zero_()
    word = torch.tensor([[1, 3, 1, 3, 1, 3]], dtype=torch.long)
    lengths = torch.tensor([6])
    halo = model._necklace_halo(6, lengths, model.blocks[0].radius)
    with torch.no_grad():
        narrow = model._strand_features(word, torch.tensor([5]), halo)
        wide = model._strand_features(word, torch.tensor([21]), halo)
    assert torch.equal(narrow, wide)


def test_boundary_features_break_translation_at_the_cylinder_edge() -> None:
    """Sliding a pattern along the strand axis is not a symmetry of `B_n`."""
    model = necklace()
    lengths = torch.tensor([6])
    halo = model._necklace_halo(6, lengths, model.blocks[0].radius)
    low = torch.tensor([[1, 2, 1, 2, 1, 2]], dtype=torch.long)
    high = torch.tensor([[9, 10, 9, 10, 9, 10]], dtype=torch.long)
    with torch.no_grad():
        near = model._strand_features(low, torch.tensor([21]), halo)
        away = model._strand_features(high, torch.tensor([21]), halo)
    assert (near - away).abs().max() > 1e-3
