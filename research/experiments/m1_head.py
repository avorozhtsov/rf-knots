"""Adapter: `rf_knots.site_head.SiteHead` behind the gate's `rank(state, sites)`.

The gate must run with hand-written baselines alone and pick the head up
unchanged, so the head is reached through one function here and nothing else in
the harness imports `torch`.

The contract, as the two modules now implement it:

```python
SiteHead(width, embedding_dim, num_kinds=9)
    forward(tokens, mask, globals_, sites, site_mask, *, budgets=None,
            strands=None) -> logits [B, S]

encoder.encode_sites(word, lengths, strands) -> {
    "tokens": [B, L, W], "mask": [B, L],
    "knot" / "isotopy" / "layout": [B, D],
}
```

Two things this module refuses to leave implicit.

**The site row has four columns.**  `(kind, position, generator, sign)`, which is
exactly `ActionSpec.decode`'s return and exactly what `ActionSpec.encode`
consumes.  `INSERT` is indexed as `3L + ((g - 1) * 2 + s) * L + p`, so
`INSERT(p, g, +1)` and `INSERT(p, g, -1)` are **distinct actions**; a
three-column row merges them and halves the `INSERT` block.  Every row is checked
against `encode` before it reaches the head, in the harness rather than only in a
test.

**The positional geometry is recorded, never defaulted.**  A default-constructed
`ScalableBraidEmbeddingEncoder` is `positional_geometry="legacy"`: absolute
Fourier features with a zero-padded seam, under which a cyclic shift of the word
-- conjugation, a no-op on the closed braid -- moves the `knot` embedding by up
to 0.377 in a unit-normalised space, against 2.2e-16 under `"necklace"`.  A gate
that scores tokens from the wrong geometry is not measuring what its report says.
So `GEOMETRY` is the harness default, the choice is a parameter, and it is
written into the report's run metadata beside the seed and the split.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass

#: `(kind, position, generator, sign)`, in the order `ActionSpec.encode` takes
#: them, so a site row round-trips to its own flat action index.  `sign` is `+1`
#: or `-1`, not a bit.
SITE_FIELDS: tuple[str, ...] = ("kind", "position", "generator", "sign")

#: Note 22 §6.4.  Never rely on the constructor's default, which is `"legacy"`.
GEOMETRY: str = "necklace"

#: A cyclic strand encoding without the seam generator is a false invariance
#: (§6.4), and the encoder raises unless the geometry is `"necklace"`.  The gate's
#: configurations are Artin, so this stays off.
CYCLIC_BAND_GENERATORS: bool = False


@dataclass(frozen=True)
class HeadStatus:
    """What of the learned path exists, and under which geometry it was run."""

    torch: bool
    site_head: bool
    encode_sites: bool
    geometry: str
    cyclic_band_generators: bool
    detail: str
    #: Read off the constructed encoder rather than assumed.  The rotary is not a
    #: constructor argument -- it follows from the geometry -- and its default has
    #: already changed once during this milestone, so the gate records the
    #: harmonics it actually ran with instead of naming a default that may move.
    rotary: dict | None = None

    @property
    def ready(self) -> bool:
        return self.torch and self.site_head and self.encode_sites

    def to_dict(self) -> dict:
        return {
            "torch_importable": self.torch,
            "rf_knots.site_head.SiteHead_importable": self.site_head,
            "encoder.encode_sites_present": self.encode_sites,
            "positional_geometry": self.geometry,
            "cyclic_band_generators": self.cyclic_band_generators,
            "rotary": self.rotary,
            "ready": self.ready,
            "detail": self.detail,
        }


def status(
    geometry: str = GEOMETRY, cyclic_band_generators: bool = CYCLIC_BAND_GENERATORS
) -> HeadStatus:
    """Probe the learned path without importing it into the hot loop."""
    missing: list[str] = []
    try:
        import torch  # noqa: F401

        has_torch = True
    except Exception as error:  # pragma: no cover - torch is installed here
        has_torch = False
        missing.append(f"torch: {error}")
    has_head = False
    try:
        from rf_knots.site_head import SiteHead  # noqa: F401

        has_head = True
    except Exception as error:
        missing.append(f"rf_knots.site_head.SiteHead: {error}")
    has_encode = False
    try:
        from rf_knots.embedding_encoder import ScalableBraidEmbeddingEncoder

        has_encode = hasattr(ScalableBraidEmbeddingEncoder, "encode_sites")
        if not has_encode:
            missing.append(
                "ScalableBraidEmbeddingEncoder has no encode_sites; the gate needs "
                "the per-cell tokens the pooling step used to discard"
            )
        elif "positional_geometry" not in inspect.signature(
            ScalableBraidEmbeddingEncoder.__init__
        ).parameters:
            missing.append(
                "the encoder takes no positional_geometry, so the geometry this "
                "run reports cannot be verified"
            )
    except Exception as error:  # pragma: no cover - the encoder is present here
        missing.append(f"rf_knots.embedding_encoder: {error}")
    return HeadStatus(
        torch=has_torch,
        site_head=has_head,
        encode_sites=has_encode,
        geometry=geometry,
        cyclic_band_generators=cyclic_band_generators,
        detail="; ".join(missing) if missing else "learned path available",
    )


def describe_rotary(encoder) -> dict:
    """What the rotary is actually doing on this encoder, by inspection.

    A per-token linear head cannot read what a rotary encodes -- the identity a
    rotary carries lives in inner products *between* tokens, and neither a
    convolution nor a single-token linear functional consumes those -- so whether
    the harmonics are live changes what the site head can possibly see.  It is
    not a constructor argument, so it is measured here and written into the
    report rather than named from memory.
    """
    out: dict = {}
    for name in ("word_harmonics", "strand_harmonics"):
        buffer = getattr(encoder, name, None)
        if buffer is None:
            out[name] = None
            continue
        values = [int(x) for x in buffer.reshape(-1).tolist()]
        out[name] = {
            "count": len(values),
            "nonzero": sum(1 for value in values if value),
            "first": values[:8],
        }
    out["active"] = any(
        isinstance(cell, dict) and cell["nonzero"] for cell in out.values()
    )
    return out


def site_features(spec, sites: Sequence) -> list[list[int]]:
    """`[kind, position, generator, sign]` per site, checked against `encode`.

    The check is here rather than only in a test: a site row that cannot
    reproduce its own flat action index means the head is being told about a
    different action space from the one the target was built in, and `aplus.py`
    -- which stores flat indices -- would disagree with the harness about what a
    site *is*.
    """
    rows = []
    for site in sites:
        row = [site.kind, site.position, site.generator, site.sign]
        flat = spec.encode(row[0], position=row[1], generator=row[2], sign=row[3])
        if flat != site.action:
            raise AssertionError(
                f"site row {row} encodes to {flat}, not to the flat action "
                f"{site.action} ({spec.describe(site.action)}) the target uses"
            )
        rows.append(row)
    return rows


def build_ranker(
    checkpoint: str | None = None,
    device: str = "cpu",
    geometry: str = GEOMETRY,
    cyclic_band_generators: bool = CYCLIC_BAND_GENERATORS,
):
    """A `rank(state, sites) -> scores` backed by `SiteHead`.

    `geometry` is passed to the encoder explicitly -- the constructor's own
    default is `"legacy"` and would silently hand the gate the wrong geometry.
    A checkpoint may name its own under `"positional_geometry"`, and a checkpoint
    that disagrees with the requested geometry raises rather than being coerced:
    scoring weights under a geometry they were not trained in measures nothing.
    """
    probe = status(geometry, cyclic_band_generators)
    if not probe.ready:
        raise RuntimeError(
            "the learned path is not available yet: " + probe.detail + ". Run the "
            "gate without --head to get the baseline table the head has to beat."
        )
    import torch

    from rf_knots.embedding_encoder import ScalableBraidEmbeddingEncoder
    from rf_knots.site_head import SiteHead

    state = torch.load(checkpoint, map_location=device) if checkpoint else {}
    stored = state.get("positional_geometry")
    if stored is not None and stored != geometry:
        raise RuntimeError(
            f"checkpoint {checkpoint} was trained under positional_geometry "
            f"{stored!r} and the gate was asked for {geometry!r}; scoring weights "
            "under a geometry they were not trained in measures nothing"
        )
    width = int(state.get("width", 96))
    embedding_dim = int(state.get("embedding_dim", 64))
    encoder = ScalableBraidEmbeddingEncoder(
        width=width,
        embedding_dim=embedding_dim,
        positional_geometry=geometry,
        cyclic_band_generators=cyclic_band_generators,
    )
    head = SiteHead(width, embedding_dim)
    if "encoder" in state:
        encoder.load_state_dict(state["encoder"])
    if "head" in state:
        head.load_state_dict(state["head"])
    encoder.eval()
    head.eval()
    build_ranker.last_rotary = describe_rotary(encoder)

    def rank(gate_state, sites) -> list[float]:
        features = site_features(gate_state.spec, sites)
        word = torch.as_tensor([list(gate_state.word)], dtype=torch.long)
        if word.numel() == 0:
            word = torch.zeros((1, 1), dtype=torch.long)
        lengths = torch.as_tensor([len(gate_state.word)], dtype=torch.long)
        strands = torch.as_tensor([gate_state.strands], dtype=torch.long)
        site_tensor = torch.as_tensor([features], dtype=torch.long)
        site_mask = torch.ones(site_tensor.shape[:2], dtype=torch.bool)
        budgets = torch.as_tensor(
            [[gate_state.remaining_moves, gate_state.remaining_crossing_changes]],
            dtype=torch.float32,
        )
        with torch.no_grad():
            batch = encoder.encode_sites(word, lengths, strands)
            globals_ = {
                name: batch[name] for name in ("knot", "isotopy", "layout")
            }
            logits = head(
                batch["tokens"],
                batch["mask"],
                globals_,
                site_tensor,
                site_mask,
                budgets=budgets,
                strands=strands,
            )
        return [float(value) for value in logits[0, : len(sites)]]

    return rank


__all__ = [
    "CYCLIC_BAND_GENERATORS",
    "GEOMETRY",
    "describe_rotary",
    "HeadStatus",
    "SITE_FIELDS",
    "build_ranker",
    "site_features",
    "status",
]
