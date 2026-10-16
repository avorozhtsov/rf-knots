"""Sampler and frozen variant checks for protocol270."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from root_value_head_ablation_v1 import VARIANTS, draw_batch, grouped_lookup


def test_three_single_factor_variants():
    assert VARIANTS == {
        "narrow": {"width": 32, "weight_decay": 0.01, "sampler": "attempt"},
        "decay": {"width": 128, "weight_decay": 0.10, "sampler": "attempt"},
        "grouped": {"width": 128, "weight_decay": 0.01, "sampler": "root_pair"},
    }


def test_grouped_sampler_selects_distinct_root_pairs():
    examples = [{"root_index": 2 * root + offset, "lambda": 10 if offset == 0 else 1000,
                 "seed": 20270000 + seed}
                for root in range(96) for offset in (0, 1) for seed in range(10)]
    lookup = grouped_lookup(examples)
    rng = torch.Generator().manual_seed(20261004)
    drawn = draw_batch(rng, len(examples), lookup, "root_pair")
    assert len(drawn) == 64
    assert len({examples[i]["root_index"] // 2 for i in drawn.tolist()}) == 64
    assert drawn.tolist() == draw_batch(torch.Generator().manual_seed(20261004),
                                        len(examples), lookup, "root_pair").tolist()


def test_attempt_sampler_uses_fit_only():
    rng = torch.Generator().manual_seed(20261004)
    drawn = draw_batch(rng, 1920, torch.empty((0, 0)), "attempt")
    assert drawn.shape == (64,)
    assert int(drawn.min()) >= 0 and int(drawn.max()) < 1920
