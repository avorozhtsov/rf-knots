"""Protocol135 resource units; no changes to the frozen protocol97 network.

Only policy-context features change. Knot/layout/isotopy embeddings, position
encoding, action semantics and parameter shapes stay unchanged. Checkpoints
must carry the feature contract: matching tensor shapes alone are insufficient.
"""

import copy

import torch

from rf_knots.site_head import SiteHead

RESOURCE_NAMES = ("remaining_moves", "remaining_cc", "length_room", "strand_room", "cc_weight")
RESOURCE_UNITS = (256.0, 2.0, 256.0, 32.0, 1000.0)


class ResourceUnitsHead(SiteHead):
    def _context_features(self, globals_, budgets, batch, device, dtype):
        if self.budget_dim != 5:
            raise ValueError("protocol135 requires the five native resource channels")
        if budgets is not None and (not torch.isfinite(budgets).all() or (budgets < 0).any()):
            raise ValueError("resources must be finite and nonnegative")
        features = super()._context_features(globals_, budgets, batch, device, dtype)
        start = len(self.global_names) * self.embedding_dim
        units = features.new_tensor(RESOURCE_UNITS)
        # Positive fixed units preserve zero/exhaustion and never truncate large
        # instances. log1p(x)/log1p(unit), not log1p(x/unit): both maps are pinned.
        return torch.cat(
            (
                features[:, :start],
                features[:, start : start + 5] / units,
                features[:, start + 5 : start + 10] / torch.log1p(units),
                features[:, start + 10 :],
            ),
            dim=-1,
        )


def with_resource_units(net, mode):
    """Return an independent model with exactly the same learned tensor values."""
    if mode not in ("raw", "units"):
        raise ValueError("unknown resource feature mode")
    result = copy.deepcopy(net)
    if mode == "units":
        old = result.head
        if type(old) is not SiteHead or old.budget_dim != 5 or not old.structural_generators:
            raise ValueError("expected the unmodified protocol97 raster head")
        head = ResourceUnitsHead(
            width=old.width,
            embedding_dim=old.embedding_dim,
            budget_dim=5,
            structural_generators=True,
        ).to(device=next(old.parameters()).device, dtype=next(old.parameters()).dtype)
        head.load_state_dict(old.state_dict(), strict=True)
        head.train(old.training)
        result.head = head
    if any(not torch.equal(v, result.state_dict()[k]) for k, v in net.state_dict().items()):
        raise ValueError("feature intervention changed initial learned tensors")
    return result
