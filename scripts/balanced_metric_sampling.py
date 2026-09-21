"""Explicit TRAIN-only metric bounds and group balancing for verified pair rows."""

from collections import defaultdict
from dataclasses import replace


def prepare_rows(rows, records, *, bounds="signature"):
    if bounds not in ("signature", "determinant") or len(rows) != len(records):
        raise ValueError("invalid bounds mode or row coverage")
    result = []
    for row, record in zip(rows, records, strict=True):
        pair = record["interval"]
        if record["split"] != "train" or row.metric != "cc":
            raise ValueError("only verified TRAIN CC rows supported")
        if (row.spec.distance_lower != pair["lower"] or row.spec.distance_upper != pair["upper"]
                or row.left_word != tuple(pair["left"]["word"])
                or row.right_word != tuple(pair["right"]["word"])
                or row.left_strands != pair["left"]["strands"]
                or row.right_strands != pair["right"]["strands"]):
            raise ValueError("materialized and verified records differ")
        lo = pair["lower"]
        if bounds == "determinant":
            # Different determinants certify different knot types, hence >=1 CC.
            lo = max(lo, int(record["left_fingerprint"][0] != record["right_fingerprint"][0]))
        if lo > pair["upper"]:
            raise ValueError("lower bound exceeds replayed upper bound")
        result.append(replace(row, spec=replace(row.spec, distance_lower=lo)))
    return result


class GroupBalancedMetricSampler:
    """Uniform (task, bound class), then query fingerprint group, then record.

    Input records must already pass the full replay/partition verifier. Groups
    are conservative invariant buckets, not knot identities. Three random()
    draws per record keep the shared rollout RNG aligned across different-size
    corpora in paired training. No per-class retries or variable randrange draws.
    """

    def __init__(self, rows, records):
        if not rows or len(rows) != len(records):
            raise ValueError("empty or mismatched metric pool")
        bins = defaultdict(lambda: defaultdict(list))
        for row, record in zip(rows, records, strict=True):
            if record["split"] != "train":
                raise ValueError("sampler cannot read heldout rows")
            lo, hi = row.spec.distance_lower, row.spec.distance_upper
            task = "short-u" if ((not row.left_word and row.left_strands == 1)
                                or (not row.right_word and row.right_strands == 1)) else "general"
            label = f"exact-{lo:g}" if lo == hi else "interval"
            side = "right" if not row.left_word and row.left_strands == 1 else "left"
            fp = record[side + "_fingerprint"]
            bins[task, label][fp[0], abs(fp[1])].append(row)
        self.bins = [tuple(tuple(g[key]) for key in sorted(g)) for _, g in sorted(bins.items())]
        self.support = {f"{task}/{label}": {"groups": len(g), "records": sum(map(len, g.values()))}
                        for (task, label), g in sorted(bins.items())}

    def sample(self, rng, count):
        if count < 0:
            raise ValueError("negative sample size")
        result = []
        for _ in range(count):
            classes = self.bins[int(rng.random() * len(self.bins))]
            group = classes[int(rng.random() * len(classes))]
            result.append(group[int(rng.random() * len(group))])
        return result
