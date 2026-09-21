"""Read-only descriptive audit of explicit-unknot TRAIN support, not model selection."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from rf_knots.embedding_normalization import fast_normalize_global_view


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(records):
    intervals = Counter()
    groups, raw, normalized = defaultdict(Counter), defaultdict(set), defaultdict(set)
    for record in records:
        if record["split"] != "train":
            raise ValueError("descriptive audit must use train rows only")
        pair = record["interval"]
        if pair["lower"] < 0 or pair["upper"] < pair["lower"]:
            raise ValueError("invalid interval")
        empty = [s for s in ("left", "right")
                 if not pair[s]["word"] and pair[s]["strands"] == 1]
        kind = "explicit_unknot" if empty else "other_pair"
        intervals[pair["metric"], kind, pair["lower"], pair["upper"]] += 1
        if not empty or pair["metric"] != "cc" or pair["lower"] != pair["upper"]:
            continue
        side = "right" if empty[0] == "left" else "left"
        view, key = pair[side], str(pair["lower"])
        fp = record[side + "_fingerprint"]
        state = tuple(view["word"]), view["strands"]
        groups[key][fp[0], abs(fp[1])] += 1
        raw[key].add(state)
        normalized[key].add(fast_normalize_global_view(*state))
    return {
        "records": len(records),
        "intervals": [{"metric": m, "kind": k, "lower": lo, "upper": hi, "count": n}
                      for (m, k, lo, hi), n in sorted(intervals.items())],
        "exact_explicit_unknot": {
            k: {"records": sum(g.values()), "fingerprint_groups": len(g),
                "raw_states": len(raw[k]), "normalized_states": len(normalized[k]),
                "groups_by_count": g.most_common()}
            for k, g in sorted(groups.items())},
        "caveat": "Descriptive support, not a proof/label audit or a causal diagnosis. "
                  "Fingerprint groups are not knot identities. Explicit empty-one-braid "
                  "anchors are only a subset of possible unknot endpoints. Normalized "
                  "states use the existing encoder preprocessing, not a canonical knot ID. "
                  "Counts are records, not actual optimizer draw counts. No evaluation rows used.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite diagnostic")
    if sha(args.corpus) != args.expected_sha256:
        raise ValueError("corpus differs from declared training input")
    sources = [Path(__file__), Path("src/rf_knots/embedding_normalization.py")]
    pins = {str(path): sha(path) for path in [args.corpus, *sources]}
    corpus = json.loads(args.corpus.read_text())
    if corpus["schema"] != "rf-knots-replay-fingerprint-corpus-v1":
        raise ValueError("unexpected corpus schema")
    result = describe(corpus["train"])
    if any(sha(Path(name)) != digest for name, digest in pins.items()):
        raise ValueError("input changed during audit")
    result.update(status="completed", purpose="train-only-descriptive-diagnostic", inputs=pins)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["exact_explicit_unknot"], indent=2))


if __name__ == "__main__":
    main()
