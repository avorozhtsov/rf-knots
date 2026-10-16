"""TRAIN-only effective-sample and compute profile for the nine-arm pilot."""

import json
from collections import Counter

from mcts_tournament_signals_v1 import _groups
from mcts_training_supervision import verified_success_traces
from tournament_s10_dataset_v1 import load_groups

from rf_knots import mcts_loop as M


def main():
    roots, groups, _ = load_groups()
    report = {}
    for arm in "BCDEFGHI":
        counter = Counter()
        lengths = []
        for (base, weight), witnesses in sorted(groups.items()):
            root = roots[base * 2]
            instance = M.Instance(
                **{**root["instance"], "word": tuple(root["instance"]["word"]), "cc_weight": weight}
            )
            traces = verified_success_traces(instance, witnesses)
            signals = _groups(
                traces,
                arm,
                seed=20261003 + base * 2 + (weight == 1000),
                k=4,
                near_margin=2,
                gap_scale=weight,
                gap_cap=4,
            )
            if signals:
                counter["contributing_groups"] += 1
            for signal in signals:
                if signal["objective"] == "lexicographic_cc_extra":
                    counter["G_cc_extra_groups"] += 1
                counter["signals"] += 1
                counter["terms"] += len(signal["coefficients"])
                lengths.append(len(signal["coefficients"]))
        counter["max_terms"] = max(lengths, default=0)
        counter["median_terms"] = sorted(lengths)[len(lengths) // 2] if lengths else 0
        report[arm] = dict(counter)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
