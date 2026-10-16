"""Complete historical149 TRAIN-pool size inventory, without neural forwards."""

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from audit_raster_train_draws import pool_statistics
from build_mcts_training_supervision import atomic, digest, sha
from graph_metric_sampling import load_graph_sampler
from mcts_joint_training import load_metric_samplers, load_policy_targets
from raster_log_training import CORPUS, GRAPH, SUPERVISION, inputs_and_sources


def sizes(states):
    states = list(states)
    if not states or any(
        not isinstance(word, tuple)
        or type(n) is not int
        or n < 1
        or any(type(g) is not int or not 0 < abs(g) < n for g in word)
        for word, n in states
    ):
        raise ValueError("invalid or empty representation inventory")
    unique = {(n, word) for word, n in states}
    return dict(
        appearances=len(states),
        distinct_raw_representations=len(unique),
        maximum_word_length=max(len(word) for word, _ in states),
        maximum_strands=max(n for _, n in states),
        at_least_385_letters=sum(len(w) >= 385 for w, _ in states),
        at_least_34_strands=sum(n >= 34 for _, n in states),
        length_histogram={str(k): v for k, v in sorted(Counter(len(w) for w, _ in states).items())},
        strands_histogram={str(k): v for k, v in sorted(Counter(n for _, n in states).items())},
        raw_representation_set_sha256=digest(sorted(unique)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".partial").exists():
        raise ValueError("new size inventory output required")
    pins, sources = inputs_and_sources()
    merge_pins(pins, sources)
    old_root = Path("artifacts/raster-train-draw-coverage-v1")
    paths = [old_root / name for name in ("report.json", "inputs.json", "pool-support.json")]
    old, old_inputs, old_pools = (json.loads(p.read_text()) for p in paths)
    if (
        old["status"] != "completed"
        or old["inputs_sha256"] != sha(paths[1])
        or old["pools_sha256"] != sha(paths[2])
    ):
        raise ValueError("prior pool audit is incomplete or changed")
    merge_pins(pins, old_inputs["inputs"])
    paths += [Path(__file__).resolve(), Path("tests/test_long_mixture_size_support.py")]
    merge_pins(pins, {str(p): sha(p) for p in paths})
    check_pins(pins)
    original, suffix = load_metric_samplers(CORPUS, SUPERVISION)
    graph = load_graph_sampler(GRAPH)
    pools, all_states = {}, []
    for name, sampler in (("original", original), ("suffix", suffix), ("graph", graph)):
        verified = pool_statistics(sampler)
        if verified != old_pools[name]:
            raise ValueError("historical TRAIN pool changed")
        rows = [r for category in sampler.bins for group in category for r in group]
        states = [
            endpoint
            for r in rows
            for endpoint in ((r.left_word, r.left_strands), (r.right_word, r.right_strands))
        ]
        pools[name] = dict(records=len(rows), **sizes(states))
        all_states.extend(states)
    positives, relative = load_policy_targets(SUPERVISION)
    policy = {}
    for name, groups in (("positive", positives), ("relative", relative)):
        states = [(row[0][1].word, row[0][1].strands) for group in groups for row in group]
        policy[name] = dict(groups=len(groups), **sizes(states))
        all_states.extend(states)
    sentinel = [((1, 1, 1), 2), ((-1, -1, -1), 2)]
    all_states.extend(sentinel)
    check_pins(pins)
    report = dict(
        schema="rf-knots-long-mixture-historical-size-support-v1",
        status="completed",
        inputs=pins,
        metric_pools=pools,
        policy_pools=policy,
        sentinel=sizes(sentinel),
        combined=sizes(all_states),
        model_forwards=0,
        optimizer_steps=0,
        scope="Every available historical149 optimizer input, not merely sampled draws. "
        "Raw representations are not knot types. New147 supply and earlier Q254 pretraining "
        "are outside this inventory;147 configured maxima must be checked separately.",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic(args.out, report)
    print(
        json.dumps(
            dict(
                status="completed",
                sha256=sha(args.out),
                sources={
                    name: {
                        k: r[k]
                        for k in (
                            "appearances",
                            "distinct_raw_representations",
                            "maximum_word_length",
                            "maximum_strands",
                        )
                    }
                    for name, r in {**pools, **policy, "combined": report["combined"]}.items()
                },
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
