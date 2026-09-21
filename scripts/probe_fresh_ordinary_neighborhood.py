"""Fixed score-blind expansion of verified non-TRAIN source states, protocol72."""

import argparse
import fcntl
import json
import os
import random
from collections import Counter
from functools import cache
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, digest, sha
from build_ordinary_graph_test_panel import cyclic_rotation
from burau_integer_check import determinant
from graph_evaluation_partition import partition_inputs
from probe_ordinary_graph_pairs import replay_cc

from rf_knots.embedding_replay_corpus import canonical, invariants, owner

SOURCE = Path("artifacts/graph-mixture-fresh-gordian-test-v1")
CONFIG = {"seed": 20261035, "attempts": 1024}


def source_pool(progress):
    states = {}
    for row in progress["rows"]:
        for rep, state in zip(row["source"]["representations"], row["states"], strict=True):
            if rep in states and states[rep] != state:
                raise ValueError("physical source representation inconsistent")
            states[rep] = state
    return {rep: states[rep] for rep in sorted(states) if len(states[rep]["word"]) >= 2}


def draw(index, pool, config=CONFIG):
    rng = random.Random(config["seed"] + index)
    rep = rng.choice(list(pool))
    positions = rng.sample(range(len(pool[rep]["word"])), 1 + index % 2)
    states = [pool[rep]]
    for position in positions:
        previous = states[-1]
        word = previous["word"].copy()
        word[position] *= -1
        states.append({"word": word, "strands": previous["strands"]})
    return (
        {"reference_rep": rep, "positions_from_reference": positions},
        list(reversed(states)),
        list(reversed(positions)),
    )


def reason(fingerprints, training, historical):
    query, reference = fingerprints[0], fingerprints[-1]
    g = query[0], abs(query[1])
    return (
        "other_owner"
        if owner(query) != "test"
        else "optimizer_collision"
        if g in training
        else "historical_query"
        if g in historical
        else "quarantine_reference"
        if owner(reference) == "quarantine"
        else "eligible"
    )


def summarize(rows):
    eligible = [r for r in rows if r["reason"] == "eligible"]
    groups = {(r["fingerprints"][0][0], abs(r["fingerprints"][0][1])) for r in eligible}
    unique = {}
    for r in eligible:
        key = digest(
            sorted(
                canonical(tuple(s["word"]), s["strands"]) for s in (r["states"][0], r["states"][-1])
            )
        )
        if key not in unique or r["upper"] < unique[key]["upper"]:
            unique[key] = r
    return {
        "attempts": len(rows),
        "reasons": dict(Counter(r["reason"] for r in rows)),
        "eligible_query_groups": sorted(groups),
        "eligible_paths": len(eligible),
        "unique_endpoint_pairs": len(unique),
        "unique_pair_intervals": dict(
            Counter(f"{r['lower']}-{r['upper']}" for r in unique.values())
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("new output required, or explicit resume")
    pins, _, training, historical = partition_inputs()
    audit = json.loads((SOURCE / "independence-audit.json").read_text())
    if audit["status"] != "passed" or audit["source_paths_checked"] != 1635:
        raise ValueError("require complete source path integrity audit")
    merge_pins(pins, audit["inputs"])
    merge_pins(
        pins,
        {
            str(p): sha(p)
            for p in (
                SOURCE / "independence-audit.json",
                Path(__file__).resolve(),
                Path("research/72-ordinary-neighborhood-fresh-query-probe.md"),
            )
        },
    )
    check_pins(pins)
    progress = json.loads((SOURCE / "progress.json").read_text())
    pool = source_pool(progress)
    if not pool:
        raise ValueError("empty physical source pool")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.out / "report.json"
    contract = {"config": CONFIG, "inputs": pins, "source_representations": len(pool)}
    previous = json.loads(path.read_text()) if args.resume else None
    if previous and previous["contract"] != contract:
        raise ValueError("resume probe contract differs")
    if previous and previous["status"] == "completed":
        print("already completed, frozen inputs verified")
        return
    rows = previous["rows"] if previous else []
    if len(rows) > CONFIG["attempts"]:
        raise ValueError("too many saved attempts")
    for i, row in enumerate(rows):
        selection, states, positions = draw(i, pool)
        if (row["index"], row["source"], row["states"], row["cc_positions"]) != (
            i,
            selection,
            states,
            positions,
        ):
            raise ValueError("saved source draw differs")

    @cache
    def fingerprint(word, n):
        fp = invariants(word, n)
        if determinant(word, n) != fp[0]:
            raise ValueError("independent determinant mismatch")
        return list(fp)

    def save(status):
        summary = summarize(rows)
        atomic(
            path,
            {
                "status": status,
                "pid": os.getpid(),
                "contract": contract,
                "rows": rows,
                "summary": summary,
                "evaluation_ready": False,
                "scope": "Diagnostic supply only. No model scores, no TRAIN data, "
                "no promotion result.",
            },
        )
        print(
            json.dumps(
                {
                    "status": status,
                    **{k: v for k, v in summary.items() if k != "eligible_query_groups"},
                    "eligible_groups": len(summary["eligible_query_groups"]),
                }
            ),
            flush=True,
        )

    save("building")
    for i in range(len(rows), CONFIG["attempts"]):
        selected, states, positions = draw(i, pool)
        actual = [
            replay_cc((tuple(a["word"]), a["strands"]), (tuple(b["word"]), b["strands"]))
            for a, b in zip(states[:-1], states[1:], strict=True)
        ]
        if actual != positions:
            raise ValueError("literal CC replay differs")
        fps = [fingerprint(tuple(s["word"]), s["strands"]) for s in states]
        lower = max(abs(fps[0][1] - fps[-1][1]) // 2, int(fps[0][0] != fps[-1][0]))
        upper = 0 if cyclic_rotation(states[0], states[-1]) is not None else len(positions)
        if lower > upper:
            raise ValueError("ordinary CC bounds contradict")
        rows.append(
            {
                "index": i,
                "source": selected,
                "states": states,
                "cc_positions": positions,
                "fingerprints": fps,
                "lower": lower,
                "upper": upper,
                "reason": reason(fps, training, historical),
            }
        )
        if len(rows) % 64 == 0:
            save("building")
    check_pins(pins)
    save("completed")


if __name__ == "__main__":
    main()
