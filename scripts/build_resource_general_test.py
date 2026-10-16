"""Protocol138 fresh general-distance panel; old74 remains immutable."""

import argparse
import fcntl
import json
import os
from collections import Counter
from functools import cache
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from audit_ordinary_graph_test_panel import trace_bounds
from build_mcts_training_supervision import atomic, sha
from build_ordinary_graph_test_panel import cyclic_rotation
from build_pure_braid_distance_panel import construct, inputs, select
from burau_integer_check import determinant
from raster_pair_training import CORPUS, PANELS

from rf_knots import reference
from rf_knots.embedding_replay_corpus import invariants

CONFIG = dict(seed=20261073, attempts=3072, per_category=32, per_query_group=4)
SCHEMA = "rf-knots-resource-general-distance-test-v1"


@cache
def fingerprint(word, strands):
    return invariants(word, strands)


@cache
def integer_determinant(word, strands):
    return determinant(word, strands)


def make_row(index):
    source, states, positions = construct(index, seed=CONFIG["seed"])
    fps = [list(fingerprint(tuple(s["word"]), s["strands"])) for s in states]
    rotation = cyclic_rotation(states[0], states[-1])
    return dict(
        index=index,
        source=source,
        states=states,
        cc_positions=positions,
        fingerprints=fps,
        cyclic_rotation=rotation,
        lower=max(abs(fps[0][1] - fps[-1][1]) // 2, int(fps[0][0] != fps[-1][0])),
        upper=0 if rotation is not None else len(positions),
    )


def audit_row(row, index):
    if row["index"] != index or (row["source"], row["states"], row["cc_positions"]) != construct(
        index, seed=CONFIG["seed"]
    ):
        raise ValueError("deterministic construction differs")
    if any(type(row[k]) is not int for k in ("lower", "upper")):
        raise ValueError("CC bounds must be integers")
    for state in row["states"]:
        if reference.num_components(tuple(state["word"]), state["strands"]) != 1:
            raise ValueError("closure is not a knot")
    if row["cc_positions"]:
        trace_bounds(row, fingerprint, integer_determinant)
    else:
        fps = []
        for state in row["states"]:
            word, n = tuple(state["word"]), state["strands"]
            fp = fingerprint(word, n)
            if fp[0] != integer_determinant(word, n):
                raise ValueError("independent determinant differs")
            fps.append(list(fp))
        if row["fingerprints"] != fps or row["lower"] != 0 or row["upper"] != 0:
            raise ValueError("zero-CC fingerprints or bounds differ")
    a, b = row["states"][0], row["states"][-1]
    rotations = [
        k
        for k in range(len(a["word"]))
        if a["strands"] == b["strands"] and a["word"][k:] + a["word"][:k] == b["word"]
    ]
    expected = min(rotations) if rotations else None
    if row["cyclic_rotation"] != expected or (row["upper"] == 0) != bool(rotations):
        raise ValueError("cyclic certificate differs")


def provenance():
    pins, groups, training, excluded = inputs()
    excluded = set(excluded)
    for path in [CORPUS, *PANELS]:
        data = json.loads(path.read_text())
        excluded.update(
            (r["left_fingerprint"][0], abs(r["left_fingerprint"][1])) for r in data["validation"]
        )
        merge_pins(pins, {str(path): sha(path)})
    for path in (
        Path(__file__).resolve(),
        Path("tests/test_resource_general_test.py"),
        Path("research/138-resource-units-fresh-general-test.md"),
        Path("scripts/audit_ordinary_graph_test_panel.py"),
        Path("scripts/build_ordinary_graph_test_panel.py"),
        Path("scripts/burau_integer_check.py"),
        Path("scripts/raster_pair_training.py"),
    ):
        merge_pins(pins, {str(path): sha(path)})
    check_pins(pins)
    return pins, groups, training, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("existing output requires explicit resume")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pins, groups, training, excluded = provenance()
    contract = dict(config=CONFIG, inputs=pins)
    path = args.out / "progress.json"
    old = json.loads(path.read_text()) if args.resume else None
    if old is not None and old["contract"] != contract:
        raise ValueError("panel resume contract differs")
    if old is not None and old["status"] == "completed":
        if sha(args.out / "panel.json") != old["panel_sha256"]:
            raise ValueError("completed panel changed")
        return
    rows = old["rows"] if old else []
    if len(rows) > CONFIG["attempts"] or old and len(rows) != old["next_index"]:
        raise ValueError("saved attempt coverage differs")
    for index, row in enumerate(rows):
        audit_row(row, index)

    def save(status, **extra):
        records, decisions, classes, count, adequate = select(rows, training, excluded, CONFIG)
        state = dict(
            status=status,
            pid=os.getpid(),
            contract=contract,
            rows=rows,
            next_index=len(rows),
            classes=classes,
            groups=count,
            selected=len(records),
            adequate=adequate,
            selection_counts=dict(Counter(decisions)),
            **extra,
        )
        atomic(path, state)
        print(
            json.dumps(
                {k: state[k] for k in ("status", "next_index", "classes", "groups", "selected")}
            ),
            flush=True,
        )

    save("building")
    for index in range(len(rows), CONFIG["attempts"]):
        row = make_row(index)
        audit_row(row, index)
        rows.append(row)
        if len(rows) % 32 == 0:
            save("building")
    records, decisions, classes, count, adequate = select(rows, training, excluded, CONFIG)
    for row in records:
        audit_row(row, row["index"])
    panel = dict(
        schema=SCHEMA,
        partition="test",
        complete=adequate,
        construction_complete=True,
        contract=contract,
        validation=records,
        size_shift=[],
        classes=classes,
        groups=count,
        training_groups={k: sorted(v) for k, v in groups.items()},
        excluded_query_groups=sorted(excluded),
        selection_decisions=decisions,
        selection_counts=dict(Counter(decisions)),
        scope="Synthetic RF-lineage query-disjoint TEST; reference endpoints may be known; "
        "Q254 pretraining exposure incompletely audited",
    )
    check_pins(pins)
    atomic(args.out / "panel.json", panel)
    save("completed", evaluation_ready=adequate, panel_sha256=sha(args.out / "panel.json"))


if __name__ == "__main__":
    main()
