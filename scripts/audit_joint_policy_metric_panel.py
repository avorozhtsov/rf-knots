"""Replay a prospective panel and trace every new optimizer state to a TRAIN witness.

Historical full-path audits are reused only with verified input hashes; this
does not claim to replay those older experiments again or identify knot types.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_mcts_training_supervision import atomic, digest, sha
from burau_integer_check import determinant

from rf_knots import mcts_loop as M
from rf_knots.embedding_evaluation import verify_evaluation_pair
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

CORPUS = Path("artifacts/embedding-diverse-short-u-corpus-v1/corpus.json")
SUPERVISION = Path("artifacts/diverse-mcts-supervision-diamonds-v1")
AUDITS = {
    "metric": Path(
        "artifacts/embedding-common-block-gordian-supplement-v1/metric-independence-audit.json"
    ),
    "rollout": Path(
        "artifacts/embedding-common-block-gordian-supplement-v1/rollout-independence-audit.json"
    ),
    "mcts": Path("artifacts/diverse-cost-mcts-training-v1/partition-replay-audit.json"),
    "certificates": SUPERVISION / "certificate-audit.json",
}


def merge_pins(pins, more):
    for name, value in more.items():
        if name in pins and pins[name] != value:
            raise ValueError(f"conflicting provenance: {name}")
        pins[name] = value


def check_pins(pins):
    for name, value in pins.items():
        if sha(Path(name)) != value:
            raise ValueError(f"audit source/input changed: {name}")


def witness_states(witness):
    """Every resource-augmented state, with invariants recomputed after each CC."""
    witness.verify()
    state = M.SearchState(witness.start.word, witness.start.strands, 0, 0)
    fp = invariants(state.word, state.strands)
    states = {state: fp}
    for step in witness.steps:
        cc = step.action.kind == "CROSSING_CHANGE"
        state = M.SearchState(
            step.after.word, step.after.strands, state.used_cc + cc, state.used_moves + 1
        )
        if cc:
            fp = invariants(state.word, state.strands)
        states[state] = fp
    if any(owner(fp) != "train" for fp in states.values()):
        raise ValueError("non-TRAIN witness in optimizer sources")
    return states


def size_stratum(word, strands):
    return {
        (False, False): "within",
        (True, False): "length_only",
        (False, True): "strands_only",
        (True, True): "both",
    }[len(word) > 208, strands > 17]


def historical_groups(historical):
    """Use the explicit schema of each pinned certificate, before expensive replay.

    The rollout input here is the extension of a full replay certificate: its
    inherited optimizer groups are named rollout_path_groups, not the base
    certificate's optimizer_path_groups. Do not silently guess or use counts.
    """
    result = []
    for name, field in (
        ("metric", "train_path_groups"),
        ("rollout", "rollout_path_groups"),
        ("mcts", "eligible_path_groups"),
    ):
        values = historical[name].get(field)
        if not isinstance(values, list) or not values:
            raise ValueError(f"missing audited group list: {name}/{field}")
        if any(
            not isinstance(fp, list)
            or len(fp) != 2
            or any(type(v) is not int for v in fp)
            or fp[0] < 1
            or fp[1] < 0
            for fp in values
        ):
            raise ValueError(f"invalid audited groups: {name}/{field}")
        result.append(set(map(tuple, values)))
    return tuple(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite independence audit")
    sources = [
        Path(__file__).resolve(),
        Path("scripts/burau_integer_check.py"),
        Path("scripts/build_mcts_training_supervision.py"),
        *sorted(Path("src/rf_knots").glob("*.py")),
    ]
    pins = {str(p): sha(p) for p in [args.panel, *sources]}
    historical = {}
    for name, path in AUDITS.items():
        audit = json.loads(path.read_text())
        if audit["status"] != "passed":
            raise ValueError("incomplete prerequisite audit")
        merge_pins(pins, {str(path): sha(path), **audit["inputs"]})
        historical[name] = audit
    old_metric, old_rollout, mcts_paths = historical_groups(historical)
    if pins.get(str(CORPUS)) != sha(CORPUS):
        raise ValueError("historical audit uses a different corpus")
    panel = json.loads(args.panel.read_text())
    if not panel["complete"] or not panel["construction_complete"]:
        raise ValueError("inadequate or incomplete prospective panel")
    merge_pins(pins, panel["contract"]["inputs"])
    excluded_path = args.panel.parent / "excluded_groups.json"
    merge_pins(pins, {str(excluded_path): sha(excluded_path)})
    check_pins(pins)
    excluded = set(map(tuple, json.loads(excluded_path.read_text())))
    queries, checked, strata = set(), [], Counter()
    for view in ("validation", "size_shift"):
        for row in panel[view]:
            witness = verify_evaluation_pair(row)
            fp = row["left_fingerprint"]
            key = fp[0], abs(fp[1])
            if owner(fp) != "test" or key in excluded:
                raise ValueError("query is not fresh TEST-owned")
            if key not in queries:
                value = determinant(witness.start.word, witness.start.strands)
                if value != fp[0]:
                    raise ValueError("independent determinant differs")
                checked.append({"fingerprint": list(key), "determinant": value})
            queries.add(key)
            if view == "validation":
                strata[size_stratum(witness.start.word, witness.start.strands)] += 1
    metric_path = SUPERVISION / "short-u-views.json"
    index = json.loads(metric_path.read_text())
    if pins.get(str(metric_path)) != sha(metric_path):
        raise ValueError("witness registry not covered by certificate audit")
    by_root, source_groups = defaultdict(dict), set()
    for witness_id, payload in index["witnesses"].items():
        if digest(payload) != witness_id:
            raise ValueError("registry witness hash differs")
        witness = UnknotWitness.from_dict(payload)
        states = witness_states(witness)
        by_root[witness.start.word, witness.start.strands].update(states)
        source_groups.update((d, abs(s)) for d, s in states.values())
    manifest_path = SUPERVISION / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if pins.get(str(manifest_path)) != sha(manifest_path):
        raise ValueError("supervision manifest not covered by certificate audit")
    counts = Counter()
    for item in manifest["conditions"]:
        path = SUPERVISION / item["file"]
        if pins.get(str(path)) != item["sha256"]:
            raise ValueError("unaudited target shard")
        targets = json.loads(path.read_text())
        instance = targets["instance"]
        root = tuple(instance["word"]), instance["strands"]
        known = by_root[root]
        for section in ("positives", "preferences"):
            for row in targets[section]:
                r = row["state"]
                state = M.SearchState(tuple(r["word"]), r["strands"], r["used_cc"], r["used_moves"])
                if state not in known:
                    raise ValueError("optimizer policy state lacks a replayed source witness")
                counts[section] += 1
        counts["conditions"] += 1
    training = old_metric | old_rollout | mcts_paths | source_groups
    if queries & training:
        raise ValueError("query intersects optimizer lineage")
    check_pins(pins)
    result = {
        "status": "passed",
        "inputs": pins,
        "test_query_groups": sorted(queries),
        "query_train_intersection": [],
        "query_previous_panel_intersection": [],
        "independent_determinants": checked,
        "source_witnesses_replayed": len(index["witnesses"]),
        "new_optimizer_states_checked": dict(counts),
        "raw_size_strata": dict(strata),
        "training_group_counts": {
            "old_metric": len(old_metric),
            "old_rollout": len(old_rollout),
            "mcts_eligible": len(mcts_paths),
            "new_sources": len(source_groups),
        },
        "historical_audits_reused": {k: str(v) for k, v in AUDITS.items()},
        "caveat": "Old full-path/certificate audits hash-verified and reused, not replayed again. "
        "Fresh panel and all new source witnesses replayed here. Conservative fingerprint "
        "groups are not knot IDs; no claim of exhaustively audited legacy exposure.",
    }
    atomic(args.out, result)
    print(
        json.dumps(
            {k: result[k] for k in ("status", "new_optimizer_states_checked", "raw_size_strata")}
        )
    )


if __name__ == "__main__":
    main()
