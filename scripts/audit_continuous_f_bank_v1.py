"""Independent retained-byte, replay, root-only F and cost audit of new bank."""

import argparse
import json
from pathlib import Path

from build_additive_root_only_f_bank_v2 import root_coefficients
from build_continuous_train_roster_v1 import exclusions
from build_mcts_training_supervision import atomic, sha
from rf_knots import mcts_loop as M
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness
from screen_legacy_unselected_root_f_v1 import cycle_free


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("immutable audit exists")
    bank = read(args.bank / "manifest.json")
    contract = bank["contract"]
    old = Path(contract["old_bank"])
    if sha(old / "manifest.json") != contract["old_manifest_sha256"]:
        raise ValueError("retained bank pin differs")
    old_manifest = read(old / "manifest.json")
    if sha(Path(contract["batch_audit"])) != contract["batch_audit_sha256"]:
        raise ValueError("batch audit SHA differs")
    excluded, states = exclusions()
    seen, f_roots, f_groups, paths_replayed = set(), 0, 0, 0
    for index, entry in enumerate(bank["roots"]):
        path = args.bank / entry["file"]
        if entry["file"] != f"root-{index:04d}.json" or sha(path) != entry["sha256"]:
            raise ValueError("new bank row SHA/order differs")
        row = read(path)
        key = tuple(row["query_group"])
        if key in seen or row["root_index"] != index:
            raise ValueError("duplicate group or root index differs")
        seen.add(key)
        f_roots += bool(row["groups"]); f_groups += len(row["groups"])
        if index < contract["old_shard_count"]:
            if entry != old_manifest["roots"][index] or sha(old / entry["file"]) != entry["sha256"]:
                raise ValueError("old row was not retained byte-identically")
            continue
        if key in excluded or row["gradient_scope"] != "exact_zero_spent_root_only":
            raise ValueError("new row group exclusion/scope differs")
        for group in row["groups"]:
            data = group["instance"]
            instance = M.Instance(**{**data, "word": tuple(data["word"])})
            fp = invariants(instance.word, instance.strands)
            if (owner(fp) != "train" or (fp[0], abs(fp[1])) != key
                    or (instance.word, instance.strands) in states
                    or group["root_index"] != index
                    or group["gradient_scope"] != "exact_zero_spent_root_only"):
                raise ValueError("new group owner/start/scope differs")
            witnesses = {}
            for saved in group["paths"]:
                actions = tuple(saved["actions"])
                w = UnknotWitness.from_actions(instance.word, instance.strands, instance.spec, actions)
                w.verify()
                if ((w.crossing_changes, w.moves) != (saved["cc"], saved["moves"])
                        or w.crossing_changes > instance.cap_crossing_changes
                        or w.moves > instance.cap_moves or not cycle_free(instance, actions)[0]):
                    raise ValueError("new complete path replay/cap/cycle differs")
                witnesses[actions] = w
                paths_replayed += 1
            for pin in group["source_shards"]:
                if sha(Path(pin["file"])) != pin["sha256"]:
                    raise ValueError("MCTS source shard SHA differs")
            result = root_coefficients(instance, witnesses)
            if (result is None or result[1] != group["root_coefficients"]
                    or len(result[1]) != group["signal_terms"]):
                raise ValueError("independent exact-root F coefficients differ")
    if f_roots != bank["summary"]["total_F_roots"] or f_groups != bank["summary"]["F_groups"]:
        raise ValueError("bank F accounting differs")
    report = {"status": "passed", "schema": "rf-knots-continuous-additive-F-bank-audit-v1",
        "manifest_sha256": sha(args.bank / "manifest.json"), "retained_old_shards": contract["old_shard_count"],
        "distinct_F_roots": f_roots, "F_groups": f_groups,
        "new_completed_paths_replayed": paths_replayed, "auditor_sha256": sha(Path(__file__))}
    args.out.mkdir(parents=True)
    atomic(args.out / "report.json", report)
    print(json.dumps({"report_sha256": sha(args.out / "report.json"), "F_roots": f_roots}), flush=True)


if __name__ == "__main__":
    main()
