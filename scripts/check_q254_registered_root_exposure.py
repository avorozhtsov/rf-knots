"""Bounded Q254 provenance/root-overlap audit, NOT proof of pretraining independence."""

import argparse
import hashlib
import json
from pathlib import Path

from check_qrun_common_witnesses import check_pins, sha, write_new

from rf_knots.embedding_replay_corpus import canonical, invariants

FROZEN = Path(
    "/Users/artemvorozhtsov/projects/unknotdb/models/"
    "q-grown-raster-axial-12-q254-frozen-20260824-v0"
)
PANELS = {
    "short-u": Path("artifacts/graph-mixture-fresh-short-u-test-v1/panel.json"),
    "general": Path("artifacts/pure-braid-short-distance-test-v1/panel.json"),
}


def processed_prefix(frozen, report):
    n = frozen["processed_count"]
    processed = report["processed"]
    if type(n) is not int or not 0 <= n <= len(processed):
        raise ValueError("invalid frozen processed count")
    prefix = processed[:n]
    digest = hashlib.sha256(json.dumps(prefix, separators=(",", ":")).encode()).hexdigest()
    if digest != frozen["processed_order_sha256"]:
        raise ValueError("later report prefix does not match frozen snapshot")
    return prefix


def bank_digest(payload):
    # Native collaborative_scientists._json_hash, not file or compact-JSON hash.
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def fingerprint_group(fp):
    return fp[0], abs(fp[1])


def compare_queries(panel, root_rows):
    groups, words = {}, {}
    for row in root_rows:
        groups.setdefault(fingerprint_group(row["fingerprint"]), []).append(row["id"])
        words.setdefault(canonical(row["word"], row["strands"]), []).append(row["id"])
    results = {}
    for split in ("validation", "size_shift"):
        if not panel.get(split):
            continue
        for endpoint in ("left", "right"):
            rows = []
            for index, r in enumerate(panel[split]):
                e = r["interval"][endpoint]
                fp = r[f"{endpoint}_fingerprint"]
                rows.append(
                    {
                        "panel_index": index,
                        "fingerprint": fp,
                        "explicit_unknot_anchor": e["word"] == [] and e["strands"] == 1,
                        "fingerprint_matches": groups.get(fingerprint_group(fp), []),
                        "cyclic_word_matches": words.get(canonical(e["word"], e["strands"]), []),
                    }
                )
            results[f"{split}/{endpoint}"] = {
                "endpoints": len(rows),
                "explicit_unknot_anchors": sum(r["explicit_unknot_anchor"] for r in rows),
                "fingerprint_collision_endpoints": sum(
                    bool(r["fingerprint_matches"]) for r in rows
                ),
                "exact_cyclic_word_overlap_endpoints": sum(
                    bool(r["cyclic_word_matches"]) for r in rows
                ),
                "rows": rows,
            }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    frozen_path = FROZEN / "manifest.json"
    frozen = json.loads(frozen_path.read_text())
    checkpoint = FROZEN / "checkpoint.pt"
    if sha(checkpoint) != frozen["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint hash differs")
    current_state = Path(frozen["source_q_state"])
    stage = current_state.parent
    manifest_path, report_path = stage / "manifest.json", stage / "report.json"
    manifest, report = json.loads(manifest_path.read_text()), json.loads(report_path.read_text())
    prefix = processed_prefix(frozen, report)
    banks = {}
    paths = [
        frozen_path,
        checkpoint,
        current_state,
        manifest_path,
        report_path,
        Path(__file__),
        Path("tests/test_q254_root_exposure.py"),
        Path("src/rf_knots/embedding_replay_corpus.py"),
        Path("src/rf_knots/seifert.py"),
        Path("src/rf_knots/reference.py"),
        Path("research/101-q254-exposure-audit-protocol.md"),
        FROZEN.parents[1] / "tools/freeze_q_policy_checkpoint.py",
    ]
    for key, copy_name in (("bank", "bank.json"), ("prior_bank", "prior-bank.json")):
        path = Path(manifest[key])
        payload = json.loads(path.read_text())
        if bank_digest(payload) != manifest[f"{key}_sha256"]:
            raise ValueError("registered bank differs from native manifest hash")
        if payload != json.loads((stage / copy_name).read_text()):
            raise ValueError("stage bank copy differs")
        banks[key] = payload["rows"]
        paths += [path, stage / copy_name]
    native_source = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench/src/pgx_mcts_bench")
    paths.append(native_source / "collaborative_scientists.py")
    inherited = manifest["initial_states"]["raster-axial-12"]
    if sha(Path(inherited["path"])) != inherited["sha256"]:
        raise ValueError("registered inherited state changed")
    paths.append(Path(inherited["path"]))
    pins = {str(p): sha(p) for p in paths}
    panels = {}
    for name, path in PANELS.items():
        audit_path = path.parent / "independence-audit.json"
        audit = json.loads(audit_path.read_text())
        if audit["status"] != "passed" or audit["inputs"].get(str(path)) != sha(path):
            raise ValueError("panel audit does not bind current panel")
        check_pins(audit["inputs"])
        for key, value in audit["inputs"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting panel provenance")
            pins[key] = value
        pins[str(audit_path)] = sha(audit_path)
        panels[name] = json.loads(path.read_text())
    check_pins(pins)
    stage_ids = [r["id"] for r in banks["bank"]]
    if stage_ids[: len(prefix)] != prefix or len(stage_ids) != manifest["representations"]:
        raise ValueError("registered stage ordering/coverage differs")
    all_rows = []
    for kind, rows in banks.items():
        for index, r in enumerate(rows):
            all_rows.append(
                {
                    "id": r["id"],
                    "word": r["word"],
                    "strands": r["strands"],
                    "fingerprint": list(invariants(tuple(r["word"]), r["strands"])),
                    "bank": kind,
                    "stage_index": index if kind == "bank" else None,
                }
            )
            if len(all_rows) % 32 == 0:
                print(json.dumps({"roots_fingerprinted": len(all_rows)}), flush=True)
    registered_prefix = [
        r for r in all_rows if r["bank"] == "prior_bank" or r["stage_index"] < len(prefix)
    ]
    comparisons = {
        name: {
            "prior_plus_completed_prefix": compare_queries(panel, registered_prefix),
            "all_registered_stage_roots": compare_queries(panel, all_rows),
        }
        for name, panel in panels.items()
    }
    check_pins(pins)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_new(
        args.out,
        {
            "status": "completed",
            "independence_proven": False,
            "inputs": pins,
            "checkpoint_sha256": frozen["checkpoint_sha256"],
            "generation_label": frozen["q_generation"],
            "frozen_stage_completed_roots": len(prefix),
            "registered_stage_roots": len(stage_ids),
            "prior_registered_roots": len(banks["prior_bank"]),
            "verified_processed_prefix": prefix,
            "frozen_source_state_sha256": frozen["source_q_state_sha256"],
            "current_source_state_sha256": pins[str(current_state)],
            "source_state_still_matches_freeze": pins[str(current_state)]
            == frozen["source_q_state_sha256"],
            "roots": all_rows,
            "comparison": comparisons,
            "limits": [
                "Generation label is not a terminal checkpoint or a best-model certificate.",
                "Fingerprint collision is not a knot-identity certificate.",
                "Registered roots do not cover all MCTS/CC trajectory states or initial pretraining.",
                "Completed-prefix metadata need not exclude partially processed next-root exposure.",
                "All-stage comparison is conservative for this bank only, not all weight history.",
                "Current training/evaluation unchanged; this is a read-only provenance audit.",
            ],
        },
    )
    print(
        json.dumps(
            {"status": "completed", "root_count": len(all_rows), "independence_proven": False}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
