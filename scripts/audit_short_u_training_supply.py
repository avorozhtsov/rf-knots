"""Independent complete-supply replay, partition, normalization and determinant audit."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from rf_knots.embedding_normalization import normalize_global_view
from rf_knots.embedding_replay_corpus import invariants, owner, path_fingerprints
from rf_knots.evidence import UnknotWitness
from burau_integer_check import determinant


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def group(fp):
    return fp[0], abs(fp[1])


def validate(row, known_groups):
    w = UnknotWitness.from_dict(row["witness"])
    w.verify()
    if not w.steps or w.steps[-1].after.word or w.steps[-1].after.strands != 1:
        raise ValueError("candidate witness does not reach the unknot")
    fp = invariants(w.start.word, w.start.strands)
    if list(fp) != row["fingerprint"] or fp[0] == 1 or group(fp) in known_groups:
        raise ValueError("candidate root fingerprint contract violated")
    path = path_fingerprints(w)
    if [list(f) for f in path] != row["path_fingerprints"] or any(owner(f) != "train" for f in path):
        raise ValueError("candidate path ownership contract violated")
    if w.crossing_changes not in (1, 2):
        raise ValueError("unexpected witness CC count")
    lo, hi = abs(fp[1]) // 2, w.crossing_changes
    strong = max(lo, int(fp[0] != 1))
    if (row["signature_interval"] != [lo, hi]
            or row["determinant_strengthened_interval"] != [strong, hi] or strong > hi):
        raise ValueError("candidate bound contract violated")
    normalized = normalize_global_view(w.start.word, w.start.strands)
    if [list(normalized.word), normalized.strands] != row["normalized"]:
        raise ValueError("fast normalization differs from replay reference")
    independent = determinant(w.start.word, w.start.strands)
    if independent != fp[0]:
        raise ValueError("independent Burau determinant mismatch")
    return {"index": row["index"], "fingerprint": list(fp),
            "path_groups": sorted({group(f) for f in path}),
            "determinant": independent, "lower": strong, "upper": hi,
            "normalized": row["normalized"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exclude-panel", type=Path, action="append", default=[])
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite supply audit")
    manifest_path = args.data / "manifest.json"
    state = read(manifest_path)
    if state["status"] != "completed" or state["next_index"] != 1024:
        raise ValueError("require completed fixed 1024-attempt supply")
    if (state["purpose"] != "train-only-supply-probe-not-training-ready"
            or sum(state["counts"].values()) != 1024):
        raise ValueError("invalid supply purpose or attempt counts")
    if [s["file"] for s in state["shards"]] != [f"candidates-{i:05d}.json" for i in range(0, 1024, 8)]:
        raise ValueError("incomplete committed shard prefix")
    pins = {**state["contract"]["inputs"], str(manifest_path): sha(manifest_path),
            str(Path(__file__)): sha(Path(__file__)),
            str(Path(__file__).with_name("burau_integer_check.py")):
                sha(Path(__file__).with_name("burau_integer_check.py")),
            **{str(p): sha(p) for p in args.exclude_panel}}
    if any(sha(Path(name)) != digest for name, digest in pins.items()):
        raise ValueError("probe source/input changed before audit")
    corpus_path = Path("artifacts/embedding-multisize-clean-corpus-v1/corpus.json")
    corpus = read(corpus_path)
    known = {group(r[s + "_fingerprint"]) for r in corpus["train"] for s in ("left", "right")}
    excluded = {group(r["left_fingerprint"]) for r in corpus["validation"]}
    for panel_path in args.exclude_panel:
        excluded.update(group(r["left_fingerprint"]) for r in read(panel_path)["validation"])
    checked, seen, indices, groups, classes, all_path_groups = [], set(), set(), Counter(), Counter(), set()
    for shard in state["shards"]:
        path = args.data / shard["file"]
        pins[str(path)] = shard["sha256"]
        if sha(path) != shard["sha256"]:
            raise ValueError("candidate shard changed")
        begin = int(path.stem.split("-")[-1])
        for row in read(path):
            if not begin <= row["index"] < begin + 8 or row["index"] in indices:
                raise ValueError("invalid or duplicate attempt index")
            result = validate(row, known)
            normalized = tuple(result["normalized"][0]), result["normalized"][1]
            g = group(result["fingerprint"])
            if normalized in seen or groups[g] >= 4:
                raise ValueError("duplicate normalized root or group cap violation")
            seen.add(normalized); indices.add(row["index"]); groups[g] += 1
            all_path_groups.update(tuple(f) for f in result["path_groups"])
            if all_path_groups & excluded:
                raise ValueError("candidate path intersects declared heldout queries")
            label = f"exact-{result['lower']}" if result["lower"] == result["upper"] else f"interval-{result['lower']}-{result['upper']}"
            classes[label] += 1
            checked.append(result)
            if len(checked) % 16 == 0:
                print(json.dumps({"replayed": len(checked), "groups": len(groups)}), flush=True)
    if (len(checked) != state["counts"].get("accepted", 0)
            or len(groups) != state["summary"]["groups"] or dict(classes) != state["summary"]["classes"]):
        raise ValueError("recomputed support differs from probe summary")
    if any(sha(Path(name)) != digest for name, digest in pins.items()):
        raise ValueError("input changed during audit")
    result = {"status": "passed", "inputs": pins, "accepted": len(checked), "groups": len(groups),
              "determinant_method": "exact unreduced Burau cofactor evaluated at t=-1",
              "classes": dict(classes), "normalized_states": len(seen),
              "train_path_groups": sorted(all_path_groups), "heldout_path_intersection": [],
              "independent_checks": checked,
              "caveat": "Fingerprint groups are not knot identities; novelty is relative to "
                        "declared train endpoints and heldout queries, not all historical experiments. "
                        "All accepted roots have independent determinant checks; signature uses "
                        "the existing Seifert implementation. Not a training or search result."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".partial")
    temp.write_text(json.dumps(result, indent=2) + "\n"); temp.replace(args.output)
    print(json.dumps({k: result[k] for k in ("status", "accepted", "groups", "classes")}), flush=True)


if __name__ == "__main__":
    main()
