"""Fixed train-only diversity probe, research/40; no model evaluation or training."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_normalization import fast_normalize_global_view
from rf_knots.embedding_replay_corpus import invariants, owner, path_fingerprints

CORPUS = Path("artifacts/embedding-multisize-clean-corpus-v1/corpus.json")
CORPUS_SHA = "04ffeca1bebccc5e8af6a7f7553b2faa2895e098041320e73bf4a2cd8093600e"
PROTOCOL = Path("research/40-diverse-short-u-supply-protocol.md")
CONFIG = dict(attempts=1024, seed=20261017, max_strands=17, min_depth=8, max_depth=48)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, payload):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def group(fp):
    return fp[0], abs(fp[1])


def bounds(fp, upper):
    signature = abs(fp[1]) // 2
    strengthened = max(signature, int(fp[0] != 1))
    if strengthened > upper:
        raise ValueError("lower bound exceeds replayed upper bound")
    return [signature, upper], [strengthened, upper]


def describe(rows):
    classes = Counter()
    for row in rows:
        lo, hi = row["determinant_strengthened_interval"]
        classes[f"exact-{lo}" if lo == hi else f"interval-{lo}-{hi}"] += 1
    return {"accepted": len(rows), "groups": len({group(r["fingerprint"]) for r in rows}),
            "classes": dict(classes),
            "max_raw_length": max((len(r["witness"]["start"]["word"]) for r in rows), default=0),
            "max_normalized_length": max((len(r["normalized"][0]) for r in rows), default=0),
            "max_raw_strands": max((r["witness"]["start"]["strands"] for r in rows), default=0),
            "max_normalized_strands": max((r["normalized"][1] for r in rows), default=0)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if sha(CORPUS) != CORPUS_SHA:
        raise ValueError("unexpected parent train corpus")
    sources = {Path(__file__).resolve().relative_to(Path.cwd()), PROTOCOL,
               *Path("src/rf_knots").rglob("*.py")}
    pins = {str(path): sha(path) for path in sorted(sources | {CORPUS})}
    contract = {"config": CONFIG, "inputs": pins}
    args.out.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = args.out / "manifest.json"
    if args.resume:
        state = read(manifest)
        if state["contract"] != contract:
            raise ValueError("resume source/input contract differs")
    else:
        state = {"contract": contract, "next_index": 0, "shards": [], "counts": {},
                 "purpose": "train-only-supply-probe-not-training-ready"}
        for path in sources:
            target = args.out / "source_snapshot" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    rows = []
    if (state["next_index"] != len(state["shards"]) * 8
            or not 0 <= state["next_index"] <= CONFIG["attempts"]
            or [s["file"] for s in state["shards"]]
            != [f"candidates-{i:05d}.json" for i in range(0, state["next_index"], 8)]):
        raise ValueError("invalid committed attempt prefix")
    for shard in state["shards"]:
        path = args.out / shard["file"]
        if sha(path) != shard["sha256"]:
            raise ValueError("committed shard changed")
        rows.extend(read(path))
    if state.get("status") == "completed":
        print("already completed")
        return
    known = {group(row[side + "_fingerprint"])
             for row in read(CORPUS)["train"] for side in ("left", "right")}
    groups = Counter(group(r["fingerprint"]) for r in rows)
    seen = {(tuple(r["normalized"][0]), r["normalized"][1]) for r in rows}
    counts = Counter(state["counts"])
    started = time.time()
    state.update(status="building", pid=os.getpid(), summary=describe(rows))
    save(manifest, state)
    for begin in range(state["next_index"], CONFIG["attempts"], 8):
        accepted = []
        for index in range(begin, min(begin + 8, CONFIG["attempts"])):
            witness = conjugate_witness(CONFIG["seed"], index, **{
                k: CONFIG[k] for k in ("max_strands", "min_depth", "max_depth")})
            fp = invariants(witness.start.word, witness.start.strands)
            g = group(fp)
            reason = ("determinant_one" if fp[0] == 1 else "nontrain_root" if owner(fp) != "train"
                      else "known_train_endpoint_group" if g in known
                      else "group_cap" if groups[g] >= 4 else None)
            if reason is None:
                normalized = fast_normalize_global_view(witness.start.word, witness.start.strands)
                if normalized in seen:
                    reason = "duplicate_normalized_state"
                else:
                    fingerprints = path_fingerprints(witness)
                    if any(owner(f) != "train" for f in fingerprints):
                        reason = "nontrain_intermediate"
                    else:
                        original, strengthened = bounds(fp, witness.crossing_changes)
                        accepted.append({"index": index, "fingerprint": fp,
                            "path_fingerprints": fingerprints, "normalized": normalized,
                            "signature_interval": original,
                            "determinant_strengthened_interval": strengthened,
                            "witness": witness.to_dict()})
                        groups[g] += 1
                        seen.add(normalized)
                        reason = "accepted"
            counts[reason] += 1
        shard = args.out / f"candidates-{begin:05d}.json"
        if shard.exists():
            if read(shard) != json.loads(json.dumps(accepted)):
                raise ValueError("uncommitted shard differs on replay")
        else:
            save(shard, accepted)
        rows.extend(accepted)
        state["shards"].append({"file": shard.name, "sha256": sha(shard)})
        state.update(next_index=min(begin + 8, CONFIG["attempts"]), counts=dict(counts),
                     summary=describe(rows), seconds_this_run=time.time() - started)
        if any(sha(Path(name)) != digest for name, digest in pins.items()):
            raise ValueError("input changed during supply probe")
        save(manifest, state)
        print(json.dumps({k: state[k] for k in ("next_index", "counts", "summary")}), flush=True)
    state["status"] = "completed"
    save(manifest, state)


if __name__ == "__main__":
    main()
