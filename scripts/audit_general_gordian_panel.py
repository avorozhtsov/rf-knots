"""Replay every test pair and train path; test query novelty is not inferred from root counts."""

import argparse
import hashlib
import json
from pathlib import Path

from rf_knots.embedding_evaluation import verify_evaluation_pair
from rf_knots.embedding_replay_corpus import path_fingerprints, verify_record
from rf_knots.invariants import determinant


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel",type=Path,required=True)
    parser.add_argument("--corpus",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--determinant-method", choices=("polynomial", "cofactor"), default="polynomial")
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite an audit")
    inputs={str(p):sha(p) for p in [args.panel,args.corpus,Path(__file__)]}
    determinant_check = determinant
    if args.determinant_method == "cofactor":
        from burau_integer_check import determinant as determinant_check
        checker_path = Path(__file__).with_name("burau_integer_check.py")
        inputs[str(checker_path)] = sha(checker_path)
    panel=json.loads(args.panel.read_text())
    if panel.get("schema")!="rf-knots-gordian-pair-panel-v1" or not panel.get("complete"):
        raise ValueError("requires an adequate completed Gordian-pair panel")
    for name,digest in panel["contract"]["inputs"].items():
        if sha(Path(name))!=digest:
            raise ValueError(f"construction input changed: {name}")
    excluded={tuple(g) for g in json.loads((args.panel.parent/"excluded_groups.json").read_text())}
    queries,reference_groups,checked=set(),set(),[]
    for view in ("validation","size_shift"):
        for r in panel[view]:
            verify_evaluation_pair(r)
            a,b=r["left_fingerprint"],r["right_fingerprint"]
            g=(a[0],abs(a[1]))
            if view=="validation" and g not in queries:
                s=r["interval"]["left"]
                value=determinant_check(tuple(s["word"]),s["strands"])
                if value!=a[0]:
                    raise ValueError("independent determinant mismatch")
                checked.append({"fingerprint":list(g),"determinant":value})
            queries.add(g);reference_groups.add((b[0],abs(b[1])))
    if queries & excluded:
        raise ValueError("test query appeared in previous panel")
    corpus=json.loads(args.corpus.read_text())
    train_groups=set()
    for i,r in enumerate(corpus["train"]):
        w=verify_record(r)
        train_groups.update((d,abs(s)) for d,s in path_fingerprints(w))
        if (i+1)%256==0:
            print(json.dumps({"train_records_replayed":i+1,"path_groups":len(train_groups)}),flush=True)
    if queries & train_groups:
        raise ValueError("test query overlaps an intermediate training state")
    if any(sha(Path(p))!=value for p,value in inputs.items()):
        raise ValueError("audit input changed during replay")
    result={"status":"passed","inputs":inputs,"test_query_groups":sorted(queries),
        "determinant_method":args.determinant_method,
        "train_path_groups":sorted(train_groups),"train_records_replayed":len(corpus["train"]),
        "test_records_replayed":sum(len(panel[v]) for v in ("validation","size_shift")),
        "query_train_path_intersection":[],"query_previous_panel_intersection":[],
        "known_reference_groups":sorted(reference_groups & train_groups),
        "independent_determinants":checked,
        "caveat":"Query novelty relative to this complete training lineage and declared excluded panels. "
                 "References may be known. Fingerprints do not identify knots; legacy experiment exposure not exhaustively audited."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(".partial")
    temporary.write_text(json.dumps(result,indent=2)+"\n")
    temporary.replace(args.output)
    print(json.dumps({"status":"passed","test_groups":len(queries),"train_path_groups":len(train_groups)}))


if __name__=="__main__":
    main()
