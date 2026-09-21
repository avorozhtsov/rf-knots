"""Score-blind short pairs ending at a nontrivial positive-braid seed (note29)."""

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil

from build_fresh_group_panel import EXCLUSIONS, atomic_json, excluded_groups, group, sha
from rf_knots.embedding_evaluation import verify_evaluation_pair
from rf_knots.embedding_replay_corpus import canonical, invariants, lift, mirror, owner
from rf_knots.interleaved_generator import InterleavedConfig, generate, positive_braid_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--attempts", type=int, default=4000)
    parser.add_argument("--max-genus", type=int, default=6)
    parser.add_argument("--max-seed-strands", type=int, default=5)
    parser.add_argument("--protocol", type=Path, default=Path("research/29-general-short-gordian-test.md"))
    parser.add_argument("--exclude-panel", type=Path, action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.attempts < 1 or args.max_genus < 1 or not 2 <= args.max_seed_strands <= 7:
        raise ValueError("invalid generation limits")
    if (args.seed,args.attempts,args.max_genus,args.max_seed_strands,args.exclude_panel)!=(20261009,4000,6,5,[]) and args.protocol==Path("research/29-general-short-gordian-test.md"):
        raise ValueError("changed generation requires a separate declared protocol")
    args.out.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.out / ".runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = list(dict.fromkeys([Path("artifacts") / n / "panel.json" for n in EXCLUSIONS] + args.exclude_panel))
    code = [Path(__file__), Path("scripts/build_fresh_group_panel.py"),
            args.protocol, *[
        Path("src/rf_knots") / n for n in ("embedding_evaluation.py", "embedding_replay_corpus.py",
        "interleaved_generator.py", "reference.py", "actions.py", "evidence.py", "seifert.py")]]
    contract = {"attempts": args.attempts, "seed": args.seed,
                "max_genus":args.max_genus,"max_seed_strands":args.max_seed_strands,
                "inputs": {str(p): sha(p) for p in sources + code}}
    excluded = excluded_groups([json.loads(p.read_text()) for p in sources])
    old = json.loads((args.out / "progress.json").read_text()) if args.resume else None
    if old and old["contract"] != contract:
        raise ValueError("resume contract differs")
    if old and old["status"] == "completed":
        print("already completed")
        return
    if not old:
        snapshot = args.out / "source_snapshot"
        snapshot.mkdir()
        for i,p in enumerate(sources + code):
            shutil.copyfile(p, snapshot / f"{i:02d}-{p.name}")
        atomic_json(args.out / "excluded_groups.json", sorted(excluded))
    records = old["records"] if old else []
    index = old["next_index"] if old else 0
    counts, groups, classes, seen = Counter(old["counts"] if old else {}), Counter(), Counter(), set()

    def category(r):
        p=r["interval"]
        return str(p["lower"]) if p["lower"] == p["upper"] else "interval"

    for r in records:
        w = verify_evaluation_pair(r)
        groups[(group(r["left_fingerprint"]), category(r))] += 1
        classes[category(r)] += 1
        seen.add(canonical(w.start.word, w.start.strands))

    def save(status):
        report={"status":status,"pid":os.getpid(),"contract":contract,"next_index":index,
                "records":records,"counts":dict(counts),"classes":dict(classes),
                "groups":len({g for g,c in groups})}
        atomic_json(args.out / "progress.json", report)
        print(json.dumps({k:report[k] for k in ("status","next_index","counts","classes","groups")}),flush=True)

    save("building")
    while index < contract["attempts"]:
        i=index
        cfg=InterleavedConfig(crossing_changes=i%3,walk_length=8,settle_walk_length=6,
            target_strands=3+(i//3)%3,target_length=18+(i//9)%8,max_len=40,max_strands=7,strand_band=1)
        span=contract["max_seed_strands"]-1
        root=positive_braid_seed(cfg.spec(),strands=2+(i//27)%span,genus=1+(i//(27*span))%contract["max_genus"],
            seed=contract["seed"]+i,churn=4)
        instance=generate(cfg,contract["seed"],i,root)
        w=instance.witness
        assert w is not None
        if (i//2)%2:
            w=mirror(w)
        prefix=len(instance.reverse_actions)
        right=w.steps[prefix-1].after if prefix else w.start
        a,b=invariants(w.start.word,w.start.strands),invariants(right.word,right.strands)
        lower,upper=abs(a[1]-b[1])//2,instance.d_upper
        if lower>upper:
            raise ValueError("signature contradicts replay bound")
        c=str(lower) if lower==upper else "interval"
        g,key=group(a),canonical(w.start.word,w.start.strands)
        reason=("other_partition" if owner(a)!="test" else "previous_group" if g in excluded
                else "quarantine_reference" if owner(b)=="quarantine"
                else "duplicate" if key in seen else "group_cap" if groups[(g,c)]>=4
                else "class_cap" if classes[c]>=32 else "accepted")
        if reason=="accepted":
            r={"split":"test","seed":contract["seed"],"index":i,"config":cfg.provenance(),
                "seed_family":root.provenance(),"mirrored":bool((i//2)%2),
                "pair_prefix_length":prefix,"left_fingerprint":list(a),"right_fingerprint":list(b),
                "witness":w.to_dict(),"interval":{"metric":"cc","left":w.start.to_dict(),
                    "right":right.to_dict(),"lower":lower,"upper":upper,"witness_semantic_length":prefix}}
            verify_evaluation_pair(r)
            records.append(r);groups[(g,c)]+=1;classes[c]+=1;seen.add(key)
        counts[reason]+=1;index+=1
        if index%100==0:
            save("building")
    save("writing_panel")
    large=[]
    for r in records:
        w,extra=lift(verify_evaluation_pair(r),25,160)
        prefix=r["pair_prefix_length"]+extra
        row={**r,"evaluation_view":"size_shift","witness":w.to_dict(),"pair_prefix_length":prefix,
             "interval":{**r["interval"],"left":w.start.to_dict(),"witness_semantic_length":prefix}}
        verify_evaluation_pair(row);large.append(row)
    adequate=len({g for g,c in groups})>=8 and len(records)>=32 and all(classes[c]>=4 for c in ("0","1","2"))
    atomic_json(args.out / "panel.json",{"schema":"rf-knots-gordian-pair-panel-v1","partition":"test",
        "construction_complete":True,"complete":adequate,"contract":contract,"validation":records,
        "size_shift":large,"classes":dict(classes),"reference_contract":"evaluation permits test references; training unchanged"})
    save("completed")


if __name__=="__main__":
    main()
