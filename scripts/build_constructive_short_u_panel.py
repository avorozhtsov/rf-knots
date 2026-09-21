"""Fixed-budget model-blind conjugate-block test generation; research note30."""

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil

from build_fresh_group_panel import EXCLUSIONS, atomic_json, excluded_groups, group, sha
from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_replay_corpus import canonical, invariants, lift, owner, verify_record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=20261010)
    parser.add_argument("--attempts",type=int,default=2000)
    parser.add_argument("--max-strands",type=int,default=8)
    parser.add_argument("--min-depth",type=int,default=4)
    parser.add_argument("--max-depth",type=int,default=16)
    parser.add_argument("--protocol",type=Path,default=Path("research/30-constructive-short-u-evaluation.md"))
    parser.add_argument("--exclude-panel",type=Path,action="append",default=[])
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.attempts < 1 or args.max_strands < 3 or args.min_depth < 0 or args.max_depth < args.min_depth:
        raise ValueError("invalid construction limits")
    if (args.seed,args.attempts,args.max_strands,args.min_depth,args.max_depth,args.exclude_panel)!=(20261010,2000,8,4,16,[]) and args.protocol==Path("research/30-constructive-short-u-evaluation.md"):
        raise ValueError("changed construction requires a separate declared protocol")
    args.out.mkdir(parents=True,exist_ok=args.resume)
    lock=(args.out/".runner.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    names=(*EXCLUSIONS,"embedding-fresh-group-test-v1","embedding-general-gordian-test-v1")
    sources=list(dict.fromkeys([Path("artifacts")/n/"panel.json" for n in names]+args.exclude_panel))
    code=[Path(__file__),Path("scripts/build_fresh_group_panel.py"),
        args.protocol,*[
        Path("src/rf_knots")/n for n in ("conjugate_witnesses.py","embedding_replay_corpus.py",
        "interleaved_generator.py","reference.py","actions.py","evidence.py","seifert.py")]]
    contract={"attempts":args.attempts,"seed":args.seed,
        "max_strands":args.max_strands,"min_depth":args.min_depth,"max_depth":args.max_depth,
        "inputs":{str(p):sha(p) for p in sources+code}}
    excluded=excluded_groups([json.loads(p.read_text()) for p in sources])
    old=json.loads((args.out/"progress.json").read_text()) if args.resume else None
    if old and old["contract"]!=contract:
        raise ValueError("resume inputs differ")
    if old and old["status"]=="completed":
        print("already completed")
        return
    if not old:
        snapshot=args.out/"source_snapshot";snapshot.mkdir()
        for i,p in enumerate(sources+code):
            shutil.copyfile(p,snapshot/f"{i:02d}-{p.name}")
        atomic_json(args.out/"excluded_groups.json",sorted(excluded))
    records=old["records"] if old else []
    counts=Counter(old["counts"] if old else {})
    groups,classes,seen=Counter(),Counter(),set()
    for r in records:
        w=verify_record(r);groups[group(r["left_fingerprint"])]+=1
        classes[str(w.crossing_changes)]+=1;seen.add(canonical(w.start.word,w.start.strands))
    index=old["next_index"] if old else 0

    def save(status):
        result={"status":status,"pid":os.getpid(),"contract":contract,"next_index":index,
            "records":records,"counts":dict(counts),"classes":dict(classes),"groups":len(groups)}
        atomic_json(args.out/"progress.json",result)
        print(json.dumps({k:result[k] for k in ("status","next_index","counts","classes","groups")}),flush=True)

    save("building")
    while index<contract["attempts"]:
        w=conjugate_witness(contract["seed"],index,max_strands=contract["max_strands"],
                            min_depth=contract["min_depth"],max_depth=contract["max_depth"])
        fp=invariants(w.start.word,w.start.strands);g=group(fp)
        key=canonical(w.start.word,w.start.strands);c=str(w.crossing_changes)
        reason=("determinant_one" if fp[0]==1 else "other_partition" if owner(fp)!="test"
            else "previous_group" if g in excluded else "duplicate" if key in seen
            else "group_cap" if groups[g]>=4 else "class_cap" if classes[c]>=32 else "accepted")
        if reason=="accepted":
            r={"split":"test","seed":contract["seed"],"index":index,
                "construction":"conjugate blocks in a destabilizable positive chain",
                "witness":w.to_dict(),"pair_prefix_length":w.moves,
                "left_fingerprint":list(fp),"right_fingerprint":[1,0],
                "interval":{"metric":"cc","left":w.start.to_dict(),
                    "right":w.steps[-1].after.to_dict(),"lower":abs(fp[1])//2,
                    "upper":w.crossing_changes,"witness_semantic_length":w.moves}}
            verify_record(r);records.append(r);groups[g]+=1;classes[c]+=1;seen.add(key)
        counts[reason]+=1;index+=1
        if index%100==0:
            save("building")
    save("writing_panel")
    large=[]
    for r in records:
        w,_=lift(verify_record(r),33,256)
        row={**r,"evaluation_view":"size_shift","witness":w.to_dict(),"pair_prefix_length":w.moves,
            "interval":{**r["interval"],"left":w.start.to_dict(),"witness_semantic_length":w.moves}}
        verify_record(row);large.append(row)
    adequate=len(groups)>=8 and len(records)>=32 and all(classes[c]>=8 for c in ("1","2"))
    atomic_json(args.out/"panel.json",{"schema":"rf-knots-gordian-pair-panel-v1","pair_kind":"short-u",
        "partition":"test","complete":adequate,"construction_complete":True,"contract":contract,
        "validation":records,"size_shift":large,"groups":len(groups),"classes":dict(classes),
        "caveat":"Synthetic conjugate-block distribution; correlated structure, CC counts are upper bounds."})
    save("completed")


if __name__=="__main__":
    main()
