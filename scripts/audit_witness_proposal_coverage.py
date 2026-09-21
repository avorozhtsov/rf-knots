"""Reference-route catalogue ranks, NOT observed MCTS coverage or an impossibility claim."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from rf_knots.actions import INSERT
from rf_knots.embedding_replay_corpus import verify_record
from rf_knots.lazy_actions import LazyActionCatalogue
from rf_knots.mcts_loop import Instance


def catalogue_rank(catalogue, action):
    common=min(len(catalogue.ordinary),catalogue.insertions)
    if action in catalogue.ordinary:
        j=catalogue.ordinary.index(action)
        rank=2*j if j<common else j+common
    else:
        kind,position,generator,sign=catalogue.spec.decode(action)
        if kind!=INSERT or catalogue.insertions==0:
            raise ValueError("action not in catalogue")
        anchor=abs(catalogue.word[position])-1 if catalogue.word else 0
        shell=2*((generator-1-anchor)%catalogue.generators)+int(sign<0)
        insertion=shell*catalogue.positions+position
        rank=2*insertion+1 if insertion<common else insertion+common
    if catalogue.action_at(rank)!=action:
        raise ValueError("catalogue inverse index mismatch")
    return rank


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("panel","report","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite diagnostic")
    panel=json.loads(args.panel.read_text());report=json.loads(args.report.read_text())
    panel_sha=hashlib.sha256(args.panel.read_bytes()).hexdigest()
    if report["status"]!="completed" or report["panel_sha256"]!=panel_sha:
        raise ValueError("require completed report and matching panel")
    nominal=report["config"]["widening_batch"]*math.ceil(math.sqrt(report["config"]["simulations"]+1))
    if nominal<1:
        raise ValueError("diagnostic requires progressive widening")
    rows=[]
    for selected in report["selected"]:
        w=verify_record(panel["validation"][selected["panel_index"]])
        instance=Instance(**{**selected["resource_caps"],"word":tuple(selected["resource_caps"]["word"])})
        if (instance.word,instance.strands)!=(w.start.word,w.start.strands):
            raise ValueError("selected root differs from witness")
        state,used=w.start,0
        steps=[]
        for step in w.steps:
            catalogue=LazyActionCatalogue(instance.spec,state.word,state.strands,used<instance.cap_crossing_changes)
            action=step.action.to_flat(instance.spec)
            steps.append({"kind":step.action.kind,"rank_zero_based":catalogue_rank(catalogue,action),
                          "catalogue_size":len(catalogue)})
            used+=int(step.action.kind=="CROSSING_CHANGE");state=step.after
        rows.append({"panel_index":selected["panel_index"],"fingerprint":selected["fingerprint"],"steps":steps})
    result={"status":"completed","panel_sha256":panel_sha,
        "report_sha256":hashlib.sha256(args.report.read_bytes()).hexdigest(),"nominal_prefix":nominal,
        "roots":len(rows),"first_step_outside":sum(r["steps"][0]["rank_zero_based"]>=nominal for r in rows),
        "any_step_outside":sum(any(s["rank_zero_based"]>=nominal for s in r["steps"]) for r in rows),
        "rows":rows,"caveat":"Static ranks along one known witness, not observed search state or proof of unreachable solution. "
        "Force-widening may exceed the nominal schedule; dominance changes offered sets; alternative routes may exist."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix(".partial");temp.write_text(json.dumps(result,indent=2)+"\n");temp.replace(args.output)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}))


if __name__=="__main__":
    main()
