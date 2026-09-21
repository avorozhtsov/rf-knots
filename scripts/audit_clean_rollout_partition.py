"""Independently replay note33 episodes and audit optimizer-eligible path groups."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.mcts_loop import Instance
from rf_knots.rollout_value import episode_states, verify_episode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--panel",type=Path,action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError("refusing to overwrite an audit")
    inputs={str(path):sha(path) for path in [args.data/"manifest.json",Path(__file__)]}
    data=json.loads((args.data/"manifest.json").read_text())
    if data["status"]!="completed" or data.get("purpose")!="training":
        raise ValueError("require complete training collection")
    inputs.update(data["inputs"])
    inputs.update(data["sources"])
    for field,file in (("roots_sha256","roots.json"),("actor_sha256","actor.pt")):
        inputs[str(args.data/file)]=data[field]
    queries=set()
    panel_counts={}
    for path in args.panel:
        audit_path=path.parent/"independence-audit.json"
        inputs.update({str(path):sha(path),str(audit_path):sha(audit_path)})
        panel=json.loads(path.read_text());audit=json.loads(audit_path.read_text())
        if not panel["complete"] or audit["status"]!="passed" or audit["inputs"][str(path)]!=sha(path):
            raise ValueError("panel metric-path audit absent or mismatched")
        corpus=data["config"]["corpus"]
        if audit["inputs"][corpus]!=data["inputs"][corpus]:
            raise ValueError("panel audit uses a different metric corpus")
        groups={(r["left_fingerprint"][0],abs(r["left_fingerprint"][1]))
                for view in ("validation","size_shift") for r in panel[view]}
        if any(owner(fp)!="test" for fp in groups):raise ValueError("panel query is not test-owned")
        queries.update(groups);panel_counts[str(path)]=len(groups)
    for name,digest in inputs.items():
        if sha(Path(name))!=digest:raise ValueError(f"input SHA mismatch: {name}")
    roots=json.loads((args.data/"roots.json").read_text())
    instances=[Instance(**{**r["instance"],"word":tuple(r["instance"]["word"])}) for r in roots]
    if len({p.uid for p in instances})!=len(instances):raise ValueError("duplicate root UID")
    pairs=defaultdict(list)
    for root in roots:pairs[root["base_root_index"]].append(root)
    if set(pairs)!=set(range(data["config"]["roots"])):raise ValueError("missing base roots")
    for pair in pairs.values():
        if len(pair)!=2 or sorted(r["instance"]["cc_weight"] for r in pair)!=[10,1000]:
            raise ValueError("missing paired cost condition")
        a,b=pair
        stripped=lambda r:{k:v for k,v in r["instance"].items() if k not in ("uid","cc_weight")}
        if stripped(a)!=stripped(b) or a["witness"]!=b["witness"]:
            raise ValueError("paired roots/resources differ")
    seen=Counter();counts=Counter();training_groups=set()
    for shard in data["shards"]:
        path=args.data/shard["file"]
        inputs[str(path)]=shard["sha256"]
        if sha(path)!=shard["sha256"]:raise ValueError("episode shard changed")
        for e in json.loads(path.read_text()):
            group,seed=e["group"],e["seed"]
            if not 0<=group<len(roots):raise ValueError("unexpected group")
            seen[group,seed]+=1
            problem=instances[group]
            verify_episode(e,problem)
            counts["episodes_replayed"]+=1
            counts["transitions_replayed"]+=e["moves"]
            counts["successes_replayed"]+=int(e["success"])
            if not e["eligible"] or seed>=6:continue
            counts["optimizer_eligible_episodes"]+=1
            counts["optimizer_eligible_successes"]+=int(e["success"])
            fp=invariants(problem.word,problem.strands)
            if list(fp)!=roots[group]["fingerprint"]:raise ValueError("root fingerprint differs")
            path_groups={fp}
            for state,action in episode_states(e,problem):
                if problem.spec.decode(action)[0]==CROSSING_CHANGE:
                    word,n=reference.apply(problem.spec,state.word,state.strands,action)
                    path_groups.add(invariants(word,n));counts["train_cc_endpoints_recomputed"]+=1
            if any(owner(fp)!="train" for fp in path_groups):
                raise ValueError("non-train group on an optimizer-eligible path")
            training_groups.update((d,abs(s)) for d,s in path_groups)
        print(json.dumps(dict(counts)),flush=True)
    expected={(g,s) for g in range(len(roots)) for s in range(data["config"]["rollouts"])}
    if set(seen)!=expected or any(n!=1 for n in seen.values()):
        raise ValueError("missing/duplicate episode coverage")
    for stored,replayed in (("episodes","episodes_replayed"),("transitions","transitions_replayed"),
                             ("successes","successes_replayed")):
        if data["counts"][stored]!=counts[replayed]:raise ValueError("collection summary differs from replay")
    if queries & training_groups:raise ValueError("test query overlaps eligible rollout state")
    if any(sha(Path(n))!=h for n,h in inputs.items()):raise ValueError("audit inputs changed")
    result={"status":"passed","inputs":inputs,"counts":dict(counts),
        "panel_query_group_counts":panel_counts,"union_query_groups":len(queries),
        "optimizer_path_groups":sorted(training_groups),"query_rollout_intersection":[],
        "caveat":"Seeds0..5 only; value accepts eligible episodes and policy a subset of complete groups. "
                 "All actions replayed; only CC endpoints need new invariants because zero-CC moves preserve them. "
                 "Fingerprint groups are not unique knot identities. This is not a model-quality result."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix(".partial");temp.write_text(json.dumps(result,indent=2)+"\n");temp.replace(args.output)
    print(json.dumps({"status":"passed",**dict(counts),"training_groups":len(training_groups)}),flush=True)


if __name__=="__main__":main()
