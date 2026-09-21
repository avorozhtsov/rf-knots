"""Run note33's frozen evaluation once both fixed-step training arms finish.

Resumption adopts a matching live child or verifies completed artifacts. Missing
partial evaluators are never silently restarted. No training process is started,
stopped, or modified by this controller.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import torch


BASE=Path("artifacts")
RUNS={"parent":BASE/"embedding-multisize-clean-joint-v1",
      **{n:BASE/f"embedding-clean-paired-{n}-v1" for n in ("value","control")}}
DATA=BASE/"embedding-clean-paired-rollouts-v1"
PANELS={n:BASE/f"embedding-clean-paired-{n}-test-v1"/"panel.json" for n in ("gordian","short-u")}


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):
    temp=path.with_suffix(".partial");temp.write_text(json.dumps(data,indent=2)+"\n");temp.replace(path)


def jobs(out,groups,roots):
    result=[]
    def add(name,script,args,output):
        result.append({"name":name,"command":[sys.executable,"-u",f"scripts/{script}.py",*map(str,args)],
                       "output":str(output)})
    for model,run in RUNS.items():
        for panel,bounds in (("gordian","original"),("short-u","original"),("short-u","determinant")):
            name=f"metric-{model}-{panel}-{bounds}";output=out/"metric"/f"{name}.json"
            add(name,"evaluate_strand_graph_panel",["--run",run,"--panel",PANELS[panel],
                "--bounds",bounds,"--output",output],output)
    for model in ("parent","value","control","length"):
        target=out/"cost"/model
        add(f"cost-{model}","evaluate_embedding_cost_search",["--run",RUNS.get(model,RUNS["parent"]),
            "--panel",PANELS["short-u"],"--out",target,"--mode","length" if model=="length" else "learned",
            "--expansions",128,"--witness-caps","--all-queries","--layout-tiebreak"],target/"report.json")
    modes=(("parent",False),("value",False),("value",True),("control",False),("control",True))
    for cost in (10,1000):
        for model,learned in modes:
            name=f"mcts-{model}-{'learned' if learned else 'constant'}-{cost}"
            output=out/"mcts"/name/"report.json"
            add(name,"evaluate_strand_graph_search",["--run",RUNS[model],"--panel",PANELS["short-u"],
                "--output",output,"--instances",groups,"--seeds",2,"--simulations",512,
                "--widening-batch",4,"--witness-caps","--cc-weight",cost,
                *(["--learned-value"] if learned else [])],output)
    for baseline,candidate in (("parent","value"),("parent","control"),("control","value"),("length","value")):
        name=f"cost-{candidate}-vs-{baseline}";output=out/"comparisons"/f"{name}.json"
        add(name,"compare_embedding_cost_search",["--baseline",out/"cost"/baseline/"report.json",
            "--candidate",out/"cost"/candidate/"report.json","--panel",PANELS["short-u"],"--output",output],output)
    for cost in (10,1000):
        for baseline,candidate,change in (("parent-constant","value-constant",False),
                ("parent-constant","control-constant",False),("control-learned","value-learned",False),
                ("value-constant","value-learned",True),("control-constant","control-learned",True),
                ("parent-constant","value-learned",True)):
            output=out/"comparisons"/f"mcts-{candidate}-vs-{baseline}-{cost}.json"
            add(output.stem,"compare_matched_mcts",["--baseline",out/"mcts"/f"mcts-{baseline}-{cost}"/"report.json",
                "--candidate",out/"mcts"/f"mcts-{candidate}-{cost}"/"report.json","--panel",PANELS["short-u"],
                "--output",output,*(["--allow-value-change"] if change else [])],output)
    rollout=out/"parent-rollouts"
    add("parent-rollouts","collect_rollout_value",["--run",RUNS["parent"],"--evaluation-panel",PANELS["short-u"],
        "--out",rollout,"--roots",roots,"--rollouts",8,"--seed",20261016],rollout/"manifest.json")
    for cost in (10,1000):
        output=out/"comparisons"/f"portfolio-vs-parent-{cost}.json"
        add(output.stem,"compare_rollout_mcts",["--data",rollout,
            "--mcts-report",out/"mcts"/f"mcts-parent-constant-{cost}"/"report.json","--output",output],output)
    return result


def check_training_manifests(manifests):
    for m in manifests:
        if m["status"]!="completed" or m["completed_steps"]!=2048:
            raise ValueError("require both declared final step2048 checkpoints")
    configs=[{k:v for k,v in m["config"].items() if k not in ("out","value_weight")} for m in manifests]
    if configs[0]!=configs[1] or [m["config"]["value_weight"] for m in manifests]!=[1,0]:
        raise ValueError("training arms are not matched")
    if configs[0]["seed"]!=20261013 or configs[0]["steps"]!=2048:
        raise ValueError("training seed/step contract differs")
    for key in ("sources","inputs","architecture","initial_checkpoint_sha256"):
        if manifests[0][key]!=manifests[1][key]:raise ValueError(f"training {key} differ")
    for m in manifests:
        if m["relative_groups"]<32 or m["long_relative_groups"]<8:
            raise ValueError("supervision gate not met")


def check_completed_prefix(completed,planned):
    if len(completed)>len(planned) or any(
        row["name"]!=job["name"] or row["output"]!=job["output"]
        for row,job in zip(completed,planned)
    ):
        raise ValueError("completed jobs must be the exact planned prefix")


def pin_checkpoints(state,current):
    if "final_checkpoints" in state and state["final_checkpoints"]!=current:
        raise ValueError("previously pinned final checkpoints changed")
    state["final_checkpoints"]=current


def processes():
    rows=subprocess.run(["ps","-axo","pid=,command="],capture_output=True,text=True,check=True).stdout.splitlines()
    result=[]
    for row in rows:
        pid,command=row.strip().split(None,1)
        try:argv=shlex.split(command)
        except ValueError:continue
        result.append((int(pid),argv,command))
    return result


def matching_children(job,listing):
    return [pid for pid,argv,_ in listing if argv[1:]==job["command"][1:]
            and Path(argv[0]).name.lower().startswith("python")]


def training_live(listing,launcher_pid):
    expected=str(DATA)
    outputs={str(RUNS[n]) for n in ("value","control")}
    def option(argv,name):
        return argv[argv.index(name)+1] if name in argv and argv.index(name)+1<len(argv) else None
    return any((Path(argv[0]).name.lower().startswith("python") and "scripts/train_rollout_value.py" in argv
                and option(argv,"--data")==expected and option(argv,"--out") in outputs)
               or (pid==launcher_pid and "for arm in value control" in command and expected in command)
               for pid,argv,command in listing if argv)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--training-launcher-pid",type=int,required=True)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=args.resume)
    lock=(args.out/"writer.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    panel=read(PANELS["short-u"])
    groups=len({(r["left_fingerprint"][0],abs(r["left_fingerprint"][1])) for r in panel["validation"]})
    planned=jobs(args.out,groups,len(panel["validation"]))
    scripts={Path(j["command"][2]) for j in planned}|{Path(__file__),Path("scripts/run_strand_graph_pilot.py")}
    source=scripts|set(Path("src/rf_knots").rglob("*.py"))
    paths=source|{Path("research/33-clean-paired-cost-rollout-protocol.md"),
        Path("research/35-clean-paired-evaluation-execution.md"),RUNS["parent"]/"checkpoint.pt",
        DATA/"manifest.json",DATA/"actor.pt",DATA/"roots.json",DATA/"training-partition-audit.json"}
    paths.update(PANELS.values());paths.update(p.parent/"independence-audit.json" for p in PANELS.values())
    inputs={str(path):sha(path) for path in sorted(paths)}
    manifest=args.out/"manifest.json"
    state=read(manifest) if args.resume else {"status":"waiting_for_training","inputs":inputs,"jobs":planned,"completed":[]}
    if state["inputs"]!=inputs or state["jobs"]!=planned:raise ValueError("controller resume inputs/jobs changed")
    check_completed_prefix(state["completed"],planned)
    for row in state["completed"]:
        if sha(Path(row["output"]))!=row["sha256"]:raise ValueError("completed output changed")
    if state["status"]=="completed":print("already completed");return
    state["pid"]=os.getpid()
    save(manifest,state)
    if not args.resume:
        for path in source:
            target=args.out/"source_snapshot"/path.resolve().relative_to(Path.cwd())
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    try:
        for path in PANELS.values():
            audit=read(path.parent/"independence-audit.json")
            if not read(path)["complete"] or audit["status"]!="passed" or audit["inputs"][str(path)]!=sha(path):
                raise ValueError("panel audit mismatch")
        rollout_audit=read(DATA/"training-partition-audit.json")
        if rollout_audit["status"]!="passed" or rollout_audit["query_rollout_intersection"]:
            raise ValueError("rollout partition audit failed")
        for name,h in rollout_audit["inputs"].items():
            if sha(Path(name))!=h:raise ValueError(f"rollout audit input changed: {name}")
        while True:
            manifests=[read(RUNS[n]/"manifest.json") if (RUNS[n]/"manifest.json").exists() else None for n in ("value","control")]
            if all(m and m["status"]=="completed" for m in manifests):break
            if not training_live(processes(),args.training_launcher_pid):
                raise RuntimeError("training sequence absent before completion; no automatic restart")
            observed=[None if m is None else {k:m.get(k) for k in ("status","completed_steps","pid")} for m in manifests]
            if state.get("training")!=observed:
                state.update(status="waiting_for_training",training=observed);save(manifest,state)
            time.sleep(15)
        check_training_manifests(manifests)
        checkpoints={}
        for name,m in zip(("value","control"),manifests,strict=True):
            run=RUNS[name]
            if sha(run/"checkpoint.pt")!=m["checkpoint_sha256"] or sha(run/"initial.pt")!=m["initial_checkpoint_sha256"]:
                raise ValueError("training tensor hash mismatch")
            if any(sha(Path(n))!=h for n,h in {**m["inputs"],**m["sources"]}.items()):
                raise ValueError("training provenance changed")
            c=torch.load(run/"checkpoint.pt",map_location="cpu",weights_only=False)
            if c["steps"]!=2048 or any(not torch.isfinite(v).all() for key in ("model","objective") for v in c[key].values()):
                raise ValueError("invalid final tensors")
            checkpoints[name]=c
        if checkpoints["value"]["rng"]!=checkpoints["control"]["rng"] or not torch.equal(checkpoints["value"]["torch_rng"],checkpoints["control"]["torch_rng"]):
            raise ValueError("training sampling streams differ")
        pin_checkpoints(state,{n:sha(RUNS[n]/"checkpoint.pt") for n in RUNS})
        (args.out/"logs").mkdir(exist_ok=True)
        for job in planned[len(state["completed"]):]:
            if any(sha(Path(n))!=h for n,h in inputs.items()):raise ValueError("evaluation inputs changed")
            output=Path(job["output"])
            live=matching_children(job,processes())
            if len(live)>1:raise RuntimeError("duplicate evaluation writers")
            prior=state.get("current",{}).get("name")==job["name"]
            if prior and not live and not output.exists():
                raise RuntimeError("previous evaluator disappeared before writing output; no automatic restart")
            if not live and output.exists() and not prior:
                raise RuntimeError("unexpected existing evaluation; refusing duplicate/overwrite")
            child=None
            if not live and not output.exists():
                with (args.out/"logs"/f"{job['name']}.log").open("xb") as log:
                    child=subprocess.Popen(job["command"],stdout=log,stderr=subprocess.STDOUT)
                live=[child.pid]
            state.update(status="evaluating",current={**job,"pid":live[0] if live else None});save(manifest,state)
            if child is not None:
                if child.wait()!=0:raise RuntimeError(f"evaluation failed: {job['name']}; inspect log")
            else:
                while matching_children(job,processes()):time.sleep(15)
            report=read(output)
            if report.get("status") not in ("completed","passed"):
                raise RuntimeError("partial evaluation left intact; no automatic restart")
            if any(sha(Path(n))!=h for n,h in inputs.items()) or any(
                sha(RUNS[n]/"checkpoint.pt")!=h for n,h in state["final_checkpoints"].items()
            ):raise ValueError("evaluation source/input/checkpoint changed while running")
            state["completed"].append({"name":job["name"],"output":str(output),"sha256":sha(output)})
            state.pop("current",None);save(manifest,state)
            print(json.dumps({"completed":job["name"],"jobs":len(state["completed"]),"total":len(planned)}),flush=True)
        state["status"]="completed";save(manifest,state)
    except Exception as error:
        state.update(status="attention_required",error=str(error));save(manifest,state);raise


if __name__=="__main__":main()
