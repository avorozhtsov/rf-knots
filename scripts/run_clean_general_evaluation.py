"""Wait for the two existing clean runs, gate final artifacts, score note29 once."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def live(pid, expected):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    return result.returncode == 0 and expected in result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    lock = (args.out / ".runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runs = [Path("artifacts/embedding-multisize-clean-" + name + "-v1") for name in ("joint", "policy")]
    panel = Path("artifacts/embedding-general-gordian-test-v1/panel.json")
    audit = panel.parent / "independence-audit.json"
    corpus = Path("artifacts/embedding-multisize-clean-corpus-v1/corpus.json")
    evaluator = Path("scripts/evaluate_strand_graph_panel.py")
    code = [Path(__file__), evaluator, Path("scripts/run_strand_graph_pilot.py"), *[
        Path("src/rf_knots") / n for n in ("embedding_evaluation.py", "embedding_checkpoint_io.py",
        "embedding_replay_corpus.py", "embedding_metric.py", "embedding_encoder.py", "strand_graph_encoder.py",
        "embedding_normalization.py", "site_head.py", "mcts_loop.py", "reference.py", "actions.py", "seifert.py")]]
    inputs = {str(p): sha(p) for p in [panel, corpus, *code]}
    state = {"status": "waiting", "pid": os.getpid(), "inputs": inputs,
             "runs": [str(p) for p in runs], "reports": []}

    def save():
        temp = args.out / "manifest.partial"
        temp.write_text(json.dumps(state, indent=2)+"\n")
        temp.replace(args.out / "manifest.json")

    save()
    while True:
        manifests = [read(p / "manifest.json") for p in runs]
        for p,m in zip(runs, manifests, strict=True):
            if m["status"] != "completed" and not live(m["pid"], f"--out {p}"):
                raise RuntimeError(f"training writer missing for {p}; not restarting it")
        if all(m["status"] == "completed" for m in manifests):
            if not audit.exists():
                raise RuntimeError("training completed but audit absent; score nothing")
            break
        state["observed_steps"] = [m.get("completed_steps", 0) for m in manifests]
        save()
        time.sleep(15)
    if any(sha(Path(p)) != h for p,h in inputs.items()):
        raise RuntimeError("evaluation inputs changed while waiting")
    evidence = read(audit)
    if evidence["status"] != "passed" or evidence["inputs"][str(panel)] != sha(panel) or evidence["inputs"][str(corpus)] != sha(corpus):
        raise RuntimeError("independent audit does not match inputs")
    if evidence["query_train_path_intersection"] or evidence["query_previous_panel_intersection"]:
        raise RuntimeError("nonempty query overlap")
    if not read(panel)["complete"]:
        raise RuntimeError("inadequate panel")
    if any(m.get("completed_steps") != 4096 for m in manifests):
        raise RuntimeError("requires declared final step4096")
    configs = [{k:v for k,v in m["config"].items() if k not in ("out", "metric_weight")} for m in manifests]
    if configs[0] != configs[1] or [m["config"]["metric_weight"] for m in manifests] != [1,0]:
        raise RuntimeError("comparison configurations differ")
    if manifests[0]["code_sha256"] != manifests[1]["code_sha256"]:
        raise RuntimeError("training source snapshots differ")
    for p,m in zip(runs,manifests,strict=True):
        if m["source_corpus_sha256"] != sha(corpus) or sha(p/"checkpoint.pt") != m["checkpoint_sha256"]:
            raise RuntimeError("final checkpoint/corpus mismatch")
        if sha(p/"initial.pt") != m["initial_checkpoint_sha256"]:
            raise RuntimeError("initial checkpoint mismatch")
    if manifests[0]["initial_checkpoint_sha256"] != manifests[1]["initial_checkpoint_sha256"]:
        raise RuntimeError("initial states differ")
    state.update(status="evaluating", audit_sha256=sha(audit),
                 final_checkpoint_sha256=[m["checkpoint_sha256"] for m in manifests])
    save()
    for name,run,initial in [("joint",runs[0],False),("policy",runs[1],False),("initial",runs[0],True)]:
        output=args.out/f"{name}.json"
        command=[sys.executable,str(evaluator),"--run",str(run),"--panel",str(panel),"--output",str(output)]
        if initial:
            command.append("--initial")
        with (args.out/f"{name}.log").open("x") as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=read(output)
        if report["status"] != "completed" or report["panel_sha256"] != sha(panel):
            raise RuntimeError("evaluation report mismatch")
        state["reports"].append({"name":name,"file":output.name,"sha256":sha(output)})
        save()
        print(json.dumps({"completed":name,"interval_error":report["validation"]["mean_interval_violation"]}),flush=True)
    state["status"]="completed"
    save()


if __name__ == "__main__":
    main()
