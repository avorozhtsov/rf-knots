"""Replay-audited paired first-found costs for identical MCTS queries/budgets."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from rf_knots.mcts_loop import Instance
from rf_knots.evidence import UnknotWitness
from rf_knots.embedding_replay_corpus import verify_record


def validate(report, panel, panel_sha):
    if report["status"] != "completed" or report["panel_sha256"] != panel_sha:
        raise ValueError("require completed reports on the pinned panel")
    if not report["selected"] or report["config"]["seeds"] < 1:
        raise ValueError("empty selection/seeds")
    view = report["partition"].rsplit(":", 1)[-1]
    selected = {}
    for item in report["selected"]:
        reference = verify_record(panel[view][item["panel_index"]])
        instance = Instance(**{**item["resource_caps"], "word": tuple(item["resource_caps"]["word"])})
        uid = instance.uid
        fp = panel[view][item["panel_index"]]["left_fingerprint"]
        if uid in selected or item["fingerprint"] != [fp[0], abs(fp[1])]:
            raise ValueError("duplicate selection or fingerprint mismatch")
        if (instance.word, instance.strands) != (reference.start.word, reference.start.strands):
            raise ValueError("selected root differs from panel")
        if instance.cap_crossing_changes != reference.crossing_changes or instance.cc_weight != report["config"]["cc_weight"]:
            raise ValueError("resource CC contract differs")
        selected[uid] = (instance, item["fingerprint"])
    expected = {(uid, seed) for uid in selected for seed in range(report["config"]["seeds"])}
    rows = {}
    for row in report["search"]["rows"]:
        key = row["instance"], row["seed"]
        if key in rows or key not in expected:
            raise ValueError("duplicate or unexpected query/seed")
        instance, fp = selected[key[0]]
        if not 0 <= row["expansions"] <= row["simulations"] <= report["config"]["simulations"]:
            raise ValueError("search count exceeds budget")
        if row["solved"]:
            w = UnknotWitness.from_dict(row["witness"])
            w.verify()
            states = [w.start] + [s.after for s in w.steps]
            if (w.start.word, w.start.strands) != (instance.word, instance.strands):
                raise ValueError("returned witness root mismatch")
            if w.moves > instance.cap_moves or w.crossing_changes > instance.cap_crossing_changes or any(
                len(s.word) > instance.spec.max_len or s.strands > instance.spec.max_strands for s in states
            ):
                raise ValueError("returned witness exceeds resource caps")
            if row["cc"] != w.crossing_changes or row["moves"] != w.moves or any(
                row[f"L{cost}"] != cost*w.crossing_changes+w.moves for cost in (10,1000)
            ):
                raise ValueError("stored costs differ from replay")
        elif any(row[k] is not None for k in ("witness", "cc", "moves", "L10", "L1000")):
            raise ValueError("failure costs must remain censored")
        rows[key] = {**row, "fingerprint": fp}
    if set(rows) != expected or report["config"]["instances"] != len(selected):
        raise ValueError("query/seed coverage incomplete")
    if report["search"]["trees"] != len(rows) or report["search"]["solved"] != sum(r["solved"] for r in rows.values()):
        raise ValueError("summary differs from rows")
    return rows


def compare(baseline, candidate, panel, panel_sha, *, allow_proposal_change=False, allow_value_change=False):
    baseline_config, candidate_config = dict(baseline["config"]), dict(candidate["config"])
    if allow_proposal_change and allow_value_change:
        raise ValueError("do not combine proposal and value-mode interventions")
    comparison_kind = "matched_search_models"
    if allow_value_change:
        if baseline_config.pop("learned_value", None) is not False or candidate_config.pop("learned_value", None) is not True:
            raise ValueError("value comparison requires explicit constant-to-learned transition")
        for report, expected in ((baseline,"constant-0.5"),(candidate,"learned")):
            if (not report.get("checkpoint_sha256") or report.get("initial") is not False
                    or report.get("uniform") is not False or report["search"].get("leaf_value") != expected):
                raise ValueError("value comparison requires pinned trained states and matching leaf mode")
        comparison_kind = ("same_model_leaf_value" if baseline["checkpoint_sha256"]==candidate["checkpoint_sha256"]
                           else "changed_model_and_leaf_value")
    if allow_proposal_change:
        comparison_kind = "same_model_site_proposals"
        if baseline_config.pop("site_proposals", None) is not False or candidate_config.pop("site_proposals", None) is not True:
            raise ValueError("proposal comparison requires explicit catalogue baseline and site candidate")
        for key in ("checkpoint_sha256", "initial", "uniform", "initial_checkpoint_sha256", "initial_state_source"):
            if key not in baseline or key not in candidate or baseline[key] != candidate[key]:
                raise ValueError("proposal comparison requires identical pinned model states")
        if not baseline["checkpoint_sha256"] or baseline["uniform"]:
            raise ValueError("proposal comparison requires a pinned neural checkpoint")
        if baseline["initial"] and (not baseline["initial_checkpoint_sha256"] or baseline["initial_state_source"] != "saved_tensors"):
            raise ValueError("proposal comparison requires saved initial tensors")
    if baseline_config != candidate_config or baseline["selected"] != candidate["selected"] or baseline["partition"] != candidate["partition"]:
        raise ValueError("matched config/selection required")
    if baseline["code_sha256"] != candidate["code_sha256"]:
        raise ValueError("evaluation code differs")
    a, b = validate(baseline,panel,panel_sha), validate(candidate,panel,panel_sha)
    if baseline["config"]["cc_weight"] not in (10,1000):
        raise ValueError("expected L10 or L1000")
    cost = int(baseline["config"]["cc_weight"])
    paired, counts, deltas = [], Counter(), []
    for key in a:
        x,y = a[key][f"L{cost}"],b[key][f"L{cost}"]
        outcome = ("both_unsolved" if x is None and y is None else "newly_solved" if x is None
                   else "lost_solution" if y is None else "lower_cost" if y<x else "higher_cost" if y>x else "equal_cost")
        counts[outcome] += 1
        if x is not None and y is not None:
            deltas.append(y-x)
        paired.append({"instance":key[0],"seed":key[1],"fingerprint":a[key]["fingerprint"],
                       "baseline_cost":x,"candidate_cost":y,"outcome":outcome})
    groups = {}
    for r in paired:
        groups.setdefault(":".join(map(str,r["fingerprint"])), []).append(r)
    compute = {}
    for name, report in (("baseline", baseline), ("candidate", candidate)):
        search = report["search"]
        compute[name] = {k: search[k] for k in ("seconds", "evaluator_requests", "action_scores")}
        compute[name].update({k: search.get(k) for k in (
            "proposal_kind", "proposal_compute", "total_neural_state_evaluations")})
        proposal = search.get("proposal_compute")
        # Historical providers were not metered: unknown overhead is not zero.
        compute[name]["total_action_scores"] = (
            search["action_scores"] + proposal["action_scores"] if proposal is not None
            else search["action_scores"] if not search.get("proposal_kind") and not search.get("factorized_proposals")
            else None
        )
    return {"status":"passed","panel_sha256":panel_sha,"cc_weight":cost,
        "comparison_kind":comparison_kind,
        "checkpoint_sha256": {"baseline": baseline.get("checkpoint_sha256"),
                              "candidate": candidate.get("checkpoint_sha256")},
        "summary":{**dict(counts),"trees":len(paired),
            "baseline_solved":sum(r["solved"] for r in a.values()),"candidate_solved":sum(r["solved"] for r in b.values()),
            "common_solved":len(deltas),"mean_cost_delta_common_only":sum(deltas)/len(deltas) if deltas else None},
        "compute":compute,
        "groups":{g:dict(Counter(r["outcome"] for r in rows)) for g,rows in groups.items()},"paired":paired,
        "caveat":"First-found witnesses, not optimal costs. Equal simulations, not equal neural work; correlated seeds/groups. "
                 "CC cost conditions policy but does not change terminal reward or make MCTS anytime. "
                 "A joint model-and-leaf change does not isolate a causal value-head effect."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","candidate","panel","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--allow-proposal-change", action="store_true",
                        help="allow only catalogue-to-site proposal change; require identical pinned weights")
    parser.add_argument("--allow-value-change", action="store_true",
                        help="allow constant-to-learned leaf change; disclose whether the model also changed")
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite a comparison")
    inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.baseline,args.candidate,args.panel)}
    result=compare(json.loads(args.baseline.read_text()),json.loads(args.candidate.read_text()),
                   json.loads(args.panel.read_text()),inputs[str(args.panel)],
                   allow_proposal_change=args.allow_proposal_change, allow_value_change=args.allow_value_change)
    result["inputs"]=inputs
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix(".partial")
    temp.write_text(json.dumps(result,indent=2)+"\n");temp.replace(args.output)
    print(json.dumps(result["summary"]))


if __name__=="__main__":
    main()
