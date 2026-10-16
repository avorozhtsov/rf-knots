"""Select one verified equal-cost TRAIN alternative per root/objective."""

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from audit_qrun_rung1_onpolicy_teacher_v1 import owned
from build_f_curriculum_roster_v1 import excluded_groups
from build_mcts_training_supervision import atomic, sha
from rf_knots import mcts_loop as M
from rf_knots import embedding_replay_corpus as corpus
from rf_knots.evidence import UnknotWitness

ROSTER = Path("artifacts/f-curriculum-roster-v3/roster.json")
TEACHER = Path("artifacts/qrun-rung1-onpolicy-teacher-v1")
AUDIT = Path("artifacts/qrun-rung1-onpolicy-teacher-audit-v2/report.json")
PROTOCOL = Path("research/292-qrun-equal-cost-diversity-pilot-v1.md")
ROSTER_SHA = "268f1c4d2e8731bcfd4cc743bed767c9864fb74b2a2022ed679fbe36af134506"
TEACHER_MANIFEST_SHA = "658c128c11fed26a639994c5d54af8ad31550f7f189be52c144cbb76c192f6a2"
AUDIT_SHA = "ee1049c12bdaca9142d13c2a8759e6bba91fe3c10d34815e1ddff3e58e1bbb86"


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new bank output must not exist")
    if (sha(ROSTER) != ROSTER_SHA or sha(TEACHER / "manifest.json") != TEACHER_MANIFEST_SHA
            or sha(AUDIT) != AUDIT_SHA):
        raise ValueError("pinned TRAIN/audit input differs")
    teacher = json.loads((TEACHER / "manifest.json").read_text())
    report = json.loads(AUDIT.read_text())
    if (teacher["status"] != "completed" or teacher["next_task"] != 2000
            or report["audited_shards"] != 2000):
        raise ValueError("teacher or independent audit incomplete")
    roots = [row for stage in json.loads(ROSTER.read_text())["stages"] for row in stage]
    if len(roots) != 1000:
        raise ValueError("roster coverage differs")
    # The independent audit already checked every saved ownership flag.
    # Recompute ownership for each selected alternative as a second gate.
    corpus.invariants = lru_cache(maxsize=20_000)(corpus.invariants)
    import audit_qrun_rung1_onpolicy_teacher_v1 as audit_module
    audit_module.invariants = corpus.invariants
    excluded = excluded_groups()
    rows = []
    counts = [[0, 0] for _ in range(4)]
    for root_index, root in enumerate(roots):
        reference = UnknotWitness.from_dict(root["witness"])
        reference.verify()
        reference_actions = tuple(step.action for step in reference.steps)
        alternatives = []
        for weight_index, weight in enumerate((10, 1000)):
            position = 2 * root_index + weight_index
            entry = teacher["shards"][position]
            path = TEACHER / f"condition-{position:04d}.json"
            if entry["file"] != path.name or sha(path) != entry["sha256"]:
                raise ValueError(f"teacher shard identity differs: {position}")
            data = json.loads(path.read_text())
            instance_dict = root["instances"][weight_index]
            instance = M.Instance(**{**instance_dict,
                                      "word": tuple(instance_dict["word"])})
            reference_cost = weight * reference.crossing_changes + reference.moves
            candidate = None
            for episode, admitted in zip(data["result"]["episodes"],
                                         data["result"]["episodes_train_owned"], strict=True):
                if not admitted or episode["continuation"]["stop_reason"] != "solved":
                    continue
                end = episode["continuation"]["end"]
                if weight * end["used_cc"] + end["used_moves"] != reference_cost:
                    continue
                actions = tuple(episode["tree_prefix"] + episode["continuation"]["actions"])
                witness = UnknotWitness.from_actions(instance.word, instance.strands,
                                                      instance.spec, actions)
                if tuple(step.action for step in witness.steps) == reference_actions:
                    continue
                payload = witness.to_dict()
                key = canonical_sha(payload)
                if candidate is None or key < candidate[0]:
                    candidate = (key, payload, episode)
            if candidate is None:
                alternatives.append(None)
                continue
            _, payload, episode = candidate
            witness = UnknotWitness.from_dict(payload)
            witness.verify()
            if ((witness.start.word, witness.start.strands) !=
                    (instance.word, instance.strands)
                    or witness.crossing_changes > instance.cap_crossing_changes
                    or witness.moves > instance.cap_moves
                    or weight * witness.crossing_changes + witness.moves != reference_cost
                    or not owned(instance, episode, excluded)):
                raise ValueError(f"selected alternative failed replay/ownership: {position}")
            alternatives.append(payload)
            counts[root["stage"]][weight_index] += 1
        rows.append({"root_index": root_index, "generator_index": root["generator_index"],
                     "stage": root["stage"], "alternatives": alternatives})
    pins = (ROSTER, TEACHER / "manifest.json", AUDIT, PROTOCOL, Path(__file__),
            Path("scripts/audit_qrun_rung1_onpolicy_teacher_v1.py"))
    payload = {"schema": "rf-knots-qrun-equal-cost-bank-v1",
               "selection": "minimum canonical witness SHA per root/objective",
               "inputs": {str(path): sha(path) for path in pins},
               "counts_by_stage_L10_L1000": counts, "rows": rows}
    args.out.mkdir(parents=True)
    atomic(args.out / "bank.json", payload)
    print(json.dumps({"bank_sha256": sha(args.out / "bank.json"),
                      "counts_by_stage_L10_L1000": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
