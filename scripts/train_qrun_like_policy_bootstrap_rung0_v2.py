"""Length-stratified v2 wrapper around the hash-pinned v1 optimizer loop."""

from pathlib import Path

import train_qrun_like_policy_bootstrap_rung0_v1 as old
from build_mcts_training_supervision import sha
from mcts_training_supervision import verified_success_traces
from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness


PROTOCOL = Path("research/280-qrun-like-bootstrap-retention-mix-v2.md")


def build_mixed_bank():
    roster = old.read(old.ROSTER)
    short = roster["stages"][0][:105]
    longer = (roster["stages"][1][:12] + roster["stages"][2][:12]
              + roster["stages"][3][:11])
    if len(short) != 105 or len(longer) != 35:
        raise ValueError("fixed v2 root supply differs")
    ordered = [root for index in range(35)
               for root in (*short[index * 3:index * 3 + 3], longer[index])]
    if len(ordered) != 140 or len({tuple(r["query_group"]) for r in ordered}) != 140:
        raise ValueError("v2 root mix is not 140 distinct groups")
    bank = []
    for root in ordered:
        witness = UnknotWitness.from_dict(root["witness"])
        witness.verify()
        paired = []
        for instance_dict in root["instances"]:
            instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
            trace = verified_success_traces(instance, [witness])[0]
            paired.append((instance, trace["rows"]))
        if tuple(row[0].cc_weight for row in paired) != (10, 1000):
            raise ValueError("resource pair differs")
        bank.append(paired)
    return bank


def main():
    # These extra CONFIG fields are included in v1's immutable contract
    # digest and therefore pin the actual v2 entrypoint and new protocol.
    old.CONFIG["source_v2_sha256"] = sha(Path(__file__))
    old.CONFIG["protocol_v2_sha256"] = sha(PROTOCOL)
    old.CONFIG["root_mix"] = {"stage0": 105, "stage1": 12,
                              "stage2": 12, "stage3": 11}
    old.build_bank = build_mixed_bank
    old.main()


if __name__ == "__main__":
    main()
