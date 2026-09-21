"""Verify every added positive action against its exact witness continuation."""

import argparse
import json
from collections import Counter
from pathlib import Path

from build_mcts_training_supervision import atomic, digest, sha
from mcts_training_diamonds import reorder_reduce_prefix

from rf_knots import mcts_loop as M
from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.cost_mcts_rollout import advance
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import SemanticAction, UnknotWitness


def state_from_dict(row):
    return M.SearchState(tuple(row["word"]), row["strands"], row["used_cc"], row["used_moves"])


def verify_certificate(instance, certificate, witness, states, actions, fps, known_cost, heldout):
    offset, span = certificate["suffix_index"], certificate["span"]
    assert 0 <= offset < len(actions) and 1 <= span <= len(actions) - offset
    state = states[offset]
    first = SemanticAction.from_dict(certificate["first_action"]).to_flat(instance.spec)
    method = certificate["method"]
    if method == "same-successor":
        assert span == 1
        alternate = (first,)
    elif method == "adjacent-diamond":
        assert span == 2
        alternate = tuple(
            SemanticAction.from_dict(a).to_flat(instance.spec)
            for a in certificate["alternate_actions"]
        )
        assert len(alternate) == 2
    elif method == "independent-reduce-prefix":
        alternate = reorder_reduce_prefix(instance, states, actions, offset, span, fps[offset])
        assert alternate is not None and len(alternate) == span
    else:
        raise ValueError("unknown certificate method")
    assert alternate[0] == first
    assert (
        instance.cc_weight * (witness.crossing_changes - state.used_cc)
        + witness.moves
        - state.used_moves
    ) == known_cost
    word, strands, cc, moves = state.word, state.strands, state.used_cc, state.used_moves
    fp = fps[offset]
    for j, action in enumerate(alternate):
        assert moves < instance.cap_moves
        assert reference.is_legal(
            instance.spec, word, strands, action, cc < instance.cap_crossing_changes
        )
        kind = instance.spec.decode(action)[0]
        word, strands = reference.apply(instance.spec, word, strands, action)
        cc += int(kind == CROSSING_CHANGE)
        moves += 1
        if kind == CROSSING_CHANGE:
            fp = invariants(word, strands)
        assert owner(fp) == "train" and (fp[0], abs(fp[1])) not in heldout
        if method == "adjacent-diamond" and j == 0:
            assert list(fp) == certificate["intermediate_fingerprint"]
    assert M.SearchState(word, strands, cc, moves) == states[offset + span]
    return state, first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--augmented", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite certificate audit")
    base = json.loads((args.base / "manifest.json").read_text())
    aug = json.loads((args.augmented / "manifest.json").read_text())
    assert base["status"] == aug["status"] == "completed"
    pins = {
        **aug["inputs"],
        str(args.augmented / "manifest.json"): sha(args.augmented / "manifest.json"),
        str(Path(__file__).resolve()): sha(Path(__file__)),
        str(args.augmented / "short-u-views.json"): base["metric_sha256"],
    }
    assert pins[str(args.base / "manifest.json")] == sha(args.base / "manifest.json")
    assert pins[str(args.data / "manifest.json")] == sha(args.data / "manifest.json")
    assert aug["metric_sha256"] == base["metric_sha256"]
    for item in aug["conditions"] + aug["certificates"]:
        pins[str(args.augmented / item["file"])] = item["sha256"]
    assert all(sha(Path(p)) == h for p, h in pins.items())
    collection = json.loads((args.data / "manifest.json").read_text())
    heldout = set(map(tuple, collection["heldout_query_groups"]))
    registry = json.loads((args.augmented / "short-u-views.json").read_text())["witnesses"]
    sources, traces = {}, {}
    for key, payload in registry.items():
        assert digest(payload) == key
        sources[key] = UnknotWitness.from_dict(payload)
        sources[key].verify()
    counts = Counter()
    assert len(aug["conditions"]) == len(aug["certificates"]) == len(base["conditions"])
    for before, after, cert_info in zip(
        base["conditions"], aug["conditions"], aug["certificates"], strict=True
    ):
        index = counts["conditions"]
        assert before["condition"] == after["condition"] == cert_info["condition"] == index
        b = json.loads((args.base / before["file"]).read_text())
        a = json.loads((args.augmented / after["file"]).read_text())
        assert {k: v for k, v in b.items() if k not in ("positives", "preferences")} == {
            k: v for k, v in a.items() if k not in ("positives", "preferences")
        }
        instance = M.Instance(**{**a["instance"], "word": tuple(a["instance"]["word"])})
        assert len(a["positives"]) == len(b["positives"])
        targets, needed = {}, set()
        for old, new in zip(b["positives"], a["positives"], strict=True):
            assert (
                old["state"] == new["state"]
                and old["known_suffix_cost"] == new["known_suffix_cost"]
            )
            state = state_from_dict(old["state"])
            assert new["actions"] == sorted(set(new["actions"]))
            assert new["positive_actions"] == sorted(set(new["positive_actions"]))
            assert (
                set(old["positive_actions"]) <= set(new["positive_actions"]) <= set(new["actions"])
            )
            assert set(old["actions"]) <= set(new["actions"])
            targets[state] = new
            needed.update(
                (state, action)
                for action in set(new["positive_actions"]) - set(old["positive_actions"])
            )
            counts["multi_positive_states"] += len(new["positive_actions"]) > 1
            counts["positive_action_slots"] += len(new["positive_actions"])
            counts["maximum_positive_set"] = max(
                counts["maximum_positive_set"], len(new["positive_actions"])
            )
        certified = set()
        certificates = json.loads((args.augmented / cert_info["file"]).read_text())
        assert len(certificates) == cert_info["count"]
        for certificate in certificates:
            assert certificate["condition"] == index
            witness = sources[certificate["witness_id"]]
            assert witness.start.word == instance.word and witness.start.strands == instance.strands
            key = (
                certificate["witness_id"],
                instance.spec.max_len,
                instance.spec.max_strands,
                instance.cap_crossing_changes,
                instance.cap_moves,
            )
            if key not in traces:
                actions = tuple(s.action.to_flat(instance.spec) for s in witness.steps)
                states = [M.SearchState(instance.word, instance.strands, 0, 0)]
                fps = [invariants(instance.word, instance.strands)]
                for action in actions:
                    states.append(advance(instance, states[-1], action))
                    fps.append(
                        invariants(states[-1].word, states[-1].strands)
                        if instance.spec.decode(action)[0] == CROSSING_CHANGE
                        else fps[-1]
                    )
                assert all(owner(fp) == "train" for fp in fps)
                traces[key] = states, actions, fps
            states, actions, fps = traces[key]
            state = states[certificate["suffix_index"]]
            result = verify_certificate(
                instance,
                certificate,
                witness,
                states,
                actions,
                fps,
                targets[state]["known_suffix_cost"],
                heldout,
            )
            assert result in needed and result not in certified
            certified.add(result)
            counts[certificate["method"]] += 1
        assert certified == needed
        assert len(a["preferences"]) == len(b["preferences"])
        for old, new in zip(b["preferences"], a["preferences"], strict=True):
            assert {k: v for k, v in old.items() if k != "actions"} == {
                k: v for k, v in new.items() if k != "actions"
            }
            assert new["actions"] == targets[state_from_dict(new["state"])]["actions"]
        counts["conditions"] += 1
        counts["certificates"] += len(certified)
        if (index + 1) % 8 == 0:
            print(json.dumps(dict(counts)), flush=True)
    for name, value in aug["augmentation_counts"].items():
        assert counts[name] == value, name
    assert all(sha(Path(p)) == h for p, h in pins.items())
    result = {
        "status": "passed",
        "inputs": pins,
        "counts": dict(counts),
        "scope": "Exact alternative continuation certificates; "
        "not global optimality or model quality.",
    }
    atomic(args.out, result)
    print(json.dumps({"status": "passed", "counts": dict(counts)}))


if __name__ == "__main__":
    main()
