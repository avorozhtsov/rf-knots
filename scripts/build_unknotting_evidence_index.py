"""Build a replay-verified cross-run unknotting-witness index.

The Semantic-v2 archive stores witnesses inside round events rather than in a
single database.  This command joins each event to its frozen braid bank,
replays every semantic path, and emits both the best witness per scientist and
the first two single-knot-mastery curriculum blocks.

KnotInfo scalar upper bounds are deliberately not converted into witnesses.
The pseudo-scientist ``knotinfo-shortest-evidence`` is present with status
``unavailable`` until an actual replayable action path is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rf_knots.actions import ActionSpec
from rf_knots.evidence import UnknotWitness

SCHEMA = "unknotting-evidence-index-v1"
PSEUDO_SCIENTIST = "knotinfo-shortest-evidence"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bank_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("rows", [])
    return {
        str(row.get("id", row.get("name"))): row
        for row in rows
        if row.get("id", row.get("name")) is not None
    }


def _replay(
    row: dict[str, Any], witness: dict[str, Any]
) -> tuple[UnknotWitness, ActionSpec]:
    word = tuple(int(value) for value in row["word"])
    strands = int(row["strands"])
    actions = [int(value) for value in witness["semantic_actions"]]
    expected = (
        int(witness["crossing_changes"]),
        int(witness["semantic_moves"]),
    )
    matches: list[tuple[UnknotWitness, ActionSpec]] = []
    for max_len in (32, 48, 64, 96, 128):
        if max_len < len(word):
            continue
        for max_strands in range(max(strands, 2), 13):
            spec = ActionSpec(max_len=max_len, max_strands=max_strands)
            if actions and max(actions) >= spec.num_actions:
                continue
            try:
                replayed = UnknotWitness.from_actions(word, strands, spec, actions)
                replayed.verify()
            except (IndexError, ValueError):
                continue
            if (replayed.crossing_changes, replayed.moves) == expected:
                matches.append((replayed, spec))
    if not matches:
        raise ValueError("no ActionSpec replays the reported witness")
    # Several larger capacities can share offsets for paths that avoid the
    # capacity-dependent insert block. Prefer the smallest compatible space and
    # record it; the semantic replay result is identical.
    return min(matches, key=lambda item: (item[1].max_len, item[1].max_strands))


def _best_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return int(row["crossing_changes"]), int(row["l10"]), str(row["source_event"])


def _canonical_knot(name: str) -> bool:
    return not name.startswith("sv2-") and not name.startswith("q4000-")


def build(results: Path) -> dict[str, Any]:
    evidence: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    paired_advantages: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    event_count = 0
    for event_path in sorted(results.rglob("events/*.json")):
        run_dir = event_path.parent.parent
        bank_path = run_dir / "bank.json"
        manifest_path = run_dir / "manifest.json"
        if not bank_path.exists():
            continue
        try:
            event = json.loads(event_path.read_text())
            if not isinstance(event, dict) or "scientists" not in event:
                continue
            rows = _bank_rows(bank_path)
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            selected = str(event["selected"])
            bank_row = rows.get(selected)
            if bank_row is None:
                raise ValueError(f"selected row {selected!r} is absent from bank")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append({"event": str(event_path), "reason": str(exc)})
            continue
        event_count += 1
        event_best: dict[str, dict[str, Any]] = {}
        for scientist, scientist_row in sorted(event["scientists"].items()):
            for ratio, evaluation in sorted((scientist_row.get("evaluation") or {}).items()):
                witness_row = evaluation.get("best_witness")
                if not witness_row or not witness_row.get("semantic_actions"):
                    continue
                try:
                    replayed, spec = _replay(bank_row, witness_row)
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(
                        {
                            "event": str(event_path),
                            "scientist": str(scientist),
                            "ratio": str(ratio),
                            "reason": str(exc),
                        }
                    )
                    continue
                record = {
                    "scientist": str(scientist),
                    "crossing_changes": replayed.crossing_changes,
                    "semantic_moves": replayed.moves,
                    "l10": 10 * replayed.crossing_changes + replayed.moves,
                    "l1000": 1000 * replayed.crossing_changes + replayed.moves,
                    "objective_ratio": float(ratio),
                    "semantic_actions": [int(value) for value in witness_row["semantic_actions"]],
                    "start": replayed.start.to_dict(),
                    "instance_id": replayed.instance_id,
                    "replay_verified": True,
                    "action_spec": {
                        "max_len": spec.max_len,
                        "max_strands": spec.max_strands,
                    },
                    "run": str(run_dir.relative_to(results)),
                    "source_event": str(event_path.relative_to(results)),
                    "source_event_sha256": _sha256(event_path),
                    "source_bank": str(bank_path.relative_to(results)),
                    "source_bank_sha256": _sha256(bank_path),
                    "source_manifest": (
                        str(manifest_path.relative_to(results))
                        if manifest_path.exists()
                        else None
                    ),
                    "source_manifest_sha256": (
                        _sha256(manifest_path) if manifest_path.exists() else None
                    ),
                    "protocol_sha256": manifest.get("protocol_sha256"),
                    "checkpoint": (manifest.get("checkpoints") or {}).get(str(scientist)),
                }
                evidence[selected][str(scientist)].append(record)
                prior = event_best.get(str(scientist))
                if prior is None or _best_key(record) < _best_key(prior):
                    event_best[str(scientist)] = record
        if len(event_best) >= 2:
            best_cc = min(row["crossing_changes"] for row in event_best.values())
            winners = [row for row in event_best.values() if row["crossing_changes"] == best_cc]
            peers = [row for row in event_best.values() if row["crossing_changes"] > best_cc]
            for winner in winners:
                if peers and winner["semantic_moves"] > 50:
                    paired_advantages.append(
                        {
                            "knot": selected,
                            "winner": winner,
                            "peer_best_crossing_changes": min(
                                row["crossing_changes"] for row in peers
                            ),
                            "peer_scientists": sorted(row["scientist"] for row in peers),
                        }
                    )

    knots: dict[str, Any] = {}
    for knot, scientists in sorted(evidence.items()):
        best = {
            scientist: min(records, key=_best_key)
            for scientist, records in sorted(scientists.items())
        }
        knots[knot] = {
            "scientists": best,
            PSEUDO_SCIENTIST: {
                "status": "unavailable",
                "reason": "pinned KnotInfo workbook contains no replayable action path",
                "rankable_by_l10": False,
            },
        }

    unique_advantages: dict[str, dict[str, Any]] = {}
    for row in paired_advantages:
        knot = row["knot"]
        if not _canonical_knot(knot):
            continue
        prior = unique_advantages.get(knot)
        if prior is None or _best_key(row["winner"]) < _best_key(prior["winner"]):
            unique_advantages[knot] = row
    ordered = sorted(
        unique_advantages.values(),
        key=lambda row: (
            int(row["winner"]["l10"]),
            int(row["winner"]["crossing_changes"]),
            row["knot"],
        ),
    )
    first = ordered[:10]
    second_ours = ordered[10:15]
    second = [
        {"slot": index + 1, "source": "ours", **row}
        for index, row in enumerate(second_ours)
    ]
    second.extend(
        {
            "slot": index + 6,
            "source": PSEUDO_SCIENTIST,
            "status": "blocked",
            "reason": "no replayable KnotInfo action path is present in the pinned sources",
        }
        for index in range(5)
    )
    return {
        "schema": SCHEMA,
        "source_results": str(results),
        "scope": {
            "event_files_joined": event_count,
            "knots_with_verified_native_evidence": len(knots),
            "replay_errors": len(errors),
            "knotinfo_replayable_evidence": 0,
        },
        "ranking": {
            "per_scientist_best": "crossing_changes, then L10",
            "curriculum_order": "L10 = 10 * crossing_changes + semantic_moves",
            "long_evidence_minimum_moves_exclusive": 50,
        },
        "curriculum": {
            "block_1": [
                {"slot": index + 1, "source": "ours", **row}
                for index, row in enumerate(first)
            ],
            "block_2": second,
        },
        "knots": knots,
        "replay_errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["scope"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
