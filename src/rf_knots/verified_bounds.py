r"""An unknotting-bound ratchet whose records can be re-checked, not just cited.

## What was wrong with the old one

`pgx_mcts_bench.bounds` stores, per knot, the fewest crossing changes anyone has
used — and in the `witness` field it stores **the knot's defining word**, not the
sequence of moves that unknotted it. So a bound can be attributed to an agent and
cannot be re-derived by anybody, including by the agent that claimed it. Its own
`HANDOFF.md` lists this as open.

That matters more than it sounds. The whole argument for this project over a
solver is that its answers are *machine-checkable*; a bound nobody can replay is
exactly the estimate the design set out to avoid. And it is not hypothetical: the
standing record on `R(3,18)#0` was 6 crossing changes against a true `u` of 2, and
nothing in the log could have revealed that, because there was nothing to check.

## What this stores instead

The full `rf_knots.evidence.UnknotWitness`: every action, every intermediate
braid state, and the action-space parameters needed to interpret them. `verify()`
replays each action through the reference implementation — checking legality,
checking the recorded state matches the replayed one, and checking the sequence
ends at the empty 1-braid — so a record is either reproducible or it is rejected
on read.

`best(path)` verifies by default. A record that fails replay is **not** silently
dropped to second place; it is reported, because a witness that does not verify is
either a bug or a false claim and both need to be seen.

## Compatibility

`knot_id` matches `pgx_mcts_bench.bounds.knot_id` exactly, so the two logs key the
same way and `import_legacy` can fold an old log in — marking every imported row
`unverified`, which is the honest description of a record with no sequence in it.
Append-only with `O_APPEND`, so concurrent claimers interleave whole lines and
nothing is ever lost; `best` folds on read.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from rf_knots.evidence import UnknotWitness

Word = tuple[int, ...]


def knot_id(word, strands: int) -> str:
    """Identity of the knot, not of the diagram. Matches the legacy log's key."""
    letters = ",".join(str(int(x)) for x in word if int(x))
    return f"b{strands}:{letters}" if letters else f"b{strands}:e"


@dataclass(frozen=True)
class Record:
    knot: str
    crossing_changes: int
    moves: int
    agent: str
    witness: UnknotWitness | None
    lower_bound: int | None = None
    note: str = ""

    @property
    def verified(self) -> bool:
        return self.witness is not None

    @property
    def exact(self) -> bool:
        """`u` is determined: the sequence meets a certified lower bound."""
        return self.lower_bound is not None and self.lower_bound == self.crossing_changes

    def beats(self, other: Record | None) -> bool:
        if other is None:
            return True
        # A verified record beats an unverified one at equal cost: the point of
        # the log is evidence, so evidence breaks ties before move count does.
        mine = (self.crossing_changes, not self.verified, self.moves)
        theirs = (other.crossing_changes, not other.verified, other.moves)
        return mine < theirs

    def to_dict(self) -> dict:
        return {
            "knot": self.knot,
            "crossing_changes": self.crossing_changes,
            "moves": self.moves,
            "agent": self.agent,
            "lower_bound": self.lower_bound,
            "note": self.note,
            "witness": self.witness.to_dict() if self.witness else None,
        }


def from_witness(
    witness: UnknotWitness, agent: str, lower_bound: int | None = None, note: str = ""
) -> Record:
    """Build a record from a replayable witness, verifying it first."""
    witness.verify()
    return Record(
        knot=knot_id(witness.start.word, witness.start.strands),
        crossing_changes=witness.crossing_changes,
        moves=witness.moves,
        agent=agent,
        witness=witness,
        lower_bound=lower_bound,
        note=note,
    )


def claim(path: Path, record: Record) -> None:
    """Append a claim. Never blocks, never overwrites, never loses a record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict())
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(handle, (line + "\n").encode())
    finally:
        os.close(handle)


def best(path: Path, verify: bool = True) -> tuple[dict[str, Record], list[str]]:
    """Fold the log into standing records, plus the list of rejected claims.

    Rejections are returned rather than logged away: a witness that fails to
    replay means either a bug in the writer or a claim that was never true, and
    both are results.
    """
    records: dict[str, Record] = {}
    rejected: list[str] = []
    if not path.exists():
        return records, rejected
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rejected.append(f"line {number}: torn or invalid JSON")
            continue
        witness = None
        if row.get("witness"):
            try:
                witness = UnknotWitness.from_dict(row["witness"])
                if verify:
                    witness.verify()
            except (ValueError, KeyError) as error:
                rejected.append(f"line {number} ({row.get('knot')}): {error}")
                continue
        record = Record(
            knot=row["knot"],
            crossing_changes=int(row["crossing_changes"]),
            moves=int(row["moves"]),
            agent=row.get("agent", ""),
            witness=witness,
            lower_bound=row.get("lower_bound"),
            note=row.get("note", ""),
        )
        if record.beats(records.get(record.knot)):
            records[record.knot] = record
    return records, rejected


def import_legacy(legacy: Path, path: Path, agent_suffix: str = " (imported)") -> int:
    """Fold a `pgx_mcts_bench.bounds` log in, marked unverified.

    Those rows store the knot's defining word in their `witness` field rather than
    a move sequence, so none of them can be replayed. They are imported anyway --
    an unverified upper bound is still an upper bound, and dropping them would
    lose the history — but they are labelled, and any verified claim of equal cost
    displaces them.
    """
    if not legacy.exists():
        return 0
    imported = 0
    for line in legacy.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        claim(
            path,
            Record(
                knot=row["knot"],
                crossing_changes=int(row["crossings"]),
                moves=int(row["moves"]),
                agent=row.get("agent", "") + agent_suffix,
                witness=None,
                note="legacy record: no move sequence stored, cannot be replayed",
            ),
        )
        imported += 1
    return imported


def report(path: Path) -> str:
    records, rejected = best(path)
    lines = [
        "# Best known unknotting bounds",
        "",
        "The fewest crossing changes anyone has used, with the sequence that did",
        "it. **verified** means the stored witness was replayed through the",
        "reference implementation on read and reached the empty 1-braid.",
        "**exact** means it also meets a certified lower bound, so `u` is",
        "determined rather than bounded.",
        "",
        "| knot | u <= | moves | verified | exact | agent |",
        "|---|---:|---:|:--:|:--:|---|",
    ]
    for key, record in sorted(records.items()):
        lines.append(
            f"| `{key[:56]}` | {record.crossing_changes} | {record.moves} | "
            f"{'yes' if record.verified else 'no'} | "
            f"{'**yes**' if record.exact else '--'} | {record.agent} |"
        )
    verified = sum(1 for r in records.values() if r.verified)
    exact = sum(1 for r in records.values() if r.exact)
    lines += [
        "",
        f"{len(records)} knots, {verified} with a replayable witness, "
        f"{exact} with `u` determined exactly.",
    ]
    if rejected:
        lines += ["", "## Rejected on read", ""]
        lines += [f"* {reason}" for reason in rejected]
    return "\n".join(lines) + "\n"
