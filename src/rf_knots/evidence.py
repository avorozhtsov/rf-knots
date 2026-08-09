"""Replayable unknotting evidence and append-only per-instance results.

The store deliberately separates three claims that old ``bounds.jsonl`` rows
conflated:

* an *instance* is the exact input diagram/word that was attempted;
* a *witness* is a complete, replayable sequence of semantic actions;
* a *lower bound* records both its value and the theorem/computation that
  produced it.

An upper bound is accepted only after replay reaches the empty one-braid.  An
exact unknotting number is reported only when a verified upper bound meets a
certified lower bound.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE, KIND_NAMES, ActionSpec

Word = tuple[int, ...]
SCHEMA = "rf-knots-evidence-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def braid_instance_id(word: Iterable[int], strands: int) -> str:
    """Stable identity of an exact braid-word instance, not merely its knot type."""
    payload = {"encoding": "braid-word-v1", "strands": int(strands),
               "word": [int(x) for x in word if int(x)]}
    return f"braid:{_digest(payload)}"


def braid_knot_key(word: Iterable[int], strands: int) -> str:
    """Human-readable source-knot key compatible with the trainer's old log."""
    letters = ",".join(str(int(x)) for x in word if int(x))
    return f"b{int(strands)}:{letters}" if letters else f"b{int(strands)}:e"


@dataclass(frozen=True)
class BraidState:
    word: Word
    strands: int
    cyclic_band_generators: bool = False

    def __post_init__(self) -> None:
        if self.strands < 1:
            raise ValueError("a braid state needs at least one strand")
        largest = self.strands if self.cyclic_band_generators else self.strands - 1
        if any(abs(x) > largest for x in self.word):
            alphabet = "B*" if self.cyclic_band_generators else "B"
            raise ValueError(
                f"word {self.word} does not belong to {alphabet}_{self.strands}"
            )
        if reference.num_components(
            self.word,
            self.strands,
            self.cyclic_band_generators,
        ) != 1:
            raise ValueError("the braid closure is a link, not a knot")

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": list(self.word),
            "strands": self.strands,
            "cyclic_band_generators": self.cyclic_band_generators,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> BraidState:
        return cls(
            tuple(int(x) for x in row["word"]),
            int(row["strands"]),
            bool(row.get("cyclic_band_generators", False)),
        )


@dataclass(frozen=True)
class SemanticAction:
    """Action encoding independent of the flat policy-head offsets."""

    kind: str
    position: int = 0
    generator: int = 1
    sign: int = 1

    @classmethod
    def from_flat(cls, spec: ActionSpec, action: int) -> SemanticAction:
        kind, position, generator, sign = spec.decode(action)
        return cls(KIND_NAMES[kind], position, generator, sign)

    def to_flat(self, spec: ActionSpec) -> int:
        try:
            kind = KIND_NAMES.index(self.kind)
        except ValueError as error:
            raise ValueError(f"unknown action kind {self.kind!r}") from error
        return spec.encode(kind, self.position, self.generator, self.sign)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "position": self.position,
                "generator": self.generator, "sign": self.sign}

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> SemanticAction:
        return cls(str(row["kind"]), int(row.get("position", 0)),
                   int(row.get("generator", 1)), int(row.get("sign", 1)))


@dataclass(frozen=True)
class WitnessStep:
    action: SemanticAction
    after: BraidState

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action.to_dict(), "after": self.after.to_dict()}

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> WitnessStep:
        return cls(SemanticAction.from_dict(row["action"]), BraidState.from_dict(row["after"]))


@dataclass(frozen=True)
class UnknotWitness:
    start: BraidState
    max_len: int
    max_strands: int
    steps: tuple[WitnessStep, ...]
    cyclic_band_generators: bool = False

    @property
    def crossing_changes(self) -> int:
        return sum(step.action.kind == KIND_NAMES[CROSSING_CHANGE] for step in self.steps)

    @property
    def moves(self) -> int:
        return len(self.steps)

    @property
    def instance_id(self) -> str:
        return braid_instance_id(self.start.word, self.start.strands)

    def verify(self) -> BraidState:
        """Replay all steps and return the terminal state, or raise on any mismatch."""
        spec = ActionSpec(
            self.max_len,
            self.max_strands,
            cyclic_band_generators=self.cyclic_band_generators,
        )
        current = self.start
        for index, step in enumerate(self.steps):
            flat = step.action.to_flat(spec)
            if not reference.is_legal(spec, current.word, current.strands, flat, True):
                raise ValueError(f"step {index} is illegal: {step.action}")
            word, strands = reference.apply(spec, current.word, current.strands, flat)
            actual = BraidState(word, strands, self.cyclic_band_generators)
            if actual != step.after:
                raise ValueError(
                    f"step {index} replay mismatch: stored {step.after}, actual {actual}"
                )
            current = actual
        if current != BraidState((), 1, self.cyclic_band_generators):
            raise ValueError(f"witness does not reach the unknot: stopped at {current}")
        return current

    @classmethod
    def from_actions(
        cls,
        start_word: Iterable[int],
        start_strands: int,
        spec: ActionSpec,
        actions: Iterable[int],
    ) -> UnknotWitness:
        current = BraidState(
            tuple(int(x) for x in start_word if int(x)),
            start_strands,
            spec.cyclic_band_generators,
        )
        steps: list[WitnessStep] = []
        for index, flat in enumerate(actions):
            if not reference.is_legal(spec, current.word, current.strands, int(flat), True):
                raise ValueError(f"action {index} is illegal: {spec.describe(int(flat))}")
            word, strands = reference.apply(spec, current.word, current.strands, int(flat))
            current = BraidState(word, strands, spec.cyclic_band_generators)
            steps.append(WitnessStep(SemanticAction.from_flat(spec, int(flat)), current))
        witness = cls(
            BraidState(
                tuple(int(x) for x in start_word if int(x)),
                start_strands,
                spec.cyclic_band_generators,
            ),
            spec.max_len,
            spec.max_strands,
            tuple(steps),
            spec.cyclic_band_generators,
        )
        witness.verify()
        return witness

    @classmethod
    def from_states(
        cls,
        states: Iterable[BraidState],
        spec: ActionSpec,
    ) -> UnknotWitness:
        """Recover a portable witness from states emitted by any controller.

        Serial head travel, scans and memory writes leave the braid unchanged and
        are omitted.  Every state-changing transition must match one legal action
        in the shared braid environment.  This is the bridge from foreign action
        spaces to replayable state-space evidence.
        """
        sequence = tuple(states)
        if not sequence:
            raise ValueError("at least one state is required")
        compact = [sequence[0]]
        compact.extend(state for state in sequence[1:] if state != compact[-1])
        steps: list[WitnessStep] = []
        for index, (before, after) in enumerate(zip(compact, compact[1:], strict=False)):
            matches = []
            for action in range(spec.num_actions):
                if not reference.is_legal(spec, before.word, before.strands, action, True):
                    continue
                word, strands = reference.apply(spec, before.word, before.strands, action)
                if (word, strands) == (after.word, after.strands):
                    matches.append(action)
            if not matches:
                raise ValueError(f"state transition {index} is not one shared braid action")
            action = min(matches)
            steps.append(WitnessStep(SemanticAction.from_flat(spec, action), after))
        witness = cls(
            sequence[0],
            spec.max_len,
            spec.max_strands,
            tuple(steps),
            spec.cyclic_band_generators,
        )
        witness.verify()
        return witness

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.to_dict(),
            "action_spec": {
                "max_len": self.max_len,
                "max_strands": self.max_strands,
                "cyclic_band_generators": self.cyclic_band_generators,
            },
            "steps": [step.to_dict() for step in self.steps],
            "crossing_changes": self.crossing_changes,
            "moves": self.moves,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> UnknotWitness:
        spec = row["action_spec"]
        return cls(
            BraidState.from_dict(row["start"]),
            int(spec["max_len"]),
            int(spec["max_strands"]),
            tuple(WitnessStep.from_dict(step) for step in row["steps"]),
            bool(spec.get("cyclic_band_generators", False)),
        )


@dataclass(frozen=True)
class LowerBoundClaim:
    value: int
    method: str
    source: str
    citation: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("lower bounds cannot be negative")
        if not self.method or not self.source:
            raise ValueError("lower bounds require method and source provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "method": self.method, "source": self.source,
                "citation": self.citation, "details": self.details}

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> LowerBoundClaim:
        return cls(int(row["value"]), str(row["method"]), str(row["source"]),
                   row.get("citation"), dict(row.get("details", {})))


@dataclass(frozen=True)
class EvidenceRecord:
    instance_id: str
    knot_key: str
    solver: str
    search_budget: dict[str, Any]
    outcome: str
    witness: UnknotWitness | None = None
    lower_bounds: tuple[LowerBoundClaim, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    benchmark: str | None = None
    checkpoint: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    def __post_init__(self) -> None:
        if self.witness is not None:
            if self.witness.instance_id != self.instance_id:
                raise ValueError("witness start does not match record instance")
            self.witness.verify()

    @property
    def certified_lower_bound(self) -> int | None:
        return max((claim.value for claim in self.lower_bounds), default=None)

    @property
    def certified_upper_bound(self) -> int | None:
        return self.witness.crossing_changes if self.witness is not None else None

    @property
    def exact_unknotting_number(self) -> int | None:
        lower, upper = self.certified_lower_bound, self.certified_upper_bound
        return upper if lower is not None and upper == lower else None

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": SCHEMA,
            "instance_id": self.instance_id,
            "knot_key": self.knot_key,
            "solver": self.solver,
            "checkpoint": self.checkpoint,
            "benchmark": self.benchmark,
            "search_budget": self.search_budget,
            "outcome": self.outcome,
            "witness": None if self.witness is None else self.witness.to_dict(),
            "lower_bounds": [claim.to_dict() for claim in self.lower_bounds],
            "metrics": self.metrics,
            "created_at": self.created_at,
        }
        return {"record_id": _digest(body), **body}

    @classmethod
    def from_dict(cls, row: dict[str, Any], verify_id: bool = True) -> EvidenceRecord:
        if row.get("schema") != SCHEMA:
            raise ValueError(f"unsupported evidence schema {row.get('schema')!r}")
        record = cls(
            instance_id=str(row["instance_id"]),
            knot_key=str(row["knot_key"]),
            solver=str(row["solver"]),
            checkpoint=row.get("checkpoint"),
            benchmark=row.get("benchmark"),
            search_budget=dict(row.get("search_budget", {})),
            outcome=str(row["outcome"]),
            witness=(None if row.get("witness") is None
                     else UnknotWitness.from_dict(row["witness"])),
            lower_bounds=tuple(LowerBoundClaim.from_dict(x)
                               for x in row.get("lower_bounds", [])),
            metrics=dict(row.get("metrics", {})),
            created_at=str(row["created_at"]),
        )
        if verify_id and record.to_dict()["record_id"] != row.get("record_id"):
            raise ValueError("evidence record hash does not match its contents")
        return record


class EvidenceStore:
    """Append-only JSONL evidence with strict reads and best-witness folding."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def append(self, record: EvidenceRecord) -> str:
        row = record.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = (_canonical_json(row) + "\n").encode()
        handle = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(handle, payload)
        finally:
            os.close(handle)
        return str(row["record_id"])

    def records(self, skip_torn_last_line: bool = True) -> list[EvidenceRecord]:
        if not self.path.exists():
            return []
        lines = self.path.read_text().splitlines()
        records: list[EvidenceRecord] = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if skip_torn_last_line and index == len(lines) - 1:
                    continue
                raise
            # A syntactically complete record with a bad hash, schema or witness
            # is corruption, even at EOF.  Only an actually torn JSON line is
            # recoverable after a crashed concurrent append.
            records.append(EvidenceRecord.from_dict(row))
        return records

    def best_witnesses(self) -> dict[str, EvidenceRecord]:
        best: dict[str, EvidenceRecord] = {}
        for record in self.records():
            if record.witness is None:
                continue
            previous = best.get(record.instance_id)
            score = (record.witness.crossing_changes, record.witness.moves)
            if previous is None or score < (
                previous.witness.crossing_changes, previous.witness.moves  # type: ignore[union-attr]
            ):
                best[record.instance_id] = record
        return best


@dataclass(frozen=True)
class SharedWitnessTarget:
    """Action-space-neutral supervision extracted from a verified witness."""

    state: BraidState
    next_state: BraidState
    crossing_change: bool
    remaining_crossing_changes: int
    remaining_moves: int


def shared_witness_targets(witness: UnknotWitness) -> tuple[SharedWitnessTarget, ...]:
    """Export state transitions and admissible cost-to-go upper bounds."""
    witness.verify()
    states = (witness.start,) + tuple(step.after for step in witness.steps)
    output: list[SharedWitnessTarget] = []
    for index, step in enumerate(witness.steps):
        tail = witness.steps[index:]
        output.append(
            SharedWitnessTarget(
                state=states[index],
                next_state=states[index + 1],
                crossing_change=step.action.kind == KIND_NAMES[CROSSING_CHANGE],
                remaining_crossing_changes=sum(
                    item.action.kind == KIND_NAMES[CROSSING_CHANGE] for item in tail
                ),
                remaining_moves=len(tail),
            )
        )
    return tuple(output)


def upper_bound_hinge(predicted_cost: Any, witness_upper_bound: Any) -> Any:
    """One-sided value loss: a witness is an upper bound, never an equality label.

    Works with Python scalars, NumPy arrays and torch tensors because all three
    implement subtraction and ``clip``/``clamp_min``-like comparison through
    multiplication by a boolean mask.
    """
    difference = predicted_cost - witness_upper_bound
    return difference * (difference > 0)
