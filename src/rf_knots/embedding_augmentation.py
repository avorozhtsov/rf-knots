"""Deterministic zero-CC augmentation and virtual hard-pair curricula."""

from __future__ import annotations

import hashlib
import math
import random
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from rf_knots.embedding_gordian import GordianPair
from rf_knots.embedding_supervision import DistanceInterval, EmbeddingRepresentation


@dataclass(frozen=True)
class ZeroCCOperation:
    kind: str
    position: int
    generator: int = 0
    sign: int = 0
    amount: int = 0


@dataclass(frozen=True)
class AugmentedRepresentation:
    word: tuple[int, ...]
    strands: int
    operations: tuple[ZeroCCOperation, ...]

    @property
    def semantic_length(self) -> int:
        return sum(operation.kind != "cyclic_shift" for operation in self.operations)


@dataclass(frozen=True)
class SyntheticPairSpec:
    index: int
    stratum: str
    metric: str
    left_representation: int
    right_representation: int
    distance_lower: float
    distance_upper: float
    base_witness_semantic_length: int
    left_seed: int
    right_seed: int
    confidence: float
    left_mirror: bool = False
    right_mirror: bool = False
    left_scramble_depth: int = -1
    right_scramble_depth: int = -1


def _seed(global_seed: int, index: int, salt: str) -> int:
    material = f"{global_seed}:{index}:{salt}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def augment_zero_cc(
    representation: EmbeddingRepresentation,
    seed: int,
    *,
    minimum_insertions: int = 6,
    maximum_insertions: int = 20,
) -> AugmentedRepresentation:
    """Apply an origin shift and replayable inverse-pair insertions."""
    if minimum_insertions < 0 or maximum_insertions < minimum_insertions:
        raise ValueError("invalid insertion range")
    rng = random.Random(seed)
    word = list(representation.word)
    operations = []
    if word:
        amount = rng.randrange(len(word))
        word = word[amount:] + word[:amount]
        operations.append(ZeroCCOperation("cyclic_shift", 0, amount=amount))
    insertions = rng.randint(minimum_insertions, maximum_insertions)
    largest_generator = max(representation.strands - 1, 1)
    for _ in range(insertions):
        position = rng.randrange(len(word) + 1)
        generator = rng.randint(1, largest_generator)
        sign = rng.choice((-1, 1))
        word[position:position] = [sign * generator, -sign * generator]
        operations.append(ZeroCCOperation("insert_inverse_pair", position, generator, sign))
    return AugmentedRepresentation(tuple(word), representation.strands, tuple(operations))


def reverse_zero_cc(
    augmented: AugmentedRepresentation,
) -> tuple[int, ...]:
    """Independently replay the exact inverse of an augmentation descriptor."""
    word = list(augmented.word)
    for operation in reversed(augmented.operations):
        if operation.kind == "insert_inverse_pair":
            expected = [
                operation.sign * operation.generator,
                -operation.sign * operation.generator,
            ]
            if word[operation.position : operation.position + 2] != expected:
                raise ValueError("inverse-pair augmentation checkpoint mismatch")
            del word[operation.position : operation.position + 2]
        elif operation.kind == "cyclic_shift":
            if word:
                amount = operation.amount % len(word)
                word = word[-amount:] + word[:-amount] if amount else word
        else:
            raise ValueError(f"unsupported zero-CC operation {operation.kind!r}")
    return tuple(word)


def random_knot_braid(seed: int, *, simple: bool) -> EmbeddingRepresentation:
    """Generate a one-component braid closure without consulting UnknotDB.

    The ascending generator prefix has an n-cycle permutation. Added generator
    squares have identity permutation, so the closure remains one component.
    """
    rng = random.Random(seed)
    strands = rng.randint(2, 4) if simple else rng.randint(3, 12)
    word = list(range(1, strands))
    square_count = rng.randint(0, 3) if simple else rng.randint(8, 30)
    for _ in range(square_count):
        position = rng.randrange(len(word) + 1)
        generator = rng.randint(1, strands - 1)
        sign = rng.choice((-1, 1))
        word[position:position] = [sign * generator, sign * generator]
    return EmbeddingRepresentation(-1, tuple(word), strands, None, "train")


def physical_representation_sha256(
    representation: EmbeddingRepresentation,
) -> bytes:
    word = representation.word
    encoded = (
        b"UKB0\x00\x00"
        + struct.pack("<H", representation.strands)
        + struct.pack("<I", len(word))
        + (struct.pack(f"<{len(word)}h", *word) if word else b"")
    )
    return hashlib.sha256(encoded).digest()


def random_absent_knot_braid(
    seed: int, *, simple: bool, existing_sha256: frozenset[bytes]
) -> EmbeddingRepresentation:
    """Deterministically rejection-sample a physical braid absent from a sidecar."""
    for nonce in range(4096):
        candidate = random_knot_braid(_seed(seed, nonce, "absent"), simple=simple)
        if physical_representation_sha256(candidate) not in existing_sha256:
            return candidate
    raise RuntimeError("could not generate a physical braid outside the pinned sidecar")


def augment_markov_zero_cc(
    representation: EmbeddingRepresentation, seed: int, *, simple: bool
) -> AugmentedRepresentation:
    """Create a distant same-knot braid via replayable braid/Markov moves."""
    rng = random.Random(seed)
    augmented = augment_zero_cc(
        representation,
        seed ^ 0x9E3779B97F4A7C15,
        minimum_insertions=0 if simple else 6,
        maximum_insertions=3 if simple else 20,
    )
    word = list(augmented.word)
    strands = augmented.strands
    operations = list(augmented.operations)
    conjugations = 1 if simple else rng.randint(1, 4)
    for _ in range(conjugations):
        generator = rng.randint(1, max(strands - 1, 1))
        sign = rng.choice((-1, 1))
        word = [sign * generator, *word, -sign * generator]
        operations.append(ZeroCCOperation("conjugate", 0, generator, sign))
    stabilizations = rng.randint(0, 1) if simple else rng.randint(1, 2)
    for _ in range(stabilizations):
        sign = rng.choice((-1, 1))
        word.append(sign * strands)
        operations.append(ZeroCCOperation("stabilize", len(word) - 1, strands, sign))
        strands += 1
    return AugmentedRepresentation(tuple(word), strands, tuple(operations))


def reverse_markov_zero_cc(
    augmented: AugmentedRepresentation,
) -> tuple[tuple[int, ...], int]:
    """Independently invert insertion, conjugation and stabilization records."""
    word = list(augmented.word)
    strands = augmented.strands
    for operation in reversed(augmented.operations):
        if operation.kind == "stabilize":
            if strands <= 1 or not word or word[-1] != operation.sign * operation.generator:
                raise ValueError("stabilization checkpoint mismatch")
            word.pop()
            strands -= 1
        elif operation.kind == "conjugate":
            expected_left = operation.sign * operation.generator
            if len(word) < 2 or word[0] != expected_left or word[-1] != -expected_left:
                raise ValueError("conjugation checkpoint mismatch")
            word = word[1:-1]
        elif operation.kind == "insert_inverse_pair":
            expected = [
                operation.sign * operation.generator,
                -operation.sign * operation.generator,
            ]
            if word[operation.position : operation.position + 2] != expected:
                raise ValueError("inverse-pair augmentation checkpoint mismatch")
            del word[operation.position : operation.position + 2]
        elif operation.kind == "cyclic_shift":
            if word:
                amount = operation.amount % len(word)
                word = word[-amount:] + word[:-amount] if amount else word
        else:
            raise ValueError(f"unsupported zero-CC operation {operation.kind!r}")
    return tuple(word), strands


def scramble_zero_cc(
    representation: EmbeddingRepresentation,
    seed: int,
    depth: int,
    *,
    mirror: bool = False,
    max_strands: int = 12,
) -> AugmentedRepresentation:
    """Apply exactly ``depth`` replayable type-preserving scramble operations."""
    if not 0 <= depth <= 6:
        raise ValueError("catalogue scramble depth must be between zero and six")
    rng = random.Random(seed)
    word = [-value for value in representation.word] if mirror else list(representation.word)
    strands = representation.strands
    operations: list[ZeroCCOperation] = []
    for _ in range(depth):
        choices = ["insert_inverse_pair", "conjugate"]
        if word:
            choices.append("cyclic_shift")
        if strands < max_strands:
            choices.append("stabilize")
        kind = rng.choice(choices)
        if kind == "cyclic_shift":
            amount = rng.randrange(1, len(word) + 1)
            word = word[amount:] + word[:amount]
            operations.append(ZeroCCOperation(kind, 0, amount=amount))
        elif kind == "insert_inverse_pair":
            position = rng.randrange(len(word) + 1)
            generator = rng.randint(1, max(strands - 1, 1))
            sign = rng.choice((-1, 1))
            word[position:position] = [sign * generator, -sign * generator]
            operations.append(ZeroCCOperation(kind, position, generator, sign))
        elif kind == "conjugate":
            generator = rng.randint(1, max(strands - 1, 1))
            sign = rng.choice((-1, 1))
            word = [sign * generator, *word, -sign * generator]
            operations.append(ZeroCCOperation(kind, 0, generator, sign))
        else:
            sign = rng.choice((-1, 1))
            word.append(sign * strands)
            operations.append(ZeroCCOperation(kind, len(word) - 1, strands, sign))
            strands += 1
    return AugmentedRepresentation(tuple(word), strands, tuple(operations))


def _procedural_cc_count(seed: int, *, simple: bool) -> int:
    if simple:
        return 1
    return random.Random(seed ^ 0xD1B54A32D192ED03).randint(1, 2)


def procedural_cc_pair(
    seed: int,
    *,
    simple: bool,
    root: EmbeddingRepresentation | None = None,
) -> tuple[EmbeddingRepresentation, AugmentedRepresentation, int]:
    """Generate a replayed CC upper-bound pair outside the proof graph."""
    root = root or random_knot_braid(seed, simple=simple)
    rng = random.Random(seed ^ 0xD1B54A32D192ED03)
    changed = list(root.word)
    cc_count = _procedural_cc_count(seed, simple=simple)
    positions = rng.sample(range(len(changed)), k=min(cc_count, len(changed)))
    for position in positions:
        changed[position] = -changed[position]
    changed_representation = EmbeddingRepresentation(
        -1, tuple(changed), root.strands, None, "train"
    )
    target = augment_markov_zero_cc(
        changed_representation, seed ^ 0x94D049BB133111EB, simple=simple
    )
    return root, target, len(positions)


class MillionPairCurriculum:
    """A deterministic virtual corpus with exactly 80 percent hard examples.

    Twenty percent comes from procedural braids absent from the proof sidecar,
    15 percent is explicitly anchored at replay checkpoints inside proof edges,
    and another 10 percent samples mirror-orbit witnesses.
    """

    def __init__(
        self,
        *,
        representation_ids: tuple[int, ...],
        edge_interior: tuple[DistanceInterval, ...],
        mirror_orbit: tuple[DistanceInterval, ...],
        exact_cc1: tuple[DistanceInterval, ...],
        exact_cc2: tuple[DistanceInterval, ...],
        natural: tuple[DistanceInterval, ...],
        existing_representation_sha256: frozenset[bytes],
        gordian: tuple[GordianPair, ...] = (),
        size: int = 1_000_000,
        seed: int = 20260829,
    ):
        if (
            not representation_ids
            or not edge_interior
            or not mirror_orbit
            or not exact_cc1
            or not exact_cc2
            or not natural
        ):
            raise ValueError("every curriculum source pool must be non-empty")
        if size <= 0 or size % 20:
            raise ValueError("curriculum size must be a positive multiple of twenty")
        self.representation_ids = representation_ids
        self.edge_interior = edge_interior
        self.mirror_orbit = mirror_orbit
        self.exact_cc1 = exact_cc1
        self.exact_cc2 = exact_cc2
        self.gordian = gordian
        self.natural = natural
        self.existing_representation_sha256 = existing_representation_sha256
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def _choose(self, rows: tuple, index: int, salt: str):
        return rows[_seed(self.seed, index, salt) % len(rows)]

    def spec(self, index: int) -> SyntheticPairSpec:
        if not 0 <= index < self.size:
            raise IndexError(index)
        twentieth = self.size // 20
        left_seed = _seed(self.seed, index, "left")
        right_seed = _seed(self.seed, index, "right")
        if index < 3 * twentieth:
            pair = self._choose(self.edge_interior, index, "edge-interior")
            return SyntheticPairSpec(
                index,
                "hard_edge_interior",
                pair.metric,
                pair.left_representation,
                pair.right_representation,
                pair.lower,
                pair.upper,
                pair.witness_semantic_length,
                left_seed,
                right_seed,
                1.0,
            )
        if index < 6 * twentieth:
            pair = self._choose(self.exact_cc1, index, "hard-cc1")
            stratum = "hard_cc1_long_detour" if index % 2 else "hard_cc1_detour"
        elif index < 7 * twentieth:
            pair = self._choose(self.exact_cc2, index, "hard-cc2")
            stratum = "hard_cc2_long_detour" if index % 2 else "hard_cc2_detour"
        elif index < 8 * twentieth and self.gordian:
            pair = self._choose(self.gordian, index, "gordian-lower-bound")
            left_choice = _seed(self.seed, index, "gordian-left-posting") % len(
                pair.left.representation_ids
            )
            right_choice = _seed(self.seed, index, "gordian-right-posting") % len(
                pair.right.representation_ids
            )
            return SyntheticPairSpec(
                index,
                "hard_gordian_catalogue",
                "cc",
                pair.left.representation_ids[left_choice],
                pair.right.representation_ids[right_choice],
                pair.lower,
                pair.upper,
                0,
                left_seed,
                right_seed,
                1.0,
                pair.left.mirror,
                pair.right.mirror,
                _seed(self.seed, index, "gordian-left-depth") % 7,
                _seed(self.seed, index, "gordian-right-depth") % 7,
            )
        elif index < 8 * twentieth:
            pair = self._choose(self.exact_cc2, index, "hard-cc2-fallback")
            stratum = "hard_cc2_long_detour" if index % 2 else "hard_cc2_detour"
        elif index < 10 * twentieth:
            pair = self._choose(self.mirror_orbit, index, "mirror-orbit")
            stratum = "hard_mirror_orbit"
        elif index < 12 * twentieth:
            representation_id = self._choose(self.representation_ids, index, "zero-cc-orbit")
            return SyntheticPairSpec(
                index,
                "hard_zero_cc_orbit",
                "cc",
                representation_id,
                representation_id,
                0.0,
                0.0,
                0,
                left_seed,
                right_seed,
                1.0,
            )
        elif index < 14 * twentieth:
            return SyntheticPairSpec(
                index,
                "hard_procedural_same_knot_cc0",
                "cc",
                -1,
                -1,
                0.0,
                0.0,
                0,
                left_seed,
                right_seed,
                1.0,
            )
        elif index < 16 * twentieth:
            simple = index % 2 == 0
            cc_upper = 1 if simple else 2
            return SyntheticPairSpec(
                index,
                "hard_procedural_cc_upper",
                "cc",
                -1,
                -1,
                0.0,
                float(cc_upper),
                cc_upper,
                left_seed,
                right_seed,
                0.7,
            )
        else:
            pair = self._choose(self.natural, index, "natural")
            stratum = "natural_replay"
        return SyntheticPairSpec(
            index,
            stratum,
            pair.metric,
            pair.left_representation,
            pair.right_representation,
            pair.lower,
            pair.upper,
            pair.witness_semantic_length,
            left_seed,
            right_seed,
            1.0,
        )


@dataclass(frozen=True)
class MaterializedSyntheticPair:
    """One pair after deterministic endpoint augmentation."""

    spec: SyntheticPairSpec
    metric: str
    left_word: tuple[int, ...]
    right_word: tuple[int, ...]
    left_strands: int
    right_strands: int
    witness_semantic_length: int


class SyntheticEmbeddingIterableDataset(IterableDataset):
    """Worker-sharded, replay-checked materialization of a virtual curriculum.

    The SQLite handle is opened inside each worker. This avoids sharing a live
    connection across ``fork``/``spawn`` and keeps the million-pair corpus
    virtual: only the current examples and one batch are resident in memory.
    """

    def __init__(
        self,
        sidecar_path: str | Path,
        curriculum: MillionPairCurriculum,
        *,
        epoch: int = 0,
        replay_check: bool = False,
        infinite: bool = False,
        start_position: int = 0,
    ):
        super().__init__()
        self.sidecar_path = str(Path(sidecar_path).resolve())
        self.curriculum = curriculum
        self.epoch = epoch
        self.replay_check = replay_check
        self.infinite = infinite
        if start_position < 0:
            raise ValueError("start_position must be non-negative")
        self.start_position = start_position

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.curriculum)

    def _materialize(self, sidecar, index: int, generation: int = 0) -> MaterializedSyntheticPair:
        spec = self.curriculum.spec(index)
        # Every epoch sees the same metric pair, but a new deterministic zero-CC
        # presentation. This expands diagram diversity without changing labels.
        epoch_salt = _seed(
            self.curriculum.seed,
            (self.epoch << 32) ^ generation,
            "epoch-generation",
        )
        procedural_seed = spec.left_seed ^ epoch_salt
        if spec.stratum == "hard_procedural_same_knot_cc0":
            simple = index % 2 == 0
            root = random_absent_knot_braid(
                procedural_seed,
                simple=simple,
                existing_sha256=self.curriculum.existing_representation_sha256,
            )
            left_variant = augment_markov_zero_cc(root, procedural_seed, simple=simple)
            right_variant = augment_markov_zero_cc(
                root, spec.right_seed ^ epoch_salt, simple=simple
            )
            return MaterializedSyntheticPair(
                spec,
                "cc",
                left_variant.word,
                right_variant.word,
                left_variant.strands,
                right_variant.strands,
                left_variant.semantic_length + right_variant.semantic_length,
            )
        if spec.stratum == "hard_procedural_cc_upper":
            simple = index % 2 == 0
            root = random_absent_knot_braid(
                procedural_seed,
                simple=simple,
                existing_sha256=self.curriculum.existing_representation_sha256,
            )
            left, right, cc_count = procedural_cc_pair(procedural_seed, simple=simple, root=root)
            # Later generations may choose one or two CC independently. The
            # stored interval is widened conservatively rather than pretending
            # that the replayed path is a lower bound.
            if cc_count > int(spec.distance_upper):
                raise ValueError("procedural CC descriptor exceeds its upper bound")
            return MaterializedSyntheticPair(
                spec,
                "cc",
                left.word,
                right.word,
                left.strands,
                right.strands,
                cc_count + right.semantic_length,
            )
        left = sidecar(spec.left_representation)
        right = sidecar(spec.right_representation)
        if spec.stratum == "hard_gordian_catalogue":
            augmented_left = scramble_zero_cc(
                left,
                spec.left_seed ^ epoch_salt,
                spec.left_scramble_depth,
                mirror=spec.left_mirror,
            )
            augmented_right = scramble_zero_cc(
                right,
                spec.right_seed ^ epoch_salt,
                spec.right_scramble_depth,
                mirror=spec.right_mirror,
            )
            if self.replay_check:
                left_root = tuple(-value for value in left.word) if spec.left_mirror else left.word
                right_root = (
                    tuple(-value for value in right.word) if spec.right_mirror else right.word
                )
                if reverse_markov_zero_cc(augmented_left) != (left_root, left.strands):
                    raise ValueError(f"left Gordian scramble {index} failed replay")
                if reverse_markov_zero_cc(augmented_right) != (right_root, right.strands):
                    raise ValueError(f"right Gordian scramble {index} failed replay")
            return MaterializedSyntheticPair(
                spec,
                "cc",
                augmented_left.word,
                augmented_right.word,
                augmented_left.strands,
                augmented_right.strands,
                augmented_left.semantic_length + augmented_right.semantic_length,
            )
        if spec.stratum.startswith("hard_"):
            augmentation_limits = (
                {"minimum_insertions": 60, "maximum_insertions": 120}
                if "_long_detour" in spec.stratum
                else {}
            )
            augmented_left = augment_zero_cc(
                left, spec.left_seed ^ epoch_salt, **augmentation_limits
            )
            augmented_right = augment_zero_cc(
                right, spec.right_seed ^ epoch_salt, **augmentation_limits
            )
            if self.replay_check:
                if reverse_zero_cc(augmented_left) != left.word:
                    raise ValueError(f"left augmentation {index} failed inverse replay")
                if reverse_zero_cc(augmented_right) != right.word:
                    raise ValueError(f"right augmentation {index} failed inverse replay")
            left_word = augmented_left.word
            right_word = augmented_right.word
            added_length = augmented_left.semantic_length + augmented_right.semantic_length
        else:
            left_word = left.word
            right_word = right.word
            added_length = 0
        return MaterializedSyntheticPair(
            spec,
            spec.metric,
            left_word,
            right_word,
            left.strands,
            right.strands,
            spec.base_witness_semantic_length + added_length,
        )

    def __iter__(self) -> Iterator[MaterializedSyntheticPair]:
        # Imported lazily so worker processes own their read-only connections.
        from rf_knots.embedding_supervision import UnknotDBEmbeddingSidecar

        worker = get_worker_info()
        start = self.start_position + (worker.id if worker else 0)
        stride = worker.num_workers if worker else 1
        multiplier = 6364136223846793005 % len(self.curriculum)
        while math.gcd(multiplier, len(self.curriculum)) != 1:
            multiplier += 2
        offset = _seed(self.curriculum.seed, self.epoch, "order") % len(self.curriculum)
        with UnknotDBEmbeddingSidecar(self.sidecar_path) as sidecar:
            cached_representation = lru_cache(maxsize=32_768)(sidecar.representation)
            position = start
            while self.infinite or position < len(self.curriculum):
                generation, within_generation = divmod(position, len(self.curriculum))
                generation_offset = _seed(
                    self.curriculum.seed,
                    (self.epoch << 32) ^ generation,
                    "order-generation",
                ) % len(self.curriculum)
                index = (multiplier * within_generation + offset + generation_offset) % len(
                    self.curriculum
                )
                yield self._materialize(cached_representation, index, generation)
                position += stride


def _pad_words(words: Sequence[tuple[int, ...]]) -> tuple[Tensor, Tensor]:
    lengths = torch.tensor([len(word) for word in words], dtype=torch.long)
    width = max(int(lengths.max().item()), 1)
    padded = torch.zeros((len(words), width), dtype=torch.long)
    for row, word in enumerate(words):
        if word:
            padded[row, : len(word)] = torch.tensor(word, dtype=torch.long)
    return padded, lengths


def collate_embedding_pairs(rows: Sequence[MaterializedSyntheticPair]) -> dict:
    """Pad variable braid words and build one GPU-transferable batch."""
    if not rows:
        raise ValueError("cannot collate an empty batch")
    left, left_lengths = _pad_words([row.left_word for row in rows])
    right, right_lengths = _pad_words([row.right_word for row in rows])
    return {
        "index": torch.tensor([row.spec.index for row in rows], dtype=torch.long),
        "stratum": tuple(row.spec.stratum for row in rows),
        "metric": tuple(row.metric for row in rows),
        "left_word": left,
        "right_word": right,
        "left_length": left_lengths,
        "right_length": right_lengths,
        "left_strands": torch.tensor([row.left_strands for row in rows], dtype=torch.long),
        "right_strands": torch.tensor([row.right_strands for row in rows], dtype=torch.long),
        "lower": torch.tensor([row.spec.distance_lower for row in rows], dtype=torch.float32),
        "upper": torch.tensor([row.spec.distance_upper for row in rows], dtype=torch.float32),
        "witness_semantic_length": torch.tensor(
            [row.witness_semantic_length for row in rows], dtype=torch.long
        ),
        "confidence": torch.tensor([row.spec.confidence for row in rows], dtype=torch.float32),
    }
