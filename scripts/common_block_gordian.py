"""Replay-defined general short-distance pairs with a common positive braid block."""

from rf_knots import reference
from rf_knots.actions import ActionSpec, COMMUTE, CROSSING_CHANGE, DESTABILIZE, INSERT, REDUCE
from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_replay_corpus import invariants, mirror
from rf_knots.evidence import UnknotWitness
from rf_knots.interleaved_generator import inverse_action


def core(seed, index, changes):
    source = 20 * index + (10 if changes == 2 else 0) + index % 10
    return conjugate_witness(seed, source, max_strands=12, min_depth=8, max_depth=32)


def combined_start(witness, genus):
    return (1,) * (2 * genus + 1) + tuple((1 if g > 0 else -1) * (abs(g) + 1)
                                         for g in witness.start.word), witness.start.strands + 1


def lifted_core_path(witness, genus, spec):
    offset = 2 * genus + 1
    start, strands = combined_start(witness, genus)
    word = start
    actions, inverses = [], []
    for step in witness.steps:
        if step.action.kind == "CROSSING_CHANGE":
            action = spec.encode(CROSSING_CHANGE, offset + step.action.position)
        elif step.action.kind == "REDUCE":
            # Core witnesses reduce interior pairs; no wrapping through H.
            if step.action.position + 1 >= len(word) - offset:
                raise ValueError("core reduction would wrap through the common block")
            action = spec.encode(REDUCE, offset + step.action.position)
        elif step.action.kind == "DESTABILIZE":
            action = spec.encode(DESTABILIZE)
        else:
            raise ValueError("unsupported core construction move")
        if not reference.is_legal(spec, word, strands, action, True):
            raise ValueError("illegal common-block core transition")
        after, n = reference.apply(spec, word, strands, action)
        inverse = inverse_action(spec, word, strands, action, after, n)
        if inverse is None:
            raise ValueError("common-block core transition is not exactly invertible")
        actions.append(action); inverses.append(inverse)
        word, strands = after, n
    if (word, strands) != ((1,) * offset, 2):
        raise ValueError("lifted core does not reach the untouched common block")
    return actions, list(reversed(inverses))


def make_pair(seed, index):
    genus, kind = 5 + (index // 3) % 8, index % 3
    left = core(seed, index, 2 if kind == 1 else 1)
    # Keep the second core's strand schedule fixed by using its explicit index.
    right_index = 20 * (index + 1024) + index % 10
    right = conjugate_witness(seed, right_index, max_strands=12, min_depth=8, max_depth=32)
    if kind == 2:
        left_fp, right_fp = invariants(left.start.word, left.start.strands), invariants(right.start.word, right.start.strands)
        if left_fp[1] * right_fp[1] > 0:
            right = mirror(right)
    a, n = combined_start(left, genus)
    b, m = combined_start(right, genus)
    spec = ActionSpec(max(len(a), len(b)) + 4, max(n, m) + 1)
    left_path, _ = lifted_core_path(left, genus, spec)
    tail = [action for _ in range(genus) for action in
            (spec.encode(CROSSING_CHANGE, 0), spec.encode(REDUCE, 0))] + [spec.encode(DESTABILIZE)]
    if kind == 2:
        right_path, backwards = lifted_core_path(right, genus, spec)
        actions = left_path + backwards + right_path + tail
        prefix = len(left_path) + len(backwards)
    elif kind == 1:
        actions = left_path + tail
        prefix = next(i + 1 for i, action in enumerate(left_path) if spec.decode(action)[0] == CROSSING_CHANGE)
    else:
        change = next((spec.encode(COMMUTE, p) for p in range(len(a))
                       if reference.is_legal(spec, a, n, spec.encode(COMMUTE, p), False)), None)
        if change is None:
            change = spec.encode(INSERT, 0, 1, 1)
        changed, new_n = reference.apply(spec, a, n, change)
        undo = inverse_action(spec, a, n, change, changed, new_n)
        if undo is None:
            raise ValueError("isotopy prefix has no exact inverse")
        a, n = changed, new_n
        actions, prefix = [undo] + left_path + tail, 1
    witness = UnknotWitness.from_actions(a, n, spec, actions)
    witness.verify()
    return witness, prefix, {"genus": genus, "attempt_kind": kind,
                              "left_core_index": 20 * index + (10 if kind == 1 else 0) + index % 10,
                              "right_core_index": right_index}
