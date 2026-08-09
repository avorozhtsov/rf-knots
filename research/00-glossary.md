# Glossary for the current experiments

This glossary defines the active, semantic-move experiment. Historical terminology
and version-by-version reports are preserved in
[`../archive/pre-semantic-moves-v1/research/00-glossary-full.md`](../archive/pre-semantic-moves-v1/research/00-glossary-full.md).

## Objective and actions

**Semantic action.** An operation that changes the mathematical braid or knot
state: for example a Reidemeister/Markov-equivalent rewrite or a crossing change.

**Charged action.** A semantic action counted in `moves`. Controller-local work is
not charged.

**Internal action.** A network-specific operation such as moving a read head,
writing private memory, or changing a finite-control state. It consumes an
internal search budget but contributes zero to the scientific objective.

**Objective.** `L_AB = A * cc + B * moves`, where `cc` is the number of crossing
changes and `moves` is the number of charged semantic actions. `L10` means
`A:B = 10:1`; `L1000` means `1000:1`.

**Objective channel.** The network input identifying the requested ratio, encoded
as a normalized `log(A/B)`. Architecture names do not contain the ratio.

**Remaining-L budget.** A network input and search constraint describing how much
objective cost remains. It is not derived solely from a local value prediction.
Search rejects a branch when a charged action would make this budget negative.

**Internal-step budget.** A separate limit on free controller work. It prevents an
agent from scanning or writing forever without corrupting the scientific cost.

## Search and learning

**Scientist.** One independently initialized solver, including its architecture,
weights, replay state, optimizer state, and search configuration.

**Architecture name.** A stable description of the information path, such as
`window-local` or `raster-axial`. Simulation count, seed, objective, and curriculum
belong in run metadata, not in this name.

**Simulations per move.** The number of MCTS simulations used to choose one action.

**Evaluation attempt.** One complete search attempt on one representation at a
specified checkpoint, seed, objective, budget, and simulations-per-move value.

**Paired attempt.** Treatment and control evaluation attempts using the same
representation, initial weights, objective, budgets, simulation count, and random
seed. Only the intended intervention differs.

**Round.** One newly selected representation.

**Training block.** Ten rounds and therefore normally ten new representations.
Adaptation decisions and paired portfolio evaluation happen at block boundaries.

**Cycle.** One frontier acquisition phase, its optimizer updates, optional
rehearsal/sharing dose, and the subsequent evaluation certificate.

**Native-refresh attempt.** A scientist's own MCTS attempt used to acquire or
improve a solution, rather than to imitate a donated route.

**F_native.** The number of frontier self-play/training cycles before the next
block evaluation. It increases when recent acquisition is too weak.

**F_old.** The number of rehearsal cycles on stored tasks. It increases when the
portfolio loses solved tasks or worsens in capped objective.

**Adaptive simulations.** A stepwise increase in simulations per move when a
paired evaluation shows that the current search allocation does not attain the
target acquisition rate. Compute is recorded separately from model identity.

**Positive episode.** A complete attempt ending in a verified unknotting witness.
All useful positions in the episode may train policy/value heads.

**Negative episode.** A failed attempt. It may train `p(solve)` and calibrated
search targets, but its actions are not treated as an expert route.

**Success-balanced replay.** Replay sampling that explicitly requests positive
and negative examples from persistent banks rather than inheriting the accidental
class ratio of the latest task. Balance is declared per loss; it is not necessarily
50:50 for every head.

**`p(solve)`.** A calibrated estimate of solving within the supplied objective and
internal budgets. Budget-censored failures and ordinary failures are labelled
separately for audit.

**Auxiliary heads.** Direct predictions of crossing changes, semantic moves, and
`p(solve)`. The objective prediction is constructed from the first two and the
requested `A:B`; it is not an unrelated free scalar.

## Sharing

**Witness / solution.** A verifier-replayable sequence of semantic actions that
unknots a particular representation. “Solved” always means within the declared
budgets; the witness itself remains valid when evaluated with a larger budget.

**Donor.** The scientist that generated a verified witness offered to another
scientist.

**Compact donor witness.** A donor witness with low semantic objective cost. The
word “compact” refers to charged semantic cost, never to donor-private head shifts
or memory operations.

**Receiver-native cost.** The best cost already obtained by the receiving
scientist on the same representation and objective, after replaying semantic
actions in the shared environment.

**Inferior donation.** A witness whose verified semantic cost is not strictly
better than the receiver's incumbent. It is logged but must not train the receiver.

**Sharing / distillation.** Training an ordinary scientist policy on a strictly
better verified donor witness. Sharing is a data source, not a separate permanent
policy adapter.

**Donation dose.** The number or fraction of optimizer updates whose policy targets
come from eligible donor witnesses in a training block.

**Route learning.** Supervised policy learning from the state-action positions of
a verified successful witness. A failed search is never called a route target.

**Canonical-route loss.** Cross-entropy of the ordinary policy on a fixed,
verifier-replayed witness. It is a diagnostic, not the scientific outcome.

**Sharing effectiveness.** The paired change caused by sharing in solved-set size
and capped objective, with compute matched. Route loss alone is insufficient.

## Evaluation and gates

**Solved set.** The representation identifiers for which at least one specified
attempt produced a verified witness. Paired reports include intersection,
treatment-only, and control-only identifiers.

**Capped objective.** For every evaluated representation, use the best verified
objective value, or a declared empirical failure cap if unsolved, then sum or
average across the fixed set. This lets a lost solve count as a regression.

**Portfolio progress.** A lexicographic safety condition: do not reduce solved-set
size; subject to that, reduce capped objective. Reports also show the two numbers
separately and the full solved-set difference.

**Retention.** Performance on a fixed bank of earlier representations after a
training block.

**Exact retention.** Bit-for-bit unchanged outputs or unchanged success on every
canary. It is a useful secondary diagnostic, not the primary scientific gate.

**Canary degradation.** Losing or worsening a previously solved diagnostic task.
It is expected locally during continual learning; it becomes unacceptable only
when portfolio progress regresses and adaptive rehearsal fails to recover it.

**Cycle certificate.** The record emitted at a block boundary: checkpoint hashes,
data counts per loss, distinct positive/negative representation counts, failure
types, compute, solved-set difference, and capped-objective change.

**Admission gate.** A preregistered paired test that a scientist must pass before
entering the long comparison. It covers native acquisition, auxiliary calibration,
retention, checkpoint reproducibility, and resumability.

**Source-disjoint split.** Train, calibration, and held-out representations are
separated by source knot/augmentation lineage, not merely by serialized braid
word, to prevent equivalent near-duplicates crossing the split.
