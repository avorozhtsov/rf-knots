# Common-domain greedy policies: completed diagnostic and replay audit

2026-09-21. Protocol82 completed all96 conditions, exit0. Independent transcript
audit passed all96, including native Q controller replay and reference replay
of failures as well as successes. Six focused tests passed; new runner/auditor
and tests passed Ruff. This is a TRAIN diagnostic, NOT a held-out promotion or
an MCTS comparison. No model weights or existing training sources changed.

Artifacts: `artifacts/common-greedy-policy-diagnostic-v1/`.
Final manifest SHA256: `71abc5a8ed031656dc6122f7b07844dbe2fcadf7faa08151f3e4ef9ec081a4cd`.
Replay audit SHA256: `227ec4ca880d38f35fdabdbe211641b64d09ce2ed3e67bc7a1b8676c9b0945f5`.
Episode files include selected actions, full before/after words, native head
positions, counters, failure reasons and replayable successful witnesses.

## Observed policy results

Same16 roots, capacities48/12, CCcap2, at most128 actual neural decisions per
attempt, no MCTS/cache/noise, no cycle avoidance, PASS suppressed. RF scores
all legal semantic edits; Q retains its local controller. Counts are NOT FLOPs.

| Frozen policy | L10 solved | L1000 solved | Neural samples, L10 / L1000 |
| --- | --- | --- | --- |
| Q254 raster | 2/16 | 1/16 | 1810 / 1927 |
| Q304 strand graph | 4/16 | 2/16 | 2014 / 1974 |
| RF joint57 | 4/16 | 4/16 | 1584 / 1585 |

Every unsuccessful attempt exhausted128 decisions; no inaccessible-action or
replay error explains these failures. Reference routing had already found
solutions in6..47 native plies. A known shorter solution does not mean the
policy can discover it without search.

Returns to an already visited exact braid state after a semantic move were
frequent: RF1412/1584 at L10 and1385/1585 at L1000; Q2541536/1675 and1406/1552;
Q304813/964 and627/780. Denominators are semantic moves, not neural decisions.
Head-only movement is excluded from those counts. These are observed repeats,
not a proof all corresponding choices are bad: Q can retain a different head
or controller context. Do not blindly prune Q states by braid word alone.
Q304 spent1050 and1194 additional decisions on controller-only movement.

## Concrete improvement over a supplied witness

Panel source index42 (`rf-joint57-0042-L10.json` and `...-L1000.json`):

- Constructive reference: CC2,10 semantic moves, L10=30 and L1000=2010.
- RF independently selected: CC1,9 semantic moves, L10=19 and L1000=1009.
- Both resulting witnesses replayed. Independent integer Burau determinant
  calculation gives3, hence the knot is nontrivial and requires at least one CC.
- Initial q0=(word length + strands -1)/2=8. The existing cost-identity checker
  verified every prefix: six REDUCE, one CC, two DESTABILIZE, zero excess.
  Therefore m >= q0+CC and L_lambda >=8+(lambda+1)*1. Both RF costs attain
  that lower bound: they are optimal for this initial representation under
  the project's declared semantic move set and cost accounting.

This is not a newly discovered knot invariant/database bound: it is a generated
TRAIN diagnostic representation, and training-history overlap is not excluded.
It nevertheless demonstrates that the policy can find a better alternative
than the supplied constructive witness; evaluation must not require exact
imitation of that witness's action sequence.

Q304 also found CC1 for index42 at L1000, with19 semantic moves and cost1019.
Q254 found the original CC2-level cost30 at L10. On index0 Q254's cost16/1006
is better than RF's available L1000 cost1008. There is no universal ordering
of the policies across roots and criteria.

## What this changes, and what remains open

The assertion that absence of Q-Run initialization alone explains RF's old1/64
MCTS result is not supported: RF is not uniformly worse on this shared short
diagnostic. Equally, this does not show Q pretraining is useless. Different
controllers, native training histories, budget features and the CC/PASS mask
prevent a causal architecture/pretraining attribution. Greedy policy success
is not MCTS strength; the old independent64-root long-word test remains separate.

The next search question is whether proper MCTS/resource allocation escapes
these repeated-state trajectories, while preserving meaningful controller
state. Freeze that comparison before scoring; count actual inference, replay
successes, and do not turn budget failures into unconditional bad-state labels.
For long inputs, retain the original architectural direction: learned local
computations plus scalable global context, including optional knot embedding
and window position. This diagnostic neither implements nor rejects it.

The primary experiment continues unchanged: graph4096/4096 complete, control
1840/4096 and advancing at this diagnostic's audit, independent evaluators
waiting for the final pair. Its distance metrics and long-input search results
remain necessary before any scientific promotion or declaration of completion.
