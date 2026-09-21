# Ordinary-neighborhood supply diagnostic: insufficient fresh support

2026-09-21. The fixed [probe72](72-ordinary-neighborhood-fresh-query-probe.md)
completed all1,024 attempts and its separate auditor replayed every draw/path.
No checkpoint was evaluated. No training data, optimizer or protocol70 test
selection was changed.

The source pool contained3,058 unique non-TRAIN physical representations from
the prior audited sidecar paths. Raw source sizes span L7..254 and N3..21;
these ranges do not imply new or independent knot types. References were chosen
uniformly by representation, not by topological complexity or invariant group.

Outcome:

| Classification | Attempts |
| --- | ---: |
| Query not invariant-owner TEST | 913 |
| TEST query group already in historical panels | 110 |
| Fresh eligible TEST query | 1 |

The one eligible result is attempt1019, reference representation176150, with
forward flips at positions18 and13 (zero-based). In query-to-reference direction
its recomputed signed fingerprints are `(249,-4) → (19,-6) → (33,-4)`.
The endpoints have equal signed signature, but different determinants, so the
certified interval is **[1,2]**, not exact2. This is one unique ordinary pair and
one conservative group `(249,4)`. It supplies no adequate independent accuracy
test and no measurement of unknotting number.

The independent audit reconstructs the complete reference population from source
SQLite bytes/splits, repeats the declared RNG draws, checks each literal signed
CC step, recalculates invariants and independent determinants, verifies ownership
and all prior-query/optimizer exclusions, and independently checks pair/group
deduplication and summary counts. It passes integrity but deliberately leaves
`evaluation_ready=false`. The protocol does not permit treating the diagnostic
itself as a model evaluation.

Artifacts:

- `artifacts/ordinary-neighborhood-fresh-query-probe-v1/report.json`, SHA
  `1506609f6b2c4d79a8d78846721157c16f8cf00d50fd58c5d565d9beec81edce`;
- `artifacts/ordinary-neighborhood-fresh-query-probe-v1/independence-audit.json`, SHA
  `83582028de2d07aa2b982febd05e8e54bf3ad7b3e433f6defeb19db6c234bf9d`.

## Consequence

Do not scale this same probe until a quota is filled or relabel its known groups
as independent. Under the fixed source/sampling rule, fresh support is too sparse.
The current short-u/full-policy comparison remains meaningful and proceeds
unchanged, but broad general-distance evidence remains missing.

The next data-source hypothesis should address this lack of diversity directly:
construct larger, nontrivial ordinary reference braids, preserve one-component
closure, and certify nearby query/reference distances through explicit1/2CC
paths. Such pairs need not come with full unknotting witnesses, because the
task is general d(K1,K2). A new experiment must freeze its construction, seed,
effort, exclusions and adequacy gate before execution; exclude this diagnostic
query group too. Report that synthetic-family coverage separately rather than
calling it a representative catalogue test or silently replacing70.
