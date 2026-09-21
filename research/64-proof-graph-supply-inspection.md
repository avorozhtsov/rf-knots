# Proof-graph supply: evidence before the next scaling step

2026-09-21. Read-only inventory during completion of experiment 57/58/60.
This is not a new dataset, training authorization gate, or an independent
quality evaluation. No graph/sidecar/checkpoint has been changed.

## Sources actually inspected

All paths below are relative to the sibling `../unknotdb/outputs/`.

| Sidecar | SHA256 |
|---|---|
| `unknotdb-embedding-pairs-v4.sqlite` | `297e420f7a97aefb15615419176102ddd31ac492acc130218c8c2e236a4bceea` |
| `unknotdb-dgkt-graph-supervision-v1.sqlite` | `d1e11ee3a44234b332b64f989ab7b680ee4de01ccc6563d942c8057abf287195` |

The first database identifies its schema as `unknotdb-embedding-pairs-v3`:
filename generation and schema version are different. Its metadata pins
proof SHA `e31028def491caab7317ab63eb9829a1e63e1fa9a512f2ccf4daac6d178ffd20`
and identification SHA
`5c9045cf39bc7b7dd4c73081b349e2073847cdd1ca2240555e19bd970a8de18c`.
These are stored provenance claims, not a new replay of the upstream graph.

## Metric supply is large but not rich in exact nonzero CC labels

Read-only SQL (`sqlite3 -readonly`) over `pairs`, grouped by metric, split,
distance_lower, distance_upper, found the following TRAIN CC rows:

| Interval | Pairs |
|---|---:|
| [0,0] | 124275 |
| [0,1] | 146710 |
| [0,2] | 27015 |
| [1,1] | 352 |
| [1,2] | 94 |
| [2,2] | 0 |

There are 203503 TRAIN representations, maximum raw word length 254 and
maximum strand count 21. These maxima need not occur on the same row.
Rows are not independent knot types or played trajectories. Sampling ten
million times from this source does not create ten million independent knots.
In particular, its CC intervals alone do not supply any certified exact-2
examples; most nonzero upper bounds admit a prediction of zero. Additional
lower bounds may strengthen some rows, but must be recomputed/verified before
making that claim. RM exact-2 labels are not CC exact-2 labels.

Crucially, metadata defines CC as the **mirror-orbit task quotient**, not
ordinary Gordian distance between the supplied oriented presentations.
TRAIN includes 9135 `mirror_orbit_task_symmetry` pairs. Such zero-cost mirror
relations cannot silently become ordinary isotopy labels. The existing
ordinary-braid evaluation distinguishes these semantics. Reuse requires
explicitly compatible targets or replay in the ordinary action system;
dropping only one provenance category does not certify the remaining graph
paths to be mirror-free.

## Policy supply has a different split contract

The DGKT graph-supervision sidecar contains 92300 TRAIN CC-solver labels and
48010 TRAIN preprocessor labels. Its documented split is a hash of the exact
physical state, **not** a knot-identity/group partition. Other representations
of the same knot may therefore occur in a different split. Copying its TRAIN
rows into this experiment would not by itself preserve our held-out groups.

Labels preserve several witnessed actions for a state when available. They
are selected from known graph routes, not unique or globally optimal actions.
Coordinate and mirror instructions do not become policy labels; theorem
macros are excluded. A bridge must preserve the exact state/action chart and
verify which continuations complete in our ordinary Artin primitive system.

## Consequence for the next experiment

These sources offer genuinely different diagram families from our current
constructive conjugate-block examples, but are not drop-in extra training.
Before any reuse:

1. Pin a complete source snapshot and reconstruct/replay compatible witnesses.
2. Reassign/filter entire known knot groups against **all** existing TRAIN and
   held-out identities, mirrored groups, fingerprints and trajectory states.
   Do not retain the source's physical-state split as an identity guarantee.
3. Keep CC/Gordian and mirror-quotient labels distinct. Recompute useful lower
   bounds and publish actual exact-0/1/2 support, including empty categories.
4. Preserve set-valued positive continuations and route provenance. Unknown
   alternatives and budget failures are not bad-action labels.
5. Measure raw L/N, normalized L/N, nontrivial diagram transformations and
   group diversity separately. N<=21/L<=254 supply alone is not large-size
   generalization; repeated zero-CC lifts are not intrinsically harder knots.

This inspection rejects the shortcut "there are hundreds of thousands of
rows, therefore simply train longer". It does not reject the embedding
architecture or proof-graph distillation. The full network and MCTS remain
the training target, without adding head-only residual adapters.

## Reproducible inventory queries

Run against the pinned files with `sqlite3 -readonly`:

```sql
SELECT metric, split, distance_lower, distance_upper, count(*)
FROM pairs GROUP BY 1,2,3,4;
SELECT split, count(*), max(word_length), max(strands)
FROM representations GROUP BY split;
SELECT provenance, count(*) FROM pairs
WHERE metric='cc' AND split='train' GROUP BY provenance;
SELECT * FROM meta;
```

For the graph-supervision database:

```sql
SELECT task, split, count(*) FROM labels GROUP BY 1,2;
```

The source contracts inspected are sibling `schema/embedding-pairs-v0.md`
and `schema/graph-supervision-v1.md`. Those contracts and an inventory are not
a substitute for the new bridge's replay/partition audit.

## Executed ordinary-pair probe and independent audit

The prospective protocol in note 65 has now completed all512 selected traces
(256 single-CC plus256 two-CC paths) from62086 eligible source-local CC edges.
No model scores or optimizer were used. Output
`artifacts/ordinary-graph-pair-supply-v1/report.json`, SHA256
`39e04b00e9e688a0d91961812aab700f9a1c980aafb77b60005ffad85b026dd3`.

An independent consumer rechecked every exact state against the source DB,
every edge ID/provenance/source split, every literal sign flip, both invariant
bounds and independent determinants, the fixed sampling selection, and all
15 current historical panel query groups. Audit **passed**:
`partition-replay-audit.json`, SHA256
`e2c53ae0e59215710d446b219c63ba9425286af273f335359a5189864b925f4b`.
All source hashes were checked again after completion. Five helper/audit tests
pass, including rejection of a free mirror and corrupted bounds.

| Ordinary interval | All selected paths | Eligible owner-TRAIN, held-out-query-disjoint |
|---|---:|---:|
| [0,1] | 53 | 49 |
| [1,1] | 203 | 101 |
| [0,2] | 69 | 52 |
| [1,2] | 135 | 68 |
| [2,2] | 52 | 13 |

283 eligible traces span34 invariant groups across endpoints/midpoints.
Their raw states have L7..42, N<=9. These are not large-size examples.
202 eligible traces have an endpoint fingerprint (1,0); that fingerprint
alone is **not** an unknot certificate, so they remain general Gordian pairs,
not automatic u(K) targets or full unknotting witnesses. Known TRAIN reference
endpoints are allowed to overlap earlier tests; held-out query groups are not.

The exact-2 evidence is newly derived from ordinary replay plus signed
signature differences, not taken from the sidecar's mirror-quotient labels.
This resolves a concrete feasibility question: the graph can supply these
labels despite its exported [2,2] count being zero. It does not establish model
improvement or an optimal local policy action.

The full eligible source graph has100500 vertices,20049 possible distinct
midpoints of degree>=2, maximum degree9, and28857 unordered two-edge paths.
These are inventory counts, not all replayed/partition-audited training rows.
A next full-source preparation can enumerate them deterministically, derive
ordinary bounds, apply the verified split/exclusion rules, deduplicate states
and pairs, then balance groups/classes before any new training. The existing
512-path probe is not to be extended opportunistically to obtain a desired
scientific result, and its non-TRAIN rows must not enter an optimizer.
