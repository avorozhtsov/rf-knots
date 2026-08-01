# Evidence, bounds and frozen benchmarks

The unit of evidence is an append-only `rf-knots-evidence-v1` JSONL record. It
retains the exact input instance, solver/checkpoint and search budget, optional
lower-bound claims with provenance, and a complete unknotting witness.

## What the verifier guarantees

`UnknotWitness` stores semantic actions (`REDUCE`, `BRAID`,
`CROSSING_CHANGE`, and so on), not flat policy-head indices. On construction and
read, every action is checked for legality, every recorded intermediate state is
recomputed, the closure is required to remain one component, and the final state
must be the empty one-braid. A corrupt hash or replay mismatch rejects the whole
record. Only an incomplete final JSON line from a crashed concurrent append may
be ignored.

The number of `CROSSING_CHANGE` actions in a verified witness certifies
`u(K) <= k`. A `LowerBoundClaim` is accepted only with its method and source.
The API reports an exact unknotting number only when the strongest recorded
lower bound equals the best verified upper bound.

`rf_knots.lower_bounds` combines the locally computed signature bound with a
pinned 2026-08-01 KnotInfo snapshot of Rasmussen `s`, Ozsváth–Szabó `tau`, and
the Nakanishi index for all 2,978 knots through 12 crossings (including the
unknot). The committed derived table records the source hash; regenerate it with
`scripts/extract_knotinfo_lower_bounds.py`, which refuses any other XLS hash.

```python
from pathlib import Path

from rf_knots.evidence import EvidenceStore

store = EvidenceStore(Path("artifacts/evidence.jsonl"))
for instance_id, record in store.best_witnesses().items():
    print(instance_id, record.certified_upper_bound, record.certified_lower_bound)
```

Validate a completed store with:

```bash
uv run rf-knots evidence-verify artifacts/evidence.jsonl
```

## Sharing between different controllers

`UnknotWitness.from_states()` translates a sequence of braid states into the
shared semantic action space. Controller-only steps—serial head travel, scanning,
register changes or tape writes—leave the braid unchanged and are omitted. A
state-changing step that cannot be replayed as one legal shared braid action is
rejected.

`shared_witness_targets()` then exposes each verified `(state, next_state)` pair,
whether it is a crossing change, and the remaining crossing-change and move
counts. These are admissible upper bounds. `upper_bound_hinge()` penalises a
cost-to-go prediction only when it is worse than the known witness; it does not
mislabel the witness as optimal.

## Frozen partitions

[`../benchmarks/rungs-v1.json`](../benchmarks/rungs-v1.json) is the first frozen
manifest. It contains the 23 distinct rung source words, not 41 repeated ladder
positions, and assigns identities to train/validation/test using a salted SHA-256
split. Its source `docs/rungs.json` hash is recorded in the manifest.

This is a calibration set. It is not an external challenge set: most knots have
published `u`. The external ReAPR/Dryad collections are catalogued separately in
[`../benchmarks/catalog.json`](../benchmarks/catalog.json); their downloaded
archive hashes and parsed-instance manifests must be pinned before a result is
reported.

## Fixed baselines

The CLI exposes three adapters:

```bash
uv run --with snappy rf-knots baseline snappy "1,-1" --strands 2
uv run --with snappy --with regina rf-knots baseline regina "1,1,1" --strands 2
uv run --with snappy rf-knots baseline reapr "..." --strands N
```

The ReAPR adapter calls the documented `knoodlesimplify` executable and therefore
also requires Knoodle to be installed. SnapPy and ReAPR success are positive
simplification evidence; failure is inconclusive. Regina runs complete
solid-torus recognition on the knot complement and returns either `unknot` or
`nontrivial`.
