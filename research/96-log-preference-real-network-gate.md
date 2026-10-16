# Full-trajectory log-preference: real-network gate before a revised experiment

2026-09-21. TRAIN-only engineering gate following the deliberately interrupted94
and numerical diagnosis95. Do not resume94, mutate its sources or use upcoming
70/74 TEST scores to select this repair.

Use exact protocol88 Q254-local initialization and the saved q254 update16
checkpoint from94. Recheck all transitive source/input hashes and the diagnosis.
Use all four complete audited relative groups, not eight sampled rows. Compute
each group's fixed reference log probabilities from the INITIAL model.

At both model states run the new two-pass full-group backward, CPU/one thread,
microbatch1, weight0.01. Record loss, log score, surviving coefficient mass,
finite parameter-gradient norms, two-pass agreement, wall time and process peak
RSS. At initialization S must be approximately0 and all groups must give a
nonzero finite parameter gradient. At update16 distinguish a nearly satisfied
group from an adverse group; do not require every global gradient to be large.
The intrinsic common comparison scale C and factor2 are defined in95 and are
not selected by TEST performance.

Independently compare the complete smallest group's one-graph autograd loss and
parameter gradient with the bounded-memory two-pass implementation at the
initial model: loss rtol1e-4/atol1e-3, gradient rtol1e-3/atol5e-5. Record actual
maximum errors; do not claim bitwise equality. A successful single-group unit
test does not replace the all-group real-network check.

No optimizer updates or scientific model selection in this gate. The initialization,
saved partial training checkpoint and all model tensors must remain unchanged.
The measured full-group costs inform the update cadence of a separately frozen
replacement training protocol; no silent change to94 is permitted. This gate
alone is not evidence of useful distances, lower L10/L1000, or a completed goal.
