# Complete protocol97 evaluation sequence

2026-09-21. This binds the implementation of the already declared97 evaluation,
not a new optimization or TEST-driven protocol change. Training continues under
its original frozen sources. No paid GPU, release, or checkpoint selection.

`run_raster_log_evaluation.py` waits for the exact live serial training launcher
and both final2048-update arms; it does not launch or restart training. It then
runs13 serial jobs: one final-pair integrity audit, eight metric reports
(two arms x saved initial/final x short-u/general panel), and two full search
runs each followed by independent replay and semantic-cost certification.

Each search covers64 roots x L10/L1000 =128 conditions, retaining exactly70's
caps, seed20261034, eight simulations, c_puct1.5, temperature1 and widening/
insertion4. The model's raw diagram embeddings produce tree priors; its learned
policy supplies continuations. There is no invented value or oracle incumbent.
Raster scoring uses microbatch2 and records actual neural work, including raw
tree anchors. Shared architecture weights are immutable throughout evaluation.

Search output is committed in eight-condition shards with hashes. Recovery must
verify model, panel, code, roots, exact committed prefix and resource contract;
an existing live worker is adopted, never duplicated. Non-resumable partial
reports require inspection. Outputs and source snapshots are preserved.

The controller computes its final summary only after all13 verified outputs.
It recomputes metric interval/constant/ranking summaries from all predictions,
paired initial-to-final metric changes, final Q254-versus-random metrics and
search outcomes, censored-path behavior and replay-certified cost floors.
Source/input hashes and completed-output hashes bind the report. The mutable
controller status file is bound via its fixed contract rather than a circular
summary/manifest hash dependency.

This comparison isolates local initialization and its resulting fixed policy
reference, not causal embedding benefit: BOTH arms have embedding channels.
There is also no initial-model search in this declared matrix. Keep that limit
explicit when interpreting final-model L10/L1000. The reused70/74 panels remain
shared TEST, and Q254 historical exposure is not fully audited. Successful
execution alone neither promotes a model nor establishes useful invariants.
