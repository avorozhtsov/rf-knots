# Additive F bank and new-source gate v1

2026-10-11. The independent b256 hard-stage audit completed 64/64 shards and
full replay/ownership/caps checks. Its report SHA is
`908a4f3c0c175aafbf5453244dafbb635cd0a4e92b0756dc60abfbd6535d5000`.
Only 2/16 new roots have F signal, both stage 2 CC1; stage 3 is 0/8.
There are three exact F contexts, 10,810.096 CPU seconds, and zero paths
cheaper than constructive reference. This bounded sample does not prove that
hard roots cannot work; it rejects blind budget scaling of this source.

Construct a separate immutable additive TRAIN bank from the independently
audited old joint-F bank (392 distinct F-usable roots), the five new roots
admitted by the b64 audit, and the two new stage-2 roots admitted by the b256
audit. Deduplicate by TRAIN query group, never by seed, lambda or braid view.
Keep old root files byte-identical, with their SHA, and recompute exact F
signals from the verified reference plus every admitted completed path for
each new root/objective. Pin both source manifests, audit reports, roster,
builder source, and all existing root SHA. Full replay, caps and owner checks
are a separate independent audit gate before optimizer use. Expected minimum
coverage is 399 distinct roots, but report the actual audited count; fail
closed on duplicate groups or lost old roots. Never alter the prior bank.

An exploratory R2-only optimizer continuation may follow in a **new**
versioned contract, preserving step-133506 model/objective/auxiliary head,
AdamW and Python/torch/MPS RNG. Use the new bank only after its full audit.
F remains the primary shared-encoder gradient with the prior 10% auxiliary
and 20% rehearsal norm ceilings. Draw old roots at least 25% of the time;
to avoid overfitting only seven new roots, initially target 80% old and 20%
new, with a bounded first block and checkpoint SHA/gradient checks. This is
continuation research, not 1k-root or hard-generalization success. A fresh
disjoint DEV contract must be frozen before selecting any later checkpoint.

In parallel, test a new TRAIN source family based on replay-complete
interleaved walks from certified positive-braid seeds, rather than the prior
single conjugate-block generator. Freeze seeds/config/quotas and heldout
exclusions before seeing F yield. First audit new root-group supply and
witnesses, then run a small balanced canonical search pilot, not bulk CPU.
If its stage-3 F yield remains poor, redesign the source again. Capped
no-solve is censored, not a negative training action. The long-term target is
30,000 different TRAIN F-usable root groups at 1k/3k/10k/30k gates, while
retaining all previous admitted roots in future banks/sampling.

Do not use viewed DEV or TEST150 for checkpoint choice. Unknotdb remains
untouched until separate ownership/dedup/replay/strict-cost staging. No
cloud purchase, database publication, commit, merge or push.
