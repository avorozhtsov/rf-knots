# F-curriculum S10 yield/CPU pilot, prospective contract

2026-10-04. The v3 candidate roster SHA256 is
`268f1c4d2e8731bcfd4cc743bed767c9864fb74b2a2022ed679fbe36af134506`.
An independent audit replayed all 1,000 constructive witnesses, checked
TRAIN ownership of roots and crossing-change states, exclusions, paired
L10/L1000 instances, distinct invariant query groups and stage quotas
`[140,286,287,287]`. These are **candidates**, not F-usable training groups.

Before bulk collection, sample eight roots in each stage without looking at
search results: the first eight SHA-ordered CC1 roots in stage 0, and the
first four CC1 plus first four CC2 roots in each of stages 1–3. For every
root and each of L10/L1000, run ten independent searches, seeds
`20270000..20270009`, each with 16 simulations, fixed canonical
pgx-mcts-bench search settings from the S10 TRAIN collection: four insertion
candidates, widening batch four, c_puct 1.5, temperature 1.0, distance
prior and cache. Use frozen actor32768 SHA256
`9b8ebee990c1986eec3cda30a00105da9819892dd8649dcc438468a197685cba`.
Total is 32 roots x 2 objectives x 10 independent searches = 640 attempts.
The search outputs are teacher data, **not neural optimizer steps**.

Write one SHA-recorded shard per attempt with full episode outcomes,
solution paths, CPU/wall timing and neural-call counts; resume only from an
unchanged manifest and source/input hashes. Independent audit must verify
all shard SHA values, replay every completed solution from the original
instance, check ownership and exclusions of all visited crossing-change
states, and count distinct solutions and exact F divergence contexts.
The pilot report must separate L10/L1000 and the four stages, and estimate
F-usable distinct-root yield and CPU per usable root. Capped unsolved
searches are censored, never negative action labels. Do not expand to the
full roster or start R2/R4 trainers until this measured yield and at least
200 audited distinct F-usable roots are available under a new frozen
contract. No TEST150 selection or claims of superiority follow from this
TRAIN pilot.
