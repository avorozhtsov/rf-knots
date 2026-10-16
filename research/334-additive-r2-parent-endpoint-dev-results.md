# Additive R2 parent/endpoint matched DEV result

2026-10-11. The prospective contract in `research/330-additive-r2-parent-endpoint-matched-dev-v1.md` completed 256/256 fixed canonical pgx search conditions. Manifest SHA256 `ca92057c121adaf2afb97fb36a59ed6f9c9f9547f75da23de925195a55d19d7f`; contract SHA256 `efba9a80b7f742e41c341180def8fa37cba83de8886b4f60aff4f2a842793333`. Independent auditor `scripts/audit_additive_r2_parent_endpoint_dev_v1.py` SHA256 `47f7354f1b2cfa1d5b8c5e8f8fdb4c9e630a4955ceef4d4a5c366bd43226035a` verified all 256 shard SHA, every pinned input SHA, starts, resource caps, cost and full replay of all 94 found solutions. Report SHA256 `43a3ed41da98e9e2812841ea6a1d33bdd566e07d7950dd735b4e61a99277fb63`.

Distinct roots solved by either of two seeds, out of 16 short plus 16 long:

| Objective | Parent step133635 | Endpoint step152964 |
| --- | ---: | ---: |
| Short L10 | 15 | 14 |
| Long L10 | 0 | 0 |
| Short L1000 | 12 | 14 |
| Long L1000 | 0 | 0 |

Endpoint minus parent is +2 distinct L1000 roots and -1 L10 root. No endpoint CC increase occurred on jointly solved attempts. Total CPU was 8226.906 vs 8289.356 seconds (+0.76% endpoint); policy neural evaluations 834,830 vs 838,109 (+0.39%). The predeclared +3 L1000 promotion threshold is **not** met, and the compute difference does not require a matched rerun. This is a tie/no promotion, not evidence that the endpoint is worse or that R2 is superior to a non-neural control. In particular, neither arm solved any of the 16 long roots at these fixed resources; no hard-root generalization claim follows. Bounded no-solves remain censored. This viewed DEV panel must not be reused to select a checkpoint.

The verified TRAIN F bank remains at 399 different roots. The root-from-scratch b64 hard micro-pilot yielded no completed solutions; a prospective reference-bridge concept is recorded in `research/333-reference-bridge-root-f-pilot-plan-v1.md`, but it has not yet been implemented or run. Before spending larger CPU or training on purported new labels, freeze an executable small-pilot contract with explicit CPU ceiling, local simulation budgets, independent solution replay, exact-root divergence and root-only leakage audit. Preserve all old bank roots, checkpoints and failures. No TEST150/viewed DEV selection, unknotdb mutation/publication, cloud purchase, commit, merge or push.
