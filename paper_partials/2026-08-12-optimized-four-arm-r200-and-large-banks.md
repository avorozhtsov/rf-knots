# Optimized four-arm R200 restart and 2,700-representation blocks

The first cloud static/no-sharing R200 process used source base `bb886413` and
reached 35 atomically recorded events. Because the evaluation optimizations live
in commit `4be0bac`, changing its imported code mid-process would have created a
hybrid arm. The legacy artifacts remain preserved at
`/srv/braid/artifacts/semantic-v2-r200/static-no-sharing`, but that compute was
stopped on 2026-08-12 and is excluded from the paired four-arm result.

All four arms restarted from their own post-R24 scientist states in a clean
checkout of `4be0bac60c7dec35f995f9a4912cedf9907a12f2`:

| arm | protocol SHA-256 |
|---|---|
| static, no sharing | `04e00201ad46ae0d7909f306d4ec76a039ce45d4a7b4fb8757ffee25d9bd0d32` |
| adaptive, no sharing | `e3c8f0991e3a748cb701ea097ea6d8e1cf151ed1907a486b251ed62d2c10f712` |
| static, sharing | `ce15a4d14fde7f32edad999e645cdb90dcc770f2e3372b86c8fb1f87b9235298` |
| adaptive, sharing | `ae4f92fb3204744fb4b51d647c51097a633c2cab17fe55e470a410627fb6f4d5` |

Every manifest is clean and records executable-source SHA-256
`7169d55a938bf5316f2dc1d07998088071f3efc38da87040d23bfe0b7599be2e`.
The common seed is 20262120; all use L10/L1000, SIM64, initial F5, adaptive
compute and rehearsal, eight self-play games, 96 optimizer steps, batch 64,
EV4, and the frozen upper-bound ACS R200 bank. Artifact roots are under
`/srv/braid/artifacts/semantic-v2-r200-optimized/`.

The continuation bank has also been frozen. It contains exactly 2,700
representations: all 1,639 compatible canonical knot-table braids plus 1,061
deterministic crossing-change-free Markov variants. The existing R200 identities
are removed from the continuation, leaving six 400-item blocks and a 100-item
tail. Each block is independently ordered by

`10 * strands + 5 * certified_unknotting_upper_bound + len(braid_word)`.

Bounds come from `dtubbenhauer/upperbounds` commit
`de66f29045e804931edd6d1c9735247f81ad68c1`, workbook SHA-256
`5f58b9d6740ed6cdb63cc728abee2ba3ac54f4427fd55127d0d6d5465f3c41d3`.
The local and Nebius block manifest SHA-256 is
`cd026f00afac50ff1aa4fa8d75dc8158b57160f9ca0a2e420d12c12aa88dc107`.
Cumulative prior banks retain R24 and every completed earlier block for replay
and rehearsal. R400-1 remains completion- and budget-gated behind all four R200
arms; this is necessary to respect the session's USD 200 cap.

The block-boundary exporter was repaired before any R400 launch. In addition to
network, optimizer, replay, best-solution bank, `F_old`, and rehearsal exposure,
it now carries each scientist's adapted `F_native` and simulation level and the
shared donation dose/healthy-streak controller. The next block rejects exported
scientist states whose copies of the sharing controller disagree; adaptive
compute therefore cannot silently reset at a group boundary.
