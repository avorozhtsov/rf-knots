# Q-run-like rung-0 matched DEV results

2026-10-05. The fixed v3 panel had 16 short and 16 long validation-owned
root representations, paired L10/L1000 caps, and no exact overlap with
TRAIN reference states of word length >=20. Shorter common simplification
tails and intermediate invariant ownership were not excluded. Panel SHA
`2e1d4a6d827fdd63235e303f889f0c73b2f8d741d64ee2307f4a18d280129a6d`.

Search contract SHA `e77a7faaedc343f67bc40c6f2d45e34ba5ed9e9e897213a9b397d2a611a4177a`:
F144 parent, short-only256 and 75/25 mixed256, each on all 32 roots × both
objectives at 64 simulations and seed 20271204. All 192 shards have matching
SHA; independent audit replayed all 93 found solutions and checked starts,
caps, CC/moves and objective costs. Manifest SHA
`91e74a076f1303ed3cec536f36dbbcc099c15aaaae46eaca24763af084acfc8e`.

| Model | Short L10 | Short L1000 | Long L10 | Long L1000 |
| --- | ---: | ---: | ---: | ---: |
| F144 parent | 11/16 | 11/16 | 4/16 | 6/16 |
| short-only256 | 14/16 | 12/16 | 1/16 | 0/16 |
| mixed256 | 13/16 | 12/16 | 4/16 | 5/16 |

Relative to F144, mixed256 gained/lost short roots 3/1 at L10 and 3/2 at
L1000; on jointly solved short roots it was cheaper/equal/dearer 2/8/0
and 4/5/0 respectively. On long roots mixed256 gained/lost 2/2 at L10
and 0/1 at L1000, with equal cost for all jointly solved long cases.
Short-only256 gained/lost 4/1 and 3/2 short roots, but lost three and
six long roots without long gains. A bounded failure is censored, not a
negative action label. Cost sums over different solved subsets must not
be interpreted as paired gains. The one-seed sample is too small for a
winner claim; short-only training has a clear long-retention warning.

For an exploratory next rung, mixed256 is the safer initialization. This
does not imply that F144 or mixed256 beats fully non-neural search. The
new rung-1 curriculum is fixed separately in protocol284, and the viewed
v3 DEV panel must not be reused for checkpoint selection. TEST150 was not
used. No superiority claim is made.
