# Q-Run weights: verified starting points for the next search comparison

2026-09-21. Follow-up to the user's questions about the 1/64 MCTS result and
whether the new scalable model inherited a strong Q-Run policy. Protocol69
continues unchanged. No Q training, checkpoint migration, deployment or
cross-lineage weight sharing was performed.

## Correction of interpretation

The current RF `SitePolicyValueNet` lineage started from scratch in protocol27;
it did not import a strong Q-Run raster/strand-graph policy. Its 1/64 result
cannot be interpreted as a result for the best Q-Run solver with embeddings.
The original proposal of pretrained raster blocks plus a higher-level graph
network has not yet been implemented by that RF architecture.

The completed RF test used 64 roots, raw L69..345/N3..24, reference witnesses
37..185 moves, only eight MCTS simulations per condition, and move caps equal
to reference witness length on **54/64** roots. There were 383 joint-policy
continuations per lambda: 382 stopped at the move cap and one solved. These
facts establish a weak policy/search result under tight resources, not its
single cause. The next already-frozen protocol70 uses more generous move caps
but retains eight simulations; it is not a causal budget sweep on the old TEST.

## Actual checkpoint and forward verification

`scripts/check_qrun_baseline_inputs.py` rechecks checkpoint manifests, compares
loaded native tensors exactly, and evaluates one native trefoil observation.
It also compares the frozen Q304 network tensors against the exact parent
state named by the completed Q305/Q314 promotion report. A changed input hash
or existing output fails closed. It uses the canonical PGX checkout, not
`pgx-mcts-bench-local-ablation`.

| Check | Q254 raster | Q304 strand graph |
| --- | --- | --- |
| Native class | RasterSerialBraidNet | StrandGraphBraidNet |
| Parameters | 248418 | 317258 |
| Configured word/strand capacities | 48 / 12 | 48 / 12 |
| Native HWC observation shape | 1 x 7 x 82 | 1 x 48 x 38 |
| Native action logits | 196 | 40 |
| Exact loaded checkpoint tensors | yes | yes |
| Finite native trefoil forward | yes | yes |
| Exact tensors of the evaluated Q304 parent | not applicable | yes |

Q254 checkpoint SHA256:
`36d1122f5494d502e556994083a1a69adf1643d5be95cd9f80ffc13b68e68d63`.
Q304 frozen checkpoint SHA256:
`0c7a3f3d5b2a08b4bc9c307958264378f8a17148438f0eae7e3d5e93b1739dfa`.
The evaluated Q304 parent's compressed state SHA256 is
`fe8b1fcbdcf09f72517e748595b21da7d535da05d15127b43a8835add805143e`;
different container hashes do not imply different model weights, hence the
explicit tensor equality check.

The first ad-hoc smoke attempt supplied HWC observations directly to Torch and
failed in the raster reshape. The corrected check uses the native oracle's
HWC-to-BCHW permutation. This was an error in the diagnostic harness, not
evidence that the checkpoint or ongoing RF training was broken. Version1 of
the completed report is preserved; version2 binds the final lint-clean script.
Reports: `artifacts/qrun-baseline-input-check-v{1,2}/report.json`.

## Existing Q evidence is not the RF test

The inspected exact-common promotion report is **BLOCKED**: on its own ten
representations with two attempts and 40 simulations, Q304 solved 8 at L10
and 6 at L1000; Q314 solved 1 at each. This supports retaining the parent
for a baseline candidate, not calling it the globally best current model.
It neither permits continuing that blocked Q lineage nor permits comparing
8/10 numerically with the separate RF 1/64 test.

## Next required experimental step

1. Use hash-bound native Q policies as actual baselines on a prospectively
   declared common supported domain. Verify split/history exposure separately;
   a new filename or an RF holdout bucket does not prove Q training independence.
2. Keep native controller moves distinct from semantic braid edits. Carry
   controller position/state, map only legal semantic actions, count neural
   evaluations and controller plies separately, and independently replay every
   resulting witness. Do not silently truncate a long input to fit L48/N12.
3. Separate policy ability from search budget using fixed diagnostic roots and
   declared resource sweeps; known TEST reuse must be labelled diagnostic.
   Add known-witness candidate coverage checks before assigning failure to
   architecture, embeddings or insufficient training.
4. To test the original long-input proposal, preserve learned local raster
   computations and introduce a declared global/window integration, or train
   a scalable policy from native teacher decisions/verified alternative paths.
   These are different experiments, not an unverified direct tensor copy into
   the current `strand_graph` model. Neither has been launched by this check.

The successful forward check is an input/compatibility gate only. No new
search success, embedding benefit, long-input transfer or production promotion
is claimed here. Current graph/control training and its independent evaluation
must still complete under their frozen contracts.
