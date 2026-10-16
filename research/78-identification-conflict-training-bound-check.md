# Named identification conflict versus current metric labels

2026-09-21. Read-only follow-up to the UnknotDB release preparation finding
`10_162` versus `10_161`. This does not resolve that identity conflict or
authorize a database release. No training source, checkpoint or label changed.

## Finding

The current protocol69 metric bounds are not obtained by equating catalogue
names. Inspection of the actual source bytes pinned by the live training
manifest shows:

- The original replay corpus is loaded through `verify_record`: its upper
  bound is the number of literal crossing changes in a replayed witness prefix;
  its lower bound is recomputed from signed signatures, then optionally
  strengthened by unequal determinants. Exact zero comes from zero-CC replay,
  not a shared attested name.
- Suffix-to-unknot rows come from replayed successful witnesses ending at the
  explicit unknot. Upper bounds count remaining CC; lower bounds use signature
  and determinant. The loader reconstructs the suffix records and compares
  them to the saved rows.
- The ordinary graph source selection does include a provenance category named
  `replayed_cc_distinct_attested_knots`, but its stored numerical bounds are
  **not copied**. The builder and independent audit recompute bounds from raw
  signed braid words, literal one/two-CC replay, signed signatures and unequal
  determinants. Signed cyclic equality alone can certify a zero upper bound.
- The source sidecar's TRAIN selection is retained, but all path states are
  additionally checked against the current invariant-based owner and historical
  query exclusions. Thus attested names can influence source availability;
  this is not a claim of complete statistical independence from identification.

The named conflict is therefore not a reason to interrupt/relabel this run:
the inspected metric-label construction has an independent certificate path.
This check does not prove every knot identification correct, resolve all
possible data bias, or establish held-out model quality.

## Live binding checks

All four hashes below matched the completed graph arm's training input pins;
all 60 current pinned training source hashes also matched. The existing full
graph audit is `passed`, `train_ready=true`, with 50618 retained pairs. This
follow-up checked its binding and implementation; it did not rerun that entire
replay audit or claim to have done so.

| Artifact | SHA256 |
| --- | --- |
| Diverse metric corpus | `f3655e7dbfcc5b6ec7448b1f747dc2a4e1235fe036ab07fec61cc18df05c8f14` |
| Audited suffix views | `e1fccfef653557ede4ee9a98e05b3c47a12a5950ab6ce7486ae65a763c0013bf` |
| Ordinary graph TRAIN SQLite | `a4554a78573d5ac73dd0ae0a92aa0ea98260ab132cde92c6ae3e58615133ff1b` |
| Full graph replay/partition audit | `aeaea4f9883d510726c35d0b14e5cf8d0f054a3333d35cfd1ed150a3433709c0` |

Implementation references: `src/rf_knots/embedding_replay_corpus.py`,
`scripts/run_strand_graph_pilot.py`, `scripts/balanced_metric_sampling.py`,
`scripts/mcts_joint_training.py`, `scripts/build_mcts_training_supervision.py`,
`scripts/build_ordinary_graph_metric_corpus.py`,
`scripts/audit_full_ordinary_graph_corpus.py`.
