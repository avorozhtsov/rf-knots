# Joint F bank: first pre-shard failure and v2 correction

2026-10-07. The first bank-builder source
`scripts/build_joint_f_bank_v1.py` created a manifest in
`artifacts/joint-f-bank-v1` but stopped at `next_root=0` before writing
any root shard: its reference-path key used `SemanticAction` objects,
which are not JSON serializable. Preserve this failed source/manifest.

The separately versioned v2 source converts the reference actions to
the same flat action IDs used by replayed search episodes via
`SemanticAction.to_flat(instance.spec)`. It otherwise retains the
research/300 scientific bank contract and writes to a separate v2
artifact. Verify a first root shard and its SHA before continuing all
1,000 roots. Neither version performs optimizer training.
