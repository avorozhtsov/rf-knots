# Semantic-v2 R200 static/no-sharing launch

On 2026-08-11 the static/no-sharing arm continued from the complete local R24
states on the same 32-vCPU Nebius instance used by the other three R24 arms.
The run name is `SV2-3S-R200-SIM64-F5-AR-EV4-NO-SHARING`; its remote artifact
root is `/srv/braid/artifacts/semantic-v2-r200/static-no-sharing`.

The frozen group contains 200 distinct representations from the source-disjoint
K3 pilot bank. Static ordering uses the original coefficients

`10 * strands + 5 * certified_u_upper_bound + len(braid_word)`.

The braid-word length, not the knot table's minimal crossing number, is the
presentation-intersection term. The frozen bank file SHA-256 is
`831917692801cb87d76b915d80e0ad3d6c2578a967e50695468a8268cc4c056e`.
The corrected manifest's canonical bank hash is
`a5fa348a0ff3204a28b40913ea04fa3d3f8ad6a0255c0fe8ca1cf3e5598096de`.
The compact 200-item upper-bound snapshot has SHA-256
`3aeedc6d210f2db4214cc4bbfa95aeff7053d8fdf1f02a9581fb01fd837ef7d4`.
It was extracted on 2026-08-11 from `dtubbenhauer/upperbounds` commit
`de66f29045e804931edd6d1c9735247f81ad68c1`, whose
`data/unknotting.xlsx` SHA-256 is
`5f58b9d6740ed6cdb63cc728abee2ba3ac54f4427fd55127d0d6d5465f3c41d3`.

Each scientist inherits its full R24 network, optimizer, replay, permanent best
solution bank, and rehearsal state. Initial-state hashes are:

- `strand-graph`: `bada5ef408c9b20efb45877136c3df44c4d6724d12ff5dd51640f5f454576e82`;
- `raster-axial`: `6f5abf8edc9800b383b75654e69c29eae0189d698f2cf34a4b38cb8c7fc95c45`;
- `cyclic-memory`: `d8053f6752c5e59c54c23a6ac4e5500d39a28be0dc9e9203e0e8663033d575b8`.

The R24 group remains in the rehearsal and retention population. R200 starts at
64 simulations per move and `F_native=5`; at ten-rung boundaries, each
scientist raises `F_native` through `5,8,12,16` if representation-objective
acquisition is below 0.80, and raises simulations through `64,128,256,512` if
attempt-level evaluation solve rate is below 0.70. `F_old` is inherited from
R24 and continues through the declared `1,2,4,8` ladder. Evaluation uses four
attempts per representation and objective, with L10 and L1000 sampled equally.

An initial lower-bound-ordered launch was stopped after one atomically committed
rung and invalidated before any continuation state was reused. The corrected
upper-bound-ordered run restarts from the unchanged post-R24 states.

The run uses seed `20262120`, three persistent scientist processes with three
Torch threads each, CPU affinity `0-29`, and Unix nice level 10 so the active R24
arms retain scheduling priority. The corrected manifest protocol hash is
`d6ca5420a96babda029b115638379ddcb4ea845174b91bda9f436f8b40426269`;
the executable-source hash is
`7480f892a1fdb641db94b118f62445e2782f4975fb0ec1fda094dcb91f38983c`.
The launch changes had not yet been committed.
