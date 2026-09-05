"""Run the first MCTS training loop and score it against its own floor.

Three arms at equal simulations -- untrained net, trained net, classical search --
because `docs/lessons.md` records that a gate whose bar sits just above the
untrained score cannot discriminate anything, and that an untrained network with
128-simulation MCTS once beat every network trained on the same curriculum.

Everything is written to `artifacts/mcts-loop/` as it is produced.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rf_knots import mcts_loop as M
from rf_knots.embedding_encoder import ScalableBraidEmbeddingEncoder

BANK = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/local-q-skm-ablation-20260817"
    "/continuation/q4000-v1-population-20260818/q254-fast6-20260823/protocol"
)
PRIOR = BANK / "prior-q204-for-q50-3-updated.json"
Q50 = BANK / "q50-3-updated.json"
EVIDENCE = Path("benchmarks/unknotting-evidence-index-20260815.json")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------- the ladder


def build_rungs(growth_len: int, cap_moves: int) -> dict[str, list[M.Instance]]:
    """Rungs A and B of the curriculum bank's own easy end, in registered order."""
    rows = M.curriculum_rows(PRIOR)
    indexed = list(enumerate(rows))
    rung_a = [
        row
        for _, row in indexed
        if row["strands"] <= 3
        and len(row["word"]) <= 8
        and row["certified_unknotting_upper_bound"] == 1
    ]
    rung_b = [
        row
        for _, row in indexed
        if row["strands"] <= 3
        and 9 <= len(row["word"]) <= 12
        and row["certified_unknotting_upper_bound"] == 1
    ]
    return {
        "A": M.instances_from_rows(
            rung_a, source="q204-A", growth_len=growth_len, cap_moves=cap_moves
        ),
        "B": M.instances_from_rows(
            rung_b, source="q204-B", growth_len=growth_len, cap_moves=cap_moves
        ),
    }


# ---------------------------------------------------- size independence check


def size_independence(net: M.SitePolicyValueNet, out: Path) -> dict:
    """Run the loop with one `state_dict` at two very different `(L, N)`.

    This is the property the whole architecture line was built for, and note 22
    §M1 asks for it to be asserted rather than assumed.  Three things are checked:
    the head's parameter count does not move, the same weights load and search at
    both sizes, and the necklace geometry is `Z_L`-equivariant under a cyclic word
    shift (the legacy default is not, by `0.377` in a unit-normalised space).
    """
    report: dict = {}
    report["parameters"] = net.parameter_report()

    # A small instance and a large one, both from the banks, sharing one net.
    small_rows = [
        row
        for row in M.curriculum_rows(PRIOR)
        if row["strands"] <= 3 and len(row["word"]) <= 8
    ][:2]
    large_rows = [row for row in M.curriculum_rows(Q50) if row["strands"] >= 11][:2]
    small = M.instances_from_rows(small_rows, source="small", cap_moves=8)
    large = [
        M.Instance(
            name=str(row["name"]),
            word=tuple(int(x) for x in row["word"]),
            strands=int(row["strands"]),
            cap_crossing_changes=int(row["certified_unknotting_upper_bound"]),
            cap_moves=8,
            # The `(512, 24)` spec of §M0.4's failed transfer, reached by slack.
            growth_len=512 - len(row["word"]),
            growth_strands=24 - int(row["strands"]),
            source="large",
            uid=f"large:{row['name']}",
        )
        for row in large_rows
    ]
    report["small_spec"] = [
        [i.spec.max_len, i.spec.max_strands, i.spec.num_actions] for i in small
    ]
    report["large_spec"] = [
        [i.spec.max_len, i.spec.max_strands, i.spec.num_actions] for i in large
    ]

    state = {k: v.clone() for k, v in net.state_dict().items()}
    evaluator = M.NetworkEvaluator(net, micro_batch=8)
    small_results = M.run_searches(small, evaluator, simulations=24, seeds=(0,))
    large_results = M.run_searches(large, evaluator, simulations=24, seeds=(0,))
    after = net.state_dict()
    report["state_dict_unchanged"] = all(
        torch.equal(state[k], after[k]) for k in state
    )
    report["ran_small"] = [r.expansions for r in small_results]
    report["ran_large"] = [r.expansions for r in large_results]
    report["parameters_after"] = net.parameter_report()
    report["parameter_count_stable"] = (
        report["parameters"] == report["parameters_after"]
    )

    # A fresh head at each size is literally the same shape: `SiteHead` has no
    # `max_len`/`max_strands` dimension to move.
    fresh = M.SitePolicyValueNet()
    fresh.load_state_dict(state)
    report["state_dict_reloads"] = True
    report["fresh_parameters"] = fresh.parameter_report()

    # Cyclic-shift equivariance of the geometry the loop is constructed with.
    # Float32, which is the column note 22 tabulates as `6.0e-08` for the necklace
    # against `0.377` for the legacy default -- and the legacy default is what a
    # caller who forgets the flag gets.
    def shift_delta(geometry: str) -> float:
        torch.manual_seed(0)
        encoder = ScalableBraidEmbeddingEncoder(positional_geometry=geometry)
        encoder.eval()
        word = torch.tensor([[1, -2, 1, 1, -2, -2, 1, 1, -2, -2]], dtype=torch.long)
        lengths = torch.tensor([word.shape[1]])
        strands = torch.tensor([3])
        rolled = torch.roll(word, shifts=3, dims=1)
        with torch.no_grad():
            a = encoder(word, lengths, strands)["knot"]
            b = encoder(rolled, lengths, strands)["knot"]
        return float((a - b).abs().max())

    report["cyclic_shift_delta"] = {
        "necklace": shift_delta("necklace"),
        "legacy": shift_delta("legacy"),
    }
    report["geometry"] = net.encoder.positional_geometry
    out.write_text(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------- evaluation


def evaluate_arm(
    name: str,
    instances: list[M.Instance],
    evaluator: M.Evaluator | None,
    *,
    simulations: int,
    seeds: tuple[int, ...],
    root_selection: str = "puct",
    time_cap: float | None = None,
) -> dict:
    start = time.monotonic()
    if evaluator is None:
        results = [
            M.classical_beam(instance, simulations=simulations, seed=seed)
            for instance in instances
            for seed in seeds
        ]
    else:
        results = M.run_searches(
            instances,
            evaluator,
            simulations=simulations,
            seeds=seeds,
            root_selection=root_selection,
            time_cap=time_cap,
            log=True,
            label=f"{name}@{simulations}",
        )
    claimed, verified = M.verify_all(results)
    elapsed = time.monotonic() - start
    moves = [len(r.actions) for r in results if r.actions is not None]
    expansions = [r.expansions for r in results]
    solved_count = sum(1 for r in results if r.solved)
    row = {
        "arm": name,
        "simulations": simulations,
        "seeds": list(seeds),
        "root_selection": root_selection if evaluator is not None else "beam",
        "instances": len(instances),
        "trees": len(results),
        "witness_rate": M.witness_rate(results),
        "claimed": claimed,
        "verified": verified,
        "mean_witness_moves": float(np.mean(moves)) if moves else None,
        "mean_expansions": float(np.mean(expansions)) if expansions else None,
        # `docs/lessons.md`: never report a conditional-on-solving cost alone.
        # An arm that abandons hard instances drops them from its own average.
        "expansions_per_solved": (
            float(np.sum(expansions) / solved_count) if solved_count else None
        ),
        "seconds": elapsed,
        "per_instance": M.per_instance_rate(results),
    }
    log(
        f"{name:>22} S={simulations:<5} rung n={len(instances)} "
        f"rate={row['witness_rate']:.3f} verified={verified}/{claimed} {elapsed:.0f}s"
    )
    return row


# ---------------------------------------------------------------- the loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/mcts-loop")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--doses", default="200,800,3200")
    parser.add_argument("--growth-len", type=int, default=2)
    parser.add_argument("--cap-moves", type=int, default=32)
    parser.add_argument("--generated", type=int, default=512)
    parser.add_argument("--evidence", type=int, default=250)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--selfplay-instances", type=int, default=128)
    parser.add_argument("--selfplay-simulations", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--net-seed", type=int, default=0)
    parser.add_argument("--time-cap", type=float, default=5400.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doses = [int(x) for x in args.doses.split(",") if x]
    seeds = tuple(range(args.seeds))
    started = time.monotonic()
    report: dict = {
        "schema": "mcts-loop-v1",
        "config": vars(args),
        "bank": {
            "prior": str(PRIOR),
            "q50": str(Q50),
            "evidence_index": str(EVIDENCE),
        },
    }

    def flush() -> None:
        report["elapsed_seconds"] = time.monotonic() - started
        (out / "report.json").write_text(json.dumps(report, indent=2, default=str))

    # -- the ladder --------------------------------------------------------
    rungs = build_rungs(args.growth_len, args.cap_moves)
    report["rungs"] = {
        name: [i.to_dict() for i in instances] for name, instances in rungs.items()
    }
    log("rungs: " + ", ".join(f"{k}={len(v)}" for k, v in rungs.items()))
    flush()

    # -- the network -------------------------------------------------------
    torch.manual_seed(args.net_seed)
    np.random.seed(args.net_seed)
    net = M.SitePolicyValueNet()
    untrained_state = {k: v.clone() for k, v in net.state_dict().items()}
    torch.save(untrained_state, out / "untrained.pt")

    log("size-independence check")
    report["size_independence"] = size_independence(net, out / "size-independence.json")
    net.load_state_dict(untrained_state)
    log(f"  {json.dumps(report['size_independence']['parameters'])}")
    log(
        f"  cyclic-shift delta necklace="
        f"{report['size_independence']['cyclic_shift_delta']['necklace']:.2e} "
        f"legacy={report['size_independence']['cyclic_shift_delta']['legacy']:.3f}"
    )
    flush()

    # -- the untrained floor and the classical control ---------------------
    rows: list[dict] = []
    for rung, instances in rungs.items():
        for dose in doses:
            rows.append(
                {"rung": rung, **evaluate_arm("classical-beam", instances, None,
                                              simulations=dose, seeds=seeds)}
            )
            flush()
            rows.append(
                {"rung": rung, **evaluate_arm(
                    "uniform-mcts", instances, M.UniformEvaluator(),
                    simulations=dose, seeds=seeds)}
            )
            flush()
            for rule in ("puct", "gumbel"):
                rows.append(
                    {"rung": rung, **evaluate_arm(
                        f"untrained-{rule}", instances, M.NetworkEvaluator(net),
                        simulations=dose, seeds=seeds, root_selection=rule)}
                )
                flush()
    report["arms"] = rows
    flush()

    # -- supervision -------------------------------------------------------
    eval_names = [i.name for rung in rungs.values() for i in rung]
    eval_keys = [(i.word, i.strands) for rung in rungs.values() for i in rung]

    log("building the training corpus")
    generated = M.generated_witness_instances(count=args.generated)
    generated = [
        (instance, witness)
        for instance, witness in generated
        if (instance.word, instance.strands) not in set(eval_keys)
    ]
    witness_samples: list[M.PolicySample] = []
    for instance, witness in generated:
        witness_samples += M.samples_from_witness(witness, name=instance.label)
    log(f"  generated instances {len(generated)} -> {len(witness_samples)} A_plus rows")

    evidence = M.evidence_index_witnesses(
        EVIDENCE, exclude_names=eval_names, exclude_keys=eval_keys, limit=args.evidence
    )
    evidence_samples: list[M.PolicySample] = []
    for name, witness in evidence:
        evidence_samples += M.samples_from_witness(witness, name=name)
    log(f"  evidence witnesses {len(evidence)} -> {len(evidence_samples)} A_plus rows")

    report["corpus"] = {
        "generated_instances": len(generated),
        "generated_aplus_rows": len(witness_samples),
        "evidence_witnesses": len(evidence),
        "evidence_aplus_rows": len(evidence_samples),
        "excluded_eval_names": sorted(set(eval_names)),
        "evidence_names_available": len(
            json.loads(EVIDENCE.read_text()).get("knots", {})
        ),
    }
    flush()

    supervised = witness_samples + evidence_samples
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    log(f"supervised warm-up on {len(supervised)} A_plus rows")
    warm = M.train_on(
        net, supervised, optimizer, batch_size=24, epochs=args.warmup_epochs,
        seed=1, log=True, label="warmup",
    )
    report["warmup"] = {
        "batches": len(warm),
        "first": vars(warm[0]) if warm else None,
        "last": vars(warm[-1]) if warm else None,
        "loss_curve": [r.total for r in warm[:: max(len(warm) // 50, 1)]],
    }
    torch.save(net.state_dict(), out / "warmup.pt")
    flush()
    log(f"  warm-up loss {warm[0].total:.4f} -> {warm[-1].total:.4f}")

    # -- self-play iterations ---------------------------------------------
    selfplay_pool = [instance for instance, _ in generated[: args.selfplay_instances]]
    iterations: list[dict] = []
    for iteration in range(args.iterations):
        if time.monotonic() - started > args.time_cap:
            log("time cap reached; stopping the self-play loop")
            break
        log(f"self-play iteration {iteration + 1}/{args.iterations}")
        evaluator = M.NetworkEvaluator(net)
        results = M.run_searches(
            selfplay_pool,
            evaluator,
            simulations=args.selfplay_simulations,
            seeds=(iteration,),
            root_selection="gumbel",
            dirichlet_fraction=0.25,
            log=True,
            label=f"selfplay{iteration}",
        )
        M.verify_all(results)
        solved = M.witness_rate(results)
        fresh: list[M.PolicySample] = []
        for result in results:
            fresh += M.samples_from_tree(result, min_visits=8, max_nodes=24)
        log(f"  self-play solved {solved:.3f}, {len(fresh)} rows")
        batch = fresh + supervised
        reports = M.train_on(
            net, batch, optimizer, batch_size=24, epochs=1,
            seed=100 + iteration, log=True, label=f"train{iteration}",
        )
        iterations.append(
            {
                "iteration": iteration,
                "selfplay_witness_rate": solved,
                "selfplay_rows": len(fresh),
                "train_rows": len(batch),
                "loss_first": vars(reports[0]) if reports else None,
                "loss_last": vars(reports[-1]) if reports else None,
            }
        )
        report["iterations"] = iterations
        torch.save(net.state_dict(), out / f"iteration-{iteration}.pt")
        flush()

    torch.save(net.state_dict(), out / "trained.pt")
    trained_state = {k: v.clone() for k, v in net.state_dict().items()}

    # -- the supervised and the self-played checkpoints, separately --------
    # `warmup` is the set-valued `A_plus` supervision alone; `trained` is that
    # plus the self-play iterations.  Reporting one number for both would make it
    # impossible to say whether the *loop* did anything.
    checkpoints = {"warmup": torch.load(out / "warmup.pt"), "trained": trained_state}
    for tag, state_dict in checkpoints.items():
        net.load_state_dict(state_dict)
        for rung, instances in rungs.items():
            for dose in doses:
                for rule in ("puct", "gumbel"):
                    rows.append(
                        {"rung": rung, **evaluate_arm(
                            f"{tag}-{rule}", instances, M.NetworkEvaluator(net),
                            simulations=dose, seeds=seeds, root_selection=rule)}
                    )
                    report["arms"] = rows
                    flush()
    net.load_state_dict(trained_state)

    log(f"done in {time.monotonic() - started:.0f}s")
    flush()


if __name__ == "__main__":
    main()
