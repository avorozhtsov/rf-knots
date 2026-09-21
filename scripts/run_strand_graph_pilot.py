"""Chirality-sensitive, replayed metric + policy smoke experiment.

This is a representation-held-out engineering pilot, NOT a knot-disjoint
promotion experiment. The historical mirror-quotiented pair corpus is not read.
No unsuccessful search is converted into a negative value label. Every training
path is replayed; pair lower bounds use signed signature differences and upper
bounds use the explicit walk back to its seed. All solutions are replay-checked.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import torch

from rf_knots import mcts_loop as M
from rf_knots import seifert
from rf_knots.embedding_augmentation import (
    MaterializedSyntheticPair,
    SyntheticPairSpec,
    collate_embedding_pairs,
)
from rf_knots.embedding_metric import MetricBatch, MetricEmbeddingObjective
from rf_knots.embedding_replay_corpus import verify_record
from rf_knots.embedding_suffix_supervision import category, collect_u_views, verify_u_view
from rf_knots.interleaved_generator import (
    InterleavedConfig,
    LipschitzValues,
    generate,
    gordian_interval,
    torus_seed,
)


def atomic_json(path, payload):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(word, strands):
    word = tuple(word)
    return strands, min(word[i:] + word[:i] for i in range(len(word))) if word else ()


def pair_row(index, group, left, right, lower, upper, length):
    spec = SyntheticPairSpec(index, group, "cc", -1, -1, lower, upper, length, 0, 0, 1.0)
    return MaterializedSyntheticPair(spec, "cc", left[0], right[0], left[1], right[1], length)


def make_corpus(count, seed, forbidden=frozenset(), large=False):
    rows, witnesses, records = [], [], []
    for index in range(count * 30):
        strength, changes = index % 4, (index // 4) % 3
        config = InterleavedConfig(
            crossing_changes=changes,
            walk_length=8,
            settle_walk_length=6,
            target_strands=6 if large else 3,
            target_length=24 if large else 12,
            max_len=40 if large else 24,
            max_strands=8 if large else 5,
            strand_band=1,
        )
        root = torus_seed(strength, config.spec())
        instance = generate(config, seed, index, root)
        if canonical(instance.word, instance.strands) in forbidden:
            continue
        assert instance.witness is not None
        instance.witness.verify()
        # Do not substitute abs(signature) or a mirror-folded knot identifier.
        left_sigma = seifert.signature(instance.word, instance.strands) if instance.word else 0
        right_sigma = -2 * strength
        interval = gordian_interval(
            instance, LipschitzValues(left_sigma // 2), LipschitzValues(right_sigma // 2)
        )
        rows.append(
            pair_row(
                index,
                f"walk-k{changes}",
                (instance.word, instance.strands),
                (root.word, root.strands),
                interval.lower,
                interval.upper,
                interval.witness_semantic_length,
            )
        )
        witnesses.append(instance.witness)
        records.append(
            {
                "config": config.provenance(),
                "seed": seed,
                "index": index,
                "interval": interval.to_dict(),
                "signature_left": left_sigma,
                "signature_right": right_sigma,
                "witness": instance.witness.to_dict(),
            }
        )
        if len(rows) == count:
            return rows, witnesses, records
    raise RuntimeError("could not fill representation-disjoint corpus")


def metric_loss(net, objective, rows, device):
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in collate_embedding_pairs(rows).items()
    }
    left = net.encoder(batch["left_word"], batch["left_length"], batch["left_strands"])
    right = net.encoder(batch["right_word"], batch["right_length"], batch["right_strands"])
    loss, _ = objective(
        left,
        right,
        MetricBatch(
            batch["metric"], batch["lower"], batch["upper"], batch["witness_semantic_length"]
        ),
    )
    pred = objective.scale("cc") * objective.cc_distance((left["knot"] * right["knot"]).sum(-1))
    return loss, pred


def load_frozen_corpus(path):
    """Recheck intervals, witnesses AND all train-path split memberships."""
    corpus = json.loads(path.read_text())
    if corpus.get("schema") != "rf-knots-replay-fingerprint-corpus-v1":
        raise ValueError("unsupported frozen replay corpus")
    result = {}
    for split in ("train", "validation", "size_shift"):
        rows, witnesses, u_rows = [], [], []
        for i, record in enumerate(corpus[split]):
            if record["split"] != split:
                raise ValueError("record/file split disagreement")
            w = verify_record(record)
            pair = record["interval"]
            left, right = pair["left"], pair["right"]
            rows.append(
                pair_row(
                    i,
                    split,
                    (tuple(left["word"]), left["strands"]),
                    (tuple(right["word"]), right["strands"]),
                    pair["lower"],
                    pair["upper"],
                    pair["witness_semantic_length"],
                )
            )
            witnesses.append(w)
            # An explicit anchor row already supplies this exact supervision.
            # Derive extra anchors only for non-unknot reference pairs.
            if w.crossing_changes <= 2 and (right["word"] or right["strands"] != 1):
                u_rows.append(
                    pair_row(
                        i,
                        f"{split}-short-u",
                        (w.start.word, w.start.strands),
                        ((), 1),
                        abs(record["left_fingerprint"][1]) // 2,
                        w.crossing_changes,
                        w.moves,
                    )
                )
        result[split] = (rows, witnesses, u_rows)
    return corpus, result


@torch.no_grad()
def metric_report(net, objective, rows, device):
    net.eval()
    records = []
    for start in range(0, len(rows), 32):
        chunk = rows[start : start + 32]
        _, pred = metric_loss(net, objective, chunk, device)
        for row, p in zip(chunk, pred.cpu().tolist(), strict=True):
            lo, hi = row.spec.distance_lower, row.spec.distance_upper
            records.append(
                {"lower": lo, "upper": hi, "prediction": p, "violation": max(lo - p, p - hi, 0.0)}
            )
    exact = [r for r in records if r["lower"] == r["upper"]]
    groups = {}
    for target in sorted({r["lower"] for r in exact}):
        group = [r for r in exact if r["lower"] == target]
        groups[str(target)] = {
            "count": len(group),
            "mae": sum(r["violation"] for r in group) / len(group),
        }
    return {
        "count": len(rows),
        "mean_interval_violation": sum(r["violation"] for r in records) / len(rows),
        "exact_groups": groups,
        "predictions": records,
    }


class PolicyEvaluator(M.NetworkEvaluator):
    def evaluate(self, requests):
        # Value has no calibrated target in this pilot; keep a common .5 leaf
        # value in every arm rather than train fake negatives from exhausted MCTS.
        return [M.Evaluation(row.priors, 0.5) for row in super().evaluate(requests)]


def search_report(
    net,
    instances,
    simulations,
    seeds,
    device,
    root_noise=0.0,
    widening_batch=0,
    learned_value=False,
    proposal_provider=None,
):
    base_evaluator = (
        (M.NetworkEvaluator(net, device) if learned_value else PolicyEvaluator(net, device))
        if net is not None
        else M.UniformEvaluator()
    )

    class MeteredEvaluator(M.Evaluator):
        def __init__(self):
            self.requests = self.action_scores = self.max_actions = 0

        def evaluate(self, requests):
            self.requests += len(requests)
            self.action_scores += sum(len(r[2]) for r in requests)
            self.max_actions = max(self.max_actions, max((len(r[2]) for r in requests), default=0))
            return base_evaluator.evaluate(requests)

    evaluator = MeteredEvaluator()
    proposal_before = proposal_provider.statistics() if callable(getattr(proposal_provider, "statistics", None)) else None
    started = time.time()
    results = M.run_searches(
        instances,
        evaluator,
        simulations=simulations,
        seeds=tuple(range(seeds)),
        dirichlet_fraction=root_noise,
        log=True,
        label="strand-graph-pilot",
        widening_batch=widening_batch,
        proposal_provider=proposal_provider,
    )
    claimed, verified = M.verify_all(results)
    assert claimed == verified
    rows = []
    for result in results:
        witness = result.witness
        rows.append(
            {
                "instance": result.instance.uid,
                "seed": result.seed,
                "solved": result.solved,
                "expansions": result.expansions,
                "simulations": result.simulations,
                "cc": witness.crossing_changes if witness else None,
                "moves": witness.moves if witness else None,
                "L10": 10 * witness.crossing_changes + witness.moves if witness else None,
                "L1000": 1000 * witness.crossing_changes + witness.moves if witness else None,
                "witness": witness.to_dict() if witness else None,
            }
        )
    proposal_compute = ({k: v - proposal_before[k] for k, v in proposal_provider.statistics().items()}
                        if proposal_before is not None else None)
    return {
        "solved": verified,
        "trees": len(rows),
        "rows": rows,
        "metric_caveat": "first-found witnesses, not cost-optimal MCTS; failures remain censored",
        "root_dirichlet_fraction": root_noise,
        "widening_batch": widening_batch,
        "factorized_proposals": isinstance(proposal_provider, M.FactorizedProposalProvider),
        "proposal_kind": type(proposal_provider).__name__ if proposal_provider is not None else None,
        "proposal_compute": proposal_compute,
        "total_neural_state_evaluations": (
            (evaluator.requests if net is not None else 0) + (proposal_compute["state_evaluations"] if proposal_compute else 0)
            if proposal_provider is None or proposal_compute is not None else None
        ),
        "leaf_value": "learned" if learned_value and net is not None else "constant-0.5",
        "evaluator_requests": evaluator.requests,
        "action_scores": evaluator.action_scores,
        "max_actions_per_request": evaluator.max_actions,
        "seconds": time.time() - started,
        "widening_caveat": "Widening consumes a simulation and re-evaluates all open priors; "
        "edge visits/values persist. Root Dirichlet noise is redrawn at each widening. "
        "Unopened actions are not negative training labels.",
        "seed_caveat": "PUCT is deterministic across seeds when root_noise=0",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=192)
    parser.add_argument("--validation-count", type=int, default=48)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--search-seeds", type=int, default=8)
    parser.add_argument("--search-instances", type=int, default=4)
    parser.add_argument("--corpus", type=Path, help="immutable invariant-group-split replay corpus")
    parser.add_argument(
        "--context-rounds",
        type=int,
        default=0,
        help="0: original trunk; -1: automatic shared circular multiscale depth",
    )
    parser.add_argument("--search-moves", type=int, default=32)
    parser.add_argument("--normalized-knot", action="store_true")
    parser.add_argument("--window-size", type=int, default=0)
    parser.add_argument("--hierarchy-levels", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--policy-candidates", type=int, default=0)
    parser.add_argument("--budget-dim", type=int, choices=(2, 5), default=2)
    parser.add_argument("--u-sampling", choices=("legacy", "roots", "dense"), default="legacy")
    parser.add_argument(
        "--root-noise", type=float, default=0.0, help="nonzero for distinct search trials"
    )
    parser.add_argument("--metric-weight", type=float, default=1.0)
    parser.add_argument("--witness-weighting", choices=("legacy", "uniform"), default="legacy")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.learning_rate <= 0:
        raise SystemExit("learning rate must be positive")
    if args.policy_candidates < 0:
        raise SystemExit("policy candidates must be nonnegative")
    if args.window_size < 0 or args.hierarchy_levels < 0 or bool(args.window_size) != bool(args.hierarchy_levels):
        raise SystemExit("window size and hierarchy levels must be positive together, or both zero")
    if args.u_sampling != "legacy" and (not args.corpus or args.batch_size < 8):
        raise SystemExit("balanced u sampling needs a frozen corpus and batch size >= 8")
    if args.metric_weight < 0:
        raise SystemExit("metric-weight must be nonnegative")
    if not 0 <= args.root_noise <= 1:
        raise SystemExit("root-noise must be in [0,1]")
    if args.out.exists() and not args.resume:
        raise SystemExit("output already exists; refusing to overwrite or duplicate a run")
    if args.resume and not (args.out / "checkpoint.pt").exists():
        raise SystemExit("resume requires a saved checkpoint")
    args.out.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.out / ".runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old_manifest = json.loads((args.out / "manifest.json").read_text()) if args.resume else None
    if old_manifest and old_manifest["status"] == "completed":
        print("already completed; no new runner or write")
        return
    started = time.time()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    anchor_rng = random.Random(args.seed + 100003)
    policy_rng = random.Random(args.seed + 200003)
    manifest = {
        "status": "building_corpus",
        "pid": os.getpid(),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"
        },
        "split_contract": "new left representations; NOT knot-disjoint; shared torus seed families",
        "distance_semantics": "ordinary Gordian intervals; mirrors NOT identified",
        "loss": "metric_weight*CC interval + set-valued replay policy; no value/failure labels",
        "code_sha256": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path("src/rf_knots/strand_graph_encoder.py"),
                Path("src/rf_knots/mcts_loop.py"),
                Path("src/rf_knots/site_head.py"),
                Path("src/rf_knots/interleaved_generator.py"),
                Path("src/rf_knots/seifert.py"),
                Path("src/rf_knots/embedding_replay_corpus.py"),
                Path("src/rf_knots/embedding_normalization.py"),
                Path("src/rf_knots/embedding_suffix_supervision.py"),
                Path("src/rf_knots/embedding_metric.py"),
                Path("src/rf_knots/embedding_encoder.py"),
                Path("src/rf_knots/reference.py"),
                Path("src/rf_knots/actions.py"),
                Path("src/rf_knots/evidence.py"),
                Path("src/rf_knots/aplus.py"),
            ]
        },
    }
    if args.corpus:
        manifest["source_corpus_sha256"] = sha(args.corpus)
    if old_manifest:
        if 'initial_checkpoint_sha256' in old_manifest:
            manifest['initial_checkpoint_sha256'] = old_manifest['initial_checkpoint_sha256']
        for field in ("config", "code_sha256", "source_corpus_sha256"):
            if manifest.get(field) != old_manifest.get(field):
                raise SystemExit(f"resume {field} mismatch; refusing to change the experiment")
    else:
        for i, source in enumerate(manifest["code_sha256"]):
            source_path = Path(source)
            target = args.out / "source_snapshot" / f"{i:02d}-{source_path.name}"
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source_path, target)
    atomic_json(args.out / "manifest.json", manifest)
    val_u, large_u = [], []
    u_groups = {}
    if args.corpus:
        source, loaded = load_frozen_corpus(args.corpus)
        train, witnesses, train_u = loaded["train"]
        validation, val_witnesses, val_u = loaded["validation"]
        large, _, large_u = loaded["size_shift"]
        if args.u_sampling == "legacy":
            train.extend(train_u)
        else:
            index = collect_u_views(source["train"], intermediate=args.u_sampling == "dense")
            # Audit every 64th compact view by directly recomputing its suffix.
            for view in index["views"][::64]:
                verify_u_view(view, source["train"], index["parent_witness_sha256"])
            index_path = args.out / "u_views.json"
            if args.resume:
                if json.loads(index_path.read_text()) != index:
                    raise RuntimeError("suffix index changed on resume")
            else:
                atomic_json(index_path, index)
            for i, view in enumerate(index["views"]):
                u_groups.setdefault(category(view), []).append(
                    pair_row(
                        i,
                        "suffix-u",
                        (tuple(view["word"]), view["strands"]),
                        ((), 1),
                        view["lower"],
                        view["upper"],
                        view["remaining_moves"],
                    )
                )
            train = [r for r in train if r.right_word or r.right_strands != 1]
            if not train or set(u_groups) != {"0", "1", "2", "interval"}:
                raise ValueError("balanced experiment requires pair pool and all four u categories")
            manifest["u_views_sha256"] = sha(index_path)
            manifest["u_views"] = index["summary"]
            manifest["u_sampling_contract"] = (
                "half batch balanced over exact 0/1/2 and interval u; "
                "remaining pair rows plus mirror sentinel; independent policy RNG"
            )
        records, val_records, large_records = (
            source[k] for k in ("train", "validation", "size_shift")
        )
        manifest.update(
            split_contract=source["split_contract"],
            source_counts=source["counts"],
            short_u_counts={
                "train": len(train_u),
                "validation": len(val_u),
                "size_shift": len(large_u),
            },
        )
    else:
        train, witnesses, records = make_corpus(args.train_count, args.seed)
        forbidden = {
            canonical(state.word, state.strands)
            for w in witnesses
            for state in [w.start, *(s.after for s in w.steps)]
        }
        validation, val_witnesses, val_records = make_corpus(
            args.validation_count, args.seed + 1, forbidden
        )
        large, _, large_records = make_corpus(
            args.validation_count, args.seed + 2, forbidden, large=True
        )
    # Explicit counterexample to the old mirror-zero contract. Verified endpoints
    # have sigma -2,+2 and u<=1 each, hence distance exactly two, via the unknot.
    assert seifert.signature((1, 1, 1), 2) == -2
    assert seifert.signature((-1, -1, -1), 2) == 2
    mirror = pair_row(-1, "mirror-trefoils", ((1, 1, 1), 2), ((-1, -1, -1), 2), 2, 2, 6)
    train.append(mirror)
    corpus = {
        "train": records,
        "validation": val_records,
        "size_shift": large_records,
        "mirror_sentinel": {"lower": 2, "upper": 2, "in_training": True},
    }
    if args.resume:
        if json.loads((args.out / "corpus.json").read_text()) != json.loads(json.dumps(corpus)):
            raise RuntimeError("regenerated corpus differs; old artifact preserved")
    else:
        atomic_json(args.out / "corpus.json", corpus)
    samples = []
    for i, witness in enumerate(witnesses):
        # The whole demonstrated suffix must fit: do not clip a longer witness
        # then falsely call its first action good under the clipped move budget.
        samples.extend(
            M.samples_from_witness(
                witness,
                name=f"train-{i}",
                cap_moves=witness.moves,
                growth_len=witness.max_len,
                growth_strands=witness.max_strands,
                include_commutation=True,
                candidate_limit=args.policy_candidates,
            )
        )
    for sample in samples:
        sample.value_target = None
    train_histogram = dict(
        Counter(f"{r.spec.distance_lower}:{r.spec.distance_upper}" for r in train)
    )
    manifest.update(
        status="prepared",
        policy_samples=len(samples),
        train_intervals=train_histogram,
        corpus_sha256=sha(args.out / "corpus.json"),
    )
    atomic_json(args.out / "manifest.json", manifest)
    architecture = dict(
        width=48,
        depth=3,
        embedding_dim=64,
        encoder_kind="strand_graph",
        graph_rounds=2,
        context_rounds=args.context_rounds,
        normalized_knot=args.normalized_knot,
    )
    if args.hierarchy_levels:
        architecture.update(window_size=args.window_size, hierarchy_levels=args.hierarchy_levels)
    if args.budget_dim != 2:
        architecture['budget_dim'] = args.budget_dim
    net = M.SitePolicyValueNet(**architecture).to(args.device)
    objective = MetricEmbeddingObjective(
        cc_rank_weight=0.1, witness_priority=args.witness_weighting == "legacy"
    ).to(args.device)
    objective.calibrate_cc_scale(reference_cosine=0.4)
    if not args.resume:
        torch.save({'model': net.state_dict(), 'objective': objective.state_dict(),
                    'architecture': architecture}, args.out / 'initial.pt')
        manifest['initial_checkpoint_sha256'] = sha(args.out / 'initial.pt')
        atomic_json(args.out / 'manifest.json', manifest)
    optimizer = torch.optim.AdamW(
        list(net.parameters()) + list(objective.parameters()), lr=args.learning_rate
    )
    resume = (
        torch.load(args.out / "checkpoint.pt", map_location=args.device, weights_only=False)
        if args.resume
        else None
    )
    before = {
        "validation": metric_report(net, objective, validation, args.device),
        "size_shift": metric_report(net, objective, large, args.device),
    }
    if val_u:
        before["short_u"] = metric_report(net, objective, val_u, args.device)
        before["short_u_size_shift"] = metric_report(net, objective, large_u, args.device)
    selected = list(range(len(val_witnesses)))
    if args.corpus:
        selected, seen = [], set()
        for i in sorted(
            range(len(val_witnesses)),
            key=lambda i: (
                val_witnesses[i].crossing_changes,
                val_witnesses[i].moves,
                len(val_witnesses[i].start.word),
                i,
            ),
        ):
            det, sigma = val_records[i]["left_fingerprint"]
            fp = (det, abs(sigma))
            if fp not in seen:
                selected.append(i)
                seen.add(fp)
        manifest["search_fingerprints"] = [
            val_records[i]["left_fingerprint"] for i in selected[: args.search_instances]
        ]
        atomic_json(args.out / "manifest.json", manifest)
    # Same start words/caps/seeds in all arms, selected before seeing net output.
    search_instances = [
        M.Instance(
            f"val-{i}",
            w.start.word,
            w.start.strands,
            w.crossing_changes,
            cap_moves=args.search_moves,
            uid=f"val-{i}",
        )
        for i in selected[: args.search_instances]
        for w in [val_witnesses[i]]
    ]
    if not resume:
        before["search_uniform"] = search_report(
            None,
            search_instances,
            args.simulations,
            args.search_seeds,
            args.device,
            args.root_noise,
        )
        before["search_untrained"] = search_report(
            net, search_instances, args.simulations, args.search_seeds, args.device, args.root_noise
        )
        atomic_json(args.out / "before.json", before)
    else:
        if sha(args.out / "corpus.json") != old_manifest["corpus_sha256"]:
            raise RuntimeError("regenerated corpus changed on resume")
        net.load_state_dict(resume["model"])
        objective.load_state_dict(resume["objective"])
        optimizer.load_state_dict(resume["optimizer"])
        rng.setstate(resume["rng"])
        if args.u_sampling != "legacy":
            anchor_rng.setstate(resume["anchor_rng"])
            policy_rng.setstate(resume["policy_rng"])
        torch.set_rng_state(resume["torch_rng"].cpu())
    for step in range(resume["steps"] if resume else 0, args.steps):
        net.train()
        if u_groups:
            anchor_count = args.batch_size // 2
            classes = ("0", "1", "2", "interval")
            anchors = [
                anchor_rng.choice(u_groups[classes[(step * anchor_count + j) % 4]])
                for j in range(anchor_count)
            ]
            batch = rng.choices(train, k=args.batch_size - anchor_count - 1) + anchors + [mirror]
        else:
            batch = rng.choices(train, k=args.batch_size - 1) + [mirror]
        policy_batch = (policy_rng if u_groups else rng).choices(samples, k=8)
        optimizer.zero_grad(set_to_none=True)
        distance_loss, _ = metric_loss(net, objective, batch, args.device)
        policy_loss, _ = M.training_loss(net, policy_batch, device=args.device, value_weight=0.0)
        loss = args.metric_weight * distance_loss + policy_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 16 == 0 or step + 1 == args.steps:
            temporary = args.out / "checkpoint.partial"
            torch.save(
                {
                    "model": net.state_dict(),
                    "objective": objective.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "architecture": architecture,
                    "steps": step + 1,
                    "rng": rng.getstate(),
                    "anchor_rng": anchor_rng.getstate(),
                    "policy_rng": policy_rng.getstate(),
                    "torch_rng": torch.get_rng_state(),
                },
                temporary,
            )
            temporary.replace(args.out / "checkpoint.pt")
            manifest.update(
                status="training",
                completed_steps=step + 1,
                metric_presentations=(step + 1) * args.batch_size,
                distance_loss=float(distance_loss.detach()),
                policy_loss=float(policy_loss.detach()),
                elapsed_seconds=time.time() - started,
            )
            atomic_json(args.out / "manifest.json", manifest)
            print(
                json.dumps(
                    {
                        k: manifest[k]
                        for k in ("status", "completed_steps", "distance_loss", "policy_loss")
                    }
                ),
                flush=True,
            )
    after = {
        "validation": metric_report(net, objective, validation, args.device),
        "size_shift": metric_report(net, objective, large, args.device),
        "mirror_training_sentinel": metric_report(net, objective, [mirror], args.device),
        "search": search_report(
            net, search_instances, args.simulations, args.search_seeds, args.device, args.root_noise
        ),
    }
    if val_u:
        after["short_u"] = metric_report(net, objective, val_u, args.device)
        after["short_u_size_shift"] = metric_report(net, objective, large_u, args.device)
    atomic_json(args.out / "after.json", after)
    manifest.update(
        status="completed",
        checkpoint_sha256=sha(args.out / "checkpoint.pt"),
        elapsed_seconds=time.time() - started,
        promotion=(
            "UNPROVEN: compare matched trained controls; first-found search is not cost optimal"
            if args.corpus
            else "NOT_ELIGIBLE: smoke size, no knot-disjoint eval, no policy-only trained control"
        ),
    )
    atomic_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
