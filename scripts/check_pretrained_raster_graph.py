"""Protocol88 real-weight unified-network initialization and gradient gate."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from check_pretrained_raster_tile_transfer import smoke_word
from check_qrun_common_witnesses import check_pins, sha, write_new
from evaluate_common_greedy_policies import load_model
from pretrained_raster_graph import GLOBALS, RasterGraphPolicy
from pretrained_raster_tiles import PretrainedRasterTiles

from rf_knots import mcts_loop as M
from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE, INSERT, STABILIZE_POS

TRANSFER = Path("artifacts/pretrained-raster-tile-transfer-v1")
SEED = 20261044


def batch_for(words, strands):
    requests = []
    for word, n in zip(words, strands, strict=True):
        instance = M.Instance(
            "gate",
            tuple(word),
            n,
            cap_crossing_changes=2,
            cap_moves=128,
            growth_len=8,
            growth_strands=4,
            cc_weight=1000,
        )
        spec = instance.spec
        if word:
            actions = [
                spec.encode(CROSSING_CHANGE, p) for p in sorted({0, len(word) // 2, len(word) - 1})
            ]
            actions.append(spec.encode(INSERT, 0, n - 1, 1))
            actions.append(spec.encode(INSERT, len(word) - 1, 1, -1))
        else:
            actions = [spec.encode(STABILIZE_POS)]
        if not all(reference.is_legal(spec, tuple(word), n, a, True) for a in actions):
            raise ValueError("engineering gate constructed illegal candidate")
        requests.append((instance, M.SearchState(tuple(word), n, 0, 0), actions))
    return M.build_batch(requests, budget_dim=5)


def forward(net, batch, heads=None):
    return net(
        batch["word"],
        batch["lengths"],
        batch["strands"],
        batch["sites"],
        batch["site_mask"],
        batch["budgets"],
        heads=heads,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    transfer_report = TRANSFER / "report.json"
    transfer = json.loads(transfer_report.read_text())
    if transfer["status"] != "passed":
        raise ValueError("local transfer gate not passed")
    pins = dict(transfer["inputs"])
    if sha(TRANSFER / "spatial-trunk.pt") != transfer["checkpoint_sha256"]:
        raise ValueError("copied local checkpoint changed")
    for p in (
        transfer_report,
        TRANSFER / "spatial-trunk.pt",
        Path(__file__),
        Path("scripts/pretrained_raster_graph.py"),
        Path("tests/test_pretrained_raster_graph.py"),
        Path("research/88-pretrained-raster-graph-integration-gate.md"),
    ):
        pins[str(p.resolve())] = sha(p)
    check_pins(pins)
    if not all(str(p.resolve()) in pins for p in (args.pgx_root / "src").rglob("*.py")):
        raise ValueError("native source scope differs")
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    torch.set_num_threads(1)
    baseline = json.loads(Path("artifacts/qrun-baseline-input-check-v2/report.json").read_text())
    q = load_model("q254", baseline, pins)
    tile = PretrainedRasterTiles(q.network.representation)
    local = torch.load(TRANSFER / "spatial-trunk.pt", map_location="cpu", weights_only=False)
    tile.load_state_dict(local["model"], strict=True)
    torch.manual_seed(SEED)
    net = RasterGraphPolicy(tile).eval()
    state_before = {k: v.clone() for k, v in net.state_dict().items()}
    shape_contract = {k: list(v.shape) for k, v in state_before.items()}
    if any(
        not torch.equal(v, state_before[f"encoder.tiles.{k}"]) for k, v in local["model"].items()
    ):
        raise ValueError("local weights differ inside unified model")
    # Synthetic differentiation probe, not a scientific optimizer update.
    batch = batch_for([[1, 1, 1], [-1, -1, -1], []], [2, 2, 1])
    output = forward(net, batch, torch.tensor([0, 1, 0]))
    effective = output["effective"]
    if not torch.isfinite(output["logits"][effective]).all():
        raise ValueError("nonfinite legal logits")
    if not torch.isneginf(output["logits"][~effective]).all():
        raise ValueError("masked logits are not minus infinity")
    probe_loss = torch.nn.functional.cross_entropy(
        output["logits"], torch.zeros(3, dtype=torch.long)
    )
    probe_vector = torch.linspace(-1, 1, 64)
    for name in GLOBALS:
        probe_loss = probe_loss + (output[name] * probe_vector).sum() / 3
    probe_loss.backward()
    groups = {
        "pretrained_tiles": "encoder.tiles.",
        "local_projection": "encoder.project.",
        "graph": "encoder.message.",
        "multiscale": "encoder.context.",
        "global_heads": "encoder.heads.",
        "position": "position.",
        "policy": "head.",
    }
    gradient_norms = {}
    for label, prefix in groups.items():
        grads = [p.grad for name, p in net.named_parameters() if name.startswith(prefix)]
        if not grads or any(g is None or not torch.isfinite(g).all() for g in grads):
            raise ValueError(f"missing/nonfinite gradient group {label}")
        gradient_norms[label] = sum(float(g.abs().sum()) for g in grads)
        if gradient_norms[label] <= 0:
            raise ValueError(f"disconnected gradient group {label}")
    net.zero_grad(set_to_none=True)
    with torch.inference_mode():
        short = batch_for([[1, 1, 1]], [2])
        small = net.encoder.encode_sites(short["word"], short["lengths"], short["strands"])
        mixed = batch_for([[1, 1, 1], smoke_word(64, 33), []], [2, 33, 1])
        mixed["word"][0, 3:] = 1000000  # ignored padding must not influence the short row
        encoded = net.encoder.encode_sites(mixed["word"], mixed["lengths"], mixed["strands"])
        for name in GLOBALS:
            torch.testing.assert_close(small[name][0], encoded[name][0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            small["tokens"][0], encoded["tokens"][0, :3], rtol=1e-5, atol=1e-6
        )
        backbone_calls = []
        hook = net.encoder.tiles.register_forward_hook(lambda *unused: backbone_calls.append(1))
        net.score_encoded(
            encoded,
            mixed["sites"],
            mixed["site_mask"],
            mixed["budgets"],
            heads=torch.tensor([0, 2, 0]),
        )
        net.score_encoded(
            encoded,
            mixed["sites"],
            mixed["site_mask"],
            mixed["budgets"],
            heads=torch.tensor([1, 5, 0]),
        )
        hook.remove()
        if backbone_calls:
            raise ValueError("head-only rescore recomputed raster")
        word, offset, head = smoke_word(64, 33), 17, 9
        original = batch_for([word], [33])
        rotated = batch_for([word[offset:] + word[:offset]], [33])
        rotated["sites"] = original["sites"].clone()
        rotated["sites"][..., 1] = (rotated["sites"][..., 1] - offset) % len(word)
        a = forward(net, original, torch.tensor([head]))
        b = forward(net, rotated, torch.tensor([(head - offset) % len(word)]))
        for name in (*GLOBALS, "logits"):
            torch.testing.assert_close(a[name], b[name], rtol=1e-5, atol=1e-5)
        covariance_error = float((a["logits"] - b["logits"]).abs().max())
        scaling = []
        for length, n in ((64, 33), (256, 65), (4096, 1025)):
            inputs = batch_for([smoke_word(length, n)], [n])
            started = time.perf_counter()
            result = forward(net, inputs, torch.tensor([length // 3]))
            elapsed = time.perf_counter() - started
            if not torch.isfinite(result["logits"][result["effective"]]).all():
                raise ValueError("nonfinite long-input policy")
            for name in GLOBALS:
                if result[name].shape != (1, 64) or not torch.isfinite(result[name]).all():
                    raise ValueError("invalid long-input global embedding")
            if {k: list(v.shape) for k, v in net.state_dict().items()} != shape_contract:
                raise ValueError("parameter shapes depend on input size")
            scaling.append(
                {
                    "length": length,
                    "strands": n,
                    "seconds": elapsed,
                    "scored_actions": int(result["effective"].sum()),
                }
            )
            print(json.dumps(scaling[-1]), flush=True)
    if any(not torch.equal(v, net.state_dict()[k]) for k, v in state_before.items()):
        raise ValueError("engineering gate changed model weights")
    check_pins(pins)
    args.out.mkdir(parents=True, exist_ok=False)
    architecture = {
        "width": 64,
        "embedding_dim": 64,
        "graph_rounds": 2,
        "tile_rows": 12,
        "tile_columns": 7,
        "tile_width": 64,
        "tile_batch": 64,
    }
    checkpoint = args.out / "initialization.pt"
    torch.save(
        {
            "model": net.state_dict(),
            "architecture": architecture,
            "seed": SEED,
            "trained": False,
            "local_source_sha256": transfer["checkpoint_sha256"],
        },
        checkpoint,
    )
    # Strict reload checks the entire saved initialization, not only its local trunk.
    restored = RasterGraphPolicy(PretrainedRasterTiles(q.network.representation)).eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored.load_state_dict(payload["model"], strict=True)
    if any(not torch.equal(v, restored.state_dict()[k]) for k, v in state_before.items()):
        raise ValueError("strict initialization reload differs")
    report = {
        "status": "passed",
        "inputs": pins,
        "checkpoint_sha256": sha(checkpoint),
        "architecture": architecture,
        "seed": SEED,
        "parameters": sum(p.numel() for p in net.parameters()),
        "pretrained_parameters": sum(p.numel() for p in net.encoder.tiles.parameters()),
        "gradient_l1_by_group": gradient_norms,
        "scaling": scaling,
        "head_rescore_backbone_calls": len(backbone_calls),
        "cyclic_policy_max_abs_error": covariance_error,
        "strict_reload_passed": True,
        "optimizer_steps": 0,
        "trained_policy": False,
        "value_head_present": False,
        "scope": (
            "Unified architecture initialization/gradient gate, not learned metric/search quality"
        ),
    }
    write_new(args.out / "report.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ("status", "parameters", "pretrained_parameters", "scope")}
        )
    )


if __name__ == "__main__":
    main()
