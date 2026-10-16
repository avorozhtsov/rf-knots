"""Protocol86 actual Q254 spatial-weight transfer, geometry and scaling gate."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from check_qrun_common_witnesses import check_pins, sha, write_new
from evaluate_common_greedy_policies import load_model
from pretrained_raster_tiles import PretrainedRasterTiles, compile_tiles, head_context

from rf_knots import reference


def smoke_word(length, strands):
    if length < strands - 1 or (length - strands + 1) % 2:
        raise ValueError("require an N-cycle backbone plus inverse pairs")
    word = list(range(1, strands))
    for i in range((length - len(word)) // 2):
        g = 1 + (i * 37 + 11) % (strands - 1)
        word.extend((g, -g))
    if reference.num_components(tuple(word), strands) != 1:
        raise ValueError("smoke closure is not a knot")
    return word


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    baseline_path = Path("artifacts/qrun-baseline-input-check-v2/report.json")
    baseline = json.loads(baseline_path.read_text())
    if baseline["status"] != "passed":
        raise ValueError("require verified Q checkpoint")
    pins = dict(baseline["inputs"])
    for path in (
        baseline_path,
        Path(__file__),
        Path("scripts/pretrained_raster_tiles.py"),
        Path("scripts/evaluate_common_greedy_policies.py"),
        Path("scripts/check_qrun_common_witnesses.py"),
        Path("research/86-pretrained-raster-tile-transfer-gate.md"),
    ):
        key, value = str(path.resolve()), sha(path)
        if key in pins and pins[key] != value:
            raise ValueError("inherited source hash changed")
        pins[key] = value
    check_pins(pins)
    if not all(str(p.resolve()) in pins for p in (args.pgx_root / "src").rglob("*.py")):
        raise ValueError("unverified native source scope")
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    torch.set_num_threads(1)
    model = load_model("q254", baseline, pins)
    teacher = model.network.representation
    net = PretrainedRasterTiles(teacher).eval()
    expected = {
        k: v
        for k, v in teacher.state_dict().items()
        if k.startswith("input.") or k.startswith("blocks.")
    }
    initial = {k: v.clone() for k, v in expected.items()}
    if set(net.state_dict()) != set(expected) or any(
        not torch.equal(v, net.state_dict()[k]) for k, v in expected.items()
    ):
        raise ValueError("spatial trunk was not copied exactly")
    parameter_count = sum(p.numel() for p in net.parameters())
    shapes = {k: list(v.shape) for k, v in net.state_dict().items()}
    native_checks, dense_checks = [], []
    small = [([1], 2), ([1, 1, 1], 2), (smoke_word(16, 5), 5), (smoke_word(25, 12), 12)]
    with torch.inference_mode():
        for word, strands in small:
            pgx = model.game.env.init_from_word(word, strands, log_ratio=math.log(1000))
            for head in sorted({0, len(word) // 2, len(word) - 1}):
                t = model.game._view(
                    pgx,
                    head,
                    model.game._no_registers(),
                    model.game._no_colours(),
                    0,
                    model.game._no_tape(),
                    reward=0.0,
                )
                observation = torch.from_numpy(np.asarray(t.observation).copy())
                observation = observation.unsqueeze(0).permute(0, 3, 1, 2).float()
                raster = observation[:, teacher.raster_start : teacher.raster_end, 0, :]
                raster = raster.reshape(1, 12, 4, 7).permute(0, 2, 1, 3)
                tile, _ = compile_tiles(word, strands, [head])
                torch.testing.assert_close(tile, raster, rtol=0, atol=0)
                native = teacher.encode_spatial(observation)[0]
                actual = net.spatial(tile)
                torch.testing.assert_close(actual, native, rtol=1e-6, atol=1e-6)
                native_checks.append(
                    {
                        "length": len(word),
                        "strands": strands,
                        "head": head,
                        "bitwise_equal": torch.equal(actual, native),
                        "max_abs_error": float((actual - native).abs().max()),
                    }
                )
        for word, strands in [*small, (smoke_word(64, 33), 33)]:
            centers = list(range(len(word)))
            tile, origins = compile_tiles(word, strands, centers)
            dense = model.game.braid_raster_planes(word, strands, max(strands, 12))
            for i, head in enumerate(centers):
                positions = (head + np.arange(7) - 3) % len(word)
                origin = int(origins[i])
                expected_tile = torch.from_numpy(dense[positions, origin : origin + 12, :])
                torch.testing.assert_close(tile[i], expected_tile.permute(2, 1, 0), rtol=0, atol=0)
            dense_checks.append(
                {"length": len(word), "strands": strands, "tiles_checked": len(centers)}
            )
    # Backprop checks trainability, but performs no optimizer update.
    net.zero_grad(set_to_none=True)
    tokens = net(smoke_word(25, 12), 12)
    tokens.square().mean().backward()
    gradients = [p.grad for p in net.parameters()]
    if any(g is None or not torch.isfinite(g).all() for g in gradients):
        raise ValueError("missing/nonfinite gradients in transferred trunk")
    if not net.input.weight.grad.abs().sum() > 0:
        raise ValueError("zero input-convolution gradient")
    net.zero_grad(set_to_none=True)
    scaling = []
    with torch.inference_mode():
        for length, strands in ((64, 33), (256, 65), (4096, 1025)):
            word = smoke_word(length, strands)
            started = time.perf_counter()
            tokens = net(word, strands)
            elapsed = time.perf_counter() - started
            if tokens.shape != (length, 128) or not torch.isfinite(tokens).all():
                raise ValueError("bad long-word token output")
            if {k: list(v.shape) for k, v in net.state_dict().items()} != shapes:
                raise ValueError("parameter shapes changed with input size")
            scaling.append(
                {
                    "length": length,
                    "strands": strands,
                    "tokens_shape": list(tokens.shape),
                    "seconds": elapsed,
                    "parameter_count": parameter_count,
                }
            )
            print(json.dumps(scaling[-1]), flush=True)
        word, offset, head = smoke_word(64, 33), 17, 9
        shifted = word[offset:] + word[:offset]
        left, right = net(word, 33), net(shifted, 33)
        torch.testing.assert_close(right, torch.roll(left, -offset, dims=0), rtol=1e-6, atol=1e-6)
        before_context = head_context(word, 33, head)
        after_context = head_context(shifted, 33, (head - offset) % len(word))
        torch.testing.assert_close(after_context, torch.roll(before_context, -offset, dims=0))
        moved_context = head_context(word, 33, head + 1)
        if torch.equal(before_context[:, :8], moved_context[:, :8]):
            raise ValueError("head-relative phases did not change")
        if not torch.equal(before_context[:, 8:], moved_context[:, 8:]):
            raise ValueError("head changed head-independent geometry")
    if any(not torch.equal(net.state_dict()[k], v) for k, v in initial.items()):
        raise ValueError("gate accidentally updated copied parameters")
    if any(not torch.equal(teacher.state_dict()[k], v) for k, v in initial.items()):
        raise ValueError("gate accidentally changed source teacher")
    if net.largest_tile_batch > 64:
        raise ValueError("unbounded tile batch")
    check_pins(pins)
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = args.out / "spatial-trunk.pt"
    torch.save(
        {
            "model": net.state_dict(),
            "source_checkpoint_sha256": baseline["results"]["q254"]["checkpoint_sha256"],
            "architecture": {"rows": 12, "columns": 7, "width": 64, "blocks": 4, "tile_batch": 64},
        },
        checkpoint,
    )
    report = {
        "status": "passed",
        "inputs": pins,
        "parameter_count": parameter_count,
        "source_policy_parameter_count": sum(p.numel() for p in model.network.parameters()),
        "checkpoint_sha256": sha(checkpoint),
        "parameter_shapes": shapes,
        "native_equivalence": native_checks,
        "dense_crop_equivalence": dense_checks,
        "finite_gradients": True,
        "weights_updated": False,
        "scaling": scaling,
        "largest_tile_batch": net.largest_tile_batch,
        "token_width": 128,
        "position_context_width": 16,
        "cyclic_covariance_passed": True,
        "scope": (
            "Pretrained spatial tokenizer only; global graph/metric/policy integration "
            "untrained and not implemented here"
        ),
    }
    write_new(args.out / "report.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ("status", "parameter_count", "largest_tile_batch", "scope")}
        )
    )


if __name__ == "__main__":
    main()
