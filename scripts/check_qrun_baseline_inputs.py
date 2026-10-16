"""Read-only checkpoint lineage and native forward checks, not a search benchmark."""

import argparse
import gzip
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def equal_states(left, right):
    return set(left) == set(right) and all(torch.equal(left[k], v) for k, v in right.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--q254-dir", type=Path, required=True)
    parser.add_argument("--q304-manifest", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    from pgx_mcts_bench.adaptive_scientists import load_scientist

    hashes = {}

    def read(path):
        path = path.resolve()
        data = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        return data

    read(Path(__file__))
    for path in sorted((args.pgx_root / "src/pgx_mcts_bench").rglob("*.py")):
        read(path)
    for path in sorted(Path("src/rf_knots").glob("*.py")):
        read(path)
    meta254 = json.loads(read(args.q254_dir / "manifest.json"))
    meta304 = json.loads(read(args.q304_manifest))
    promotion = json.loads(read(args.promotion))
    parent = promotion["results"]["q304-parent"]
    state_path = Path(parent["state"])
    blob = read(state_path)
    if hashes[str(state_path.resolve())] != parent["state_sha256"]:
        raise ValueError("evaluated Q304 parent state changed")
    state = torch.load(io.BytesIO(gzip.decompress(blob)), map_location="cpu", weights_only=False)
    scientist = state.get("scientists", {}).get("strand-graph-12", state.get("scientist"))
    parent_weights = scientist["network"]
    torch.set_num_threads(1)
    results = {}
    specifications = [
        ("q254", "raster-axial-12", args.q254_dir / "checkpoint.pt", meta254),
        ("q304", "strand-graph-12", Path(meta304["checkpoint"]), meta304),
    ]
    for label, name, path, metadata in specifications:
        data = read(path)
        if hashes[str(path.resolve())] != metadata["checkpoint_sha256"]:
            raise ValueError(f"{label} checkpoint differs from its manifest")
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        weights = payload.get("network", payload)
        loaded = load_scientist(
            name, path, seed=20260921, device="cpu", simulations=0,
            require_factorized=True, objective_budget_channel=True,
        )
        loaded.network.eval()
        game = loaded.game
        pgx_state = game.env.init_from_word([1, 1, 1], 2, log_ratio=math.log(1000))
        transition = game._view(
            pgx_state, 0, game._no_registers(), game._no_colours(), 0,
            game._no_tape(), reward=0.0,
        )
        # Native adapter observations are HWC, while Torch expects BCHW.
        tensor = torch.from_numpy(np.asarray(transition.observation).copy())
        tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2).float()
        with torch.inference_mode():
            logits, value = loaded.network(tensor)
        results[label] = {
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "native_network": type(loaded.network).__name__,
            "loaded_weights_exact": equal_states(loaded.network.state_dict(), weights),
            "same_tensors_as_evaluated_q304_parent": (
                equal_states(weights, parent_weights) if label == "q304" else None
            ),
            "max_len": loaded.config.game.max_len,
            "max_strands": loaded.config.game.max_strands,
            "observation_shape_hwc": list(transition.observation.shape),
            "logits_shape": list(logits.shape),
            "legal_actions": int(np.asarray(transition.legal_actions).sum()),
            "finite_forward": bool(torch.isfinite(logits).all() and torch.isfinite(value).all()),
            "parameter_count": sum(p.numel() for p in loaded.network.parameters()),
        }
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"input changed during check: {path}")
    passed = all(r["loaded_weights_exact"] and r["finite_forward"] for r in results.values())
    passed = passed and results["q304"]["same_tensors_as_evaluated_q304_parent"]
    report = {
        "status": "passed" if passed else "not_ready",
        "results": results, "inputs": hashes,
        "promotion_status": promotion["status"], "promotion_summary": promotion["summary"],
        "training_performed": False,
        "scope": "Native weights and one trefoil forward only; no search/transfer quality claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
