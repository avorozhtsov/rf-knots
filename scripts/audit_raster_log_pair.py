"""Fail-closed final training audit for97; independent metrics/search still required."""

import argparse
import json
import sys
from pathlib import Path

import torch
from check_qrun_common_witnesses import check_pins, sha, write_new
from raster_log_training import CONFIG, digest, matched_initializations, validate_relative_counts


def audit_arm(directory, arm, expected_state, architecture, parent):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "completed" or manifest["completed_steps"] != CONFIG["steps"]:
        raise ValueError(f"{arm} is incomplete")
    if manifest["config"] != {**CONFIG, "arm": arm} or manifest["architecture"] != architecture:
        raise ValueError("arm configuration/architecture differs")
    if manifest["inputs"] != parent["inputs"] or manifest["sources"] != parent["sources"]:
        raise ValueError("arm and launcher provenance differ")
    if manifest["contract_sha256"] != digest(
        {"config": manifest["config"], "inputs": manifest["inputs"], "sources": manifest["sources"]}
    ):
        raise ValueError("arm contract digest differs")
    for name, key in (
        ("initial.pt", "initial_checkpoint_sha256"),
        ("progress.pt", "progress_sha256"),
        ("checkpoint.pt", "checkpoint_sha256"),
    ):
        if sha(directory / name) != manifest[key]:
            raise ValueError(f"{arm}/{name} hash differs")
    if manifest["checkpoint_sha256"] != manifest["progress_sha256"]:
        raise ValueError("final and last progress checkpoint differ")
    initial = torch.load(directory / "initial.pt", map_location="cpu", weights_only=False)
    final = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
    if set(initial["model"]) != set(expected_state) or any(
        not torch.equal(value, initial["model"][key]) for key, value in expected_state.items()
    ):
        raise ValueError("initial model differs from exact matched initialization")
    if (
        final["steps"] != CONFIG["steps"]
        or final["architecture"] != architecture
        or final["contract_sha256"] != manifest["contract_sha256"]
        or final["draw_digest"] != manifest["draw_digest"]
    ):
        raise ValueError("checkpoint and manifest contract/step/draw identity differs")
    if set(final["model"]) != set(expected_state) or any(
        value.shape != expected_state[key].shape
        or value.dtype != expected_state[key].dtype
        or not torch.isfinite(value).all()
        for key, value in final["model"].items()
    ):
        raise ValueError("invalid trained model shapes or nonfinite tensors")
    if not all(torch.isfinite(v).all() for v in final["objective"].values()):
        raise ValueError("nonfinite objective")
    optimizer_steps = []
    groups = final["optimizer"]["param_groups"]
    if (
        len(groups) != 1
        or groups[0]["lr"] != CONFIG["lr"]
        or groups[0]["weight_decay"] != CONFIG["weight_decay"]
    ):
        raise ValueError("optimizer hyperparameters differ from fixed contract")
    for state in final["optimizer"]["state"].values():
        if any(
            isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
            for value in state.values()
        ):
            raise ValueError("nonfinite optimizer state")
        step = float(state["step"])
        if step != int(step) or not 0 < step <= CONFIG["steps"]:
            raise ValueError("invalid optimizer update count")
        optimizer_steps.append(int(step))
    if not optimizer_steps or max(optimizer_steps) != CONFIG["steps"]:
        raise ValueError("no optimizer parameter reached the final update")
    for key, per_step in (
        ("metric_sampled_rows", 32),
        ("positive_sampled_states", 8),
    ):
        if manifest[key] != per_step * CONFIG["steps"]:
            raise ValueError("sample accounting differs")
    counts = final["relative_group_update_counts"]
    validate_relative_counts(counts, 4, CONFIG["steps"])
    sizes = [334, 52, 74, 54]
    if (
        manifest["relative_group_update_counts"] != counts
        or manifest["relative_group_sizes"] != sizes
        or manifest["relative_updates"] != sum(counts)
        or manifest["relative_full_states"]
        != sum(n * size for n, size in zip(counts, sizes, strict=True))
    ):
        raise ValueError("full-group relative accounting differs")
    changed = [
        key
        for key, value in final["model"].items()
        if not torch.equal(value, initial["model"][key])
    ]
    for prefix in ("encoder.tiles.", "encoder.heads.", "head."):
        if not any(key.startswith(prefix) for key in changed):
            raise ValueError(f"no trained tensor changed in {prefix}")
    return final, {
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "initial_sha256": manifest["initial_checkpoint_sha256"],
        "manifest_sha256": sha(manifest_path),
        "changed_tensor_keys": changed,
        "optimizer_state_entries": len(optimizer_steps),
        "optimizer_step_counts": sorted(set(optimizer_steps)),
        "completed_steps": final["steps"],
        "draw_digest": final["draw_digest"],
        "relative_group_update_counts": counts,
        "relative_full_states": manifest["relative_full_states"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    parent_path = args.run / "manifest.json"
    parent = json.loads(parent_path.read_text())
    if parent["status"] != "completed" or parent["completed_arms"] != ["q254", "random"]:
        raise ValueError("full training pair is not complete")
    check_pins(parent["inputs"])
    check_pins(parent["sources"])
    if parent["config"] != CONFIG or parent["contract_sha256"] != digest(
        {"config": CONFIG, "inputs": parent["inputs"], "sources": parent["sources"]}
    ):
        raise ValueError("launcher contract changed")
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    _, states, architecture, differences = matched_initializations(parent["inputs"])
    finals, rows = {}, {}
    for arm in ("q254", "random"):
        finals[arm], rows[arm] = audit_arm(args.run / arm, arm, states[arm], architecture, parent)
    if (
        finals["q254"]["relative_group_update_counts"]
        != finals["random"]["relative_group_update_counts"]
        or finals["q254"]["rng"] != finals["random"]["rng"]
        or not torch.equal(finals["q254"]["torch_rng"], finals["random"]["torch_rng"])
        or finals["q254"]["draw_digest"] != finals["random"]["draw_digest"]
    ):
        raise ValueError("paired data/RNG coverage differs")
    pins = {
        **parent["inputs"],
        **parent["sources"],
        str(parent_path): sha(parent_path),
        str(Path(__file__).resolve()): sha(Path(__file__)),
    }
    for arm in rows:
        for filename in ("initial.pt", "checkpoint.pt", "manifest.json"):
            path = args.run / arm / filename
            pins[str(path)] = sha(path)
    check_pins(pins)
    write_new(
        args.out,
        {
            "status": "passed",
            "inputs": pins,
            "arms": rows,
            "initialization_changed_keys": differences,
            "data_rng_aligned": True,
            "scope": "Training integrity only; independent distance/search not implied",
        },
    )
    print(json.dumps({"status": "passed", "arms": list(rows), "data_rng_aligned": True}))


if __name__ == "__main__":
    main()
