"""Final135 training integrity gate; does not claim distance or search success."""

import argparse
import json
import sys
from pathlib import Path

import torch
from check_qrun_common_witnesses import check_pins, sha, write_new
from raster_log_training import digest, matched_initializations, validate_relative_counts
from train_raster_resource_pair import CONFIG, verify_checkpoint

from rf_knots.embedding_metric import MetricEmbeddingObjective


def check_tensors(actual, expected, *, exact=False):
    if actual.keys() != expected.keys():
        raise ValueError("tensor key coverage differs")
    for key, value in actual.items():
        if (
            value.shape != expected[key].shape
            or value.dtype != expected[key].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"tensor shape/dtype/finiteness differs: {key}")
        if exact and not torch.equal(value, expected[key]):
            raise ValueError(f"initial tensor differs: {key}")


def check_optimizer(payload, named_parameters, step):
    state = payload["optimizer"]
    groups = state["param_groups"]
    if (
        len(groups) != 1
        or groups[0]["params"] != list(range(len(named_parameters)))
        or groups[0]["lr"] != CONFIG["lr"]
        or groups[0]["weight_decay"] != CONFIG["weight_decay"]
        or groups[0]["betas"] != (0.9, 0.999)
        or groups[0]["eps"] != 1e-8
    ):
        raise ValueError("optimizer configuration/parameter coverage differs")
    expected = (
        set()
        if step == 0
        else {i for i, (name, _) in enumerate(named_parameters) if name != "objective.rm_scale_raw"}
    )
    if state["state"].keys() != expected:
        raise ValueError("optimizer state coverage differs")
    for index, values in state["state"].items():
        if float(values["step"]) != step:
            raise ValueError("optimizer step differs")
        for key in ("exp_avg", "exp_avg_sq"):
            value = values[key]
            parameter = named_parameters[index][1]
            if (
                value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError("optimizer moments differ or are nonfinite")


def audit_arm(directory, arm, parent, expected_state, architecture, parameters):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if (
        manifest["status"] != "completed"
        or manifest["phase"] != "completed"
        or manifest["completed_steps"] != CONFIG["steps"]
        or manifest["contract_sha256"] != parent["contract_sha256"]
        or manifest["arm"] != arm
    ):
        raise ValueError("arm incomplete or contract differs")
    pins = {str(path): sha(path)}
    payloads = {}
    objective_state = MetricEmbeddingObjective().state_dict()
    for stage, key, step in (
        ("initial", "initial_checkpoint", 0),
        ("final", "checkpoint", CONFIG["steps"]),
    ):
        entry = manifest[key]
        payload = verify_checkpoint(directory, entry, parent["contract_sha256"], arm)
        if payload["steps"] != step or payload["architecture"] != architecture:
            raise ValueError("stage step/architecture differs")
        check_tensors(payload["model"], expected_state, exact=stage == "initial")
        check_tensors(payload["objective"], objective_state, exact=stage == "initial")
        check_optimizer(payload, parameters, step)
        validate_relative_counts(payload["relative_counts"], 4, step)
        pins[str(directory / entry["file"])] = entry["sha256"]
        payloads[stage] = payload
    initial, final = payloads["initial"], payloads["final"]
    if initial["draw_digest"] != digest([]):
        raise ValueError("initial draw trace nonempty")
    if (
        final["draw_digest"] != manifest["draw_digest"]
        or final["relative_counts"] != manifest["relative_counts"]
    ):
        raise ValueError("manifest/checkpoint sampling differs")
    if not torch.equal(final["objective"]["rm_scale_raw"], initial["objective"]["rm_scale_raw"]):
        raise ValueError("unused RM scale changed")
    changed = [k for k in expected_state if not torch.equal(final["model"][k], expected_state[k])]
    for prefix in ("encoder.tiles.", "encoder.heads.", "head."):
        if not any(k.startswith(prefix) for k in changed):
            raise ValueError(f"no trained tensor changed in {prefix}")
    expected_reports = {"dev-000000.json", "dev-002048.json"}
    if set(manifest["development_sha256"]) != expected_reports:
        raise ValueError("development coverage differs")
    for filename, checksum in manifest["development_sha256"].items():
        report_path = directory / filename
        if sha(report_path) != checksum:
            raise ValueError("development report changed")
        report = json.loads(report_path.read_text())
        expected_entry = manifest[
            "initial_checkpoint" if filename == "dev-000000.json" else "checkpoint"
        ]
        if (
            report["count"] != 256
            or len(report["records"]) != 256
            or report["checkpoint"] != expected_entry
            or report["contract_sha256"] != parent["contract_sha256"]
            or report["arm"] != arm
        ):
            raise ValueError("development binding/coverage differs")
        pins[str(report_path)] = checksum
    return (
        payloads,
        pins,
        dict(
            initial=manifest["initial_checkpoint"],
            final=manifest["checkpoint"],
            steps=final["steps"],
            optimizer_states=len(final["optimizer"]["state"]),
            changed_model_tensor_keys=changed,
            draw_digest=final["draw_digest"],
            relative_counts=final["relative_counts"],
            metric_draws=32 * final["steps"],
            positive_draws=8 * final["steps"],
            reference_sha256=manifest["reference_sha256"],
        ),
    )


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
    if (
        parent["status"] != "completed"
        or parent["completed_arms"] != CONFIG["arms"]
        or parent["config"] != CONFIG
        or parent["current_arm"] is not None
    ):
        raise ValueError("both fixed training arms must be complete")
    if parent["contract_sha256"] != digest(
        dict(config=CONFIG, inputs=parent["inputs"], sources=parent["sources"])
    ):
        raise ValueError("training contract differs")
    pins = {
        **parent["inputs"],
        **parent["sources"],
        str(parent_path): sha(parent_path),
        str(Path(__file__).resolve()): sha(Path(__file__)),
        "tests/test_resource_training_audit.py": sha(Path("tests/test_resource_training_audit.py")),
    }
    check_pins(pins)
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    net, states, architecture, _ = matched_initializations(parent["inputs"])
    objective = MetricEmbeddingObjective()
    parameters = list(net.named_parameters()) + [
        ("objective." + name, p) for name, p in objective.named_parameters()
    ]
    results, payloads = {}, {}
    for arm in CONFIG["arms"]:
        payloads[arm], added, results[arm] = audit_arm(
            args.run / arm, arm, parent, states["q254"], architecture, parameters
        )
        for name, checksum in added.items():
            if name in pins and pins[name] != checksum:
                raise ValueError("conflicting audit input")
            pins[name] = checksum
    a, b = (payloads[arm]["final"] for arm in CONFIG["arms"])
    if (
        a["draw_digest"] != b["draw_digest"]
        or a["rng"] != b["rng"]
        or a["relative_counts"] != b["relative_counts"]
        or not torch.equal(a["torch_rng"], b["torch_rng"])
        or not torch.equal(a["mps_rng"], b["mps_rng"])
    ):
        raise ValueError("paired data/random state differs")
    check_pins(pins)
    write_new(
        args.out,
        dict(
            status="passed",
            inputs=pins,
            arms=results,
            initial_tensors_identical=True,
            data_rng_aligned=True,
            scope="Training integrity; independent distance/search still required",
        ),
    )
    print(json.dumps(dict(status="passed", arms=list(results))), flush=True)


if __name__ == "__main__":
    main()
