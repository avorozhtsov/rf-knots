"""Post-hoc parameter-coordinate attribution of frozen118 gradient vectors.

No forwards, optimizer, search, source changes or feature interventions.
"""

import argparse
import json
from pathlib import Path

import torch
from check_qrun_common_witnesses import check_pins, sha, write_new

ROOT = Path("artifacts/raster-task-gradient-diagnostic-v1")
REPORT_SHA = "f4911c95002d5db7cc076e07b633b7324bf9f9db06142a77592e391da494fdb2"


def attribute(weight, bias, total_squared_norm):
    weight, bias = weight.double(), bias.double()
    if weight.shape != (64, 208) or bias.shape != (64,):
        raise ValueError("require protocol97 context linear dimensions")
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("nonfinite saved gradient")
    squared = float(weight.square().sum())
    if not torch.isfinite(torch.tensor(total_squared_norm)) or total_squared_norm < squared:
        raise ValueError("invalid whole-network squared norm")
    columns = {
        "global_knot": (0, 64),
        "global_isotopy": (64, 128),
        "global_layout": (128, 192),
        "raw_moves": (192, 193),
        "raw_cc": (193, 194),
        "raw_length_room": (194, 195),
        "raw_strand_room": (195, 196),
        "raw_lambda": (196, 197),
        "log_resources": (197, 202),
        "exhaustion_flags": (202, 207),
        "present": (207, 208),
    }
    norms = {
        key: float(weight[:, start:stop].square().sum()) for key, (start, stop) in columns.items()
    }
    raw = sum(norms[key] for key in columns if key.startswith("raw_"))
    lam = norms["raw_lambda"]
    identity_error = (weight[:, 196] - 1000 * bias).abs()
    return dict(
        column_squared_norms=norms,
        first_linear_squared_norm=squared,
        total_squared_norm=total_squared_norm,
        raw_resources_share_linear=raw / squared if squared else None,
        raw_lambda_share_linear=lam / squared if squared else None,
        raw_lambda_share_whole=lam / total_squared_norm if total_squared_norm else None,
        # Algebraic parameter-coordinate change ONLY; not a new policy or an AdamW step.
        coordinate_only_norm_ratio=((total_squared_norm - lam + lam / 1e6) / total_squared_norm)
        ** 0.5
        if total_squared_norm
        else None,
        lambda_1000_bias_identity_max_abs_error=float(identity_error.max()),
        lambda_1000_bias_identity_relative_l2_error=float(identity_error.norm()) / lam**0.5
        if lam
        else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    if sha(ROOT / "report.json") != REPORT_SHA:
        raise ValueError("different gradient diagnostic")
    report = json.loads((ROOT / "report.json").read_text())
    inputs = json.loads((ROOT / "inputs.json").read_text())
    if sha(ROOT / "inputs.json") != report["inputs_sha256"]:
        raise ValueError("changed diagnostic input report")
    check_pins(inputs["inputs"])
    selection_path = ROOT / "selection.json"
    if sha(selection_path) != report["selection_sha256"]:
        raise ValueError("changed selection")
    selection = json.loads(selection_path.read_text())
    if not all(
        row["request"]["instance"]["cc_weight"] == 1000
        for group in selection["full_relative_groups"]
        for row in group
    ):
        raise ValueError("relative constant-lambda premise failed")
    pins = {
        str(ROOT / "report.json"): REPORT_SHA,
        str(ROOT / "inputs.json"): sha(ROOT / "inputs.json"),
        str(selection_path): sha(selection_path),
        str(Path(__file__).resolve()): sha(__file__),
        "tests/test_raster_resource_gradients.py": sha("tests/test_raster_resource_gradients.py"),
    }
    rows = []
    for case, expected in report["cases"].items():
        path = ROOT / case / "report.json"
        if sha(path) != expected:
            raise ValueError("changed diagnostic case")
        case_report = json.loads(path.read_text())
        vectors_path = path.parent / "gradients.pt"
        if sha(vectors_path) != case_report["vectors_sha256"]:
            raise ValueError("changed saved gradients")
        pins[str(path)], pins[str(vectors_path)] = expected, sha(vectors_path)
        vectors = torch.load(vectors_path, map_location="cpu", weights_only=True)
        for task, vector in vectors.items():
            layout = {p["name"]: p for p in case_report["layout"][task]}

            def tensor(name, layout=layout, vector=vector):
                item = layout["net.head.context_film.0." + name]
                return vector[item["start"] : item["stop"]].reshape(item["shape"])

            result = attribute(
                tensor("weight"), tensor("bias"), float(vector.double().square().sum())
            )
            if not task.startswith("relative"):
                result["lambda_1000_bias_identity_max_abs_error"] = None
                result["lambda_1000_bias_identity_relative_l2_error"] = None
            rows.append(dict(case=case, task=task, **result))
    check_pins(pins)
    write_new(
        args.out,
        dict(
            status="completed",
            post_hoc=True,
            inputs=pins,
            rows=rows,
            neural_forwards=0,
            optimizer_steps=0,
            scope="Parameter-coordinate attribution, not feature importance, causal intervention "
            "or optimizer equivalence.",
        ),
    )
    print(json.dumps(dict(status="completed", rows=len(rows), sha256=sha(args.out))))


if __name__ == "__main__":
    main()
