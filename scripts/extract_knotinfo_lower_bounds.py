"""Extract theorem-producing lower-bound invariants from a pinned KnotInfo XLS.

Usage:
    uv run --with pandas --with xlrd python scripts/extract_knotinfo_lower_bounds.py \
        /path/to/knotinfo_data_complete.xls
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src" / "rf_knots" / "data" / "lower_bounds.json"
EXPECTED_SHA256 = "bd454dcb6bcd5effe205b27ca9de172bb21cf87ce190e15f870e1b07a714ccbe"


def lower_endpoint(raw: str) -> int | None:
    numbers = [int(x) for x in re.findall(r"-?\d+", raw)]
    return min(numbers) if numbers else None


def main(path: str) -> None:
    import pandas as pd

    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"KnotInfo snapshot hash mismatch: {digest}")
    frame = pd.read_excel(source, header=0, dtype=str).fillna("")
    values = {}
    for row in frame.to_dict(orient="records"):
        name = row["name"].strip()
        match = re.fullmatch(r"(\d+)(?:[an])?_(\d+)", name)
        if match is None or int(match.group(1)) > 12:
            continue
        s_raw = row["rasmussen_invariant"].strip()
        tau_raw = row["ozsvath_szabo_tau_invariant"].strip()
        nakanishi_raw = row["nakanishi_index"].strip()
        entry = {}
        if re.fullmatch(r"-?\d+", s_raw):
            entry["rasmussen_s"] = int(s_raw)
        if re.fullmatch(r"-?\d+", tau_raw):
            entry["ozsvath_szabo_tau"] = int(tau_raw)
        nakanishi = lower_endpoint(nakanishi_raw)
        if nakanishi is not None:
            entry["nakanishi_lower"] = nakanishi
            entry["nakanishi_raw"] = nakanishi_raw
        if entry:
            values[name] = entry
    payload = {
        "schema": "rf-knots-lower-bounds-v1",
        "source": {
            "name": "KnotInfo",
            "url": "https://knotinfo.org/knotinfo_data_complete.xls",
            "retrieved_at": "2026-08-01",
            "sha256": digest,
        },
        "theorems": {
            "rasmussen_s": "abs(s(K))/2 <= u(K)",
            "ozsvath_szabo_tau": "abs(tau(K)) <= u(K)",
            "nakanishi_lower": "Nakanishi index <= u(K)",
        },
        "values": values,
    }
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(values)} knots")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("expected path to knotinfo_data_complete.xls")
    main(sys.argv[1])
