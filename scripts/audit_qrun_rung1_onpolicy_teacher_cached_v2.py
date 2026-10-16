"""Run the independent v1 teacher audit with bounded pure-invariant caching.

This changes no teacher data or audit rule. It only avoids recomputing the
same knot invariants for repeated (word, strands) states across episodes.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import audit_qrun_rung1_onpolicy_teacher_v1 as audit


audit.invariants = lru_cache(maxsize=20_000)(audit.invariants)


def chunked_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunked-out", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()
    if args.chunk_size <= 0 or args.chunk_size % 2:
        raise ValueError("chunk size must be a positive even number")
    manifest_sha = audit.sha(audit.ROOT / "manifest.json")
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.chunked_out.mkdir(parents=True, exist_ok=True)
    totals = defaultdict(Counter)
    for start in range(0, 2000, args.chunk_size):
        limit = min(args.chunk_size, 2000 - start)
        path = args.chunked_out / f"chunk-{start:04d}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if (record["audit_source_sha256"] != source_sha
                    or record["manifest_sha256"] != manifest_sha
                    or record["start_shard"] != start
                    or record["audited_shards"] != limit):
                raise ValueError(f"saved audit chunk differs: {path}")
        else:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                sys.argv = [sys.argv[0], "--start", str(start), "--limit", str(limit)]
                audit.main()
            record = json.loads(output.getvalue())
            if (record["manifest_sha256"] != manifest_sha
                    or record["start_shard"] != start
                    or record["audited_shards"] != limit):
                raise ValueError(f"audit chunk differs: {start}")
            record["audit_source_sha256"] = source_sha
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
            os.replace(temporary, path)
        for key, counts in record["by_stage_lambda"].items():
            totals[key].update(counts)
        print(f"audited {start + limit}/2000", flush=True)
    report = {"audited_shards": 2000,
              "audit_source_sha256": source_sha,
              "manifest_sha256": manifest_sha,
              "by_stage_lambda": {key: dict(value) for key, value in sorted(totals.items())}}
    path = args.chunked_out / "report.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, sort_keys=True) + "\n")
    os.replace(temporary, path)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    if "--chunked-out" in sys.argv:
        chunked_main()
    else:
        audit.main()
