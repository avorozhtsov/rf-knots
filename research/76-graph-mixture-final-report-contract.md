# Final report for the graph-mixture experiment

2026-09-21. This is an execution/reporting contract, **not a model result**.
The training and evaluation contracts remain unchanged: [69](69-graph-mixture-training-protocol.md),
[70](70-graph-mixture-prospective-evaluation.md) and
[74](74-pure-braid-short-distance-test-protocol.md).

## What must finish

`scripts/summarize_graph_mixture_experiment.py` requires the complete declared
14-job short-u/search matrix and the separate three-model general-distance
matrix. It refuses a partial or changed matrix. It rechecks output hashes,
audit provenance, final checkpoint identities, identical search roots/resources,
and the supplemental comparison against its raw predictions. It does not start
or restart workers, score intermediate weights, or change model selection.

Once both matrices finish, run from the RF Knots repository:

```sh
.venv/bin/python scripts/summarize_graph_mixture_experiment.py \
  --out artifacts/graph-mixture-summary-v1
```

The destination must not already exist. The summary is not an extra running
controller: the two existing evaluation controllers own their respective queues.

## Scientific distinctions retained

- Short-u originals, their zero-CC lifted representations, and general short
  Gordian pairs stay separate. Group overlap is reported, not counted as
  independent replication. The original empty sidecar TEST remains unavailable.
- Constant0/1/2 controls and equal-query-group errors remain in the report.
  Exact0 coordinate checks, exact1/exact2 and interval labels are not conflated.
  The Markdown overview displays exact2 error and row count for both families;
  absent exact2 short-u evidence is shown as unavailable, never as zero error.
- Search reports solution coverage, newly/lost solved conditions, and paired
  L10/L1000 only on common successes. A censored search has no observed solution
  cost, not zero or an invented penalty. Actual work counts are retained.
- Every successful solution and each reference witness is replayed. The report
  checks root identity, crossing-change and move caps, action encoding resources,
  and both reported costs against that replay.
- A gap to a known reference is different from a gap to a certified lower bound.
  With q0=(word_length+strand_count-1)/2 and a certified u lower bound, the
  semantic lower cost is q0+(lambda+1)*u_lower. Equality certifies an optimum
  for the exact start/action system. Otherwise a lower-bound gap is not called
  an optimality regret. A successful search can itself attain the bound even
  when the reference does not.
- A necessary cost envelope larger than the search cap is a resource limitation,
  not evidence that a better solution exists or was missed.

This is a one-training-seed experiment. It does not establish million-scale RL,
general knot-distribution quality, or successful solving at arbitrary size. The
architecture's absence of a fixed word/strand ceiling is a different claim from
learned quality beyond its training sizes. No automatic promotion is encoded.

## Implementation checks and live snapshot

The focused report/semantic-floor/prior-summary suite passed26 tests. The new
tests cover complete-matrix requirements, replayed resource/cost corruption,
censored costs, reference-versus-solution optimality, and preservation of the
separate metric families/views and uniform-search control.

The broader training/auditor/evaluation/report suite then passed70 tests. Its
only warning was unavailable optional Plink tkinter GUI support; this does not
affect the headless experiment.

At the recorded live snapshot, graph training had saved1920/4096 steps, with
CPU time and checkpoint/log updates advancing. Its paired old-data control was
still queued behind it. The fresh short-u builder had processed8100/10000
attempts and retained64 groups; it must still finish all prescribed attempts.
The core evaluation controller was waiting for those gates, and the supplemental
controller was waiting for the final training-pair audit. No final model score
for this experiment was available. Existing training, sources pinned by live
runs, and all unrelated dirty/generated work were preserved.

## Completed model-blind reference cost audit

After both short-u data audits passed, the final-report replay routine was run
on the completed panel and the frozen70 roots, with no model predictions or
search solutions. It rechecked the full data certificate's inputs and replayed
all64 reference witnesses. The panel SHA was
`3432122621600c892070970d469f52a55c9ca59d0a0d42fa6f539786f37c4f84`.

- All64 references have zero semantic excess: moves=q0+CC. Their q0 values
  range34..217, reference moves35..219 and declared move caps168..900.
- For40 roots (32 exact1 and8 exact2), the reference reaches the certified
  lower cost for **both** lambda10 and lambda1000. Here matching the reference
  is already optimal in this action system; beating it would contradict a
  checked bound or indicate a bug.
- The remaining24 roots have certified u in[1,2] and a two-CC witness. They
  do not have a proof that the reference cost is minimal.
- The necessary move envelope for any solution costing at most the reference
  fits the current move cap for all64 lambda10 conditions. For lambda1000 it
  exceeds the current cap on exactly the24 interval-labelled roots.

The last item does not prove that a better solution exists. It means the finite
lambda1000 search cannot rule out a longer, lower-CC route solely from a failure
at the prescribed cap. Conversely, a sufficiently large move cap is not a
guarantee of discovery: simulations and action proposals remain finite too.
No caps or sampling rules were changed after seeing this reference-only result.

Reproduce from the repository with the current working-tree routine
(no final model or final search report is needed for this reference-only call):

```sh
PYTHONPATH=scripts .venv/bin/python - <<'PY'
import json
from pathlib import Path
from audit_joint_policy_metric_panel import check_pins
from evaluate_graph_mixture_cost_mcts import PANEL, build_roots
from summarize_graph_mixture_experiment import certify_costs

audit = json.loads((PANEL.parent / "independence-audit.json").read_text())
assert audit["status"] == "passed" and audit["evaluation_ready"] is True
check_pins(audit["inputs"])
panel = json.loads(PANEL.read_text())
result = certify_costs(panel, build_roots(panel), {})
print(json.dumps(result["reference_summary"], indent=2))
PY
```

The combined summary command above still requires the complete model/search
matrix. This reference-only check does not bypass that gate or constitute model
accuracy or search-performance evidence.
