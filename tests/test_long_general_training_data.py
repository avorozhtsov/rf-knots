import importlib
import json
import random
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("long_general_training_data")


def fixture_records(module):
    """Structural sampler fixtures, not certified mathematical claims."""
    records = []
    for band, length in enumerate((60, 120, 220)):
        for c in range(3):
            word = [1] * (length - 5) + [2] * (c + 1) + [-1] * (4 - c)
            if c:
                states = [dict(word=word, strands=3)]
                for pos in range(c):
                    after = states[-1]["word"].copy()
                    after[pos] *= -1
                    states.append(dict(word=after, strands=3))
            else:
                states = [dict(word=word, strands=3), dict(word=word[1:] + word[:1], strands=3)]
            fps = [[3, 2] for _ in states]
            row = dict(
                index=band * 3 + c,
                source=dict(length_band=band),
                split="train",
                states=states,
                fingerprints=fps,
                left_fingerprint=fps[0],
                right_fingerprint=fps[-1],
                lower=c,
                upper=c,
                cc_positions=list(range(c)),
                interval=dict(
                    metric="cc",
                    left=states[0],
                    right=states[-1],
                    lower=c,
                    upper=c,
                    witness_semantic_length=c,
                ),
            )
            row["pair_key"] = module.pair_key(row)
            records.append(row)
    return records


def policy_fixtures():
    instance = M.Instance("fixture", (1, 1, 1), 2, 1)
    state = M.SearchState(instance.word, 2, 0, 0)
    request = (instance, state, (instance.spec.encode(CROSSING_CHANGE, 0),))
    positives = [[(request, (0,))]]
    groups = [[(request, 0, (-1.0) ** i) for i in range(n)] for n in (3, 19)]
    return positives, [(g, torch.full((len(g),), -1.0)) for g in groups]


def old_sampler(module):
    class Sampler:
        def sample(self, rng, count):
            return [
                module.pair_row(
                    rng.randrange(100000), "old-fixture", ((1, 1, 1), 2), ((-1, -1, -1), 2), 2, 2, 2
                )
                for _ in range(count)
            ]

    return Sampler()


def test_materialization_and_fixed_band_probabilities(module):
    records = fixture_records(module)
    sampler = module.LongGeneralSampler(records, set())
    assert list(sampler.support) == ["0", "1", "2"]
    for classes in sampler.support.values():
        assert set(classes) == {"exact-0", "exact-1", "exact-2"}
        assert all(v["probability"] == pytest.approx(1 / 9) for v in classes.values())
    for record in records:
        row = module.materialize(record)
        assert row.spec.distance_lower == record["lower"]
        assert row.left_word == tuple(record["states"][0]["word"])


@pytest.mark.parametrize(
    "mutation",
    ["split", "middle_owner", "excluded", "duplicate", "band", "size", "bound", "missing_class"],
)
def test_reject_invalid_train_sources(module, mutation):
    records = deepcopy(fixture_records(module))
    excluded = set()
    if mutation == "split":
        records[0]["split"] = "test"
    elif mutation == "middle_owner":
        records[2]["fingerprints"][1] = [49, 0]
    elif mutation == "excluded":
        excluded = {(3, 2)}
    elif mutation == "duplicate":
        records.append(records[0])
    elif mutation == "band":
        records[0]["source"]["length_band"] = 3
    elif mutation == "size":
        records[0]["source"]["length_band"] = 2
    elif mutation == "bound":
        records[0]["interval"]["upper"] = 1
    else:
        records.pop()
    with pytest.raises(ValueError):
        module.LongGeneralSampler(records, excluded)


def test_sampler_consumes_exactly_five_draws_and_swaps_endpoints(module):
    sampler = module.LongGeneralSampler(fixture_records(module), set())

    class Constant:
        def __init__(self, swap):
            self.calls, self.swap = 0, swap

        def random(self):
            self.calls += 1
            return self.swap if self.calls % 5 == 0 else 0.0

    left, right = Constant(0.9), Constant(0.1)
    a, b = sampler.sample(left, 3), sampler.sample(right, 3)
    assert left.calls == right.calls == 15
    for x, y in zip(a, b, strict=True):
        assert x.left_word == y.right_word and x.right_word == y.left_word
        assert x.spec == y.spec
    assert sampler.sample(left, 0) == [] and left.calls == 15
    for n in (-1, 1.0, True):
        with pytest.raises(ValueError):
            sampler.sample(left, n)


def test_paired_draw_changes_only_eight_graph_slots_and_preserves_rng(module):
    sampler = old_sampler(module)
    long = module.LongGeneralSampler(fixture_records(module), set())
    positives, refs = policy_fixtures()
    a, b = random.Random(100), random.Random(100)
    la, lb = random.Random(101), random.Random(101)
    for _ in range(64):
        x = module.paired_draw("control", a, la, sampler, sampler, sampler, long, positives, refs)
        y = module.paired_draw("long", b, lb, sampler, sampler, sampler, long, positives, refs)
        assert len(x.metrics) == len(y.metrics) == 32
        assert x.metrics[:23] == y.metrics[:23] and x.metrics[31:] == y.metrics[31:]
        assert all(len(row.left_word) > 48 for row in y.metrics[23:31])
        assert all(len(row.left_word) == 3 for row in x.metrics[23:31])
        assert x.positives == y.positives and x.relative is y.relative
        assert x.group_index == y.group_index
        assert x.shared_digest == y.shared_digest and x.selected_digest != y.selected_digest
        assert a.getstate() == b.getstate() and la.getstate() == lb.getstate()

        def weights(rows):
            return sum(
                module.witness_priority_weight(
                    r.metric, r.spec.distance_upper, r.witness_semantic_length
                )
                for r in rows
            )

        assert weights(x.metrics) == weights(y.metrics)


def test_both_rng_states_resume_exactly(module):
    sampler = old_sampler(module)
    long = module.LongGeneralSampler(fixture_records(module), set())
    positives, refs = policy_fixtures()
    rng, lrng = random.Random(42), random.Random(43)
    args = (sampler, sampler, sampler, long, positives, refs)
    for _ in range(7):
        module.paired_draw("long", rng, lrng, *args)
    a, b = random.Random(), random.Random()
    a.setstate(rng.getstate())
    b.setstate(lrng.getstate())
    x, y = module.paired_draw("long", rng, lrng, *args), module.paired_draw("long", a, b, *args)
    assert x.metrics == y.metrics and x.positives == y.positives
    assert (x.shared_digest, x.selected_digest, x.group_index) == (
        y.shared_digest,
        y.selected_digest,
        y.group_index,
    )


def test_missing_or_unready_audit_never_loads(module, tmp_path):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"schema": module.SCHEMA}))
    with pytest.raises(FileNotFoundError):
        module.load_long_source(tmp_path)
    audit = tmp_path / "consumer-audit.json"
    for payload in (
        dict(inputs={}, status="failed"),
        dict(inputs={}, status="passed", training_ready=False),
        dict(inputs={}, status="passed", training_ready=True, minimum_support_ready=False),
    ):
        audit.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="completed ready"):
            module.load_long_source(tmp_path)


def test_ready_loader_materializes_train_not_test_and_binds_consumer(module, tmp_path, monkeypatch):
    # Small structural fixture tests the successful loader path. Its mock
    # consumer is not a certificate or authority for a real optimizer run.
    monkeypatch.setitem(module.DATA_CONFIG, "min_pairs", dict(train=1, validation=0, test=0))
    monkeypatch.setitem(module.DATA_CONFIG, "min_groups", dict(train=1, validation=0, test=0))
    records = dict(train=fixture_records(module), validation=[], test=[])
    summary, gaps = module.supply_summary(records)
    assert not gaps
    corpus = dict(
        schema=module.SCHEMA,
        contract=dict(config=module.DATA_CONFIG, inputs={}),
        construction_complete=True,
        minimum_support_ready=True,
        optimizer_ready=False,
        records=records,
        summary=summary,
        support_gaps=[],
        excluded_query_groups=[],
        training_groups=dict(old=[[3, 2]]),
    )
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(corpus))
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps(dict(status="completed")))
    required = [
        path,
        progress,
        Path("scripts/build_long_general_cc_supply.py").resolve(),
        Path("scripts/audit_long_general_cc_supply.py").resolve(),
    ]
    audit = dict(
        inputs={str(p): module.sha(p) for p in required},
        status="passed",
        training_ready=True,
        minimum_support_ready=True,
        corpus=str(path),
        corpus_sha256=module.sha(path),
        attempts=module.DATA_CONFIG["attempts"],
        selected_certificates_checked=9,
        summary=summary,
        support_gaps=[],
    )
    audit_path = tmp_path / "consumer-audit.json"
    audit_path.write_text(json.dumps(audit))
    sampler, dev, pins = module.load_long_source(tmp_path)
    assert len(sampler.bins) == 3 and dev == ([], [])
    assert pins[str(audit_path)] == module.sha(audit_path)
    audit["corpus_sha256"] = "changed"
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="hash-bound"):
        module.load_long_source(tmp_path)


def test_draw_unknown_arm_and_reweighting_rejected(module):
    sampler = old_sampler(module)
    long = module.LongGeneralSampler(fixture_records(module), set())
    positives, refs = policy_fixtures()
    args = (random.Random(4), random.Random(5), sampler, sampler, sampler, long, positives, refs)
    with pytest.raises(ValueError, match="unknown"):
        module.paired_draw("other", *args)

    class BadSource:
        def sample(self, rng, count):
            return [
                module.pair_row(i, "wrong-weight", ((1, 1, 1), 2), ((-1, -1, -1), 2), 1, 1, 30)
                for i in range(count)
            ]

    with pytest.raises(ValueError, match="normalization"):
        module.paired_draw(
            "long",
            random.Random(1),
            random.Random(2),
            sampler,
            sampler,
            sampler,
            BadSource(),
            positives,
            refs,
        )


def test_configuration_preserves_base(module):
    base = importlib.import_module("raster_log_training")
    assert module.CONFIG["seed"] == 20261076 and base.CONFIG["seed"] == 20261047
    assert module.CONFIG["steps"] == base.CONFIG["steps"] == 2048
    assert all(
        module.CONFIG[k] == base.CONFIG[k]
        for k in (
            "lr",
            "weight_decay",
            "grad_clip",
            "positive_weight",
            "relative_weight",
            "relative_every",
        )
    )
    assert asdict(module.materialize(fixture_records(module)[0]))["metric"] == "cc"
