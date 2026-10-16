import fcntl
import importlib
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("build_long_mixture_short_u_test")


@lru_cache(maxsize=1)
def real_record(module):
    for index in range(100):
        witness = module.conjugate_witness(
            module.CONFIG["seed"],
            index,
            **{k: module.CONFIG[k] for k in ("max_strands", "min_depth", "max_depth")},
        )
        fp = module.invariants(witness.start.word, witness.start.strands)
        if fp[0] == 1 or module.owner(fp) != "test":
            continue
        return dict(
            split="test",
            seed=module.CONFIG["seed"],
            index=index,
            construction="conjugate blocks in a destabilizable positive chain",
            witness=witness.to_dict(),
            pair_prefix_length=witness.moves,
            left_fingerprint=list(fp),
            right_fingerprint=[1, 0],
            interval=dict(
                metric="cc",
                left=witness.start.to_dict(),
                right=witness.steps[-1].after.to_dict(),
                lower=abs(fp[1]) // 2,
                upper=witness.crossing_changes,
                witness_semantic_length=witness.moves,
            ),
        )
    raise AssertionError("fixed test fixture did not find a nontrivial TEST witness")


def test_registry_covers_intermediates_and_both_new_heldout_queries(module):
    corpus = dict(
        excluded_query_groups=[[13, 2]],
        training_groups=dict(old=[[3, 2]]),
        records=dict(
            train=[dict(fingerprints=[[5, 0], [7, -2], [9, 0]])],
            validation=[dict(left_fingerprint=[11, -2])],
            test=[dict(left_fingerprint=[15, 4])],
        ),
    )
    result = module.registry_groups(corpus)
    assert result == {(3, 2), (5, 0), (7, 2), (9, 0), (11, 2), (13, 2), (15, 4)}
    registry = module.registry(result)
    assert len(registry["validation"]) == 7 and registry["size_shift"] == []
    assert registry["schema"] == "query-group-exclusion-registry-not-labelled-examples"


def test_real_new_lift_and_separate_consumer(module):
    row = real_record(module)
    enlarged = module.lifted_record(row)
    result = module.audit_records(dict(validation=[row], size_shift=[enlarged]), set())
    assert result["selected_roots"] == 1 and not result["evaluation_ready"]
    assert result["endpoint_lengths"][0][1] >= 385
    assert result["endpoint_strands"][0][1] >= 34
    assert (
        module.verify_record(enlarged).crossing_changes
        == module.verify_record(row).crossing_changes
    )


@pytest.mark.parametrize("mutation", ["excluded", "duplicate", "coverage", "seed", "lift"])
def test_reject_changed_panel_evidence(module, mutation):
    row = deepcopy(real_record(module))
    enlarged = module.lifted_record(row)
    panel = dict(validation=[row], size_shift=[enlarged])
    excluded = set()
    if mutation == "excluded":
        excluded.add(module.group(row["left_fingerprint"]))
    elif mutation == "duplicate":
        panel["validation"].append(row)
        panel["size_shift"].append(enlarged)
    elif mutation == "coverage":
        panel["size_shift"] = []
    elif mutation == "seed":
        row["seed"] += 1
    else:
        # An original or legacy256/33 view cannot masquerade as the new lift.
        panel["size_shift"] = [row]
    with pytest.raises(ValueError):
        module.audit_records(panel, excluded)


def test_live_generator_lock_refuses_duplicate(module, tmp_path):
    directory = tmp_path / "generator"
    directory.mkdir()
    with (directory / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="still active"):
            module.generator_unlocked(directory)
    module.generator_unlocked(directory)


def test_missing_parent_does_not_create_apparent_run(module, monkeypatch, tmp_path):
    source, output = tmp_path / "missing-source", tmp_path / "not-started"
    monkeypatch.setattr("sys.argv", ["builder", "--long-data", str(source), "--out", str(output)])
    with pytest.raises(FileNotFoundError):
        module.main()
    assert not output.exists()


def test_seed_and_lift_do_not_mutate_old_protocol(module):
    old = importlib.import_module("build_resource_short_u_test")
    assert module.CONFIG["seed"] == 20261078 and old.CONFIG["seed"] == 20261072
    assert module.LIFT == dict(length=385, strands=34)
