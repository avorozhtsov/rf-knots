import json
from pathlib import Path
import horizon48_feasibility_v3 as new
import continuous_random_stream_v2 as old

def test_distribution_is_distinct_and_old_unchanged():
    assert new.HORIZON == 48 and old.HORIZON == 24

def test_all_prior_failed_rosters_excluded():
    priors = new.prior_rosters()
    assert len(priors) == 57
    assert any('batch-000052' in str(p) for p in priors)

def test_frozen_bounded_resources():
    p = json.loads(new.PROTOCOL.read_text())
    assert p['search']['conditions'] == 32
    assert p['search']['total_CPU_upper_bound'] == 640
    assert p['supply']['max_CPU_seconds'] == 30
