import importlib.util
from pathlib import Path

import pytest

from rf_knots.actions import ActionSpec
from rf_knots.lazy_actions import LazyActionCatalogue


@pytest.mark.parametrize("word,n",[((),1),((1,1,1),2),((1,2,-1,-2),3)])
@pytest.mark.parametrize("crossing",[False,True])
def test_inverse_rank_matches_every_catalogue_action(word,n,crossing):
    p=Path(__file__).parents[1]/"scripts/audit_witness_proposal_coverage.py"
    s=importlib.util.spec_from_file_location("coverage_audit",p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    for slack in (0,4):
        catalogue=LazyActionCatalogue(ActionSpec(max(1,len(word)+slack),n+1),word,n,crossing)
        for rank in range(len(catalogue)):
            assert m.catalogue_rank(catalogue,catalogue.action_at(rank))==rank
