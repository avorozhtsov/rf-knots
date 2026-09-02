"""The invariant a certified *lower* bound has to satisfy: it must not be wrong.

An overshooting lower bound is the dangerous direction. `rf_knots.unknot_search`
prunes on `max(abs(sigma)/2, abs(tau), Montesinos)` [research/22 section 1.3], so
a bound that is too large deletes a real witness and reports nothing rather than
reporting something false -- the failure mode that is invisible from the outside.

The defect that motivated this file: `seifert.branched_cover_homology` returned
`[1, 87]` for a knot whose `H_1(Sigma_2)` is the cyclic group `Z/87`, and
`computed_claims` read the length-2 list as "non-cyclic" and certified `u >= 2`.
Two independent checks below, one that needs nothing but the repository and one
that needs the unknotdb proof snapshot.
"""

from __future__ import annotations

import json
import math
import random
import signal
from importlib.resources import files
from pathlib import Path

import pytest

from rf_knots import seifert
from rf_knots.lower_bounds import computed_claims, strongest

pytest.importorskip("spherogram", reason="certified bounds need the 'bounds' extra")

UNKNOTDB = Path("/Users/artemvorozhtsov/projects/unknotdb")
PROOF = UNKNOTDB / "outputs/unknotdb-dgkt-witness-coverage-v1-final.sqlite"
PAIRS = UNKNOTDB / "outputs/unknotdb-embedding-pairs-v4.sqlite"


def _is_squarefree(number: int) -> bool:
    factor = 2
    while factor * factor <= number:
        if number % (factor * factor) == 0:
            return False
        while number % factor == 0:
            number //= factor
        factor += 1
    return True


def test_the_montesinos_claim_never_fires_on_a_squarefree_determinant():
    """`abs(H_1(Sigma_2)) = det(K)`, and a squarefree order forces a cyclic group.

    So the determinant is an independent certificate that the Montesinos bound
    must stay silent, computed by a different routine (Bareiss elimination) than
    the invariant factors. On the bundled table this covers 2207 of 2870 knots,
    and the broken chain violated it on 58 of them -- this is the check that
    would have caught the defect without any external snapshot.
    """
    table = json.loads(
        files("rf_knots").joinpath("data/knot_table.json").read_text()
    )["knots"]
    checked = 0
    squarefree = 0
    violations = []
    for name, row in table.items():
        word, strands = tuple(row["braid"]), row["strands"]
        try:
            factors = seifert.branched_cover_homology(word, strands)
            order = seifert.determinant(word, strands)
        except (ValueError, seifert.BackendUnavailable):
            continue
        checked += 1
        # the Smith form itself: a divisibility chain, no units, right order
        assert 1 not in factors, (name, factors)
        assert math.prod(factors) == order, (name, factors, order)
        for first, second in zip(factors, factors[1:], strict=False):
            assert second % first == 0, (name, factors)
        if _is_squarefree(order):
            squarefree += 1
            if len(factors) > 1:
                violations.append((name, order, factors))
    assert checked > 2500
    assert squarefree > 2000
    assert violations == []


def _evaluate(representation, seconds: float = 2.0):
    """The claim for one node, or `None` when the exact elimination is too slow.

    A handful of the wide representations (21 strands, 254 letters) blow the
    intermediate integers up far enough that the elimination takes minutes. They
    are skipped rather than waited on; the exhaustive pass over the whole
    falsification set lives in `artifacts/embedding-landmark/check_montesinos.py`.
    """

    class Timeout(Exception):
        pass

    def fire(_signal, _frame):
        raise Timeout

    previous = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return strongest(computed_claims(representation.word, representation.strands))
    except Timeout:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.skipif(
    not (PROOF.exists() and PAIRS.exists()),
    reason="needs the unknotdb proof snapshot for the replayed upper bounds",
)
def test_no_computed_lower_bound_exceeds_a_replayed_upper_bound():
    """`lower <= u <= upper`: a certified bound above a *replayed* `u_upper` is a
    contradiction, and the contradiction is always the lower bound's fault.

    The sample is weighted onto `u_upper <= 1`, where the Montesinos claim is the
    only bound that can overshoot and where a false `u >= 2` costs a witness. It
    is a sample and not the whole graph because `computed_claims` runs `tau`, at
    roughly 90 nodes a second; the exhaustive pass over all 9,716 nodes with a
    replayed `u_upper <= 1` is `artifacts/embedding-landmark/check_montesinos.py`,
    which is what pins the count at zero rather than at "none in the sample".
    """
    from rf_knots.embedding_route_retrieval import load_graph_representations

    representations = load_graph_representations(PROOF, PAIRS)
    unknotting_one = [rep for rep in representations if rep.u_upper <= 1]
    assert len(unknotting_one) > 9000

    rng = random.Random(20260901)
    sample = rng.sample(unknotting_one, 2500) + rng.sample(representations, 500)
    checked = 0
    contradictions = []
    for representation in sample:
        value = _evaluate(representation)
        if value is None:
            continue
        checked += 1
        if value > representation.u_upper:
            contradictions.append(
                (representation.node_id, representation.knot_id, value,
                 representation.u_upper)
            )
    assert checked > len(sample) * 0.95
    assert contradictions == []
