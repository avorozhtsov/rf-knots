import unittest
from rf_knots.actions import ActionSpec
from rf_knots.evidence import UnknotWitness
from build_short_horizon_random_roster_v5 import suffix

class SuffixTests(unittest.TestCase):
    def test_too_short(self):
        self.assertIsNone(suffix(UnknotWitness.from_actions((),1,ActionSpec(32,4),[])))
    def test_empty_exact(self):
        w=UnknotWitness.from_actions((),1,ActionSpec(32,4),[])
        self.assertEqual(suffix(w,0).start,w.start)
    def test_empty_proof_replay(self):
        w=UnknotWitness.from_actions((),1,ActionSpec(32,4),[])
        self.assertEqual(suffix(w,0).verify(),w.start)
