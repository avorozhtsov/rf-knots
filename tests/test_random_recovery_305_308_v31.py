import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import recover_random_window_305_308_v31 as r


class RecoveryTests(unittest.TestCase):
    def test_exact_missing_nine(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); d = root/'batch-000305'; d.mkdir()
            for n in r.condition_names()[:7]: (d/n).touch()
            with patch.object(r, 'ROOT', root), patch.object(r, 'WINDOW', root/'window'):
                r.preflight_layout()
                self.assertEqual(r.missing_conditions(d), r.condition_names()[7:])
                (root/'batch-000306').mkdir()
                with self.assertRaises(AssertionError): r.preflight_layout()

    def test_unknown_never_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td); (d/'search-4-10-20261023461.json').touch()
            with self.assertRaises(AssertionError): r.missing_conditions(d)

    def test_only_supply_flag_may_change_after_pass(self):
        s = dict(status='STOPPED_LEARNER', exit_code=0, stop_reason='storage_gate',
            supply_stopped=True, main_intake_stopped=False, round=20, spec='same', counters=[14,15])
        with self.assertRaises(AssertionError): r.repaired_state(s, False)
        fixed = r.repaired_state(s, True)
        self.assertEqual({**fixed, 'supply_stopped': True}, s)
        self.assertTrue(s['supply_stopped'])
        with self.assertRaises(AssertionError): r.repaired_state({**s,'exit_code':137}, True)

    def test_original_gate_unchanged(self):
        self.assertFalse(r.p.window_gate(7, 1))
        self.assertTrue(r.p.window_gate(8, 2400))
        self.assertFalse(r.p.window_gate(8, 2400.1))


if __name__ == '__main__': unittest.main()
