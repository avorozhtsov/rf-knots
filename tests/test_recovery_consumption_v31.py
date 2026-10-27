import unittest
from check_recovery_consumption_v31 import consumed_ids


class ConsumptionTests(unittest.TestCase):
    def test_ready_and_old_draws_are_not_new_consumption(self):
        self.assertEqual(consumed_ids(['new'], ['old'], ['old']), set())
        self.assertEqual(consumed_ids(['new'], ['old','new'], ['old']), set())
        self.assertEqual(consumed_ids(['new'], ['old','new'], ['old','new']), {'new'})
