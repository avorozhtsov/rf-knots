import unittest
from hourly_checkpoint_policy_v1 import CheckpointPolicy


class PolicyTests(unittest.TestCase):
    def test_hourly_not_step_based(self):
        p = CheckpointPolicy(10)
        self.assertFalse(p.due(3609, 20000))
        self.assertTrue(p.due(3610, 1))
        p.committed(3610)
        self.assertFalse(p.due(3611, 1024))
        self.assertTrue(p.due(7210, 2))

    def test_graceful_stop_and_no_duplicate(self):
        p = CheckpointPolicy(0)
        self.assertTrue(p.due(20, 4, stopping=True))
        self.assertFalse(p.due(4000, 0, stopping=True))

    def test_bad_interval(self):
        for value in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                CheckpointPolicy(0, value)


if __name__ == '__main__':
    unittest.main()
