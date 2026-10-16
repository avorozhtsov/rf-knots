import copy
import json
import unittest
from train_continuous_hourly_r2_v17 import validate


class GateTests(unittest.TestCase):
    def setUp(self):
        with open('artifacts/continuous-r2-diagnostic-v16/failure.json') as f:
            self.recorded = json.load(f)['metrics']

    def test_recorded_small_finite_gradient(self):
        validate(self.recorded)

    def test_zero_and_nonfinite_rejected(self):
        for x in [0., 1e-13, 1e-12, float('nan'), float('inf')]:
            d = copy.deepcopy(self.recorded)
            d['F_encoder_grad_norm'] = x
            with self.assertRaises(ValueError): validate(d)

    def test_bad_losses_and_ratios_rejected(self):
        for key, value in [('F_loss', -1.), ('F_loss', float('nan')),
                           ('aux_scale', 1.), ('rehearsal_scale', 1.),
                           ('aux_scale', float('nan'))]:
            d = copy.deepcopy(self.recorded)
            d[key] = value
            with self.assertRaises(ValueError): validate(d)


if __name__ == '__main__': unittest.main()
