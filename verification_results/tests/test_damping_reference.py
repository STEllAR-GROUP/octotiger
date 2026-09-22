import unittest
import numpy as np
from verification_results import runner
from verification_results.radiation.damping_reference import discreteMean, dampingChecks


class DampingReferenceTests(unittest.TestCase):
    def history(self, n, method='BE'):
        p = runner.descriptors()['radiation.skinner_ostriker.damped_wave'][1]['parameters']
        rate = p['c_cm_s']*p['chi_cm_inverse']; dt = p['final_time_s']/n
        factor = 1/(1+rate*dt) if method == 'BE' else np.exp(-rate*dt)
        h = np.zeros(n+1, dtype=[('rad_dt', float), ('mean_E', float)])
        h['rad_dt'][1:] = dt; h['mean_E'] = factor**np.arange(n+1)
        return h, p

    def test_backward_euler_is_first_order_without_altering_reference(self):
        errors = []
        for n in (32, 64, 128):
            h, p = self.history(n)
            self.assertTrue(dampingChecks(h, p)['backward_euler_mean'])
            errors.append(abs(h['mean_E'][-1]-np.exp(-.1)))
        orders = np.log2(np.array(errors[:-1])/errors[1:])
        self.assertTrue(np.all(abs(orders-1) < .01), orders)

    def test_wrong_source_or_step_history_is_rejected(self):
        h, p = self.history(64, method='exponential')
        self.assertFalse(dampingChecks(h, p)['backward_euler_mean'])
        h, p = self.history(64); h['rad_dt'][12] *= 1.01
        self.assertFalse(dampingChecks(h, p)['backward_euler_mean'])

    def test_variable_steps_and_input_validation(self):
        h, p = self.history(4); h['rad_dt'][1:] *= [.5, 1.5, .75, 1.25]
        h['mean_E'] = discreteMean(h, p)
        self.assertTrue(dampingChecks(h, p)['backward_euler_mean'])
        h['rad_dt'][2] = 0
        with self.assertRaises(ValueError): discreteMean(h, p)


if __name__ == '__main__': unittest.main()
