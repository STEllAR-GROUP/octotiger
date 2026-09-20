"""Independent discrete reference for the uniform mean of an absorbed beam.

Periodic transport conserves the spatial mean. With theta=1, constant absorption
and velocity terms off, backward Euler therefore gives exactly
    mean(E)_n = mean(E)_0 * product_j (1 + c_hat*chi_a*dt_j)^(-1).
Its global error is O(dt), even when source-free transport is second order.
This supplements, and does not replace, the continuum exponential reference.
"""
import numpy as np


def discrete_mean(history, parameters):
    rate = parameters['c_cm_s']*parameters['reduced_light_speed_ratio']*parameters['chi_cm_inverse']
    dt = np.asarray(history['rad_dt'][1:])
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0) or rate < 0:
        raise ValueError('Backward-Euler reference requires finite positive steps and nonnegative absorption')
    if parameters['radiation_options']['source_theta'] != 1 or parameters['radiation_options']['velocity_terms']:
        raise ValueError('This discrete reference applies only to backward Euler with velocity terms off')
    return float(history['mean_E'][0])*np.r_[1., np.cumprod(1/(1+rate*dt))]


def damping_checks(history, parameters):
    expected = discrete_mean(history, parameters)
    scale = max(abs(float(history['mean_E'][0])), 1.)
    return {'backward_euler_mean': bool(np.all(np.isfinite(history['mean_E'])) and
                                        np.max(abs(history['mean_E']-expected)) <= 5e-12*scale)}
