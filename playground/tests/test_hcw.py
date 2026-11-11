import numpy as np

from spacesim.dynamics.hcw import hcw_accel, hcw_propagate, mean_motion


def rk4(state, n, dt, steps):
    def f(s):
        return np.concatenate([s[3:], hcw_accel(s[:3], s[3:], n)])
    s = state.copy()
    for _ in range(steps):
        k1 = f(s); k2 = f(s + dt / 2 * k1); k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return s


def test_mean_motion_leo():
    period_min = 2 * np.pi / mean_motion(400e3) / 60
    assert 92.0 < period_min < 93.0


def test_closed_form_matches_integration():
    n = mean_motion(400e3)
    s0 = np.array([10.0, -5.0, 2.0, 0.01, -0.02, 0.005])
    t = 1800.0
    assert np.allclose(hcw_propagate(s0, n, t), rk4(s0, n, 1.0, 1800), atol=1e-6)


def test_natural_motion_ellipse_is_periodic():
    n = mean_motion(400e3)
    x0 = 5.0
    s0 = np.array([x0, 0.0, 0.0, 0.0, -2 * n * x0, 0.0])  # drift-free condition
    period = 2 * np.pi / n
    assert np.allclose(hcw_propagate(s0, n, period), s0, atol=1e-9)
