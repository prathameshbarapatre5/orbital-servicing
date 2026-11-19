"""Attitude control: keep the sensor boresight (+X) on the client.

This is the vehicle's own ADCS, not the RL policy. It computes a pointing
target, a PD torque from the attitude error, and a momentum-dump torque request
for the thrusters when the wheels approach saturation.
"""
import math

import numpy as np


def _cross(a, b):
    """3-vector cross product without np.cross overhead (hot path, called every ADCS update)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def pointing_frame(pos, target, up=(0.0, 0.0, 1.0)):
    """Rotation (columns = body axes in world): +X toward target, +Z toward `up`."""
    x = np.asarray(target, float) - np.asarray(pos, float)
    x = x / (math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) + 1e-12)
    ux, uy, uz = up
    if abs(ux * x[0] + uy * x[1] + uz * x[2]) > 0.95:
        ux, uy, uz = (1.0, 0.0, 0.0) if abs(x[0]) < 0.9 else (0.0, 1.0, 0.0)
    k = ux * x[0] + uy * x[1] + uz * x[2]
    z = np.array([ux - k * x[0], uy - k * x[1], uz - k * x[2]])
    z = z / math.sqrt(z[0] * z[0] + z[1] * z[1] + z[2] * z[2])
    R = np.empty((3, 3))
    R[:, 0], R[:, 1], R[:, 2] = x, _cross(z, x), z
    return R


def attitude_error(R, R_des):
    """Small-angle error vector (body frame) and total angle in rad."""
    E = R_des.T @ R
    e = 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    c = (E[0, 0] + E[1, 1] + E[2, 2] - 1.0) / 2.0
    angle = math.acos(min(1.0, max(-1.0, c)))
    return e, angle


class ADCS:
    def __init__(self, spec):
        I = np.asarray(spec.inertia, float)
        wn, z = spec.adcs.bandwidth, spec.adcs.damping
        self.kp, self.kd = I * wn ** 2, 2 * z * wn * I
        self.dump_threshold, self.dump_gain = spec.adcs.dump_threshold, spec.adcs.dump_gain

    def torque(self, R, omega_body, R_des, omega_des_world=None):
        """PD on attitude error, with line-of-sight rate feed-forward when tracking a moving target."""
        e, angle = attitude_error(R, R_des)
        w_des = R.T @ omega_des_world if omega_des_world is not None else np.zeros(3)
        return -self.kp * e - self.kd * (omega_body - w_des), angle


    def dump_request(self, wheel_momentum_body, saturation):
        """External torque that unloads the wheels, only once they are fairly full."""
        if np.max(saturation) < self.dump_threshold:
            return np.zeros(3)
        return -self.dump_gain * wheel_momentum_body


def los_rate(pos, vel, target_pos, target_vel):
    """Angular velocity (world) of the line of sight from pos to the target."""
    d = np.asarray(target_pos, float) - np.asarray(pos, float)
    dv = np.asarray(target_vel, float) - np.asarray(vel, float)
    return _cross(d, dv) / max(float(d @ d), 1e-9)
