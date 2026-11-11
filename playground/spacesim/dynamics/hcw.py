"""Hill-Clohessy-Wiltshire (HCW) relative orbital dynamics.

Frame: LVLH centered on the client spacecraft's reference orbit.
  x = radial (away from Earth), y = along-track, z = orbit normal.
MuJoCo's world frame IS this LVLH frame, so HCW terms are applied as
fictitious forces on every free body each physics step.
"""
import numpy as np

MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6378137.0        # m


def mean_motion(altitude_m: float) -> float:
    """Mean motion n [rad/s] of a circular orbit at the given altitude."""
    return float(np.sqrt(MU_EARTH / (R_EARTH + altitude_m) ** 3))


def hcw_accel(pos: np.ndarray, vel: np.ndarray, n: float) -> np.ndarray:
    """HCW acceleration for positions/velocities of shape (..., 3)."""
    x, z = pos[..., 0], pos[..., 2]
    vx, vy = vel[..., 0], vel[..., 1]
    ax = 3.0 * n**2 * x + 2.0 * n * vy
    ay = -2.0 * n * vx
    az = -(n**2) * z
    return np.stack([ax, ay, az], axis=-1)


def hcw_stm(n: float, t: float) -> np.ndarray:
    """6x6 closed-form state transition matrix for state [x,y,z,vx,vy,vz]."""
    c, s, nt = np.cos(n * t), np.sin(n * t), n * t
    phi = np.zeros((6, 6))
    phi[0:3, 0:3] = [[4 - 3 * c, 0, 0],
                     [6 * (s - nt), 1, 0],
                     [0, 0, c]]
    phi[0:3, 3:6] = [[s / n, 2 * (1 - c) / n, 0],
                     [-2 * (1 - c) / n, (4 * s - 3 * nt) / n, 0],
                     [0, 0, s / n]]
    phi[3:6, 0:3] = [[3 * n * s, 0, 0],
                     [-6 * n * (1 - c), 0, 0],
                     [0, 0, -n * s]]
    phi[3:6, 3:6] = [[c, 2 * s, 0],
                     [-2 * s, 4 * c - 3, 0],
                     [0, 0, c]]
    return phi


def hcw_propagate(state: np.ndarray, n: float, t: float) -> np.ndarray:
    """Closed-form far-field propagation (no thrust, no contact).

    Use for the far-field regime of the two-timescale design: large steps
    during long approaches, then hand off to MuJoCo inside a capture radius.
    """
    return hcw_stm(n, t) @ np.asarray(state, dtype=float)
