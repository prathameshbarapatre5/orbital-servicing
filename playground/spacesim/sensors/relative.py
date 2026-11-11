"""Relative navigation sensing and line-of-sight checks."""
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class SensorConfig:
    pos_noise: float = 0.02   # m, 1-sigma
    vel_noise: float = 0.005  # m/s, 1-sigma


class RelativeSensor:
    def __init__(self, cfg: SensorConfig):
        self.cfg = cfg

    def measure(self, rel_pos, rel_vel, rng):
        rel_pos = np.asarray(rel_pos, dtype=float)
        rel_vel = np.asarray(rel_vel, dtype=float)
        return (rel_pos + rng.normal(0.0, self.cfg.pos_noise, rel_pos.shape),
                rel_vel + rng.normal(0.0, self.cfg.vel_noise, rel_vel.shape))


def _ray(model, data, pnt, vec, bodyexclude):
    """mj_ray wrapper tolerant of the optional `normal` output argument."""
    geomid = np.array([-1], dtype=np.int32)
    pnt = np.asarray(pnt, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    try:
        dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, bodyexclude, geomid)
    except TypeError:
        normal = np.zeros(3, dtype=np.float64)
        dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, bodyexclude, geomid, normal)
    return float(dist), int(geomid[0])


def line_of_sight(model, data, p_from, p_to, from_body, to_body) -> bool:
    """True if nothing other than `to_body` blocks the segment p_from -> p_to."""
    delta = np.asarray(p_to, dtype=float) - np.asarray(p_from, dtype=float)
    length = np.linalg.norm(delta)
    if length < 1e-9:
        return True
    dist, gid = _ray(model, data, p_from, delta / length, from_body)
    if dist < 0 or dist >= length - 1e-6:
        return True
    return int(model.geom_bodyid[gid]) == to_body
