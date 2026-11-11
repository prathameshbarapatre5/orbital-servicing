"""Translational thrusters with saturation, deadband and a fuel budget.

v0: thrust commanded per LVLH axis (no attitude control yet).
v1: body-frame thruster layout + reaction wheels.
"""
from dataclasses import dataclass

import numpy as np

G0 = 9.80665


@dataclass
class ThrusterConfig:
    max_thrust: float = 2.0   # N per axis
    isp: float = 220.0        # s (monoprop-ish)
    fuel_kg: float = 0.1      # usable propellant per servicer
    deadband: float = 0.05    # |cmd| below this -> no firing (min impulse bit proxy)


class TranslationalThrusters:
    def __init__(self, cfg: ThrusterConfig, num_agents: int):
        self.cfg = cfg
        self.num_agents = num_agents
        self.fuel = np.full(num_agents, cfg.fuel_kg)

    def reset(self):
        self.fuel[:] = self.cfg.fuel_kg

    def compute(self, cmd: np.ndarray, dt: float):
        """cmd: (N, 3) in [-1, 1]. Returns (force (N,3) [N], fuel_used (N,) [kg])."""
        cmd = np.clip(np.asarray(cmd, dtype=float), -1.0, 1.0)
        cmd = np.where(np.abs(cmd) < self.cfg.deadband, 0.0, cmd)
        force = cmd * self.cfg.max_thrust
        # One thruster pair per axis -> propellant scales with L1 norm.
        demand = np.abs(force).sum(axis=1) * dt / (self.cfg.isp * G0)
        scale = np.ones(self.num_agents)
        has_demand = demand > 0
        scale[has_demand] = np.minimum(1.0, self.fuel[has_demand] / demand[has_demand])
        force *= scale[:, None]
        used = demand * scale
        self.fuel = np.maximum(self.fuel - used, 0.0)
        return force, used

    @property
    def fuel_fraction(self) -> np.ndarray:
        return self.fuel / self.cfg.fuel_kg
