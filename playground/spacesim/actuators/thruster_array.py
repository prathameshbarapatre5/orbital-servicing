"""Real thruster layout: allocation, pulse limits and fuel, from a SpacecraftSpec.

Agents command a desired force; the allocator finds throttles u in [0, 1] for
the actual thrusters so that the produced body wrench matches the request
(force, plus any torque for momentum dumping). Whatever the geometry cannot
deliver simply isn't delivered: that is the design's real capability.
"""
import numpy as np

G0 = 9.80665


class ThrusterArray:
    def __init__(self, spec, num_agents, torque_weight=10.0, reg=1e-6, iters=80):
        th = spec.thrusters
        com = np.asarray(spec.com, dtype=float)
        self.pos = np.array([t.pos for t in th], dtype=float) - com
        d = np.array([t.dir for t in th], dtype=float)
        self.dir = d / np.linalg.norm(d, axis=1, keepdims=True)
        self.tmax = np.array([t.max_thrust for t in th], dtype=float)
        self.isp = np.array([t.isp for t in th], dtype=float)
        self.min_on = np.array([t.min_on_time for t in th], dtype=float)
        self.n = len(th)
        self.num_agents = num_agents
        # wrench per unit throttle: [force; torque about CoM]
        self.B = np.vstack([self.dir.T, np.cross(self.pos, self.dir).T]) * self.tmax  # 6 x n
        self.W = np.diag([1, 1, 1, torque_weight, torque_weight, torque_weight])
        self.reg, self.iters = reg, iters
        self.fuel_capacity = spec.propellant_kg
        self.reset()

    def reset(self, health=None):
        self.fuel = np.full(self.num_agents, self.fuel_capacity)
        self.health = np.ones((self.num_agents, self.n)) if health is None else np.asarray(health, float)
        self.last_u = np.zeros((self.num_agents, self.n))

    def allocate(self, wrench, health):
        """Bounded least squares: min ||W(B u - w)||^2 + reg ||u||^2, 0 <= u <= health."""
        Bw = self.W @ (self.B * health)
        tw = self.W @ wrench
        H = Bw.T @ Bw + self.reg * np.eye(self.n)
        g0 = Bw.T @ tw
        step = 1.0 / np.linalg.eigvalsh(H).max()
        u = np.clip(np.linalg.lstsq(Bw, tw, rcond=None)[0], 0, 1) * health
        for _ in range(self.iters):
            u = np.clip(u - step * (H @ u - g0), 0, 1) * health
        return u

    def command(self, i, wrench_body, dt):
        """Throttles for agent i, after minimum on-time and fuel limits.
        Returns (force_body, torque_body, fuel_used, u)."""
        u = self.allocate(np.asarray(wrench_body, float), self.health[i])
        u = np.where(u * dt < self.min_on, 0.0, u)              # pulses below the minimum impulse bit
        mdot = (u * self.tmax / (self.isp * G0)).sum()
        used = mdot * dt
        if used > self.fuel[i] and used > 0:
            u *= self.fuel[i] / used
            used = self.fuel[i]
        self.fuel[i] -= used
        self.last_u[i] = u
        w = self.B @ u
        return w[:3], w[3:], used, u

    @property
    def fuel_fraction(self):
        return self.fuel / max(self.fuel_capacity, 1e-12)
