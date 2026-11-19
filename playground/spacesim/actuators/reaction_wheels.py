"""Reaction wheels: torque and momentum limits, with momentum bookkeeping."""
import numpy as np


class ReactionWheels:
    def __init__(self, spec, num_agents):
        w = spec.wheels
        self.A = np.array([x.axis for x in w], dtype=float).T            # 3 x k
        self.A /= np.linalg.norm(self.A, axis=0, keepdims=True)
        self.Ainv = np.linalg.pinv(self.A)
        self.tmax = np.array([x.max_torque for x in w], dtype=float)
        self.hmax = np.array([x.max_momentum for x in w], dtype=float)
        self.num_agents = num_agents
        self.reset()

    def reset(self):
        self.h = np.zeros((self.num_agents, len(self.tmax)))           # wheel momentum, per wheel

    def apply(self, i, torque_body_cmd, dt):
        """Torque delivered to the bus (body frame). Wheels spin the opposite way."""
        tw = np.clip(self.Ainv @ torque_body_cmd, -self.tmax, self.tmax)
        # the bus gets +tw, the wheels absorb -tw; respect momentum limits
        h_new = np.clip(self.h[i] - tw * dt, -self.hmax, self.hmax)
        tw = (self.h[i] - h_new) / dt
        self.h[i] = h_new
        return self.A @ tw

    def momentum_body(self, i):
        return self.A @ self.h[i]

    def saturation(self, i):
        return np.abs(self.h[i]) / self.hmax
