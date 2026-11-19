"""Privileged scripted inspector: approach a hold point, then circle the client.

Each inspector circles in a different plane, so together they see most faces
as the client tumbles. Use it to check the coverage reward is achievable with
the current fuel, keep-out and range settings before training.
"""
import numpy as np

PLANE_AXES = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]),
              np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0, 1.0]) / np.sqrt(3)]


class ScriptedInspector:
    def __init__(self, env, radius=6.0, period=180.0, gain=0.15, vmax=0.4, kv=0.5):
        self.env = env
        self.radius, self.w = radius, 2 * np.pi / period
        self.gain, self.vmax, self.kv = gain, vmax, kv
        self.name = "scripted_inspector"
        self.reset()

    def reset(self):
        self.plan = None

    def _init_plan(self, pos):
        self.plan = []
        for i, p in enumerate(pos):
            a = PLANE_AXES[i % len(PLANE_AXES)]
            u = p - a * (a @ p)
            if np.linalg.norm(u) < 1e-3:
                u = np.cross(a, [0.3, 0.5, 0.8])
            u /= np.linalg.norm(u)
            self.plan.append({"u": u, "b": np.cross(a, u), "phi": 0.0, "circling": False})

    def act(self, obs):
        env = self.env
        pos, vel = env._servicer_states()
        cpos, cvel = env._free_state(env.ids.client_joint)
        rel = pos - cpos
        if self.plan is None:
            self._init_plan(rel)
        scale = env.cfg.servicer.mass / getattr(env, "cmd_force", env.cfg.thrusters.max_thrust)
        actions = {}
        for i, a in enumerate(env.possible_agents):
            if a not in env.agents:
                continue
            P = self.plan[i]
            if not P["circling"]:
                target, vff = P["u"] * self.radius, np.zeros(3)
                if np.linalg.norm(target - rel[i]) < 0.8:
                    P["circling"] = True
            else:
                P["phi"] += self.w * env.ctrl_dt
                c, s = np.cos(P["phi"]), np.sin(P["phi"])
                target = (P["u"] * c + P["b"] * s) * self.radius
                vff = (-P["u"] * s + P["b"] * c) * self.radius * self.w
            v_des = self.gain * (target - rel[i])
            sp = np.linalg.norm(v_des)
            if sp > self.vmax:
                v_des *= self.vmax / sp
            v_des = v_des + vff + cvel
            actions[a] = np.clip((v_des - vel[i]) * self.kv * scale, -1, 1).astype(np.float32)
        return actions
