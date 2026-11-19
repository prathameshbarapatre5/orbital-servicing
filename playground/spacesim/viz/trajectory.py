"""Trajectory export: replay any run (scripted or trained policy) in the web demo
or in Blender. Positions are relative to the client, in the LVLH frame
(x radial, y along-track, z orbit normal). Client attitude is a MuJoCo quaternion (w, x, y, z).
"""
import gzip
import json

import numpy as np

FORMAT = "spacesim-trajectory"
VERSION = 1


def _r(a, nd=4):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


class TrajectoryLogger:
    def __init__(self, env, policy_name="unknown"):
        self.env = env
        self.policy_name = policy_name
        self.meta, self.frames, self._ret = None, [], 0.0

    def start(self, seed=None):
        e, c = self.env, self.env.cfg
        self.meta = {
            "scenario": c.name, "policy": self.policy_name, "comm_policy": c.comms.policy,
            "dt": e.ctrl_dt, "num_servicers": e.N, "seed": seed,
            "client_half_size": list(c.client.half_size),
            "servicer_half_size": list(c.servicer.half_size),
            "ports_local": [list(p) for p in c.client.ports],
            "client_omega0": list(c.client.init_omega),
            "fuel_kg": c.thrusters.fuel_kg,
            "task": "inspection" if hasattr(e, "team_coverage") else "docking",
        }
        self.frames, self._ret = [], 0.0
        self.record(0.0)

    def record(self, reward=0.0):
        e = self.env
        m, d = e.model, e.data
        pos, vel = e._servicer_states()
        qa = m.jnt_qposadr[e.ids.client_joint]
        cpos, quat = d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7]
        self._ret += float(reward)
        self.frames.append({
            "t": int(e.t),
            "p": _r(pos - cpos),
            "v": _r(vel),
            "q": _r(quat, 6),
            "f": _r(e.last_thrust, 3),
            "ev": [[int(s), int(r), st] for s, r, st in e.channel.last_events],
            "dk": [int(k) for k in e._docked_ports()],
            "fuel": _r(getattr(e, "fuel_fraction", e.thrusters.fuel_fraction), 3),
            "ret": round(self._ret, 3),
        })
        if getattr(e, "spec", None) is not None:
            qs = [d.qpos[m.jnt_qposadr[j] + 3:m.jnt_qposadr[j] + 7] for j in e.ids.servicer_joints]
            self.frames[-1]["sq"] = _r(qs, 5)
            self.frames[-1]["th"] = _r(e.tarray.last_u, 3)
        if hasattr(e, "team_coverage"):
            self.frames[-1]["cov"] = round(float(e.team_coverage), 4)

    def to_dict(self):
        return {"format": FORMAT, "version": VERSION, "meta": self.meta, "frames": self.frames}

    def save(self, path):
        path = str(path)
        data = json.dumps(self.to_dict(), separators=(",", ":"))
        if path.endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(data)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
        return path


def run_and_log(env, agent, seed=0, policy_name=None):
    """Play one episode with `agent` and return a filled TrajectoryLogger."""
    log = TrajectoryLogger(env, policy_name or getattr(agent, "name", type(agent).__name__))
    obs, _ = env.reset(seed=seed)
    agent.reset()
    log.start(seed)
    info = {}
    while env.agents:
        obs, rew, term, trunc, info = env.step(agent.act(obs))
        log.record(next(iter(rew.values())))
    return log, (next(iter(info.values())) if info else {})
