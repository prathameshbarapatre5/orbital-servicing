"""Simple agents for visualization and sanity checks.

ScriptedDocking is a PRIVILEGED expert: it reads ground truth from the env.
Its job is to prove the task is solvable with the current reward, docking
tolerances and fuel budget, and to give you something to watch before
training produces a policy. If it can't dock, a learned policy can't either.
"""
import numpy as np


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def reset(self):
        pass

    def act(self, obs):
        return {a: self.env.action_space(a).sample() for a in self.env.agents}


class ZeroAgent(RandomAgent):
    def act(self, obs):
        return {a: np.zeros(3, dtype=np.float32) for a in self.env.agents}


class ScriptedDocking:
    """Two-phase guidance: fly to a standoff point outside the assigned port,
    then close in slowly, matching the port's velocity (client may tumble)."""

    def __init__(self, env, standoff=3.0, gain=0.15, vmax=0.6, vmax_final=0.15, kv=0.5):
        self.env = env
        self.standoff, self.gain, self.kv = standoff, gain, kv
        self.vmax, self.vmax_final = vmax, vmax_final
        self.reset()

    def reset(self):
        self.final = np.zeros(self.env.N, dtype=bool)
        self.assign = None

    def act(self, obs):
        env = self.env
        pos, vel = env._servicer_states()
        ppos, pvel = env._port_states()
        cpos, _ = env._free_state(env.ids.client_joint)
        if self.assign is None:  # lock assignment for the episode
            _, self.assign = env._assignment()
        scale = env.cfg.servicer.mass / env.cfg.thrusters.max_thrust
        actions = {}
        for i, a in enumerate(env.possible_agents):
            if a not in env.agents:
                continue
            k = self.assign[i]
            normal = ppos[k] - cpos
            normal /= np.linalg.norm(normal) + 1e-9
            target = ppos[k] + (0.0 if self.final[i] else self.standoff) * normal
            err = target - pos[i]
            if not self.final[i] and np.linalg.norm(err) < 0.5:
                self.final[i] = True
            v_des = self.gain * err
            vmax = self.vmax_final if self.final[i] else self.vmax
            speed = np.linalg.norm(v_des)
            if speed > vmax:
                v_des *= vmax / speed
            v_des += pvel[k]
            actions[a] = np.clip((v_des - vel[i]) * self.kv * scale, -1, 1).astype(np.float32)
        return actions


def make_agent(name, env):
    agents = {"random": RandomAgent, "zero": ZeroAgent, "scripted": ScriptedDocking}
    if name not in agents:
        raise ValueError(f"unknown agent {name}; choose from {sorted(agents)}")
    return agents[name](env)
