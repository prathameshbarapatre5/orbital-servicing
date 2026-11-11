"""PettingZoo ParallelEnv: cooperative multi-servicer rendezvous and docking.

The only part MARL libraries see. Everything below (dynamics, thrusters,
sensors, comms) stays library-agnostic.

Per-agent observation (all in LVLH, noisy where sensed):
  own pos/vel relative to client (6) | client angular velocity (3)
  each port's pos/vel relative to self (6P) | own fuel fraction (1)
  one comm slot per other agent: payload (6+P) + valid flag + age (8+P each)

Global state for a centralized critic (CTDE / MAPPO): see `state()`.
"""
import itertools

import mujoco
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from spacesim.actuators.thrusters import TranslationalThrusters
from spacesim.comms.channel import Channel
from spacesim.comms.policies import make_comm_policy
from spacesim.config import ScenarioConfig
from spacesim.dynamics.hcw import hcw_accel, mean_motion
from spacesim.sensors.relative import RelativeSensor, line_of_sight
from spacesim.spacecraft.builder import build_model


class OrbitalServicingEnv(ParallelEnv):
    metadata = {"name": "orbital_servicing_v0", "render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, cfg: ScenarioConfig | None = None, render_mode=None):
        self.cfg = cfg or ScenarioConfig()
        c = self.cfg
        self.render_mode = render_mode
        self.model, self.ids = build_model(c)
        self.data = mujoco.MjData(self.model)
        self.n_orbit = mean_motion(c.orbit_altitude_m)
        self.N = c.num_servicers
        self.P = len(c.client.ports)
        if self.P < self.N:
            raise ValueError("need at least as many docking ports as servicers")

        self.possible_agents = [f"servicer_{i}" for i in range(self.N)]
        self.agents = []
        self.ctrl_dt = c.physics_dt * c.control_decimation

        self.payload_dim = 6 + self.P
        self.thrusters = TranslationalThrusters(c.thrusters, self.N)
        self.sensor = RelativeSensor(c.sensors)
        self.channel = Channel(c.comms, self.N, self.payload_dim)
        self.comm_policy = make_comm_policy(c.comms)
        self.rng = np.random.default_rng()

        self._perms = list(itertools.permutations(range(self.P), self.N))
        self._all_bodies = [self.ids.client_body] + list(self.ids.servicer_bodies)
        self._all_joints = [self.ids.client_joint] + list(self.ids.servicer_joints)
        self._servicer_set = set(self.ids.servicer_bodies)

        obs_dim = 6 + 3 + 6 * self.P + 1 + (self.N - 1) * (self.payload_dim + 2)
        state_dim = 6 * self.N + 13 + 6 * self.P + self.N
        self._obs_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self._act_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.state_space = spaces.Box(-np.inf, np.inf, (state_dim,), np.float32)
        self._renderer = None

    # ---- PettingZoo API -------------------------------------------------
    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d, c = self.model, self.data, self.cfg
        mujoco.mj_resetData(m, d)

        qa, da = m.jnt_qposadr[self.ids.client_joint], m.jnt_dofadr[self.ids.client_joint]
        d.qpos[qa:qa + 7] = [0, 0, 0, 1, 0, 0, 0]
        d.qvel[da:da + 6] = [0, 0, 0, *c.client.init_omega]  # free-joint angvel is body frame

        for j, p in zip(self.ids.servicer_joints, self._sample_positions()):
            qa, da = m.jnt_qposadr[j], m.jnt_dofadr[j]
            d.qpos[qa:qa + 7] = [*p, 1, 0, 0, 0]
            v = self.rng.normal(size=3)
            d.qvel[da:da + 3] = v / (np.linalg.norm(v) + 1e-9) * self.rng.uniform(0, c.init.max_speed)
            d.qvel[da + 3:da + 6] = 0.0
        mujoco.mj_forward(m, d)

        self.thrusters.reset()
        self.channel.reset()
        if hasattr(self.comm_policy, "reset"):
            self.comm_policy.reset()
        self._slots = np.zeros((self.N, self.N, self.payload_dim), dtype=np.float32)
        self._rx_step = np.full((self.N, self.N), -np.inf)
        self._known_intents = np.full((self.N, self.N), -1, dtype=int)
        self.t = 0
        self.agents = list(self.possible_agents)
        self._prev_cost, _ = self._assignment()
        self._prev_docked = 0

        obs = self._observations()
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        m, d, c, w = self.model, self.data, self.cfg, self.cfg.reward
        cmd = np.stack([np.asarray(actions.get(a, np.zeros(3)), dtype=float)
                        for a in self.possible_agents])
        thrust, fuel_used = self.thrusters.compute(cmd, self.ctrl_dt)

        contact_pen = 0.0
        for _ in range(c.control_decimation):
            self._apply_forces(thrust)
            mujoco.mj_step(m, d)
            contact_pen += self._contact_penalty()
        contact_pen /= c.control_decimation
        mujoco.mj_forward(m, d)  # sync derived quantities (site poses, velocities)
        self.t += 1

        self._run_comms()

        cost, _ = self._assignment()
        docked_port = self._docked_ports()
        n_docked = int(np.sum(docked_port >= 0))
        success = n_docked == self.N and len(set(docked_port.tolist())) == self.N
        pos, _ = self._servicer_states()
        oob = bool(np.any(np.linalg.norm(pos, axis=1) > c.max_range))

        reward = (w.progress * (self._prev_cost - cost)
                  - w.fuel * fuel_used.sum() / c.thrusters.fuel_kg
                  - w.time
                  - contact_pen
                  + w.dock * (n_docked - self._prev_docked))
        if success:
            reward += w.success
        if oob:
            reward -= w.out_of_bounds
        self._prev_cost, self._prev_docked = cost, n_docked

        terminated = success or oob
        truncated = (not terminated) and self.t >= c.max_steps

        obs = self._observations()
        rewards = {a: float(reward) for a in self.agents}  # shared team reward
        terms = {a: terminated for a in self.agents}
        truncs = {a: truncated for a in self.agents}
        info = {
            "success": success,
            "out_of_bounds": oob,
            "n_docked": n_docked,
            "assignment_cost": cost,
            "comms_step": dict(self.channel.step_stats),
            "comms_episode": dict(self.channel.episode_stats),
        }
        infos = {a: dict(info, fuel_fraction=float(self.thrusters.fuel_fraction[i]))
                 for i, a in enumerate(self.agents)}
        if terminated or truncated:
            self.agents = []
        return obs, rewards, terms, truncs, infos

    def state(self):
        m, d = self.model, self.data
        pos, vel = self._servicer_states()
        qa, da = m.jnt_qposadr[self.ids.client_joint], m.jnt_dofadr[self.ids.client_joint]
        ppos, pvel = self._port_states()
        return np.concatenate([
            pos.ravel(), vel.ravel(),
            d.qpos[qa:qa + 7], d.qvel[da:da + 6],
            ppos.ravel(), pvel.ravel(),
            self.thrusters.fuel_fraction,
        ]).astype(np.float32)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ---- physics helpers -----------------------------------------------
    def _free_state(self, joint):
        qa, da = self.model.jnt_qposadr[joint], self.model.jnt_dofadr[joint]
        return self.data.qpos[qa:qa + 3].copy(), self.data.qvel[da:da + 3].copy()

    def _servicer_states(self):
        pv = [self._free_state(j) for j in self.ids.servicer_joints]
        return np.array([p for p, _ in pv]), np.array([v for _, v in pv])

    def _port_states(self):
        res = np.zeros(6)
        pos, vel = [], []
        for s in self.ids.port_sites:
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, s, res, 0)
            pos.append(self.data.site_xpos[s].copy())
            vel.append(res[3:6].copy())  # res = [angular, linear], world frame
        return np.array(pos), np.array(vel)

    def _apply_forces(self, thrust):
        d, m = self.data, self.model
        d.xfrc_applied[:] = 0.0
        for k, (b, j) in enumerate(zip(self._all_bodies, self._all_joints)):
            p, v = self._free_state(j)
            f = m.body_mass[b] * hcw_accel(p, v, self.n_orbit)
            if k > 0:
                f = f + thrust[k - 1]
            d.xfrc_applied[b, :3] = f

    def _contact_penalty(self):
        m, d, w = self.model, self.data, self.cfg.reward
        pen = 0.0
        for k in range(d.ncon):
            con = d.contact[k]
            b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
            s1, s2 = b1 in self._servicer_set, b2 in self._servicer_set
            if s1 and s2:
                pen += w.servicer_contact
            elif s1 or s2:
                v1 = self._free_state(self._body_joint(b1))[1]
                v2 = self._free_state(self._body_joint(b2))[1]
                excess = np.linalg.norm(v1 - v2) - w.safe_contact_speed
                if excess > 0:
                    pen += w.impact * excess
        return pen

    def _body_joint(self, body):
        return int(self.model.body_jntadr[body])

    # ---- task helpers --------------------------------------------------
    def _sample_positions(self):
        c = self.cfg.init
        out = []
        while len(out) < self.N:
            v = self.rng.normal(size=3)
            p = v / np.linalg.norm(v) * self.rng.uniform(c.radius_min, c.radius_max)
            if all(np.linalg.norm(p - q) >= c.min_separation for q in out):
                out.append(p)
        return out

    def _assignment(self):
        """Min-cost one-to-one servicer->port assignment (brute force; N is small)."""
        pos, _ = self._servicer_states()
        ppos, _ = self._port_states()
        D = np.linalg.norm(pos[:, None, :] - ppos[None, :, :], axis=-1)
        rows = np.arange(self.N)
        costs = [D[rows, list(perm)].sum() for perm in self._perms]
        best = int(np.argmin(costs))
        return float(costs[best]), self._perms[best]

    def _docked_ports(self):
        dk = self.cfg.docking
        pos, vel = self._servicer_states()
        ppos, pvel = self._port_states()
        out = np.full(self.N, -1)
        for i in range(self.N):
            dist = np.linalg.norm(ppos - pos[i], axis=1)
            k = int(np.argmin(dist))
            if dist[k] < dk.pos_tol and np.linalg.norm(pvel[k] - vel[i]) < dk.vel_tol:
                out[i] = k
        return out

    def _intents(self, pos, ppos):
        """Each servicer's locally chosen target: its nearest port."""
        D = np.linalg.norm(pos[:, None, :] - ppos[None, :, :], axis=-1)
        return np.argmin(D, axis=1)

    def _run_comms(self):
        pos, vel = self._servicer_states()
        ppos, _ = self._port_states()
        client_pos = self._free_state(self.ids.client_joint)[0]
        intents = self._intents(pos, ppos)
        payloads = np.zeros((self.N, self.payload_dim), dtype=np.float32)
        payloads[:, 0:3] = pos - client_pos
        payloads[:, 3:6] = vel
        payloads[np.arange(self.N), 6 + intents] = 1.0

        targets = self.comm_policy.select(pos, intents, self._known_intents, self.rng)
        bodies = self.ids.servicer_bodies

        def los(s, r):
            return line_of_sight(self.model, self.data, pos[s], pos[r], bodies[s], bodies[r])

        for s, r, p in self.channel.step(targets, payloads, pos, self.t, self.rng, los):
            self._slots[r, s] = p
            self._rx_step[r, s] = self.t
            self._known_intents[r, s] = int(np.argmax(p[6:]))

    def _observations(self):
        c = self.cfg
        pos, vel = self._servicer_states()
        cpos, cvel = self._free_state(self.ids.client_joint)
        da = self.model.jnt_dofadr[self.ids.client_joint]
        omega = self.data.qvel[da + 3:da + 6].copy()
        ppos, pvel = self._port_states()
        obs = {}
        for i, a in enumerate(self.agents):
            rp, rv = self.sensor.measure(pos[i] - cpos, vel[i] - cvel, self.rng)
            pp, pv = self.sensor.measure(ppos - pos[i], pvel - vel[i], self.rng)
            slots = []
            for j in range(self.N):
                if j == i:
                    continue
                age = self.t - self._rx_step[i, j]
                valid = float(np.isfinite(age) and age <= c.comms.max_age_steps)
                age_n = min(age, c.comms.max_age_steps) / c.comms.max_age_steps if valid else 1.0
                slots.append(np.concatenate([self._slots[i, j] * valid, [valid, age_n]]))
            obs[a] = np.concatenate([
                rp, rv, omega, pp.ravel(), pv.ravel(),
                [self.thrusters.fuel_fraction[i]], *slots,
            ]).astype(np.float32)
        return obs


def make_env(scenario_path=None, **overrides):
    cfg = ScenarioConfig.from_yaml(scenario_path) if scenario_path else ScenarioConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return OrbitalServicingEnv(cfg)
