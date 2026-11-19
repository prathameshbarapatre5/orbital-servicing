"""Inspection RPO: inspectors image a tumbling client under changing lighting.

Agents: inspector_0..N-1 (6U CubeSats). No docking. The team must image
`coverage_target` of the client surface while staying outside the keep-out
sphere, avoiding each other, and spending little fuel.

Two fidelity levels:
  * no spec: idealized point-mass inspectors with per-axis LVLH thrust (fast, simple)
  * servicer.spec set: the real design. 6-DOF rigid body with the CAD mass
    properties, the actual thruster layout (allocation, minimum impulse bit, Isp
    fuel use, optional failures), reaction wheels with torque and momentum limits,
    an onboard ADCS that points the +X payload at the client (and dumps wheel
    momentum with the thrusters), and coverage from the real sensor suite
    (boresight, FOV, range, sunlight).
In both modes the policy commands a desired force in LVLH (3D in [-1, 1]); with a
spec, the vehicle delivers only what its thrusters can.

Per-agent observation:
  own pos/vel rel. client (6, noisy) | client quat (4) | client omega (3)
  sun direction (3) | own face coverage (6) | known team face coverage (6)
  keep-out margin (1) | fuel (1) | comm slots (N-1) x (payload 18 + valid + age)
Payload: pos (3), vel (3), current target face one-hot (6), own face coverage (6).
TSB treats "imaging the same face" as a conflict worth talking about.
"""
import mujoco
import numpy as np
from gymnasium import spaces

from spacesim.comms.channel import Channel
from spacesim.envs.servicing_env import OrbitalServicingEnv
from spacesim.sensors.relative import line_of_sight
from spacesim.actuators.reaction_wheels import ReactionWheels
from spacesim.actuators.thruster_array import ThrusterArray
from spacesim.control.adcs import ADCS, los_rate, pointing_frame
from spacesim.dynamics.hcw import hcw_accel
from spacesim.tasks.coverage import box_surface_points, face_fractions, face_normals, visible, visible_sensors
from spacesim.tasks.inspection_config import InspectionScenarioConfig


class InspectionEnv(OrbitalServicingEnv):
    metadata = {"name": "orbital_inspection_v0", "render_modes": ["rgb_array"], "render_fps": 2}

    def __init__(self, cfg: InspectionScenarioConfig | None = None, render_mode=None):
        cfg = cfg or InspectionScenarioConfig()
        self.spec = None
        if cfg.servicer.spec:
            from spacesim.spacecraft.spec import load_spec
            self.spec = load_spec(cfg.servicer.spec)
            sv = cfg.servicer
            sv.half_size, sv.mass = list(self.spec.bus_half_size), self.spec.mass
            sv.inertia, sv.com = list(self.spec.inertia), list(self.spec.com)
            if self.spec.visual_mesh and not sv.visual_mesh:
                sv.visual_mesh = self.spec.visual_mesh
            cfg.thrusters.fuel_kg = self.spec.propellant_kg
        super().__init__(cfg, render_mode)
        t = cfg.task
        self.possible_agents = [f"inspector_{i}" for i in range(self.N)]
        self.P = 6  # "intents" are client faces
        self.payload_dim = 6 + 6 + 6
        self.channel = Channel(cfg.comms, self.N, self.payload_dim)
        self._pts, self._nrm, self._face = box_surface_points(cfg.client.half_size, t.coverage.grid)
        self.K = len(self._pts)
        self.sun_dir = np.array([1.0, 0.0, 0.0])
        extra, extra_state = 0, 0
        if self.spec:
            self.tarray = ThrusterArray(self.spec, self.N)
            self.wheels = ReactionWheels(self.spec, self.N)
            self.adcs = ADCS(self.spec)
            self.cmd_force = float(self.spec.max_axis_force.max())   # N at |action| = 1
            self.n_thr = self.tarray.n
            extra = 4 + 3 + 3 + 1 + self.n_thr + 1   # attitude, rates, wheel momentum, pointing err, health, delivered
            extra_state = self.N * (4 + 3)
            self.point_err = np.zeros(self.N)
            self.delivered = np.ones(self.N)
        obs_dim = 6 + 4 + 3 + 3 + 6 + 6 + 1 + 1 + extra + (self.N - 1) * (self.payload_dim + 2)
        state_dim = 6 * self.N + 7 + 3 + 6 + self.N + extra_state
        self._obs_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.state_space = spaces.Box(-np.inf, np.inf, (state_dim,), np.float32)

    # ---- episode ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c, dr = self.cfg, self.cfg.dr
        if dr.enabled:
            axis = self.rng.normal(size=3)
            c.client.init_omega = list(axis / np.linalg.norm(axis) * self.rng.uniform(0, dr.omega_max))
            c.init.radius_min, c.init.radius_max = dr.spawn_min, dr.spawn_max
        sun = self.rng.normal(size=3) if dr.enabled and dr.random_sun else np.array([1.0, 0.3, 0.2])
        self.sun_dir = sun / np.linalg.norm(sun)
        self.own_mask = np.zeros((self.N, self.K), dtype=bool)
        self.team_mask = np.zeros(self.K, dtype=bool)
        self._known_face_cov = np.zeros((self.N, 6))
        self.team_coverage = 0.0
        self.collided = False
        obs, infos = super().reset(seed=None, options=options)
        if self.spec:
            self._spec_reset()
            obs = self._observations()
        return obs, infos

    # ---- design-driven (spec) vehicle --------------------------------------
    @property
    def fuel_fraction(self):
        return self.tarray.fuel_fraction if self.spec else self.thrusters.fuel_fraction

    def _rot(self, axis, ang):
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K

    def _spec_reset(self):
        m, d, dr = self.model, self.data, self.cfg.dr
        health = (self.rng.random((self.N, self.n_thr)) >= dr.thruster_fail_prob).astype(float) \
            if dr.enabled else np.ones((self.N, self.n_thr))
        self.tarray.reset(health)
        self.wheels.reset()
        cpos = self._free_state(self.ids.client_joint)[0]
        for j in self.ids.servicer_joints:
            qa, da = m.jnt_qposadr[j], m.jnt_dofadr[j]
            p = d.qpos[qa:qa + 3].copy()
            err = np.radians(dr.attitude_error_deg) * self.rng.random() if dr.enabled else 0.0
            R = pointing_frame(p, cpos) @ self._rot(self.rng.normal(size=3), err)
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, R.flatten())
            d.qpos[qa + 3:qa + 7] = q
            d.qvel[da + 3:da + 6] = 0.0
        mujoco.mj_forward(m, d)
        self.point_err[:] = 0.0
        self.delivered[:] = 1.0

    def _servicer_rot(self, i):
        return self.data.xmat[self.ids.servicer_bodies[i]].reshape(3, 3).copy()

    def _spec_command(self, cmd):
        """Allocate this control step's thruster throttles from the desired LVLH force."""
        Fb, Tb, used = np.zeros((self.N, 3)), np.zeros((self.N, 3)), np.zeros(self.N)
        for i in range(self.N):
            R = self._servicer_rot(i)
            f_des = R.T @ (np.clip(cmd[i], -1, 1) * self.cmd_force)
            dump = self.adcs.dump_request(self.wheels.momentum_body(i), self.wheels.saturation(i))
            f, t, u_used, _ = self.tarray.command(i, np.r_[f_des, dump], self.ctrl_dt)
            Fb[i], Tb[i], used[i] = f, t, u_used
            n = np.linalg.norm(f_des)
            self.delivered[i] = min(1.0, (f @ f_des) / n ** 2) if n > 1e-9 else 1.0
        return Fb, Tb, used

    def _apply_spec_forces(self, Fb, Tb, dt):
        m, d = self.model, self.data
        d.xfrc_applied[:] = 0.0
        cb, cj = self.ids.client_body, self.ids.client_joint
        cp, cv = self._free_state(cj)
        d.xfrc_applied[cb, :3] = m.body_mass[cb] * hcw_accel(cp, cv, self.n_orbit)
        for i, (b, j) in enumerate(zip(self.ids.servicer_bodies, self.ids.servicer_joints)):
            p, v = self._free_state(j)
            R = d.xmat[b].reshape(3, 3)
            omega = d.qvel[m.jnt_dofadr[j] + 3:m.jnt_dofadr[j] + 6]
            tau, self.point_err[i] = self.adcs.torque(R, omega, pointing_frame(p, cp), los_rate(p, v, cp, cv))
            t_rw = self.wheels.apply(i, tau, dt)
            d.xfrc_applied[b, :3] = R @ Fb[i] + m.body_mass[b] * hcw_accel(p, v, self.n_orbit)
            d.xfrc_applied[b, 3:] = R @ (Tb[i] + t_rw)

    def step(self, actions):
        m, d, c = self.model, self.data, self.cfg
        t, w = c.task, c.task.reward
        cmd = np.stack([np.asarray(actions.get(a, np.zeros(3)), dtype=float) for a in self.possible_agents])
        if self.spec:
            Fb, Tb, fuel_used = self._spec_command(cmd)
            self.last_thrust = np.array([self._servicer_rot(i) @ Fb[i] for i in range(self.N)])
            for _ in range(c.control_decimation):
                self._apply_spec_forces(Fb, Tb, c.physics_dt)
                mujoco.mj_step(m, d)
                self.collided |= self._client_contact()
        else:
            thrust, fuel_used = self.thrusters.compute(cmd, self.ctrl_dt)
            self.last_thrust = thrust
            for _ in range(c.control_decimation):
                self._apply_forces(thrust)
                mujoco.mj_step(m, d)
                self.collided |= self._client_contact()
        mujoco.mj_forward(m, d)
        self.t += 1

        new_frac = self._update_coverage()
        self._run_comms()

        pos, _ = self._servicer_states()
        cpos = self._free_state(self.ids.client_joint)[0]
        dist = np.linalg.norm(pos - cpos, axis=1)
        keepout = np.clip(t.keepout_radius - dist, 0, None).sum()
        pair = [np.linalg.norm(pos[i] - pos[j]) for i in range(self.N) for j in range(i + 1, self.N)]
        too_close = sum(p < t.min_separation for p in pair)
        oob = bool(np.any(dist > c.max_range))
        success = self.team_coverage >= t.coverage_target

        reward = (w.coverage * new_frac - w.keepout * keepout - w.separation * too_close
                  - w.fuel * fuel_used.sum() / c.thrusters.fuel_kg - w.time)
        if success:
            reward += w.success
        if self.collided:
            reward -= w.collision
        if oob:
            reward -= w.out_of_bounds

        terminated = success or self.collided or oob
        truncated = (not terminated) and self.t >= c.max_steps
        obs = self._observations()
        info = {"success": success, "coverage": round(self.team_coverage, 4), "collision": self.collided,
                "out_of_bounds": oob, "n_docked": 0, "comms_step": dict(self.channel.step_stats),
                "comms_episode": dict(self.channel.episode_stats)}
        rewards = {a: float(reward) for a in self.agents}
        terms = {a: terminated for a in self.agents}
        truncs = {a: truncated for a in self.agents}
        if self.spec:
            info.update(pointing_error_deg=np.degrees(self.point_err).round(2).tolist(),
                        thrust_delivered=self.delivered.round(3).tolist(),
                        wheel_saturation=[float(self.wheels.saturation(i).max()) for i in range(self.N)])
        infos = {a: dict(info, fuel_fraction=float(self.fuel_fraction[i]))
                 for i, a in enumerate(self.agents)}
        if terminated or truncated:
            self.agents = []
        return obs, rewards, terms, truncs, infos

    # ---- task helpers -----------------------------------------------------
    def _client_frame(self):
        b = self.ids.client_body
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def _update_coverage(self):
        cpos, R = self._client_frame()
        pts_w = self._pts @ R.T + cpos
        nrm_w = self._nrm @ R.T
        pos, _ = self._servicer_states()
        seen_any = np.zeros(self.K, dtype=bool)
        cc = self.cfg.task.coverage
        for i in range(self.N):
            if self.spec:
                seen = visible_sensors(pts_w, nrm_w, pos[i], self._servicer_rot(i), self.spec.sensors,
                                       self.sun_dir, cc.max_view_deg, cc.min_sun_cos)
            else:
                seen = visible(pts_w, nrm_w, pos[i], self.sun_dir, cc)
            self.own_mask[i] |= seen
            seen_any |= seen
        new = seen_any & ~self.team_mask
        self.team_mask |= seen_any
        self.team_coverage = float(self.team_mask.mean())
        return float(new.sum()) / self.K

    def _client_contact(self):
        m, d = self.model, self.data
        cb = self.ids.client_body
        for k in range(d.ncon):
            con = d.contact[k]
            b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
            if (b1 == cb and b2 in self._servicer_set) or (b2 == cb and b1 in self._servicer_set):
                return True
        return False

    # compatibility hooks used by overlay / comms / trajectory export
    def _port_states(self):
        """Face centers (and velocities) stand in for 'ports': intents are faces."""
        cpos, R = self._client_frame()
        half = np.asarray(self.cfg.client.half_size)
        centers = face_normals() * np.repeat(half, 2)[:, None] @ R.T + cpos
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.ids.client_body, res, 0)
        omega, v = res[:3], res[3:]
        vel = np.array([v + np.cross(omega, p - cpos) for p in centers])
        return centers, vel

    def _intents(self, pos, ppos):
        """Each inspector's current target = the client face it is looking at most directly."""
        cpos, R = self._client_frame()
        normals = face_normals() @ R.T
        rel = pos - cpos
        rel /= np.linalg.norm(rel, axis=1, keepdims=True) + 1e-9
        return np.argmax(rel @ normals.T, axis=1)

    def _docked_ports(self):
        return np.full(self.N, -1)

    def _assignment(self):
        return 0.0, tuple(range(self.N))

    def _run_comms(self):
        pos, vel = self._servicer_states()
        cpos = self._free_state(self.ids.client_joint)[0]
        intents = self._intents(pos, None)
        own_cov = np.array([face_fractions(self.own_mask[i], self._face) for i in range(self.N)])
        payloads = np.zeros((self.N, self.payload_dim), dtype=np.float32)
        payloads[:, 0:3] = pos - cpos
        payloads[:, 3:6] = vel
        payloads[np.arange(self.N), 6 + intents] = 1.0
        payloads[:, 12:18] = own_cov
        targets = self.comm_policy.select(pos, intents, self._known_intents, self.rng)
        bodies = self.ids.servicer_bodies

        def los(s, r):
            return line_of_sight(self.model, self.data, pos[s], pos[r], bodies[s], bodies[r])

        for s, r, p in self.channel.step(targets, payloads, pos, self.t, self.rng, los):
            self._slots[r, s] = p
            self._rx_step[r, s] = self.t
            self._known_intents[r, s] = int(np.argmax(p[6:12]))
            self._known_face_cov[r] = np.maximum(self._known_face_cov[r], p[12:18])
        for i in range(self.N):
            self._known_face_cov[i] = np.maximum(self._known_face_cov[i], own_cov[i])

    def _observations(self):
        c = self.cfg
        pos, vel = self._servicer_states()
        cpos, cvel = self._free_state(self.ids.client_joint)
        qa, da = self.model.jnt_qposadr[self.ids.client_joint], self.model.jnt_dofadr[self.ids.client_joint]
        quat = self.data.qpos[qa + 3:qa + 7].copy()
        omega = self.data.qvel[da + 3:da + 6].copy()
        obs = {}
        for i, a in enumerate(self.agents):
            rp, rv = self.sensor.measure(pos[i] - cpos, vel[i] - cvel, self.rng)
            own = face_fractions(self.own_mask[i], self._face) if hasattr(self, "own_mask") else np.zeros(6)
            margin = np.linalg.norm(pos[i] - cpos) - c.task.keepout_radius
            slots = []
            for j in range(self.N):
                if j == i:
                    continue
                age = self.t - self._rx_step[i, j]
                valid = float(np.isfinite(age) and age <= c.comms.max_age_steps)
                age_n = min(age, c.comms.max_age_steps) / c.comms.max_age_steps if valid else 1.0
                slots.append(np.concatenate([self._slots[i, j] * valid, [valid, age_n]]))
            extra = []
            if self.spec:
                sqa, sda = self.model.jnt_qposadr[self.ids.servicer_joints[i]], self.model.jnt_dofadr[self.ids.servicer_joints[i]]
                extra = [self.data.qpos[sqa + 3:sqa + 7], self.data.qvel[sda + 3:sda + 6],
                         self.wheels.h[i] / self.wheels.hmax, [self.point_err[i] / np.pi],
                         self.tarray.health[i], [self.delivered[i]]]
            obs[a] = np.concatenate([rp, rv, quat, omega, self.sun_dir, own, self._known_face_cov[i],
                                     [margin], [self.fuel_fraction[i]], *extra, *slots]).astype(np.float32)
        return obs

    def state(self):
        pos, vel = self._servicer_states()
        cpos = self._free_state(self.ids.client_joint)[0]
        qa, da = self.model.jnt_qposadr[self.ids.client_joint], self.model.jnt_dofadr[self.ids.client_joint]
        team = face_fractions(self.team_mask, self._face)
        extra = []
        if self.spec:
            for j in self.ids.servicer_joints:
                sqa, sda = self.model.jnt_qposadr[j], self.model.jnt_dofadr[j]
                extra += [self.data.qpos[sqa + 3:sqa + 7], self.data.qvel[sda + 3:sda + 6]]
        return np.concatenate([(pos - cpos).ravel(), vel.ravel(), self.data.qpos[qa + 3:qa + 7],
                               self.data.qvel[da + 3:da + 6], self.sun_dir, team,
                               self.fuel_fraction, *extra]).astype(np.float32)
