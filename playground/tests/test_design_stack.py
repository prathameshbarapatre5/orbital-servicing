"""Design-driven vehicle: spec loading, thruster allocation, wheels, ADCS, sensors."""
import os

import numpy as np
import pytest

from spacesim.actuators.reaction_wheels import ReactionWheels
from spacesim.actuators.thruster_array import ThrusterArray
from spacesim.control.adcs import ADCS, attitude_error, pointing_frame
from spacesim.spacecraft.spec import load_spec
from spacesim.tasks.coverage import box_surface_points, visible_sensors

SPEC = "spacesim/spacecraft/specs/ins_a000_inspector.yaml"


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC)


def test_spec_from_cad(spec):
    assert len(spec.thrusters) == 8 and len(spec.wheels) == 3 and len(spec.sensors) == 4
    assert os.path.exists(spec.visual_mesh), f"visual mesh not found: {spec.visual_mesh}"
    # canted-corner layout: 4 x 0.1 N x cos(30) along X, 4 x 0.1 N x 0.354 along Y and Z
    assert np.allclose(spec.max_axis_force, [0.3464, 0.1416, 0.1416], atol=2e-3)


def test_full_6dof_authority(spec):
    ta = ThrusterArray(spec, 1)
    assert np.linalg.matrix_rank(ta.B) == 6
    for w in ([0.2, 0, 0, 0, 0, 0], [0, -0.1, 0, 0, 0, 0], [0, 0, 0.1, 0, 0, 0], [0, 0, 0, 0, 0, 0.003]):
        out = ta.B @ ta.allocate(np.array(w, float), ta.health[0])
        assert np.allclose(out, w, atol=2e-3)


def test_failed_thrusters_reduce_capability(spec):
    ta = ThrusterArray(spec, 1)
    health = np.ones(8)
    health[[4, 5, 6, 7]] = 0          # lose the four aft thrusters (the +X pushers)
    out = ta.B @ ta.allocate(np.array([0.2, 0, 0, 0, 0, 0]), health)
    assert out[0] < 0.01


def test_min_impulse_bit_and_fuel(spec):
    ta = ThrusterArray(spec, 1)
    f, t, used, u = ta.command(0, np.array([1e-4, 0, 0, 0, 0, 0]), dt=0.01)
    assert np.allclose(u, 0) and used == 0      # too short to fire
    f, t, used, u = ta.command(0, np.array([0.3, 0, 0, 0, 0, 0]), dt=0.5)
    assert used > 0 and ta.fuel[0] < spec.propellant_kg


def test_wheels_saturate(spec):
    rw = ReactionWheels(spec, 1)
    for _ in range(2000):
        rw.apply(0, np.array([0, 0, 0.01]), 0.01)
    assert rw.saturation(0)[2] == pytest.approx(1.0)
    assert np.allclose(rw.apply(0, np.array([0, 0, 0.01]), 0.01), 0)   # no more torque that way


def test_adcs_slews_to_target(spec):
    I, rw, ad = np.array(spec.inertia), ReactionWheels(spec, 1), ADCS(spec)

    def rot(a, th):
        a = np.asarray(a, float) / np.linalg.norm(a)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    R, w, Rd = rot([0.3, 1, 0.2], np.radians(60)), np.zeros(3), pointing_frame([0, 0, 0], [10, 0, 0])
    for _ in range(4000):
        tau, _ = ad.torque(R, w, Rd)
        tb = rw.apply(0, tau, 0.01)
        w = w + (tb - np.cross(w, I * w)) / I * 0.01
        if np.linalg.norm(w) > 0:
            R = R @ rot(w, np.linalg.norm(w) * 0.01)
    assert np.degrees(attitude_error(R, Rd)[1]) < 1.0


def test_sensor_fov_matters(spec):
    pts, nrm, _ = box_surface_points([1, 1, 1], 4)
    sun = np.array([1.0, 0, 0])
    facing = pointing_frame([8, 0, 0], [0, 0, 0])
    away = pointing_frame([8, 0, 0], [16, 0, 0])
    assert visible_sensors(pts, nrm, [8, 0, 0], facing, spec.sensors, sun, 70, 0.1).sum() > 0
    assert visible_sensors(pts, nrm, [8, 0, 0], away, spec.sensors, sun, 70, 0.1).sum() == 0


def test_spec_env_runs():
    pytest.importorskip("mujoco")
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test
    from spacesim.envs.registry import build_env, load_config
    env = build_env(load_config("spacesim/scenarios/inspection_2x.yaml"))
    assert env.spec is not None and env.n_thr == 8
    parallel_api_test(env, num_cycles=30)
    obs, _ = env.reset(seed=0)
    obs, rew, term, trunc, info = env.step({a: np.array([1.0, 0, 0]) for a in env.agents})
    i = next(iter(info.values()))
    assert len(i["pointing_error_deg"]) == env.N and 0 <= i["thrust_delivered"][0] <= 1.0
