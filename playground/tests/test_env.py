"""Needs mujoco and pettingzoo installed."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("pettingzoo")

from pettingzoo.test import parallel_api_test  # noqa: E402

from spacesim.envs.servicing_env import OrbitalServicingEnv  # noqa: E402
from spacesim.sensors.relative import line_of_sight  # noqa: E402


def test_pettingzoo_api():
    parallel_api_test(OrbitalServicingEnv(), num_cycles=200)


def test_hcw_drift_without_thrust():
    env = OrbitalServicingEnv()
    env.reset(seed=0)
    s0 = env.state()
    for _ in range(50):
        env.step({a: np.zeros(3) for a in env.agents})
    assert np.all(np.isfinite(env.state())) and not np.allclose(env.state(), s0)


def test_state_and_obs_shapes():
    env = OrbitalServicingEnv()
    obs, _ = env.reset(seed=1)
    for a, o in obs.items():
        assert o.shape == env.observation_space(a).shape
    assert env.state().shape == env.state_space.shape


def test_client_blocks_los():
    env = OrbitalServicingEnv()
    env.reset(seed=0)
    m, d = env.model, env.data
    for j, p in zip(env.ids.servicer_joints, [[10, 0, 0], [-10, 0, 0], [0, 10, 0]]):
        qa = m.jnt_qposadr[j]
        d.qpos[qa:qa + 7] = [*p, 1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    pos, _ = env._servicer_states()
    b = env.ids.servicer_bodies
    assert not line_of_sight(m, d, pos[0], pos[1], b[0], b[1])  # through the client
    assert line_of_sight(m, d, pos[0], pos[2], b[0], b[2])      # clear path
