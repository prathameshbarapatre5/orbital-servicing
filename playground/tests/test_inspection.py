import numpy as np
import pytest

from spacesim.tasks.coverage import CoverageConfig, box_surface_points, face_fractions, visible


def test_box_points_and_faces():
    pts, nrm, face = box_surface_points([1, 1, 1], 4)
    assert pts.shape == (96, 3) and set(face.tolist()) == set(range(6))
    assert np.allclose(np.abs(pts[np.arange(96), np.argmax(np.abs(nrm), 1)]), 1.0)


def test_visibility_rules():
    pts, nrm, face = box_surface_points([1, 1, 1], 4)
    cfg = CoverageConfig(lwir=False)
    sun = np.array([1.0, 0, 0])
    seen = visible(pts, nrm, [8, 0, 0], sun, cfg)          # in front of the lit +x face
    assert face_fractions(seen, face)[1] == 1.0 and seen[face != 1].sum() == 0
    dark = visible(pts, nrm, [-8, 0, 0], sun, cfg)         # in front of the unlit -x face
    assert dark.sum() == 0
    ir = visible(pts, nrm, [-3.5, 0, 0], sun, CoverageConfig(lwir=True))  # LWIR sees it up close
    assert face_fractions(ir, face)[0] == 1.0
    far = visible(pts, nrm, [30, 0, 0], sun, cfg)          # beyond camera range
    assert far.sum() == 0


mujoco = pytest.importorskip("mujoco")
pytest.importorskip("pettingzoo")


def _env():
    from spacesim.envs.registry import build_env, load_config
    return build_env(load_config("spacesim/scenarios/inspection_2x.yaml"))


def test_inspection_api():
    from pettingzoo.test import parallel_api_test
    parallel_api_test(_env(), num_cycles=60)


def test_scripted_inspector_gains_coverage():
    from spacesim.agents.scripted import make_agent
    env = _env()
    agent = make_agent("inspector", env)
    obs, _ = env.reset(seed=0)
    agent.reset()
    info = {}
    while env.agents:
        obs, rew, term, trunc, info = env.step(agent.act(obs))
    cov = next(iter(info.values()))["coverage"]
    assert cov > 0.3, f"scripted inspectors only imaged {cov:.0%}"
