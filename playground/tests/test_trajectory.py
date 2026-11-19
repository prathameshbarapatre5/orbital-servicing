"""Trajectory export (sim tooling for the web demo and Blender)."""
import json

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("pettingzoo")

from spacesim.agents.scripted import make_agent  # noqa: E402
from spacesim.envs.registry import build_env, load_config  # noqa: E402
from spacesim.viz.trajectory import FORMAT, run_and_log  # noqa: E402


@pytest.mark.parametrize("scenario,agent", [(None, "scripted"),
                                            ("spacesim/scenarios/inspection_2x.yaml", "inspector")])
def test_trajectory_roundtrip(tmp_path, scenario, agent):
    env = build_env(load_config(scenario))
    env.cfg.max_steps = 20
    log, _ = run_and_log(env, make_agent(agent, env), seed=3)
    data = json.load(open(log.save(tmp_path / "traj.json")))
    assert data["format"] == FORMAT and data["meta"]["num_servicers"] == env.N
    assert len(data["frames"]) == 21
    f = data["frames"][-1]
    assert len(f["p"]) == env.N and len(f["q"]) == 4
