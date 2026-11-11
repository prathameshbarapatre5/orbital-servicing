"""Watch episodes live in the MuJoCo 3D viewer, with comm links drawn in.

  python scripts/watch.py --agent scripted --policy tsb
  python scripts/watch.py --agent random --policy broadcast --speed 4

Agents: random | zero | scripted (privileged expert). A trained policy plugs
in later through the same reset()/act(obs) interface.
"""
import argparse
import time

import mujoco
import mujoco.viewer

from spacesim.agents.scripted import make_agent
from spacesim.config import ScenarioConfig
from spacesim.envs.servicing_env import OrbitalServicingEnv
from spacesim.viz.overlay import draw_overlay


def launch(model, data):
    try:
        return mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    except TypeError:
        return mujoco.viewer.launch_passive(model, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--policy", default=None, choices=[None, "tsb", "random_k", "broadcast", "none"])
    ap.add_argument("--agent", default="scripted", choices=["random", "zero", "scripted"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed vs real time")
    args = ap.parse_args()

    cfg = ScenarioConfig.from_yaml(args.scenario) if args.scenario else ScenarioConfig()
    if args.policy:
        cfg.comms.policy = args.policy
    if args.max_steps:
        cfg.max_steps = args.max_steps
    env = OrbitalServicingEnv(cfg)
    agent = make_agent(args.agent, env)

    with launch(env.model, env.data) as v:
        v.cam.lookat[:] = 0.0
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 45.0, 135.0, -25.0
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=ep)
            agent.reset()
            ret, info = 0.0, {}
            while env.agents and v.is_running():
                t0 = time.time()
                obs, rew, term, trunc, info = env.step(agent.act(obs))
                ret += next(iter(rew.values()))
                with v.lock():
                    v.user_scn.ngeom = 0
                    draw_overlay(v.user_scn, env)
                v.sync()
                wait = env.ctrl_dt / args.speed - (time.time() - t0)
                if wait > 0:
                    time.sleep(wait)
            if not v.is_running():
                break
            last = next(iter(info.values())) if info else {}
            print(f"ep {ep}: return={ret:.2f} success={last.get('success')} "
                  f"docked={last.get('n_docked')} sent={last.get('comms_episode', {}).get('sent')}")
    env.close()


if __name__ == "__main__":
    main()
