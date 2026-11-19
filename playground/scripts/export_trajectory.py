"""Export episodes as JSON for the web demo (drag the file onto the page) or Blender.
No rendering, so it runs headless and fast.

  python scripts/export_trajectory.py --agent scripted --episodes 3
  python scripts/export_trajectory.py --scenario spacesim/scenarios/inspection_2x.yaml --agent inspector --gz
"""
import argparse
from pathlib import Path

from spacesim.agents.scripted import make_agent
from spacesim.envs.registry import build_env, load_config
from spacesim.viz.trajectory import run_and_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--policy", default=None, choices=[None, "tsb", "random_k", "broadcast", "none"],
                    help="comm policy")
    ap.add_argument("--agent", default="scripted",
                    help="random | zero | scripted | inspector")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="trajectories")
    ap.add_argument("--gz", action="store_true", help="write .json.gz (much smaller)")
    args = ap.parse_args()

    cfg = load_config(args.scenario)
    if args.policy:
        cfg.comms.policy = args.policy
    env = build_env(cfg)
    agent = make_agent(args.agent, env)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = getattr(agent, "name", args.agent)
    for ep in range(args.episodes):
        seed = args.seed + ep
        log, info = run_and_log(env, agent, seed=seed, policy_name=tag)
        path = out / f"{cfg.name}_{cfg.comms.policy}_{tag}_seed{seed}.json{'.gz' if args.gz else ''}"
        log.save(path)
        print(f"{path}: steps={len(log.frames) - 1} success={info.get('success')} "
              f"docked={info.get('n_docked')} sent={info.get('comms_episode', {}).get('sent')}")
    env.close()


if __name__ == "__main__":
    main()
