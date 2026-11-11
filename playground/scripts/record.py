"""Record episodes to MP4 (headless-friendly: export MUJOCO_GL=egl).

  python scripts/record.py --agent scripted --policy tsb --episodes 2
"""
import argparse
from pathlib import Path

from spacesim.agents.scripted import make_agent
from spacesim.config import ScenarioConfig
from spacesim.envs.servicing_env import OrbitalServicingEnv
from spacesim.viz.record import record_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--policy", default=None, choices=[None, "tsb", "random_k", "broadcast", "none"])
    ap.add_argument("--agent", default="scripted", choices=["random", "zero", "scripted"])
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--speedup", type=float, default=2.0)
    ap.add_argument("--out", default="videos")
    args = ap.parse_args()

    cfg = ScenarioConfig.from_yaml(args.scenario) if args.scenario else ScenarioConfig()
    if args.policy:
        cfg.comms.policy = args.policy
    if args.max_steps:
        cfg.max_steps = args.max_steps
    env = OrbitalServicingEnv(cfg)
    agent = make_agent(args.agent, env)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        path = out / f"{cfg.name}_{cfg.comms.policy}_{args.agent}_ep{ep}.mp4"
        res = record_episode(env, agent, str(path), seed=ep, speedup=args.speedup)
        print(f"{path}: {res}")
    env.close()


if __name__ == "__main__":
    main()
