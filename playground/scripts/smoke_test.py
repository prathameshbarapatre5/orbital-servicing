"""Random-policy rollout: checks the stack runs end to end and prints comms stats.

  python scripts/smoke_test.py --scenario spacesim/scenarios/rendezvous_3v1.yaml --policy tsb
"""
import argparse
import time

import numpy as np

from spacesim.envs.registry import build_env, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--policy", default=None, choices=[None, "tsb", "broadcast", "none", "random_k"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--agent", default="random", help="random | zero | scripted | inspector")
    args = ap.parse_args()

    cfg = load_config(args.scenario)
    if args.policy:
        cfg.comms.policy = args.policy
    env = build_env(cfg)

    from spacesim.agents.scripted import make_agent
    agent = make_agent(args.agent, env)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        agent.reset()
        ret, steps, t0 = 0.0, 0, time.time()
        while env.agents:
            actions = agent.act(obs)
            obs, rew, term, trunc, info = env.step(actions)
            ret += next(iter(rew.values()))
            steps += 1
        last = next(iter(info.values()))
        cs = last["comms_episode"]
        print(f"ep {ep}: return={ret:8.2f} steps={steps} sps={steps / (time.time() - t0):.0f} "
              f"success={last['success']} docked={last['n_docked']} cov={last.get('coverage', '-')} | comms sent={cs['sent']} "
              f"delivered={cs['delivered']} los_drop={cs['dropped_los']} range_drop={cs['dropped_range']}")
    env.close()


if __name__ == "__main__":
    main()
