"""Random-policy rollout: checks the stack runs end to end and prints comms stats.

  python scripts/smoke_test.py --scenario spacesim/scenarios/rendezvous_3v1.yaml --policy tsb
"""
import argparse
import time

import numpy as np

from spacesim.config import ScenarioConfig
from spacesim.envs.servicing_env import OrbitalServicingEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--policy", default=None, choices=[None, "tsb", "broadcast", "none", "random_k"])
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    cfg = ScenarioConfig.from_yaml(args.scenario) if args.scenario else ScenarioConfig()
    if args.policy:
        cfg.comms.policy = args.policy
    env = OrbitalServicingEnv(cfg)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        ret, steps, t0 = 0.0, 0, time.time()
        while env.agents:
            actions = {a: env.action_space(a).sample() for a in env.agents}
            obs, rew, term, trunc, info = env.step(actions)
            ret += next(iter(rew.values()))
            steps += 1
        last = next(iter(info.values()))
        cs = last["comms_episode"]
        print(f"ep {ep}: return={ret:8.2f} steps={steps} sps={steps / (time.time() - t0):.0f} "
              f"success={last['success']} docked={last['n_docked']} | comms sent={cs['sent']} "
              f"delivered={cs['delivered']} los_drop={cs['dropped_los']} range_drop={cs['dropped_range']}")
    env.close()


if __name__ == "__main__":
    main()
