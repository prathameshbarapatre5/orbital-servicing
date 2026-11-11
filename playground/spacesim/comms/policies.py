"""Who-talks-to-whom policies. Swap these for clean comms ablations.

TSBHeuristic is a PLACEHOLDER for the team's Targeted Selective Broadcasting
mechanism: replace `relevance()` (or the whole class) with the real TSB logic,
or make the gate learnable by adding a comm head to the agent's action.
"""
import numpy as np

from spacesim.comms.channel import CommsConfig, TSBConfig


class NoComm:
    name = "none"

    def select(self, positions, intents, known_intents, rng):
        return [[] for _ in range(len(positions))]


class FullBroadcast:
    name = "broadcast"

    def select(self, positions, intents, known_intents, rng):
        n = len(positions)
        return [[j for j in range(n) if j != i] for i in range(n)]


class RandomK:
    """Fair baseline: same message budget as TSB (k per step), no selectivity."""
    name = "random_k"

    def __init__(self, k):
        self.k = k

    def select(self, positions, intents, known_intents, rng):
        n = len(positions)
        return [[int(j) for j in rng.choice([j for j in range(n) if j != i],
                                            size=min(self.k, n - 1), replace=False)]
                for i in range(n)]


class TSBHeuristic:
    """Event-triggered, targeted sending.

    Relevance = proximity + port conflict + bonus for neighbors not heard from yet.
    A message goes out only if there is news (intent changed since last told,
    or refresh_steps elapsed), to the top-k neighbors above threshold.
    """
    name = "tsb"

    def __init__(self, cfg: TSBConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self._last_told = None
        self._since = None

    def relevance(self, i, positions, intents, known_intents):
        d = np.linalg.norm(positions - positions[i], axis=1)
        score = np.exp(-d / self.cfg.dist_scale)
        conflict = (known_intents[i] == intents[i]) & (known_intents[i] >= 0)
        score = score + self.cfg.conflict_weight * conflict
        score = score + self.cfg.unknown_bonus * (known_intents[i] < 0)
        score[i] = -np.inf
        return score

    def select(self, positions, intents, known_intents, rng):
        n = len(positions)
        if self._last_told is None:
            self._last_told = np.full((n, n), -1)
            self._since = np.full((n, n), np.inf)
        self._since += 1
        out = []
        for i in range(n):
            score = self.relevance(i, positions, intents, known_intents)
            news = (self._last_told[i] != intents[i]) | (self._since[i] > self.cfg.refresh_steps)
            score = np.where(news, score, -np.inf)
            order = np.argsort(-score)
            chosen = [int(j) for j in order[: self.cfg.k] if score[j] > self.cfg.threshold]
            for j in chosen:
                self._last_told[i, j] = intents[i]
                self._since[i, j] = 0
            out.append(chosen)
        return out


def make_comm_policy(cfg: CommsConfig):
    if cfg.policy == "none":
        return NoComm()
    if cfg.policy == "broadcast":
        return FullBroadcast()
    if cfg.policy == "random_k":
        return RandomK(cfg.tsb.k)
    if cfg.policy == "tsb":
        return TSBHeuristic(cfg.tsb)
    raise ValueError(f"unknown comm policy: {cfg.policy}")
