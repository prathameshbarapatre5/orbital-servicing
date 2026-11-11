"""Communication channel: range, line-of-sight, latency, random drops, bandwidth.

Every message attempt is logged so communication efficiency (one of the
project's headline metrics) can be reported per step and per episode.
`last_events` keeps who-sent-to-whom this step for visualization.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TSBConfig:
    k: int = 1                    # max targets per sender per step
    threshold: float = 0.3        # min relevance to transmit at all
    dist_scale: float = 10.0      # m, proximity relevance falloff
    conflict_weight: float = 1.0  # bonus when sender and receiver target the same port
    unknown_bonus: float = 0.5    # neighbors whose intent we don't know yet are worth contacting
    refresh_steps: int = 10       # resend unchanged info at most every 1 s


@dataclass
class CommsConfig:
    policy: str = "tsb"            # "tsb" | "random_k" | "broadcast" | "none"
    range_m: float = 30.0
    latency_steps: int = 1         # in control steps
    drop_prob: float = 0.05
    max_msgs_per_sender: int = 2   # bandwidth cap per control step
    require_los: bool = True
    max_age_steps: int = 20        # for normalizing message age in observations
    tsb: TSBConfig = field(default_factory=TSBConfig)


STAT_KEYS = ("attempted", "sent", "dropped_bandwidth", "dropped_range",
             "dropped_los", "dropped_random", "delivered", "bytes_sent")


class Channel:
    def __init__(self, cfg: CommsConfig, num_agents: int, payload_dim: int):
        self.cfg = cfg
        self.num_agents = num_agents
        self.payload_dim = payload_dim
        self.reset()

    def reset(self):
        self._queue = []  # (deliver_step, sender, receiver, payload)
        self.step_stats = {k: 0 for k in STAT_KEYS}
        self.episode_stats = {k: 0 for k in STAT_KEYS}
        self.last_events = []  # (sender, receiver, status) for this step

    def step(self, targets, payloads, positions, t, rng, los_fn=None):
        """Send this step's messages and return those due for delivery.

        targets:   list (len N) of receiver index lists chosen by the comm policy
        payloads:  (N, payload_dim) array, one payload per sender
        positions: (N, 3) sender/receiver positions
        los_fn:    optional callable (sender, receiver) -> bool
        Returns a list of (sender, receiver, payload) delivered at step t.
        Event statuses: "sent", "range", "los", "random", "bandwidth".
        """
        st = {k: 0 for k in STAT_KEYS}
        events = []
        payloads = np.asarray(payloads, dtype=np.float32)
        for s, recvs in enumerate(targets):
            recvs = [r for r in dict.fromkeys(recvs) if r != s]
            st["attempted"] += len(recvs)
            if len(recvs) > self.cfg.max_msgs_per_sender:
                st["dropped_bandwidth"] += len(recvs) - self.cfg.max_msgs_per_sender
                events += [(s, r, "bandwidth") for r in recvs[self.cfg.max_msgs_per_sender:]]
                recvs = recvs[: self.cfg.max_msgs_per_sender]
            st["sent"] += len(recvs)
            st["bytes_sent"] += len(recvs) * payloads[s].nbytes
            for r in recvs:
                if np.linalg.norm(positions[s] - positions[r]) > self.cfg.range_m:
                    st["dropped_range"] += 1
                    events.append((s, r, "range"))
                elif self.cfg.require_los and los_fn is not None and not los_fn(s, r):
                    st["dropped_los"] += 1
                    events.append((s, r, "los"))
                elif rng.random() < self.cfg.drop_prob:
                    st["dropped_random"] += 1
                    events.append((s, r, "random"))
                else:
                    self._queue.append((t + self.cfg.latency_steps, s, r, payloads[s].copy()))
                    events.append((s, r, "sent"))

        due = [q for q in self._queue if q[0] <= t]
        self._queue = [q for q in self._queue if q[0] > t]
        st["delivered"] = len(due)
        self.step_stats = st
        self.last_events = events
        for k in STAT_KEYS:
            self.episode_stats[k] += st[k]
        return [(s, r, p) for _, s, r, p in due]
