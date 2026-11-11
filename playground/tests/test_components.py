import numpy as np

from spacesim.actuators.thrusters import ThrusterConfig, TranslationalThrusters
from spacesim.comms.channel import Channel, CommsConfig, TSBConfig
from spacesim.comms.policies import FullBroadcast, NoComm, RandomK, TSBHeuristic


def test_thruster_deadband_and_fuel_limit():
    th = TranslationalThrusters(ThrusterConfig(max_thrust=2.0, fuel_kg=1e-4, deadband=0.1), 2)
    f, used = th.compute(np.array([[0.05, 0, 0], [1, 1, 1]]), dt=1.0)
    assert np.allclose(f[0], 0) and used[0] == 0
    assert used[1] <= 1e-4 + 1e-12 and th.fuel[1] >= 0
    f2, _ = th.compute(np.array([[0, 0, 0], [1, 1, 1]]), dt=1.0)
    assert np.allclose(f2[1], 0)


def test_channel_latency_range_bandwidth():
    cfg = CommsConfig(range_m=10.0, latency_steps=2, drop_prob=0.0, max_msgs_per_sender=1)
    ch = Channel(cfg, 3, payload_dim=4)
    pos = np.array([[0, 0, 0], [5, 0, 0], [50, 0, 0]], float)
    rng = np.random.default_rng(0)
    pay = np.arange(12, dtype=float).reshape(3, 4)
    out = ch.step([[1, 2], [], []], pay, pos, t=0, rng=rng)
    assert out == [] and ch.step_stats["dropped_bandwidth"] == 1
    assert ch.step([[2], [], []], pay, pos, t=1, rng=rng) == []
    assert ch.step_stats["dropped_range"] == 1
    out = ch.step([[], [], []], pay, pos, t=2, rng=rng)
    assert len(out) == 1 and out[0][0] == 0 and out[0][1] == 1


def test_baseline_policies():
    pos = np.array([[0, 0, 0], [1, 0, 0], [40, 0, 0]], float)
    intents = np.array([0, 0, 1])
    known = np.full((3, 3), -1)
    rng = np.random.default_rng(0)
    assert NoComm().select(pos, intents, known, rng) == [[], [], []]
    assert FullBroadcast().select(pos, intents, known, rng)[0] == [1, 2]
    sel = RandomK(1).select(pos, intents, known, rng)
    assert all(len(s) == 1 and i not in s for i, s in enumerate(sel))


def test_tsb_cold_start_event_trigger_and_refresh():
    pos = np.array([[0, 0, 0], [1, 0, 0], [40, 0, 0]], float)
    intents = np.array([0, 0, 1])
    rng = np.random.default_rng(0)
    cfg = TSBConfig(k=1, threshold=0.3, dist_scale=10.0, unknown_bonus=0.5, refresh_steps=10)
    tsb = TSBHeuristic(cfg)
    unknown = np.full((3, 3), -1)
    sel = tsb.select(pos, intents, unknown, rng)
    assert sel[0] == [1] and sel[2] != []
    known = np.tile(intents, (3, 1))
    np.fill_diagonal(known, -1)
    sel = tsb.select(pos, intents, known, rng)
    assert sel[0] == [] and sel[2] == []
    sent_0, sent_2 = False, False
    for _ in range(cfg.refresh_steps + 1):
        sel = tsb.select(pos, intents, known, rng)
        sent_0 |= 1 in sel[0]
        sent_2 |= bool(sel[2])
    assert sent_0 and not sent_2
    tsb.reset()
    tsb.select(pos, intents, known, rng)
    sel = tsb.select(pos, np.array([2, 0, 1]), known, rng)
    assert sel[0] != []
