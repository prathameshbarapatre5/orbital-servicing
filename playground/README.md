# spacesim: space robotics layer on MuJoCo 4.0

Multi-agent in-orbit servicing env for MARL (MAPPO / CTDE) with a pluggable
comms layer for Targeted Selective Broadcasting (TSB) ablations.

## Setup
MuJoCo 4.0 needs a newer Python than 3.10. Use uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt && uv pip install -e .
pytest tests/ -v          # 11 tests should pass
```
Headless machines (cluster): `export MUJOCO_GL=egl` before rendering.

## Run
```bash
# sanity rollout + comms stats
python scripts/smoke_test.py --policy tsb          # tsb | random_k | broadcast | none

# watch live in 3D (green = sent, red = dropped, blue = target port)
python scripts/watch.py --agent scripted --policy tsb

# record MP4s to videos/
python scripts/record.py --agent scripted --policy tsb --episodes 2
```
Agents: `random`, `zero`, `scripted` (privileged expert, proves the task is solvable).

## Scenarios
| File | Use |
|---|---|
| `easy_3v1.yaml` | No tumble, close spawn. Train here first. |
| `rendezvous_3v1.yaml` | Main task: tumbling client, 3 servicers, 4 ports |
| `rendezvous_3v1_wide.yaml` | Wide spawn, exercises comm range/LOS drops |

## Layout
| Module | What it does |
|---|---|
| `dynamics/hcw.py` | HCW relative motion in LVLH; closed-form STM |
| `spacecraft/builder.py` | Builds client, servicers, ports with `mjSpec` |
| `actuators/thrusters.py` | Per-axis thrust, deadband, Isp fuel budget |
| `sensors/relative.py` | Noisy relative nav, `mj_ray` line-of-sight |
| `comms/channel.py` | Range, LOS, latency, drops, bandwidth, message log |
| `comms/policies.py` | `none`, `random_k`, `broadcast`, `tsb` (placeholder heuristic) |
| `envs/servicing_env.py` | PettingZoo `ParallelEnv`, `state()` for centralized critic |
| `agents/scripted.py` | Random, zero, scripted docking expert |
| `viz/overlay.py`, `viz/record.py` | 3D comm overlays, MP4 recording |

## Task and reward
Shared team reward: assignment-cost progress, fuel, time, contact/impact
penalties, dock bonus (undock negates it), success bonus, out-of-bounds penalty.
Weights live in `RewardConfig` (`spacesim/config.py`).
Success = every servicer at a distinct port within position and velocity tolerance.

## Conventions
- World frame = LVLH centered on the client orbit (x radial, y along-track, z orbit normal)
- Gravity off; HCW terms applied via `xfrc_applied` each physics step
- Physics 100 Hz, control 10 Hz, 60 s episodes

## Plug in the real TSB
Replace `TSBHeuristic.relevance()` / `select()` in `comms/policies.py`, or add a
comm head to the action space and pass the learned gate into `Channel.step()`.

## Roadmap
- MAPPO training config (BenchMARL) + Slurm scripts
- Attitude control, body-frame thrusters, reaction wheels
- Debris / keep-out zones (threat perception), sensor FOV
- Two-timescale handoff (closed-form far field, MuJoCo near field)
