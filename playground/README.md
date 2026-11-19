# orbital-servicing

Multi-agent reinforcement learning for in-orbit servicing, with Targeted Selective Broadcasting (TSB) and a custom simulation playground.

- `spacesim/`: the simulation playground, a space robotics layer on MuJoCo 4.0

## spacesim

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

## Inspection RPO task (inspectors only)

Two INS-A000 6U inspectors image a tumbling client. No docking: the team must image
85% of the client surface while respecting a 3 m keep-out sphere, avoiding each other,
and saving fuel. Sun direction and tumble are randomized every episode.

```bash
python scripts/smoke_test.py --scenario spacesim/scenarios/inspection_2x.yaml --agent inspector
python scripts/watch.py      --scenario spacesim/scenarios/inspection_2x.yaml --agent inspector
python scripts/export_trajectory.py --scenario spacesim/scenarios/inspection_2x.yaml --agent inspector
```
Train on `inspection_2x_easy.yaml` first (no tumble, fixed Sun, close spawn).

Sensing model: the visible camera needs the surface sunlit and within 2 to 15 m; LWIR works
unlit but only within 4 m. Dark faces therefore force close, risky passes near the keep-out
sphere, so lighting, tumble and fuel genuinely trade off. The scripted inspector (fixed 6 m
circles) averages about 70% coverage, a baseline a trained policy should beat.

## Design-driven vehicles (spacecraft specs)

`spacesim/spacecraft/specs/ins_a000_inspector.yaml` describes the real inspector, extracted
from the Fusion CAD: mass properties, the 8 canted corner thrusters (position, force direction,
thrust, Isp, minimum on-time), 3 reaction wheels, the +X sensor suite (camera, LWIR,
telescope, ToF with FOV and range) and ADCS tuning. Point a scenario at it:

```yaml
servicer:
  spec: ../spacecraft/specs/ins_a000_inspector.yaml
```

What the stack then does every step:
- **Thruster allocation:** the policy's desired LVLH force is mapped onto the real thrusters
  (bounded least squares, 6-DOF), with minimum impulse bit, Isp-based fuel use and optional
  random thruster failures (`dr.thruster_fail_prob`). Unreachable force is simply not delivered.
- **6-DOF dynamics:** CAD inertia, thruster torques, gyroscopic coupling.
- **ADCS:** reaction-wheel PD with line-of-sight rate feed-forward keeps the +X payload on the
  client; thrusters dump wheel momentum near saturation.
- **Sensors:** coverage only counts what a real sensor sees (boresight, FOV, range, sunlight).
- **Observations add:** attitude, body rates, wheel momentum, pointing error, thruster health,
  fraction of commanded thrust delivered.

Values marked ASSUMED in the spec (mass, inertia, thrust level, propellant, wheel limits) should
be replaced from Fusion physical properties and component datasheets.

## Bring your own models

**Spacecraft meshes** (dataset GLB/GLTF/OBJ/STL):
```bash
pip install trimesh
python scripts/convert_mesh.py data/client.glb spacesim/scenarios/assets/client.obj --units m
python scripts/convert_mesh.py data/servicer.glb spacesim/scenarios/assets/servicer.obj --fit 0.6
```
The script prints a YAML snippet (physics box size + suggested ports). In the scenario:
```yaml
client:
  visual_mesh: assets/client.obj   # relative to the YAML file
  half_size: [1.1, 0.9, 0.9]       # physics box: fit the bus, not the solar panels
servicer:
  visual_mesh: assets/servicer.obj
```
Meshes are visual-only (no collisions, no mass, viewer group 2); physics keeps the boxes, so training speed is unchanged.

**Web demo:** drag `.glb` models onto the left half (client) or right half (servicers), and drop
an exported trajectory `.json(.gz)` anywhere to replay your policy with the full renderer.

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
| `viz/trajectory.py` | Trajectory export for the web demo and Blender |
| `envs/inspection_env.py` | Inspection RPO task (coverage, lighting, keep-out, domain randomization) |
| `tasks/coverage.py` | Surface sampling and camera / LWIR visibility rules |
| `agents/inspector_scripted.py` | Scripted circumnavigation baseline |
| `spacecraft/spec.py`, `spacecraft/specs/` | Vehicle design specs (thrusters, wheels, sensors, mass properties) |
| `actuators/thruster_array.py` | Thruster allocation, minimum impulse bit, fuel, failures |
| `actuators/reaction_wheels.py` | Wheel torque and momentum limits |
| `control/adcs.py` | Pointing control, LOS feed-forward, momentum dumping |

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
- Attitude control, body-frame thrusters, reaction wheels
- Debris / keep-out zones (threat perception), sensor FOV
- Two-timescale handoff (closed-form far field, MuJoCo near field)
