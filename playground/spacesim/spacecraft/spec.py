"""Spacecraft design specs: the properties agents fly with.

A spec file describes one vehicle from its CAD: mass properties, every thruster
(position, force direction, thrust, Isp, minimum on-time), reaction wheels,
sensors (boresight, field of view, range, whether they need sunlight) and the
ADCS tuning. Units: SI, body frame, positions relative to the bus geometric center.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass
class ThrusterSpec:
    name: str
    pos: list                  # m, body frame
    dir: list                  # unit force direction ON the spacecraft (opposite the exhaust)
    max_thrust: float = 0.1    # N
    isp: float = 220.0         # s
    min_on_time: float = 0.02  # s, shorter pulses are dropped (minimum impulse bit)


@dataclass
class WheelSpec:
    axis: list
    max_torque: float = 0.002     # N m
    max_momentum: float = 0.015   # N m s


@dataclass
class SensorSpec:
    name: str
    kind: str = "visible"         # visible | lwir | tof | telescope
    boresight: list = field(default_factory=lambda: [1.0, 0.0, 0.0])
    fov_deg: float = 40.0         # full cone angle
    r_min: float = 2.0
    r_max: float = 15.0
    needs_sun: bool = True
    counts_for_coverage: bool = True


@dataclass
class ADCSSpec:
    bandwidth: float = 0.4        # rad/s, closed-loop natural frequency
    damping: float = 0.9
    dump_threshold: float = 0.6   # start thruster momentum dumping at this wheel-momentum fraction
    dump_gain: float = 0.05       # 1/s


@dataclass
class SpacecraftSpec:
    name: str
    mass: float
    bus_half_size: list
    inertia: list                 # [Ixx, Iyy, Izz] about the CoM, kg m^2
    com: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    propellant_kg: float = 0.5
    visual_mesh: str | None = None
    thrusters: list = field(default_factory=list)
    wheels: list = field(default_factory=list)
    sensors: list = field(default_factory=list)
    adcs: ADCSSpec = field(default_factory=ADCSSpec)
    notes: dict = field(default_factory=dict)

    @property
    def max_axis_force(self):
        """Best-case force along +/- each body axis using all thrusters at full throttle."""
        d = np.array([t.dir for t in self.thrusters]) * np.array([t.max_thrust for t in self.thrusters])[:, None]
        return np.maximum(np.clip(d, 0, None).sum(0), np.clip(-d, 0, None).sum(0))


def _expand_thrusters(raw):
    """Supports explicit entries or a `mirror` block that generates symmetric sets."""
    out = []
    for t in raw:
        if "mirror" in t:
            base = np.array(t["pos"], dtype=float)
            exhaust = np.array(t["exhaust"], dtype=float)
            exhaust /= np.linalg.norm(exhaust)
            k = 0
            for sx in (1, -1):
                for sy in (1, -1):
                    for sz in (1, -1):
                        s = np.array([sx, sy, sz])
                        if not all(s[i] == 1 or t["mirror"][i] for i in range(3)):
                            continue
                        out.append(ThrusterSpec(
                            name=f"{t.get('name', 'thr')}_{k}", pos=(base * s).tolist(),
                            dir=(-exhaust * s).tolist(), max_thrust=t.get("max_thrust", 0.1),
                            isp=t.get("isp", 220.0), min_on_time=t.get("min_on_time", 0.02)))
                        k += 1
        else:
            out.append(ThrusterSpec(**t))
    return out


def load_spec(path):
    path = Path(path)
    with open(path) as f:
        d = yaml.safe_load(f)
    spec = SpacecraftSpec(
        name=d["name"], mass=d["mass"], bus_half_size=d["bus_half_size"], inertia=d["inertia"],
        com=d.get("com", [0.0, 0.0, 0.0]), propellant_kg=d.get("propellant_kg", 0.5),
        visual_mesh=d.get("visual_mesh"),
        thrusters=_expand_thrusters(d.get("thrusters", [])),
        wheels=[WheelSpec(**w) for w in d.get("wheels", [])],
        sensors=[SensorSpec(**s) for s in d.get("sensors", [])],
        adcs=ADCSSpec(**d.get("adcs", {})), notes=d.get("notes", {}))
    if spec.visual_mesh and not Path(spec.visual_mesh).is_absolute():
        spec.visual_mesh = str((path.parent / spec.visual_mesh).resolve())
    return spec
