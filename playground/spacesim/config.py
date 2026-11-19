"""Scenario configuration: dataclasses with a YAML loader."""
import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from spacesim.actuators.thrusters import ThrusterConfig
from spacesim.comms.channel import CommsConfig
from spacesim.sensors.relative import SensorConfig


@dataclass
class ClientConfig:
    half_size: list = field(default_factory=lambda: [1.0, 1.0, 1.0])  # m
    mass: float = 500.0                                                # kg
    init_omega: list = field(default_factory=lambda: [0.0, 0.0, 0.02])  # rad/s, body frame (tumble)
    # Docking ports in the client body frame. Must be >= num_servicers.
    ports: list = field(default_factory=lambda: [
        [1.35, 0.0, 0.0], [-1.35, 0.0, 0.0], [0.0, 1.35, 0.0], [0.0, -1.35, 0.0]])
    # Optional visual-only mesh (OBJ/STL). Physics keeps the box above.
    visual_mesh: str | None = None
    visual_scale: float = 1.0
    visual_rgba: list = field(default_factory=lambda: [0.8, 0.8, 0.85, 1.0])


@dataclass
class ServicerConfig:
    half_size: list = field(default_factory=lambda: [0.3, 0.3, 0.3])
    mass: float = 50.0
    visual_mesh: str | None = None
    visual_scale: float = 1.0
    visual_rgba: list = field(default_factory=lambda: [0.85, 0.65, 0.25, 1.0])
    # Optional design spec (thrusters, wheels, sensors, mass properties). When set,
    # it overrides half_size, mass, inertia and propellant, and enables 6-DOF flight.
    spec: str | None = None
    inertia: list | None = None   # [Ixx, Iyy, Izz] kg m^2 (filled from spec)
    com: list | None = None


@dataclass
class InitConfig:
    radius_min: float = 8.0
    radius_max: float = 20.0
    min_separation: float = 2.0
    max_speed: float = 0.05


@dataclass
class DockingConfig:
    pos_tol: float = 0.25   # m
    vel_tol: float = 0.05   # m/s relative to the port


@dataclass
class RewardConfig:
    progress: float = 1.0        # per meter of reduced assignment cost
    fuel: float = 2.0            # per full tank used
    time: float = 0.01           # per step
    servicer_contact: float = 0.5
    impact: float = 5.0          # per m/s above safe speed on the client
    safe_contact_speed: float = 0.1
    dock: float = 5.0            # per newly docked servicer (negated on undock)
    success: float = 50.0
    out_of_bounds: float = 20.0


@dataclass
class ScenarioConfig:
    name: str = "rendezvous_3v1"
    num_servicers: int = 3
    orbit_altitude_m: float = 400_000.0
    physics_dt: float = 0.01
    control_decimation: int = 10     # control step = 0.1 s
    max_steps: int = 600             # 60 s episodes
    max_range: float = 60.0          # m from client before out-of-bounds
    client: ClientConfig = field(default_factory=ClientConfig)
    servicer: ServicerConfig = field(default_factory=ServicerConfig)
    init: InitConfig = field(default_factory=InitConfig)
    thrusters: ThrusterConfig = field(default_factory=ThrusterConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    comms: CommsConfig = field(default_factory=CommsConfig)
    docking: DockingConfig = field(default_factory=DockingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_yaml(cls, path):
        path = Path(path)
        with open(path) as f:
            cfg = _build(cls, yaml.safe_load(f) or {})
        # mesh paths in a scenario file are relative to that file
        for part in (cfg.client, cfg.servicer):
            if part.visual_mesh and not Path(part.visual_mesh).is_absolute():
                part.visual_mesh = str((path.parent / part.visual_mesh).resolve())
        if cfg.servicer.spec and not Path(cfg.servicer.spec).is_absolute():
            cfg.servicer.spec = str((path.parent / cfg.servicer.spec).resolve())
        return cfg


def _build(cls, data):
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        val, typ = data[f.name], hints[f.name]
        kwargs[f.name] = _build(typ, val) if dataclasses.is_dataclass(typ) else val
    unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**kwargs)
