"""Scenario config for the inspection RPO task."""
from dataclasses import dataclass, field

from spacesim.config import ScenarioConfig
from spacesim.tasks.coverage import CoverageConfig


@dataclass
class InspectionRewardConfig:
    coverage: float = 20.0        # per full client surface newly imaged (team)
    success: float = 20.0
    keepout: float = 1.0          # per m inside the keep-out sphere, per step
    separation: float = 0.5       # per step two inspectors are too close
    fuel: float = 2.0             # per full tank used
    time: float = 0.005
    collision: float = 20.0       # hitting the client ends the episode
    out_of_bounds: float = 10.0


@dataclass
class InspectionTaskConfig:
    coverage_target: float = 0.85     # team fraction of the client surface imaged
    keepout_radius: float = 3.0       # m from client center
    min_separation: float = 2.0       # m between inspectors
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    reward: InspectionRewardConfig = field(default_factory=InspectionRewardConfig)


@dataclass
class DomainRandConfig:
    enabled: bool = True
    omega_max: float = 0.03           # rad/s, random tumble axis and rate
    random_sun: bool = True           # random Sun direction (lighting) per episode
    spawn_min: float = 15.0           # m
    spawn_max: float = 30.0
    thruster_fail_prob: float = 0.0   # per thruster per episode (spec-driven vehicles)
    attitude_error_deg: float = 30.0  # initial pointing error (spec-driven vehicles)


@dataclass
class InspectionScenarioConfig(ScenarioConfig):
    task: InspectionTaskConfig = field(default_factory=InspectionTaskConfig)
    dr: DomainRandConfig = field(default_factory=DomainRandConfig)
