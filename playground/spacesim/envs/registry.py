"""Pick the right config class and env from a scenario file."""
import yaml

from spacesim.config import ScenarioConfig


def load_config(path=None):
    """A scenario YAML with a top-level `task:` block is an inspection scenario."""
    if path is None:
        return ScenarioConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if "task" in data or "dr" in data:
        from spacesim.tasks.inspection_config import InspectionScenarioConfig
        return InspectionScenarioConfig.from_yaml(path)
    return ScenarioConfig.from_yaml(path)


def build_env(cfg, render_mode=None):
    from spacesim.tasks.inspection_config import InspectionScenarioConfig
    if isinstance(cfg, InspectionScenarioConfig):
        from spacesim.envs.inspection_env import InspectionEnv
        return InspectionEnv(cfg, render_mode)
    from spacesim.envs.servicing_env import OrbitalServicingEnv
    return OrbitalServicingEnv(cfg, render_mode)
