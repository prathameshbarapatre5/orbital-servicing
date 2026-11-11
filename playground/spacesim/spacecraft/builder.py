"""Procedural MuJoCo model construction with mjSpec (no hand-written XML)."""
from dataclasses import dataclass

import mujoco


@dataclass
class ModelIds:
    client_body: int
    client_joint: int
    servicer_bodies: list
    servicer_joints: list
    port_sites: list


def build_model(cfg):
    spec = mujoco.MjSpec()
    spec.modelname = cfg.name
    spec.option.timestep = cfg.physics_dt
    spec.option.gravity = [0.0, 0.0, 0.0]  # orbital effects come from HCW forces
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    light = spec.worldbody.add_light()
    light.pos = [0.0, 0.0, 40.0]
    light.dir = [0.0, 0.0, -1.0]

    # Client spacecraft (servicing target).
    client = spec.worldbody.add_body()
    client.name = "client"
    cj = client.add_freejoint()
    cj.name = "client_free"
    hull = client.add_geom()
    hull.name = "client_hull"
    hull.type = mujoco.mjtGeom.mjGEOM_BOX
    hull.size = list(cfg.client.half_size)
    hull.mass = cfg.client.mass
    hull.rgba = [0.75, 0.75, 0.8, 1.0]
    for k, p in enumerate(cfg.client.ports):
        site = client.add_site()
        site.name = f"port_{k}"
        site.pos = list(p)
        site.size = [0.08, 0.08, 0.08]
        site.rgba = [0.1, 0.9, 0.2, 1.0]

    # Servicers.
    for i in range(cfg.num_servicers):
        body = spec.worldbody.add_body()
        body.name = f"servicer_{i}"
        body.pos = [0.0, 0.0, 10.0 + 2.0 * i]  # overwritten at reset
        j = body.add_freejoint()
        j.name = f"servicer_{i}_free"
        g = body.add_geom()
        g.name = f"servicer_{i}_hull"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = list(cfg.servicer.half_size)
        g.mass = cfg.servicer.mass
        g.rgba = [0.2, 0.45, 0.9, 1.0]

    model = spec.compile()

    def bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    def jid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def sid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)

    ids = ModelIds(
        client_body=bid("client"),
        client_joint=jid("client_free"),
        servicer_bodies=[bid(f"servicer_{i}") for i in range(cfg.num_servicers)],
        servicer_joints=[jid(f"servicer_{i}_free") for i in range(cfg.num_servicers)],
        port_sites=[sid(f"port_{k}") for k in range(len(cfg.client.ports))],
    )
    return model, ids
