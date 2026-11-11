"""3D overlays drawn on top of the MuJoCo scene.

  green line  = message sent this step
  red line    = message dropped (range / LOS / random / bandwidth)
  thin blue   = each servicer's current target port (its intent)

Works for both the live viewer (viewer.user_scn) and offscreen video
(renderer.scene). Call once per frame, after the scene is updated.
"""
import mujoco
import numpy as np

SENT = (0.1, 0.95, 0.3, 0.9)
DROPPED = (0.95, 0.15, 0.15, 0.7)
INTENT = (0.3, 0.55, 1.0, 0.35)


def _add_line(scn, a, b, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if hasattr(mujoco, "mjv_connector"):
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
    else:  # older API name
        mujoco.mjv_makeConnector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, *a, *b)
    scn.ngeom += 1


def draw_overlay(scn, env, comms=True, intents=True):
    pos, _ = env._servicer_states()
    if intents:
        ppos, _ = env._port_states()
        for i, k in enumerate(env._intents(pos, ppos)):
            _add_line(scn, pos[i], ppos[k], 0.015, INTENT)
    if comms:
        for s, r, status in env.channel.last_events:
            if status == "sent":
                _add_line(scn, pos[s], pos[r], 0.05, SENT)
            else:
                _add_line(scn, pos[s], pos[r], 0.025, DROPPED)
