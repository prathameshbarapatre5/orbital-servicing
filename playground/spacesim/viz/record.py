"""Offscreen episode recording to MP4 (works headless on the cluster).

On machines without a display: export MUJOCO_GL=egl
Use from training: every N iterations, record_episode(env, policy_agent, path)
and log the file to TensorBoard / W&B.
"""
import mujoco
import numpy as np

from spacesim.viz.overlay import draw_overlay


def make_camera(distance=45.0, azimuth=135.0, elevation=-25.0):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = 0.0  # LVLH origin = client
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def record_episode(env, agent, path, seed=0, width=960, height=720, speedup=2.0, overlay=True):
    import imageio.v2 as imageio

    m = env.model
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, height)
    renderer = mujoco.Renderer(m, height, width)
    cam = make_camera()

    obs, _ = env.reset(seed=seed)
    agent.reset()
    fps = max(1, int(round(speedup / env.ctrl_dt)))
    ret, info = 0.0, {}
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
        while env.agents:
            obs, rew, term, trunc, info = env.step(agent.act(obs))
            ret += next(iter(rew.values()))
            cam.azimuth += 0.15  # slow orbit for a nicer 3D view
            renderer.update_scene(env.data, camera=cam)
            if overlay:
                draw_overlay(renderer.scene, env)
            writer.append_data(renderer.render())
    renderer.close()
    last = next(iter(info.values())) if info else {}
    return {"return": ret, "success": last.get("success"), "n_docked": last.get("n_docked")}
