"""Inspection coverage geometry (pure numpy, no MuJoCo).

The client surface is sampled as points with outward normals. A point counts as
imaged when an inspector sees it:
  visible camera: point is sunlit, inside the camera range window, and viewed
                  within max_view_deg of its normal
  LWIR (thermal): works unlit, but only at short range
The client is convex, so the normal test also handles self-occlusion.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class CoverageConfig:
    grid: int = 4                 # points per face edge -> 6 * grid^2 samples
    r_min: float = 2.0            # m, closer than this the camera can't focus
    r_vis: float = 15.0           # m, visible camera max useful range
    r_ir: float = 4.0             # m, LWIR max useful range (forces close passes for unlit faces)
    lwir: bool = True
    max_view_deg: float = 70.0    # max angle between surface normal and view ray
    min_sun_cos: float = 0.1      # sunlit if normal . sun > this


def box_surface_points(half, grid):
    """Points, normals and face ids on a box (body frame). Face id = 2*axis + (sign > 0)."""
    half = np.asarray(half, dtype=float)
    ticks = (np.arange(grid) + 0.5) / grid * 2 - 1  # cell centers in [-1, 1]
    pts, nrm, face = [], [], []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for sign in (-1.0, 1.0):
            n = np.zeros(3)
            n[axis] = sign
            for a in ticks:
                for b in ticks:
                    p = np.zeros(3)
                    p[axis] = sign * half[axis]
                    p[u] = a * half[u]
                    p[v] = b * half[v]
                    pts.append(p)
                    nrm.append(n)
                    face.append(2 * axis + (sign > 0))
    return np.array(pts), np.array(nrm), np.array(face)


def face_normals():
    """Outward normal per face id, matching box_surface_points ordering."""
    out = np.zeros((6, 3))
    for axis in range(3):
        out[2 * axis, axis] = -1.0
        out[2 * axis + 1, axis] = 1.0
    return out


def visible(points_w, normals_w, cam_pos, sun_dir, cfg: CoverageConfig):
    """Boolean mask of surface points imaged from cam_pos this step."""
    ray = np.asarray(cam_pos)[None, :] - points_w
    dist = np.linalg.norm(ray, axis=1)
    cos_view = np.einsum("ij,ij->i", normals_w, ray) / np.maximum(dist, 1e-9)
    facing = cos_view >= np.cos(np.radians(cfg.max_view_deg))
    lit = normals_w @ np.asarray(sun_dir) > cfg.min_sun_cos
    cam = lit & facing & (dist >= cfg.r_min) & (dist <= cfg.r_vis)
    if cfg.lwir:
        cam |= facing & (dist >= cfg.r_min) & (dist <= cfg.r_ir)
    return cam


def face_fractions(mask, face_ids):
    return np.array([mask[face_ids == f].mean() for f in range(6)])


def visible_sensors(points_w, normals_w, cam_pos, R_body, sensors, sun_dir, max_view_deg, min_sun_cos):
    """Coverage from a real sensor suite: each sensor has a boresight (body frame),
    a field of view, a range window and may need sunlight. Returns a boolean mask."""
    ray = np.asarray(cam_pos)[None, :] - points_w           # surface -> camera
    dist = np.linalg.norm(ray, axis=1)
    cos_view = np.einsum("ij,ij->i", normals_w, ray) / np.maximum(dist, 1e-9)
    facing = cos_view >= np.cos(np.radians(max_view_deg))
    lit = normals_w @ np.asarray(sun_dir) > min_sun_cos
    look = -ray / np.maximum(dist, 1e-9)[:, None]            # camera -> surface
    out = np.zeros(len(points_w), dtype=bool)
    for s in sensors:
        if not s.counts_for_coverage:
            continue
        b = R_body @ np.asarray(s.boresight, float)
        b /= np.linalg.norm(b)
        in_fov = look @ b >= np.cos(np.radians(s.fov_deg / 2))
        ok = facing & in_fov & (dist >= s.r_min) & (dist <= s.r_max)
        if s.needs_sun:
            ok &= lit
        out |= ok
    return out
