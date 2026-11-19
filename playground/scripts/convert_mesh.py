"""Prepare a dataset spacecraft model for spacesim.

Converts GLB/GLTF/OBJ/STL/PLY to a centered OBJ (MuJoCo-friendly), optionally rescales
it, and prints a YAML snippet with suggested docking-port positions on the bounding box.

  pip install trimesh
  python scripts/convert_mesh.py data/tango.glb assets/client.obj --units m
  python scripts/convert_mesh.py data/servicer.glb assets/servicer.obj --fit 0.6
"""
import argparse

import numpy as np


def main():
    import trimesh

    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--units", choices=["m", "cm", "mm"], default="m", help="units of the source model")
    ap.add_argument("--fit", type=float, default=None, help="scale so the largest dimension equals this (m)")
    ap.add_argument("--port-gap", type=float, default=0.35, help="port offset outside the box (m)")
    args = ap.parse_args()

    mesh = trimesh.load(args.src, force="mesh")
    mesh.apply_scale({"m": 1.0, "cm": 0.01, "mm": 0.001}[args.units])
    lo, hi = mesh.bounds
    mesh.apply_translation(-(lo + hi) / 2)
    if args.fit:
        mesh.apply_scale(args.fit / float(np.max(hi - lo)))
    lo, hi = mesh.bounds
    mesh.export(args.dst)
    half = (hi - lo) / 2
    g = args.port_gap
    print(f"wrote {args.dst}  size (m): {np.round(hi - lo, 3).tolist()}  faces: {len(mesh.faces)}")
    print("\n# scenario YAML snippet (physics box approximates the bus; adjust to the main body, not the panels)")
    print(f"client:\n  visual_mesh: {args.dst}\n  half_size: {np.round(half, 3).tolist()}\n  ports:")
    for p in ([half[0] + g, 0, 0], [-half[0] - g, 0, 0], [0, half[1] + g, 0], [0, -half[1] - g, 0]):
        print(f"    - {np.round(p, 3).tolist()}")
    if len(mesh.faces) > 200_000:
        print("\nnote: >200k faces, consider decimating in Blender for viewer performance")


if __name__ == "__main__":
    main()
