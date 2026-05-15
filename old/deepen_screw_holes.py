#!/usr/bin/env python3
"""
Post-process: deepen pad screw holes in the pan body.

Loads the pan body STL, reads screw hole positions from notepad_properties.json,
and drills M2 pilot holes deeper into the drum body. Uses voxel carving for
speed on the large mesh.

Usage:
    python deepen_screw_holes.py [--depth=15] [--res=0.5] [--input=data/pan_body.stl]

    --depth  Total hole depth from pocket floor (mm, default 15)
    --res    Voxel resolution (mm, default 0.5 — should match pan body)
"""

import sys
import json
import math
import numpy as np
import trimesh

# Defaults
DEPTH = 15.0        # mm — total depth from pocket floor (was 5mm)
RES = 0.5           # mm — must match pan body voxel resolution
PILOT_R = 1.0       # mm — M2 pilot hole radius
CSINK_TOP_R = 2.0   # mm — countersink head radius
CSINK_PLUG_DEPTH = 2.0
CSINK_TAPER_DEPTH = 1.0
POCKET_DEPTH = 5.0
INPUT = "data/pan_body.stl"
OUTPUT = None  # defaults to overwrite input


def main():
    global DEPTH, RES, INPUT, OUTPUT

    for arg in sys.argv[1:]:
        if arg.startswith('--depth='):
            DEPTH = float(arg.split('=')[1])
        elif arg.startswith('--res='):
            RES = float(arg.split('=')[1])
        elif arg.startswith('--input='):
            INPUT = arg.split('=')[1]
        elif arg.startswith('--output='):
            OUTPUT = arg.split('=')[1]

    if OUTPUT is None:
        OUTPUT = INPUT

    print(f"Deepening screw holes in {INPUT}")
    print(f"  Depth: {DEPTH}mm from pocket floor (was 5mm)")
    print(f"  Resolution: {RES}mm")

    # Load screw hole positions and transform to body coordinates
    from generate_pan_body import compute_leveling_rotation
    R_level = compute_leveling_rotation("data/Tenor Pan only.obj")
    offset = np.array(
        json.load(open('data/pan_centroid_offset.json'))['centroid_offset_mm'])

    props = json.load(open('data/notepads/notepad_properties.json'))
    screw_holes = []
    for p in props:
        for h in p.get('hole_positions', []):
            h_body = R_level @ (np.array(h) - offset)
            screw_holes.append(h_body.tolist())
    print(f"  {len(screw_holes)} screw holes (body coords)")

    # Load pan body
    print(f"  Loading {INPUT}...", end="", flush=True)
    mesh = trimesh.load(INPUT)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    print(f" {len(mesh.faces)/1e6:.1f}M faces")

    # Voxelize
    print(f"  Voxelizing at {RES}mm...", end="", flush=True)
    vox = mesh.voxelized(RES)
    grid = vox.matrix.copy()  # boolean 3D array
    transform = vox.transform
    inv_transform = np.linalg.inv(transform)
    print(f" grid {grid.shape}")

    def world_to_voxel(xyz):
        """Convert world coords to voxel indices."""
        h = np.array([xyz[0], xyz[1], xyz[2], 1.0])
        v = inv_transform @ h
        return int(round(v[0])), int(round(v[1])), int(round(v[2]))

    nx, ny, nz = grid.shape
    n_carved = 0

    # For each screw hole, carve from pocket_floor down to pocket_floor - DEPTH
    print(f"  Carving deeper holes...", end="", flush=True)
    for hx, hy, hz in screw_holes:
        pocket_floor = hy - POCKET_DEPTH

        # Voxel Y range for the deeper hole
        y_top = pocket_floor
        y_bot = pocket_floor - DEPTH

        # Voxel index range for the countersink area
        r_max = CSINK_TOP_R
        for dx_mm in np.arange(-r_max - RES, r_max + RES, RES):
            wx = hx + dx_mm
            for dz_mm in np.arange(-r_max - RES, r_max + RES, RES):
                wz = hz + dz_mm
                r_sq = dx_mm * dx_mm + dz_mm * dz_mm

                if r_sq <= PILOT_R * PILOT_R:
                    # Full depth pilot hole
                    for dy_mm in np.arange(y_bot, y_top, RES):
                        ix, iy, iz = world_to_voxel([wx, dy_mm, wz])
                        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                            if grid[ix, iy, iz]:
                                grid[ix, iy, iz] = False
                                n_carved += 1
                elif r_sq <= CSINK_TOP_R * CSINK_TOP_R:
                    # Countersink region (only near pocket_floor)
                    r = math.sqrt(r_sq)
                    if r <= PILOT_R:
                        taper_extra = CSINK_TAPER_DEPTH
                    else:
                        taper_extra = CSINK_TAPER_DEPTH * (CSINK_TOP_R - r) / (CSINK_TOP_R - PILOT_R)
                    max_depth = CSINK_PLUG_DEPTH + taper_extra
                    for dy_mm in np.arange(pocket_floor - max_depth, y_top, RES):
                        ix, iy, iz = world_to_voxel([wx, dy_mm, wz])
                        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                            if grid[ix, iy, iz]:
                                grid[ix, iy, iz] = False
                                n_carved += 1

    print(f" {n_carved} voxels removed")

    # Extract mesh from modified voxel grid
    print(f"  Extracting mesh...", end="", flush=True)
    from skimage.measure import marching_cubes
    verts_mc, faces_mc, _, _ = marching_cubes(grid.astype(float), level=0.5)
    # Apply the voxel transform to get back to world coords
    ones = np.ones((len(verts_mc), 1))
    h_verts = np.hstack([verts_mc, ones])
    world_verts = (transform @ h_verts.T).T[:, :3]
    result = trimesh.Trimesh(vertices=world_verts, faces=faces_mc)
    trimesh.repair.fix_normals(result)
    print(f" {len(result.faces)/1e6:.1f}M faces, vol={result.is_volume}")

    # Save
    print(f"  Saving {OUTPUT}...")
    result.export(OUTPUT)
    print("Done.")


if __name__ == "__main__":
    main()
