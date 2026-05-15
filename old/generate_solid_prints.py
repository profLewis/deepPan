#!/usr/bin/env python3
"""
Generate solid (watertight manifold) versions of pads and mounts for 3D printing.

Two-stage approach:
  1. Boolean-union disjoint bodies (exact geometry).
  2. If that doesn't yield a volume, voxelize at 0.1mm and remesh.

Output files have _solid suffix in solid/ subdirectories.
"""

import trimesh
import numpy as np
from pathlib import Path


VOXEL_PITCH = 0.1  # mm — high resolution for printing fidelity


def make_solid(mesh, label=""):
    """Repair a mesh into a watertight solid suitable for printing."""
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_winding(mesh)

    if not mesh.is_volume:
        bodies = mesh.split(only_watertight=False)
        if len(bodies) > 1:
            print(f"    {label}: {len(bodies)} bodies — boolean union...", end="", flush=True)
            result = bodies[0]
            trimesh.repair.fix_normals(result)
            trimesh.repair.fill_holes(result)
            for b in bodies[1:]:
                trimesh.repair.fix_normals(b)
                trimesh.repair.fill_holes(b)
                try:
                    result = result.union(b, engine='manifold')
                except Exception:
                    result = trimesh.util.concatenate([result, b])
            mesh = result
            print(" done")
        else:
            trimesh.repair.fix_normals(mesh)
            trimesh.repair.fill_holes(mesh)

    # If still not a volume, fall back to voxel remesh then drill
    if not mesh.is_volume:
        print(f"    {label}: voxel remesh at {VOXEL_PITCH}mm...", end="", flush=True)
        vox = mesh.voxelized(VOXEL_PITCH)
        mesh = vox.marching_cubes
        mesh.apply_transform(vox.transform)  # map voxel indices back to world coords
        trimesh.repair.fix_normals(mesh)
        print(" done")
        # Drill M2 holes AFTER voxelizing (mesh is now a proper volume)
        mesh = _drill_m2_holes(mesh, label)

    return mesh


def _drill_m2_holes(mesh, label):
    """Drill M2 clearance holes through push cap or mount base assemblies."""

    M2_R = 1.1  # M2 clearance = 2.2mm diameter
    holes = []  # list of (x, y, z_mid, length) for each drill

    if 'push_cap' in label or 'central_push' in label:
        try:
            from generate_central_mount import (
                PLATE_X_OFFSET, PLATE_Y_OFFSET, PLATE_THICKNESS,
                PCB_HOLE_GRID, PCB_BOSS_HEIGHT, PLATE_Z_TOP, PLATE_HEIGHT,
            )
        except ImportError:
            return mesh
        ht = PLATE_THICKNESS / 2
        yo = PLATE_Y_OFFSET
        half_grid = PCB_HOLE_GRID / 2
        xo = PLATE_X_OFFSET
        z_center = PLATE_Z_TOP - PLATE_HEIGHT / 2
        y_top = ht + yo + PCB_BOSS_HEIGHT + 2.0
        y_bot = -ht + yo - 2.0
        bore_len = y_top - y_bot
        bore_mid = (y_top + y_bot) / 2
        for hx, hz in [(-half_grid + xo, z_center - half_grid),
                        (+half_grid + xo, z_center - half_grid),
                        (+half_grid + xo, z_center + half_grid),
                        (-half_grid + xo, z_center + half_grid)]:
            # Drill along Y axis (R90 rotated)
            holes.append(('Y', hx, bore_mid, hz, bore_len))

    elif 'mount_base' in label:
        try:
            from generate_mount_base import (
                PCB_HOLE_GRID, PCB_HOLE_DIAMETER, BASE_HEIGHT, PCB_BOSS_HEIGHT,
            )
        except ImportError:
            return mesh
        half_grid = PCB_HOLE_GRID / 2
        z_bot = -BASE_HEIGHT - PCB_BOSS_HEIGHT - 1.0
        z_top = 1.0
        bore_len = z_top - z_bot
        bore_mid = (z_top + z_bot) / 2
        for cx, cy in [(-half_grid, -half_grid), (half_grid, -half_grid),
                        (half_grid, half_grid), (-half_grid, half_grid)]:
            # Drill along Z axis (no rotation needed)
            holes.append(('Z', cx, cy, bore_mid, bore_len))

    else:
        return mesh

    if not holes:
        return mesh

    print(f"    {label}: drilling {len(holes)} M2 holes...", end="", flush=True)
    for h in holes:
        axis, *coords_and_len = h
        if axis == 'Y':
            hx, ym, hz, blen = coords_and_len
            drill = trimesh.creation.cylinder(radius=M2_R, height=blen, sections=48)
            R90 = np.eye(4)
            R90[1, 1] = 0; R90[1, 2] = -1
            R90[2, 1] = 1; R90[2, 2] = 0
            drill.apply_transform(R90)
            T = np.eye(4)
            T[0, 3] = hx; T[1, 3] = ym; T[2, 3] = hz
            drill.apply_transform(T)
        else:  # Z axis
            cx, cy, zm, blen = coords_and_len
            drill = trimesh.creation.cylinder(radius=M2_R, height=blen, sections=48)
            T = trimesh.transformations.translation_matrix([cx, cy, zm])
            drill.apply_transform(T)
        try:
            mesh = mesh.difference(drill, engine='manifold')
        except Exception:
            pass
    print(" done")
    return mesh


def process_file(src_path, dst_path):
    """Load STL, make solid, save."""
    try:
        mesh = trimesh.load(str(src_path))
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    except Exception as e:
        print(f"  SKIP {src_path.name}: {e}")
        return False

    label = src_path.stem
    n_before = len(mesh.faces)

    mesh = make_solid(mesh, label)

    is_volume = mesh.is_volume
    status = "OK" if is_volume else "WARN"
    print(f"  {label}: {n_before}→{len(mesh.faces)}f  vol={is_volume}  {status}")

    mesh.export(str(dst_path))
    return True


def main():
    print("=" * 60)
    print("Generating solid meshes for 3D printing")
    print(f"  Voxel fallback pitch: {VOXEL_PITCH}mm")
    print("=" * 60)

    # ── Pads ──
    pad_dir = Path("data/notepads")
    pad_stls = sorted(pad_dir.glob("notepad_[CIO]*.stl"))
    pad_stls = [p for p in pad_stls if '_centered' not in p.stem]

    print(f"\nPads ({len(pad_stls)} files):")
    solid_dir = pad_dir / "solid"
    solid_dir.mkdir(exist_ok=True)
    for stl in pad_stls:
        dst = solid_dir / stl.name.replace(".stl", "_solid.stl")
        process_file(stl, dst)

    # ── Mounts ──
    mount_dir = Path("data/mounts")
    mount_stls = sorted(mount_dir.glob("*.stl"))

    print(f"\nMounts ({len(mount_stls)} files):")
    mount_solid_dir = mount_dir / "solid"
    mount_solid_dir.mkdir(exist_ok=True)
    for stl in mount_stls:
        dst = mount_solid_dir / stl.name.replace(".stl", "_solid.stl")
        process_file(stl, dst)

    # ── Screw cap ──
    cap_path = pad_dir / "cap.stl"
    if cap_path.exists():
        print(f"\nScrew cap:")
        dst = solid_dir / "cap_solid.stl"
        process_file(cap_path, dst)

    print(f"\nSolid files written to:")
    print(f"  Pads:   {solid_dir}/")
    print(f"  Mounts: {mount_solid_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
