#!/usr/bin/env python3
"""
Carve assembly_view pad/mount/sleeve objects from pantry_thick.stl.

For each Pad_* object in assembly_view.obj, voxelizes a LOCAL crop of
pantry_thick at high resolution, subtracts the pad's volume, and
re-extracts the mesh. Writes intermediate results after each pad.

No full-grid voxelization — each pad is processed independently in its
own small bounding box at very fine resolution (0.1mm default).

Usage:
    python carve_pads.py [--res=0.1] [--sigma=0.3]
"""

import numpy as np
import struct
import gc
import sys
import math
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import marching_cubes
from collections import defaultdict


def read_stl_tris(path):
    """Read binary STL → list of triangles [(v0,v1,v2), ...]."""
    with open(path, 'rb') as f:
        f.read(80)
        n_tri = struct.unpack('<I', f.read(4))[0]
        tris = []
        for _ in range(n_tri):
            data = struct.unpack('<12fH', f.read(50))
            v0 = np.array(data[3:6])
            v1 = np.array(data[6:9])
            v2 = np.array(data[9:12])
            tris.append((v0, v1, v2))
    return tris


def write_stl(path, tris):
    """Write binary STL from list of (v0,v1,v2) triangles."""
    with open(path, 'wb') as f:
        f.write(path.encode()[:80].ljust(80, b'\0'))
        f.write(struct.pack('<I', len(tris)))
        for v0, v1, v2 in tris:
            n = np.cross(v1 - v0, v2 - v0)
            nl = np.linalg.norm(n)
            if nl > 0:
                n /= nl
            f.write(struct.pack('<3f', *n))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))


def read_obj_objects(path):
    """Read OBJ with named objects → dict of name → (verts Nx3, faces Mx3+)."""
    all_verts = []
    objects = {}
    current = None
    current_faces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                p = line.split()
                all_verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith('o '):
                if current and current_faces:
                    objects[current] = current_faces
                current = line[2:].strip()
                current_faces = []
            elif line.startswith('f ') and current is not None:
                indices = [int(tok.split('/')[0]) - 1 for tok in line.split()[1:]]
                current_faces.append(indices)
    if current and current_faces:
        objects[current] = current_faces
    all_verts = np.array(all_verts)
    return all_verts, objects


def voxelize_tris_in_box(tris_list, x_range, y_range, z_range):
    """Fill voxels INSIDE a set of triangles using column-based ray casting.

    For each XZ column, cast a ray in +Y and count crossings to determine
    inside/outside. Fill voxels that are inside the closed surface.
    """
    res_x = x_range[1] - x_range[0]
    res_y = y_range[1] - y_range[0]
    res_z = z_range[1] - z_range[0]
    nx, ny, nz = len(x_range), len(y_range), len(z_range)
    grid = np.zeros((nx, ny, nz), dtype=np.uint8)

    # For each triangle, rasterize into the XZ columns and record Y crossings
    # Use a simpler approach: for each XZ column, find min/max Y of nearby
    # triangles and fill between them
    # Actually use a heightmap approach: top Y and bottom Y per column
    col_ymax = np.full((nx, nz), -1e9)
    col_ymin = np.full((nx, nz), 1e9)

    for v0, v1, v2 in tris_list:
        # Bounding box of triangle in grid coords
        xs = [v0[0], v1[0], v2[0]]
        zs = [v0[2], v1[2], v2[2]]
        ys = [v0[1], v1[1], v2[1]]
        ix_lo = max(0, int((min(xs) - x_range[0]) / res_x) - 1)
        ix_hi = min(nx - 1, int((max(xs) - x_range[0]) / res_x) + 1)
        iz_lo = max(0, int((min(zs) - z_range[0]) / res_z) - 1)
        iz_hi = min(nz - 1, int((max(zs) - z_range[0]) / res_z) + 1)
        y_lo = min(ys)
        y_hi = max(ys)
        for ix in range(ix_lo, ix_hi + 1):
            for iz in range(iz_lo, iz_hi + 1):
                if y_hi > col_ymax[ix, iz]:
                    col_ymax[ix, iz] = y_hi
                if y_lo < col_ymin[ix, iz]:
                    col_ymin[ix, iz] = y_lo

    # Fill columns between ymin and ymax
    for ix in range(nx):
        for iz in range(nz):
            if col_ymax[ix, iz] > -1e8:
                iy_lo = max(0, int((col_ymin[ix, iz] - y_range[0]) / res_y))
                iy_hi = min(ny, int((col_ymax[ix, iz] - y_range[0]) / res_y) + 1)
                if iy_hi > iy_lo:
                    grid[ix, iy_lo:iy_hi, iz] = 1
    return grid


def tris_to_list(verts, faces):
    """Convert indexed mesh to triangle list."""
    tris = []
    for face in faces:
        # Triangulate
        for i in range(1, len(face) - 1):
            tris.append((verts[face[0]].copy(),
                         verts[face[i]].copy(),
                         verts[face[i+1]].copy()))
    return tris


def main():
    res = 0.1
    sigma = 0.3
    for arg in sys.argv[1:]:
        if arg.startswith('--res='):
            res = float(arg.split('=')[1])
        elif arg.startswith('--sigma='):
            sigma = float(arg.split('=')[1])

    print(f"Carve pads from pantry_thick.stl (res={res}mm, sigma={sigma})")
    print("=" * 60)

    # Load pantry_thick as triangle list
    print("\nLoading pantry_thick.stl...")
    pantry_tris = read_stl_tris('data/pantry_thick.stl')
    print(f"  {len(pantry_tris)} triangles")

    # Load assembly view objects
    print("Loading assembly_view.obj...")
    asm_verts, asm_objects = read_obj_objects('data/notepads/assembly_view.obj')
    pad_names = sorted([n for n in asm_objects if n.startswith('Pad_')])
    mount_names = sorted([n for n in asm_objects if n.startswith('MountBase_')])
    sleeve_names = sorted([n for n in asm_objects if n.startswith('Sleeve_')])
    print(f"  {len(pad_names)} pads, {len(mount_names)} mounts, {len(sleeve_names)} sleeves")

    # For each pad, also include its mount and sleeve
    pad_groups = {}
    for pname in pad_names:
        note = pname.replace('Pad_', '')
        group_names = [pname]
        mname = f'MountBase_{note}'
        sname = f'Sleeve_{note}'
        if mname in asm_objects:
            group_names.append(mname)
        if sname in asm_objects:
            group_names.append(sname)
        pad_groups[note] = group_names

    # Process each pad
    current_tris = list(pantry_tris)  # working copy

    for pi, (note, group_names) in enumerate(sorted(pad_groups.items())):
        print(f"\n[{pi+1}/{len(pad_groups)}] Carving {note} ({', '.join(group_names)})...")

        # Collect all triangles for this pad group
        pad_tris = []
        for gname in group_names:
            faces = asm_objects[gname]
            pad_tris.extend(tris_to_list(asm_verts, faces))
        if not pad_tris:
            print("  No geometry, skipping")
            continue

        # Bounding box of the pad group (with generous margin so pantry
        # mesh around the pad is fully captured)
        pad_margin = 5.0  # mm
        all_pts = np.array([p for tri in pad_tris for p in tri])
        bb_min = all_pts.min(axis=0) - pad_margin
        bb_max = all_pts.max(axis=0) + pad_margin
        print(f"  BBox: X=[{bb_min[0]:.1f},{bb_max[0]:.1f}] "
              f"Y=[{bb_min[1]:.1f},{bb_max[1]:.1f}] "
              f"Z=[{bb_min[2]:.1f},{bb_max[2]:.1f}]")

        # Local grid for this pad
        x_range = np.arange(bb_min[0], bb_max[0] + res, res)
        y_range = np.arange(bb_min[1], bb_max[1] + res, res)
        z_range = np.arange(bb_min[2], bb_max[2] + res, res)
        lnx, lny, lnz = len(x_range), len(y_range), len(z_range)
        print(f"  Local grid: {lnx}x{lny}x{lnz} = {lnx*lny*lnz/1e6:.1f}M voxels")

        # Voxelize the pad group
        pad_grid = voxelize_tris_in_box(pad_tris, x_range, y_range, z_range)
        n_pad_vox = int(pad_grid.sum())
        print(f"  Pad voxels: {n_pad_vox/1e3:.0f}K")

        if n_pad_vox == 0:
            print("  Empty pad voxels, skipping")
            continue

        # Voxelize the pantry triangles in the SAME local box
        # Only include pantry triangles that overlap the bounding box
        local_pantry_tris = []
        for v0, v1, v2 in current_tris:
            tri_min = np.minimum(np.minimum(v0, v1), v2)
            tri_max = np.maximum(np.maximum(v0, v1), v2)
            if (tri_max[0] >= bb_min[0] and tri_min[0] <= bb_max[0] and
                tri_max[1] >= bb_min[1] and tri_min[1] <= bb_max[1] and
                tri_max[2] >= bb_min[2] and tri_min[2] <= bb_max[2]):
                local_pantry_tris.append((v0, v1, v2))
        print(f"  Local pantry tris: {len(local_pantry_tris)}")

        pantry_grid = voxelize_tris_in_box(local_pantry_tris, x_range, y_range, z_range)
        n_pantry_vox = int(pantry_grid.sum())
        print(f"  Pantry voxels in box: {n_pantry_vox/1e3:.0f}K")

        # Subtract: remove pad voxels from pantry
        overlap = pantry_grid & pad_grid
        n_removed = int(overlap.sum())
        carved_grid = pantry_grid & ~pad_grid
        n_carved = int(carved_grid.sum())
        print(f"  Overlap: {n_removed/1e3:.0f}K, Remaining: {n_carved/1e3:.0f}K")
        del pantry_grid, pad_grid, overlap; gc.collect()

        if n_carved == 0 or n_removed == 0:
            print("  No change, skipping")
            del carved_grid; gc.collect()
            continue

        # SDF extract the carved local grid
        spacing = (res, res, res)
        padded = np.pad(carved_grid, 1, mode='constant', constant_values=0)
        del carved_grid; gc.collect()

        pb = padded.view(np.bool_)
        sdf = distance_transform_edt(pb, sampling=spacing).astype(np.float32)
        np.logical_not(pb, out=pb)
        sdf -= distance_transform_edt(pb, sampling=spacing).astype(np.float32)
        del padded; gc.collect()
        if sigma > 0:
            gaussian_filter(sdf, sigma=sigma, output=sdf)
        new_verts, new_faces, _, _ = marching_cubes(sdf, level=0.0, spacing=spacing)
        del sdf; gc.collect()

        # Offset to world coordinates
        new_verts[:, 0] += x_range[0] - res
        new_verts[:, 1] += y_range[0] - res
        new_verts[:, 2] += z_range[0] - res

        # Replace pantry triangles in this bounding box with the carved mesh
        # Keep all pantry tris OUTSIDE the box, replace those inside
        kept_tris = []
        for v0, v1, v2 in current_tris:
            centroid = (v0 + v1 + v2) / 3
            if (centroid[0] >= bb_min[0] and centroid[0] <= bb_max[0] and
                centroid[1] >= bb_min[1] and centroid[1] <= bb_max[1] and
                centroid[2] >= bb_min[2] and centroid[2] <= bb_max[2]):
                continue  # remove — replaced by carved version
            kept_tris.append((v0, v1, v2))

        # Add the carved mesh triangles
        for face in new_faces:
            v0 = new_verts[face[0]]
            v1 = new_verts[face[1]]
            v2 = new_verts[face[2]]
            kept_tris.append((v0.copy(), v1.copy(), v2.copy()))

        print(f"  Tris: {len(current_tris)} -> {len(kept_tris)} "
              f"(removed {len(current_tris)-len(kept_tris)+len(new_faces)}, "
              f"added {len(new_faces)})")
        current_tris = kept_tris

        # Write intermediate result
        tmp_path = f'data/pantry_carved_tmp_{pi+1:02d}_{note}.stl'
        write_stl(tmp_path, current_tris)
        print(f"  -> {tmp_path} ({len(current_tris)} tri)")

    # Final output
    out_path = 'data/pantry_carved.stl'
    print(f"\nWriting final: {out_path} ({len(current_tris)} tri)")
    write_stl(out_path, current_tris)
    print("Done!")


if __name__ == "__main__":
    main()
