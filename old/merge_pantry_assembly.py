#!/usr/bin/env python3
"""
Merge pantry.stl (pan surface) with assembly_view_solid.stl (hardware).

The pantry surface becomes a thin shell (thickened inward by PAN_THICKNESS).
The assembly solid objects are voxelized and unioned with the shell.
Result: the pan surface with hardware protruding inward from it.

Usage:
    python merge_pantry_assembly.py [--res=0.5] [--sigma=0.5] [--thickness=5]
"""

import numpy as np
import struct
import gc
import sys
import math
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import marching_cubes


def read_stl_verts(path):
    """Read binary STL, return all triangle vertices as Nx3."""
    with open(path, 'rb') as f:
        f.read(80)
        n_tri = struct.unpack('<I', f.read(4))[0]
        verts = np.empty((n_tri * 3, 3), dtype=np.float32)
        for i in range(n_tri):
            data = struct.unpack('<12fH', f.read(50))
            for vi in range(3):
                verts[i*3 + vi] = data[3 + vi*3: 6 + vi*3]
    return verts, n_tri


def write_stl(path, verts, faces):
    with open(path, 'wb') as f:
        f.write(b'Merged pantry+assembly'.ljust(80, b'\0'))
        f.write(struct.pack('<I', len(faces)))
        for tri in faces:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            n = np.cross(v1 - v0, v2 - v0)
            nl = np.linalg.norm(n)
            if nl > 0:
                n /= nl
            f.write(struct.pack('<3f', *n))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))


def main():
    res = 0.5
    sigma = 0.5
    y_factor = 4
    thickness = 5.0  # shell thickness (mm)
    for arg in sys.argv[1:]:
        if arg.startswith('--res='):
            res = float(arg.split('=')[1])
        elif arg.startswith('--sigma='):
            sigma = float(arg.split('=')[1])
        elif arg.startswith('--thickness='):
            thickness = float(arg.split('=')[1])
        elif arg.startswith('--y-factor='):
            y_factor = int(arg.split('=')[1])

    y_res = res / y_factor

    print("=" * 60)
    print(f"Merge pantry + assembly (res={res}mm, Y={y_res}mm, "
          f"shell={thickness}mm)")
    print("=" * 60)

    # Load pantry surface
    print("\nLoading pantry.stl...")
    pan_v, pan_ntri = read_stl_verts('data/pantry.stl')
    pan_r = np.sqrt(pan_v[:, 0]**2 + pan_v[:, 2]**2)
    drum_r = float(pan_r.max())
    print(f"  {pan_ntri} tri, R=[0,{drum_r:.1f}], "
          f"Y=[{pan_v[:,1].min():.1f},{pan_v[:,1].max():.1f}]")

    # Load assembly
    print("Loading assembly_view_solid.stl...")
    asm_v, asm_ntri = read_stl_verts('data/notepads/assembly_view_solid.stl')
    print(f"  {asm_ntri} tri, Y=[{asm_v[:,1].min():.1f},{asm_v[:,1].max():.1f}]")

    # Grid bounds
    y_min = min(float(pan_v[:, 1].min()), float(asm_v[:, 1].min())) - 5
    y_max = float(pan_v[:, 1].max())
    margin = res * 2
    x_range = np.arange(-drum_r - margin, drum_r + margin + res, res)
    z_range = np.arange(-drum_r - margin, drum_r + margin + res, res)
    y_range = np.arange(y_min - margin, y_max + margin + y_res, y_res)
    nx, ny, nz = len(x_range), len(y_range), len(z_range)
    print(f"\nGrid: {nx}x{ny}x{nz} = {nx*ny*nz/1e6:.0f}M voxels")

    # Compute pantry surface heightmap
    print("Computing pantry heightmap...")
    XI, ZI = np.meshgrid(x_range, z_range, indexing='ij')
    RR = np.sqrt(XI**2 + ZI**2)
    inside_drum = RR <= drum_r

    # Deduplicate pantry vertices for griddata
    pan_unique, pan_inv = np.unique(
        np.round(pan_v[:, [0, 2]], 3), axis=0, return_inverse=True)
    # For duplicate XZ positions, take max Y (the surface)
    pan_y_unique = np.full(len(pan_unique), -1e9)
    for i, yi in enumerate(pan_inv):
        if pan_v[i, 1] > pan_y_unique[yi]:
            pan_y_unique[yi] = pan_v[i, 1]

    height_map = griddata(pan_unique, pan_y_unique, (XI, ZI), method='linear')
    height_map[np.isnan(height_map) & inside_drum] = y_max
    del XI, ZI, pan_unique, pan_y_unique

    # Shell: fill from (surface - thickness) to surface per column
    print(f"Filling shell ({thickness}mm thick)...")
    grid = np.zeros((nx, ny, nz), dtype=np.uint8)
    for ix in range(nx):
        for iz in range(nz):
            if not inside_drum[ix, iz]:
                continue
            h = height_map[ix, iz]
            if np.isnan(h):
                continue
            iy_top = min(ny, int((h - y_range[0]) / y_res) + 1)
            iy_bot = max(0, int((h - thickness - y_range[0]) / y_res))
            if iy_top > iy_bot:
                grid[ix, iy_bot:iy_top, iz] = 1
    n_shell = int(grid.sum())
    print(f"  Shell: {n_shell/1e6:.1f}M voxels")
    del height_map, inside_drum, RR

    # Voxelize assembly solid objects into the grid
    # For each assembly triangle vertex, mark the voxel
    # Use a filled approach: for each XZ column that has assembly vertices,
    # fill from min_Y to max_Y of assembly at that column
    print("Voxelizing assembly into grid...")

    # Bin assembly vertices into XZ columns, track Y range per column
    asm_col_ymin = np.full((nx, nz), 1e9)
    asm_col_ymax = np.full((nx, nz), -1e9)
    for vi in range(len(asm_v)):
        ix = int((asm_v[vi, 0] - x_range[0]) / res + 0.5)
        iz = int((asm_v[vi, 2] - z_range[0]) / res + 0.5)
        if 0 <= ix < nx and 0 <= iz < nz:
            y_val = asm_v[vi, 1]
            if y_val < asm_col_ymin[ix, iz]:
                asm_col_ymin[ix, iz] = y_val
            if y_val > asm_col_ymax[ix, iz]:
                asm_col_ymax[ix, iz] = y_val
    # Spread to neighbors for coverage
    from scipy.ndimage import minimum_filter, maximum_filter
    asm_col_ymin_spread = minimum_filter(asm_col_ymin, size=3)
    asm_col_ymax_spread = maximum_filter(asm_col_ymax, size=3)

    n_asm = 0
    for ix in range(nx):
        for iz in range(nz):
            if asm_col_ymin_spread[ix, iz] > 1e8:
                continue
            iy_bot = max(0, int((asm_col_ymin_spread[ix, iz] - y_range[0]) / y_res))
            iy_top = min(ny, int((asm_col_ymax_spread[ix, iz] - y_range[0]) / y_res) + 1)
            if iy_top > iy_bot:
                before = int(grid[ix, iy_bot:iy_top, iz].sum())
                grid[ix, iy_bot:iy_top, iz] = 1
                n_asm += (iy_top - iy_bot) - before
    del asm_col_ymin, asm_col_ymax, asm_col_ymin_spread, asm_col_ymax_spread
    print(f"  Assembly: {n_asm/1e6:.1f}M voxels added")
    print(f"  Total: {int(grid.sum())/1e6:.1f}M voxels")

    # Extract via SDF
    print("\nSDF extraction...")
    spacing = (res, y_res, res)
    padded = np.pad(grid, 1, mode='constant', constant_values=0)
    del grid; gc.collect()

    print("  Interior EDT...")
    pb = padded.view(np.bool_)
    sdf = distance_transform_edt(pb, sampling=spacing).astype(np.float32)
    print("  Exterior EDT...")
    np.logical_not(pb, out=pb)
    sdf -= distance_transform_edt(pb, sampling=spacing).astype(np.float32)
    del padded; gc.collect()

    if sigma > 0:
        print(f"  Gaussian sigma={sigma}...")
        gaussian_filter(sdf, sigma=sigma, output=sdf)

    print("  Marching cubes...")
    verts, faces, _, _ = marching_cubes(sdf, level=0.0, spacing=spacing)
    del sdf; gc.collect()

    verts[:, 0] += x_range[0] - res
    verts[:, 1] += y_range[0] - y_res
    verts[:, 2] += z_range[0] - res
    print(f"  {len(verts)}v, {len(faces)}f")

    # Export
    out_stl = 'data/pantry_merged.stl'
    print(f"\nWriting {out_stl}...")
    write_stl(out_stl, verts, faces)

    out_obj = 'data/pantry_merged.obj'
    with open(out_obj, 'w') as f:
        f.write("# Pantry + assembly merged\n# Units: mm\n\n")
        f.write("o PantryMerged\n")
        for v in verts:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        f.write("\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"  {out_obj}")
    print("\nDone!")


if __name__ == "__main__":
    main()
