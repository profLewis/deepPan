#!/usr/bin/env python3
"""
Subtract assembly_view_solid.stl from pantry_thick.stl in one pass.

Voxelizes both meshes at the same resolution, subtracts, extracts via SDF.

Usage:
    python subtract_assembly.py [--res=0.2] [--sigma=0.5]
"""

import numpy as np
import struct
import gc
import sys
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import marching_cubes


def read_stl_verts(path):
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
        f.write(path.encode()[:80].ljust(80, b'\0'))
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


def voxelize_heightmap(v, x_range, y_range, z_range):
    """Voxelize a closed mesh via column min/max Y heightmap."""
    res_x = x_range[1] - x_range[0]
    res_y = y_range[1] - y_range[0]
    res_z = z_range[1] - z_range[0]
    nx, ny, nz = len(x_range), len(y_range), len(z_range)

    col_ymin = np.full((nx, nz), 1e9, dtype=np.float32)
    col_ymax = np.full((nx, nz), -1e9, dtype=np.float32)

    for vi in range(len(v)):
        ix = int((v[vi, 0] - x_range[0]) / res_x + 0.5)
        iz = int((v[vi, 2] - z_range[0]) / res_z + 0.5)
        if 0 <= ix < nx and 0 <= iz < nz:
            y = v[vi, 1]
            if y < col_ymin[ix, iz]:
                col_ymin[ix, iz] = y
            if y > col_ymax[ix, iz]:
                col_ymax[ix, iz] = y

    # Spread to fill gaps (mesh vertices may not cover every voxel column)
    from scipy.ndimage import maximum_filter, minimum_filter
    col_ymax = maximum_filter(col_ymax, size=3)
    col_ymin = minimum_filter(col_ymin, size=3)

    grid = np.zeros((nx, ny, nz), dtype=np.uint8)
    for ix in range(nx):
        for iz in range(nz):
            if col_ymax[ix, iz] > -1e8:
                iy_lo = max(0, int((col_ymin[ix, iz] - y_range[0]) / res_y))
                iy_hi = min(ny, int((col_ymax[ix, iz] - y_range[0]) / res_y) + 1)
                if iy_hi > iy_lo:
                    grid[ix, iy_lo:iy_hi, iz] = 1
    return grid


def main():
    res = 0.2
    sigma = 0.5
    for arg in sys.argv[1:]:
        if arg.startswith('--res='):
            res = float(arg.split('=')[1])
        elif arg.startswith('--sigma='):
            sigma = float(arg.split('=')[1])

    print(f"Subtract assembly from pantry_thick (res={res}mm, sigma={sigma})")
    print("=" * 60)

    # Load meshes
    print("\nLoading pantry_thick.stl...")
    pan_v, pan_n = read_stl_verts('data/pantry_thick.stl')
    print(f"  {pan_n} tri, Y=[{pan_v[:,1].min():.1f},{pan_v[:,1].max():.1f}]")

    print("Loading assembly_view_solid.stl...")
    asm_v, asm_n = read_stl_verts('data/notepads/assembly_view_solid.stl')
    print(f"  {asm_n} tri, Y=[{asm_v[:,1].min():.1f},{asm_v[:,1].max():.1f}]")

    # Grid covering both
    all_v = np.vstack([pan_v, asm_v])
    margin = res * 3
    bb_min = all_v.min(axis=0) - margin
    bb_max = all_v.max(axis=0) + margin
    del all_v

    x_range = np.arange(bb_min[0], bb_max[0] + res, res).astype(np.float32)
    y_range = np.arange(bb_min[1], bb_max[1] + res, res).astype(np.float32)
    z_range = np.arange(bb_min[2], bb_max[2] + res, res).astype(np.float32)
    nx, ny, nz = len(x_range), len(y_range), len(z_range)
    print(f"\nGrid: {nx}x{ny}x{nz} = {nx*ny*nz/1e6:.0f}M voxels")

    # Voxelize pantry
    print("Voxelizing pantry...")
    pan_grid = voxelize_heightmap(pan_v, x_range, y_range, z_range)
    n_pan = int(pan_grid.sum())
    print(f"  {n_pan/1e6:.1f}M voxels")
    del pan_v

    # Voxelize assembly
    print("Voxelizing assembly...")
    asm_grid = voxelize_heightmap(asm_v, x_range, y_range, z_range)
    n_asm = int(asm_grid.sum())
    print(f"  {n_asm/1e6:.1f}M voxels")
    del asm_v

    # Subtract
    overlap = int((pan_grid & asm_grid).sum())
    print(f"\nSubtracting... (overlap: {overlap/1e6:.1f}M)")
    result = pan_grid & ~asm_grid
    n_result = int(result.sum())
    print(f"  Result: {n_result/1e6:.1f}M voxels")
    del pan_grid, asm_grid; gc.collect()

    # SDF extract
    print("\nSDF extraction...")
    spacing = (res, res, res)
    padded = np.pad(result, 1, mode='constant', constant_values=0)
    del result; gc.collect()

    pb = padded.view(np.bool_)
    print("  Interior EDT...")
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
    verts[:, 1] += y_range[0] - res
    verts[:, 2] += z_range[0] - res
    print(f"  {len(verts)}v, {len(faces)}f")

    # Export
    out = 'data/pantry_carved.stl'
    print(f"\nWriting {out}...")
    write_stl(out, verts, faces)

    out_obj = 'data/pantry_carved.obj'
    with open(out_obj, 'w') as f:
        f.write("# Pantry - assembly\n# Units: mm\n\no PantryCarved\n")
        for v in verts:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        f.write("\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"  {len(faces)} tri")
    print("Done!")


if __name__ == "__main__":
    main()
