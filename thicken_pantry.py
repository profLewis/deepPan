#!/usr/bin/env python3
"""
Thicken pantry.stl inward (into drum interior) to create a solid body
that contains assembly_view_solid.stl.

The outer surface matches pantry.stl exactly. Material extends inward
(downward in the bowl) deep enough that all assembly components fit
within the volume.

Usage:
    python thicken_pantry.py [--res=0.5] [--sigma=0.5]
"""

import numpy as np
import struct
import gc
import sys
import math
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import marching_cubes


def read_stl(path):
    """Read binary STL, return (vertices Nx3, triangles Mx3 indices, normals Mx3)."""
    with open(path, 'rb') as f:
        f.read(80)
        n_tri = struct.unpack('<I', f.read(4))[0]
        all_verts = []
        normals = []
        for _ in range(n_tri):
            data = struct.unpack('<12fH', f.read(50))
            normals.append(data[0:3])
            for vi in range(3):
                all_verts.append(data[3 + vi*3: 6 + vi*3])
    verts = np.array(all_verts)
    normals = np.array(normals)
    # Deduplicate vertices
    unique, inverse = np.unique(np.round(verts, 4), axis=0, return_inverse=True)
    tris = inverse.reshape(-1, 3)
    return unique, tris, normals


def write_stl(path, verts, faces):
    with open(path, 'wb') as f:
        f.write(b'Thickened pantry'.ljust(80, b'\0'))
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
    for arg in sys.argv[1:]:
        if arg.startswith('--res='):
            res = float(arg.split('=')[1])
        elif arg.startswith('--sigma='):
            sigma = float(arg.split('=')[1])
        elif arg.startswith('--y-factor='):
            y_factor = int(arg.split('=')[1])

    y_res = res / y_factor

    print("=" * 60)
    print(f"Thicken pantry.stl (res={res}mm, Y={y_res}mm, sigma={sigma})")
    print("=" * 60)

    # Load pantry surface
    print("\nLoading pantry.stl...")
    pan_v, pan_t, pan_n = read_stl('data/pantry.stl')
    print(f"  {len(pan_v)} unique vertices, {len(pan_t)} triangles")
    pan_r = np.sqrt(pan_v[:, 0]**2 + pan_v[:, 2]**2)
    drum_r = float(pan_r.max())
    y_min = float(pan_v[:, 1].min())
    y_max = float(pan_v[:, 1].max())
    print(f"  R: [0, {drum_r:.1f}]mm, Y: [{y_min:.1f}, {y_max:.1f}]mm")

    # Load assembly to find how deep we need to go
    print("\nLoading assembly_view_solid.stl...")
    asm_v, asm_t, _ = read_stl('data/notepads/assembly_view_solid.stl')
    asm_y_min = float(asm_v[:, 1].min())
    asm_y_max = float(asm_v[:, 1].max())
    print(f"  {len(asm_v)} vertices, Y: [{asm_y_min:.1f}, {asm_y_max:.1f}]mm")

    # Bottom of thickened body: enough margin below assembly
    y_bottom = asm_y_min - 10.0  # 10mm margin below deepest assembly point
    print(f"  Thickened body Y range: [{y_bottom:.1f}, {y_max:.1f}]mm")

    # Build voxel grid
    margin = res * 2
    x_range = np.arange(-drum_r - margin, drum_r + margin + res, res)
    z_range = np.arange(-drum_r - margin, drum_r + margin + res, res)
    y_range = np.arange(y_bottom - margin, y_max + margin + y_res, y_res)
    nx, ny, nz = len(x_range), len(y_range), len(z_range)
    print(f"\n  Grid: {nx}x{ny}x{nz} = {nx*ny*nz/1e6:.0f}M voxels")

    # Compute surface heightmap from pantry mesh vertices
    # For each XZ position, the max Y of any nearby surface vertex = the surface
    print("  Computing surface heightmap...")
    XI, ZI = np.meshgrid(x_range, z_range, indexing='ij')
    RR = np.sqrt(XI**2 + ZI**2)
    inside_drum = RR <= drum_r

    # Use griddata to interpolate surface Y from mesh vertices
    height_map = griddata(pan_v[:, [0, 2]], pan_v[:, 1], (XI, ZI), method='linear')
    # NaN inside drum = drum wall area, use max Y there
    height_map[np.isnan(height_map) & inside_drum] = y_max
    col_top = np.full((nx, nz), y_bottom)
    valid = ~np.isnan(height_map) & inside_drum
    col_top[valid] = height_map[valid]
    iy_top = np.clip(((col_top - y_range[0]) / y_res).astype(np.int32) + 1, 0, ny)
    del XI, ZI, height_map, col_top; gc.collect()

    # Fill: solid from y_bottom to surface per column
    print("  Filling solid...")
    grid = np.zeros((nx, ny, nz), dtype=np.uint8)
    for ix in range(nx):
        for iz in range(nz):
            iy_end = iy_top[ix, iz]
            if iy_end > 0 and inside_drum[ix, iz]:
                grid[ix, :iy_end, iz] = 1
    n_solid = int(grid.sum())
    print(f"  {n_solid / 1e6:.0f}M voxels filled")
    del inside_drum, iy_top, RR; gc.collect()

    # Extract mesh via SDF
    print("\nExtracting mesh (SDF)...")
    spacing = (res, y_res, res)
    padded = np.pad(grid, 1, mode='constant', constant_values=0)
    del grid; gc.collect()

    print("  SDF interior...")
    sdf = distance_transform_edt(padded.view(np.bool_), sampling=spacing).astype(np.float32)
    print("  SDF exterior...")
    pb = padded.view(np.bool_)
    np.logical_not(pb, out=pb)
    sdf -= distance_transform_edt(pb, sampling=spacing).astype(np.float32)
    del padded; gc.collect()

    if sigma > 0:
        print(f"  Gaussian sigma={sigma}...")
        gaussian_filter(sdf, sigma=sigma, output=sdf)

    print("  Marching cubes...")
    verts, faces, _, _ = marching_cubes(sdf, level=0.0, spacing=spacing)
    del sdf; gc.collect()

    # Offset to world coordinates
    verts[:, 0] += x_range[0] - res
    verts[:, 1] += y_range[0] - y_res
    verts[:, 2] += z_range[0] - res
    print(f"  {len(verts)} vertices, {len(faces)} triangles")

    # Export
    out_path = 'data/pantry_thickened.stl'
    print(f"\nWriting {out_path}...")
    write_stl(out_path, verts, faces)
    print(f"  {len(faces)} triangles")

    # Also OBJ
    obj_path = 'data/pantry_thickened.obj'
    with open(obj_path, 'w') as f:
        f.write("# Thickened pantry\n# Units: mm\n\no ThickenedPantry\n")
        for v in verts:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        f.write("\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"  {obj_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
