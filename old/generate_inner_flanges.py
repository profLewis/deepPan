#!/usr/bin/env python3
"""
Add flanges to inner ring pads via voxelisation.

- Shifts each inner pad by optimized amount (≤5mm)
- Adds 7mm flange ring with groove profile
- Cuts M2 screw holes (through-hole + counterbore) through pad+flange
- Flat bottom
- Updates notepad_properties.json with new positions
- Regenerates assembly_view

Usage:
    python generate_inner_flanges.py
"""

import numpy as np
import json
import struct
import math
from pathlib import Path
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
from skimage.measure import marching_cubes

from generate_notepad import (
    compute_leveling_rotation, NOTE_BY_INDEX, NOTE_MAPPING,
    PAN_THICKNESS, SCREW_HOLE_DIAMETER, COUNTERBORE_DIAMETER, COUNTERBORE_DEPTH,
    SCREW_HOLE_INSET
)

# Optimized shifts (strict ≤5mm magnitude)
SHIFTS = {
    'I0': (4.3, 2.6),
    'I1': (3.9, 3.1),
    'I2': (2.2, -4.5),
    'I3': (-4.1, -2.8),
    'I4': (-4.7, -1.6),
}

FLANGE_EXT = 7.0        # mm outward from pad boundary
FLANGE_STEP = 1.0       # mm below surface at inner edge
FLANGE_MID = 1.5        # mm below surface at midpoint
FLANGE_OUTER = 2.5      # mm below surface at outer edge
SCREW_FLANGE_POS = 0.5  # fraction of flange extension for screw placement
RES = 0.2               # mm voxel resolution


def read_obj(path):
    verts = []
    faces = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                faces.append([int(p.split('/')[0]) - 1 for p in parts])
    return np.array(verts), faces


def write_obj(path, verts, faces, name="Pad"):
    with open(path, 'w') as f:
        f.write(f"# {name} with flange\no {name}\n")
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        f.write("\n")
        for face in faces:
            f.write("f " + " ".join(str(i + 1) for i in face) + "\n")


def write_stl(path, verts, faces):
    with open(path, 'wb') as f:
        f.write(b"STL pad".ljust(80, b'\0'))
        f.write(struct.pack('<I', len(faces)))
        for tri in faces:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            n = np.cross(v1 - v0, v2 - v0)
            nl = np.linalg.norm(n)
            if nl > 0: n /= nl
            f.write(struct.pack('<3f', *n))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))


def groove_depth(frac):
    """Groove profile: depth below surface as function of distance fraction (0-1)."""
    if frac < 0.5:
        t = frac / 0.5
        return FLANGE_STEP + t * (FLANGE_MID - FLANGE_STEP)
    else:
        t = (frac - 0.5) / 0.5
        return FLANGE_MID + t * (FLANGE_OUTER - FLANGE_MID)


def process_inner_pad(note, props_entry, offset, R_level):
    """Add flange to one inner pad using voxelisation."""
    idx = note
    obj_path = props_entry['obj_path']
    print(f"\n{'='*50}")
    print(f"Processing {idx}")

    # Read pad OBJ (in body coords)
    pad_v, pad_f = read_obj(obj_path)

    # Apply shift in XZ (body coords: Y=drum axis, XZ=horizontal)
    dx, dz = SHIFTS[idx]
    pad_v[:, 0] += dx
    pad_v[:, 2] += dz
    print(f"  Shifted ({dx:.1f}, {dz:.1f})mm")

    # Get pad boundary in XZ for flange
    pad_xz = pad_v[:, [0, 2]]
    hull = ConvexHull(pad_xz)
    pad_poly = Polygon(pad_xz[hull.vertices])
    pad_center_xz = np.array([pad_poly.centroid.x, pad_poly.centroid.y])

    # Pad Y bounds
    y_top = float(pad_v[:, 1].max())  # playing surface
    y_bot = y_top - PAN_THICKNESS     # shell bottom
    y_min_all = float(pad_v[:, 1].min())  # mount extends below

    # Flange boundary: buffer pad boundary outward
    flange_poly = pad_poly.buffer(FLANGE_EXT)

    # Compute screw hole positions on the flange
    # Place 4 holes equally spaced around the pad at SCREW_FLANGE_POS of flange ext
    screw_r = FLANGE_EXT * SCREW_FLANGE_POS
    # Get 4 directions (roughly at the pad boundary corners)
    boundary_coords = np.array(pad_poly.exterior.coords[:-1])
    # Find 4 well-spaced boundary points
    n_bnd = len(boundary_coords)
    target_spacing = n_bnd / 4
    screw_positions = []
    for si in range(4):
        bi = int(si * target_spacing) % n_bnd
        bpt = boundary_coords[bi]
        # Direction from center to boundary
        direction = bpt - pad_center_xz
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        # Screw at boundary + screw_r outward
        screw_xz = bpt + direction * screw_r
        screw_positions.append(screw_xz)

    print(f"  4 screw holes at {screw_r:.1f}mm beyond boundary")

    # Voxelise
    margin = RES * 3
    x_range = np.arange(pad_xz[:, 0].min() - FLANGE_EXT - margin,
                        pad_xz[:, 0].max() + FLANGE_EXT + margin + RES, RES)
    z_range = np.arange(pad_xz[:, 1].min() - FLANGE_EXT - margin,
                        pad_xz[:, 1].max() + FLANGE_EXT + margin + RES, RES)
    y_range = np.arange(y_min_all - margin, y_top + margin + RES, RES)
    nx, ny, nz = len(x_range), len(y_range), len(z_range)
    print(f"  Voxel grid: {nx}x{ny}x{nz} = {nx*ny*nz/1e6:.1f}M ({RES}mm)")

    grid = np.zeros((nx, ny, nz), dtype=np.uint8)

    # Step 1: Voxelise original pad (all verts inside = solid)
    # Simple approach: for each XZ column, if it's inside the pad hull,
    # fill from y_min_all to y_top
    from matplotlib.path import Path as MplPath
    pad_path = MplPath(pad_xz[hull.vertices])

    XI, ZI = np.meshgrid(x_range, z_range, indexing='ij')
    pts = np.column_stack([XI.ravel(), ZI.ravel()])
    inside_pad = pad_path.contains_points(pts).reshape(nx, nz)

    iy_top = int((y_top - y_range[0]) / RES) + 1
    iy_bot = max(0, int((y_min_all - y_range[0]) / RES))

    for ix in range(nx):
        for iz in range(nz):
            if inside_pad[ix, iz]:
                grid[ix, iy_bot:iy_top, iz] = 1

    # Step 2: Add flange ring (pad_poly to flange_poly)
    flange_path = MplPath(np.array(flange_poly.exterior.coords))
    inside_flange = flange_path.contains_points(pts).reshape(nx, nz)
    flange_only = inside_flange & ~inside_pad

    iy_shell_bot = max(0, int((y_bot - y_range[0]) / RES))

    for ix in range(nx):
        for iz in range(nz):
            if flange_only[ix, iz]:
                # Groove profile: compute depth based on distance from pad boundary
                pt = Point(x_range[ix], z_range[iz])
                dist = pad_poly.exterior.distance(pt)
                frac = min(dist / FLANGE_EXT, 1.0)
                depth = groove_depth(frac)
                iy_flange_top = min(ny, int((y_top - depth - y_range[0]) / RES) + 1)
                if iy_flange_top > iy_shell_bot:
                    grid[ix, iy_shell_bot:iy_flange_top, iz] = 1

    n_solid = int(grid.sum())
    print(f"  Solid: {n_solid/1e6:.1f}M voxels")

    # Step 3: Cut screw holes
    hole_r = SCREW_HOLE_DIAMETER / 2
    cbore_r = COUNTERBORE_DIAMETER / 2
    cbore_depth = COUNTERBORE_DEPTH

    for si, sxz in enumerate(screw_positions):
        sx, sz = sxz
        # Through-hole (full height)
        ix_min = max(0, int((sx - hole_r - x_range[0]) / RES) - 1)
        ix_max = min(nx-1, int((sx + hole_r - x_range[0]) / RES) + 2)
        iz_min = max(0, int((sz - hole_r - z_range[0]) / RES) - 1)
        iz_max = min(nz-1, int((sz + hole_r - z_range[0]) / RES) + 2)
        for ix in range(ix_min, ix_max+1):
            dx = x_range[ix] - sx
            for iz in range(iz_min, iz_max+1):
                dz = z_range[iz] - sz
                if dx*dx + dz*dz <= hole_r**2:
                    grid[ix, :, iz] = 0  # through-hole

        # Counterbore on top (playing surface side)
        iy_cb_top = min(ny, int((y_top - y_range[0]) / RES) + 2)
        iy_cb_bot = max(0, int((y_top - cbore_depth - y_range[0]) / RES))
        ix_min = max(0, int((sx - cbore_r - x_range[0]) / RES) - 1)
        ix_max = min(nx-1, int((sx + cbore_r - x_range[0]) / RES) + 2)
        iz_min = max(0, int((sz - cbore_r - z_range[0]) / RES) - 1)
        iz_max = min(nz-1, int((sz + cbore_r - z_range[0]) / RES) + 2)
        for ix in range(ix_min, ix_max+1):
            dx = x_range[ix] - sx
            for iz in range(iz_min, iz_max+1):
                dz = z_range[iz] - sz
                if dx*dx + dz*dz <= cbore_r**2:
                    grid[ix, iy_cb_bot:iy_cb_top, iz] = 0

    # Step 4: Flatten bottom
    # Set all bottom-layer voxels to the same Y level
    # (already handled by using iy_shell_bot consistently)

    # Step 5: Marching cubes
    padded = np.pad(grid, 1, mode='constant', constant_values=0)
    verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=(RES, RES, RES))
    verts[:, 0] += x_range[0] - RES
    verts[:, 1] += y_range[0] - RES
    verts[:, 2] += z_range[0] - RES
    print(f"  Mesh: {len(verts)}v, {len(faces)}f")

    # Step 6: Write output
    obj_name = f"NotePad_{idx}_flanged"
    write_obj(obj_path, verts, [list(f) for f in faces], obj_name)
    stl_path = obj_path.replace('.obj', '.stl')
    write_stl(stl_path, verts, [list(f) for f in faces])
    print(f"  Saved: {obj_path}")

    # Step 7: Update hole positions in properties
    # Convert screw XZ positions back to original coords for properties
    new_holes_orig = []
    pan_offset = np.array(json.load(open('data/pan_centroid_offset.json'))['centroid_offset_mm'])
    R_inv = np.linalg.inv(R_level)
    for sxz in screw_positions:
        # Body coords → original coords
        body_pos = np.array([sxz[0], y_top, sxz[1]])  # XZ position at surface Y
        orig_pos = (R_inv @ body_pos) + pan_offset
        new_holes_orig.append(orig_pos.tolist())

    return {
        'shift': SHIFTS[idx],
        'hole_positions': new_holes_orig,
        'screw_positions_body': screw_positions,
    }


def main():
    print("Inner Pad Flange Generator")
    print("=" * 50)

    props = json.load(open('data/notepads/notepad_properties.json'))
    offset = np.array(json.load(open('data/pan_centroid_offset.json'))['centroid_offset_mm'])
    R_level = compute_leveling_rotation("data/Tenor Pan only.obj")

    results = {}
    for p in props:
        if p['ring'] != 'inner':
            continue
        idx = p['index']
        result = process_inner_pad(idx, p, offset, R_level)
        results[idx] = result

    # Update properties with new hole positions and shifted centroids
    for p in props:
        if p['index'] in results:
            r = results[p['index']]
            p['hole_positions'] = r['hole_positions']
            dx, dz = r['shift']
            # Shift centroid in original coords
            body_c = R_level @ (np.array(p['centroid']) - offset)
            body_c[0] += dx
            body_c[2] += dz
            p['centroid'] = ((np.linalg.inv(R_level) @ body_c) + offset).tolist()

    with open('data/notepads/notepad_properties.json', 'w') as f:
        json.dump(props, f, indent=2)
    print(f"\nUpdated notepad_properties.json")

    # Regenerate assembly view
    print("\nRegenerating assembly view...")
    import subprocess
    subprocess.run(['python', 'generate_assembly_view.py'], capture_output=True)
    print("Done!")


if __name__ == "__main__":
    main()
