#!/usr/bin/env python3
"""
Generate solid screw-hole cylinder objects for the pan body.

For each screw hole in the pan body (structural + base attachment), creates
a small tube (bushing) mesh: a hollow cylinder with the screw hole in the
centre.  These are exported as a combined OBJ/STL for visualization and
assembly verification.

The hole in each cylinder is already carved into the voxel grid by
generate_pan_body.py — this script just creates the visible tube objects.

Usage:
    python generate_screw_cylinders.py
"""

import numpy as np
import struct
import math
import json

from generate_cylinders import (
    CYL_OUTER_R, CYL_INNER_R,
    N_STRUTS, STRUT_HALF_W,
    STRUT_ANGLES_RAD, WIRE_CYL_ANGLES_RAD,
    SCREW_PILOT_R, SCREW_CLEAR_R, SCREW_DEPTH, SCREW_EDGE_MAX,
    CYL_SCREW_R_INNER, CYL_SCREW_R_OUTER, CYL_SCREW_N,
    STRUT_SCREW_SPACING, STRUT_SCREW_R_START,
)
from generate_notepad import compute_leveling_rotation, NOTE_BY_INDEX
from generate_sector import extract_bowl_surface

# Screw cylinder dimensions
N_SIDES = 16         # polygon sides for cylinder mesh
BUSHING_OUTER_R = 3.0  # mm — outer radius of the bushing tube
# Inner radius = SCREW_PILOT_R (M3 = 1.25mm) or PILOT_R (M2 = 1.0mm)


def make_tube(center_xz, y_bot, y_top, outer_r, inner_r, n_sides=N_SIDES):
    """Create a tube (hollow cylinder) mesh.

    Returns (verts, faces) as numpy arrays.
    Verts are in world coordinates.
    """
    cx, cz = center_xz
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    verts = []
    # Outer ring bottom (0..n-1), outer ring top (n..2n-1)
    for y in [y_bot, y_top]:
        for i in range(n_sides):
            verts.append([cx + outer_r * cos_a[i], y, cz + outer_r * sin_a[i]])
    # Inner ring bottom (2n..3n-1), inner ring top (3n..4n-1)
    for y in [y_bot, y_top]:
        for i in range(n_sides):
            verts.append([cx + inner_r * cos_a[i], y, cz + inner_r * sin_a[i]])

    n = n_sides
    faces = []
    for i in range(n):
        j = (i + 1) % n
        # Outer wall
        faces.append([i, j, j + n])
        faces.append([i, j + n, i + n])
        # Inner wall (reversed winding)
        faces.append([2*n + i, 2*n + i + n, 2*n + j + n])
        faces.append([2*n + i, 2*n + j + n, 2*n + j])
        # Top annulus (outer top to inner top)
        faces.append([i + n, j + n, 3*n + j])
        faces.append([i + n, 3*n + j, 3*n + i])
        # Bottom annulus (outer bottom to inner bottom, reversed)
        faces.append([i, 2*n + j, j])
        faces.append([i, 2*n + i, 2*n + j])

    return np.array(verts), faces


def _write_obj(path, all_objects):
    """Write combined OBJ with named objects."""
    with open(path, 'w') as f:
        f.write("# Screw hole cylinders\n# Units: mm\n\n")
        vert_off = 0
        for obj_name, verts, faces in all_objects:
            f.write(f"o {obj_name}\n")
            for v in verts:
                f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
            for face in faces:
                f.write(f"f {' '.join(str(i + 1 + vert_off) for i in face)}\n")
            vert_off += len(verts)
            f.write("\n")


def _write_stl(path, all_objects):
    """Write combined STL."""
    # Collect all triangles
    all_tris = []
    for _, verts, faces in all_objects:
        for face in faces:
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            all_tris.append((v0, v1, v2))

    with open(path, 'wb') as f:
        f.write(b'Screw cylinders STL'.ljust(80, b'\0'))
        f.write(struct.pack('<I', len(all_tris)))
        for v0, v1, v2 in all_tris:
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
    print("Generate Screw Cylinders")
    print("=" * 50)

    # Get geometry parameters
    bowl_v, bowl_f, _, _, _ = extract_bowl_surface("data/Tenor Pan only.obj")
    R_level = compute_leveling_rotation("data/Tenor Pan only.obj")
    bowl_v = (R_level @ bowl_v.T).T
    y_min = float(bowl_v[:, 1].min())
    drum_r = float(np.sqrt(bowl_v[:, 0]**2 + bowl_v[:, 2]**2).max())

    BASE_THICKNESS = 10.0
    sy_base = y_min + BASE_THICKNESS

    all_objects = []
    idx = 0

    # ── Structural screws: cylinder rings ─────────────────────────
    print("\nCylinder screws...")
    for ring_r in [CYL_SCREW_R_INNER, CYL_SCREW_R_OUTER]:
        for wa in WIRE_CYL_ANGLES_RAD:
            sx = ring_r * math.cos(wa)
            sz = ring_r * math.sin(wa)
            v, f = make_tube((sx, sz), sy_base, sy_base + SCREW_DEPTH,
                             BUSHING_OUTER_R, SCREW_PILOT_R)
            all_objects.append((f"ScrewCyl_struct_{idx}", v, f))
            idx += 1
    print(f"  {idx} cylinder screws (R={CYL_SCREW_R_INNER},{CYL_SCREW_R_OUTER}mm)")

    # ── Structural screws: struts ─────────────────────────────────
    print("Strut screws...")
    n_strut_start = idx
    for sa in STRUT_ANGLES_RAD:
        cos_a = math.cos(sa)
        sin_a = math.sin(sa)
        r_pos = STRUT_SCREW_R_START
        while r_pos < drum_r - 10:
            for t_off in [-(STRUT_HALF_W - SCREW_EDGE_MAX),
                          STRUT_HALF_W - SCREW_EDGE_MAX]:
                sx = r_pos * cos_a + t_off * (-sin_a)
                sz = r_pos * sin_a + t_off * cos_a
                v, f = make_tube((sx, sz), sy_base, sy_base + SCREW_DEPTH,
                                 BUSHING_OUTER_R, SCREW_PILOT_R)
                all_objects.append((f"ScrewCyl_strut_{idx}", v, f))
                idx += 1
            r_pos += STRUT_SCREW_SPACING
    print(f"  {idx - n_strut_start} strut screws")

    # ── Base attachment screws ────────────────────────────────────
    print("Base attachment screws...")
    BASE_SCREW_N = 8
    BASE_SCREW_INSET = 12.0
    BASE_SCREW_PILOT_R = 1.25
    BASE_SCREW_DEPTH = 8.0
    base_screw_r = drum_r - BASE_SCREW_INSET
    n_base_start = idx
    for si in range(BASE_SCREW_N):
        angle = 2 * math.pi * si / BASE_SCREW_N
        sx = base_screw_r * math.cos(angle)
        sz = base_screw_r * math.sin(angle)
        v, f = make_tube((sx, sz), sy_base, sy_base + BASE_SCREW_DEPTH,
                         BUSHING_OUTER_R, BASE_SCREW_PILOT_R)
        all_objects.append((f"ScrewCyl_base_{idx}", v, f))
        idx += 1
    print(f"  {idx - n_base_start} base screws (R={base_screw_r:.0f}mm)")

    # ── Pad screw holes ───────────────────────────────────────────
    print("Pad screw holes...")
    props = json.load(open('data/notepads/notepad_properties.json'))
    offset = np.array(
        json.load(open('data/pan_centroid_offset.json'))['centroid_offset_mm'])
    PILOT_R = 1.0
    PILOT_DEPTH = 5.0
    PAN_THICKNESS = 5.0
    n_pad_start = idx
    for p in props:
        for hp in p.get('hole_positions', []):
            h = R_level @ (np.array(hp) - offset)
            hx, hy, hz = h[0], h[1], h[2]
            pocket_floor = hy - PAN_THICKNESS
            y_bot = pocket_floor - PILOT_DEPTH
            y_top = pocket_floor
            v, f = make_tube((hx, hz), y_bot, y_top,
                             BUSHING_OUTER_R, PILOT_R)
            all_objects.append((f"ScrewCyl_pad_{idx}", v, f))
            idx += 1
    print(f"  {idx - n_pad_start} pad screws")

    # ── Export ────────────────────────────────────────────────────
    print(f"\nTotal: {len(all_objects)} screw cylinders")

    obj_path = 'data/notepads/screw_cylinders.obj'
    _write_obj(obj_path, all_objects)
    n_tri = sum(len(f) for _, _, f in all_objects)
    print(f"  {obj_path} ({n_tri} faces)")

    stl_path = 'data/notepads/screw_cylinders.stl'
    _write_stl(stl_path, all_objects)
    print(f"  {stl_path} ({n_tri} tri)")

    print("\nDone!")


if __name__ == "__main__":
    main()
