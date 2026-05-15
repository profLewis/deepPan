#!/usr/bin/env python3
"""
Extract the drum wall (rim) from the raw pan OBJ, transform it to match
the pan_body / assembly_view coordinate system, and export as OBJ + STL.

The rim is identified by classify_drum_wall (faces at R > 225mm with
near-vertical normals). The output is in the same coordinate system as
pan_body*.obj and pan_central_piece*.obj (centered, leveled, mm).

Usage:
    python extract_rim.py
"""

import numpy as np
import struct
import math

from generate_sector import extract_bowl_surface
from generate_notepad import compute_leveling_rotation
from generate_quarter import classify_drum_wall, reindex_mesh


def _write_obj(path, verts, faces, obj_name="DrumRim"):
    with open(path, 'w') as f:
        f.write(f"# Drum wall (rim)\n# Units: mm\n\n")
        f.write(f"o {obj_name}\n")
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        f.write("\n")
        for face in faces:
            idx_str = " ".join(str(i + 1) for i in face)
            f.write(f"f {idx_str}\n")


def _write_stl(path, verts, faces):
    with open(path, 'wb') as f:
        f.write(b'Drum rim STL'.ljust(80, b'\0'))
        # Triangulate faces (some may be quads)
        triangles = []
        for face in faces:
            for i in range(1, len(face) - 1):
                triangles.append((face[0], face[i], face[i + 1]))
        f.write(struct.pack('<I', len(triangles)))
        for tri in triangles:
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
    obj_path = "data/Tenor Pan only.obj"

    print("Extract Drum Rim")
    print("=" * 50)

    # Phase 1: Load bowl surface (centered + scaled to mm)
    print("\nPhase 1: Load + extract bowl surface...")
    bowl_v, bowl_f, face_mat, face_group, pan_offset = extract_bowl_surface(obj_path)

    # Phase 2: Apply leveling rotation (drum axis -> Y)
    print("\nPhase 2: Level...")
    R_level = compute_leveling_rotation(obj_path)
    bowl_v = (R_level @ bowl_v.T).T

    # Phase 3: Classify rim — include the full skirt, rolled top edge,
    # and the transition inward up to where the outer pad tops begin.
    # The trim circle is the inner R of the outer pad playing surfaces.
    print("\nPhase 3: Classify rim...")

    # Find inner R of outer pad top surfaces from assembly view
    a_verts_list = []
    a_objs = {}
    a_cur = None
    with open('data/notepads/assembly_view.obj') as af:
        for line in af:
            line = line.strip()
            if line.startswith('v '):
                p = line.split()
                a_verts_list.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith('o '):
                a_cur = line[2:]
                a_objs[a_cur] = set()
            elif line.startswith('f ') and a_cur:
                for p in line.split()[1:]:
                    try:
                        a_objs[a_cur].add(int(p.split('/')[0]) - 1)
                    except ValueError:
                        pass
    a_verts_arr = np.array(a_verts_list)

    # Pad top surfaces: vertices within 5mm of each pad's Y max
    pad_top_r_min = float('inf')
    for name, vis in a_objs.items():
        if not name.startswith('Pad_O'):
            continue
        pv = a_verts_arr[sorted(vis)]
        y_max = float(pv[:, 1].max())
        top = pv[pv[:, 1] > y_max - 5.0]
        r_min = float(np.sqrt(top[:, 0]**2 + top[:, 2]**2).min())
        pad_top_r_min = min(pad_top_r_min, r_min)
    print(f"  Outer pad top inner R = {pad_top_r_min:.1f}mm (trim circle)")

    dw_faces = []
    for fi, face in enumerate(bowl_f):
        fc = bowl_v[face].mean(axis=0)
        r = math.sqrt(fc[0]**2 + fc[2]**2)
        if r > pad_top_r_min:
            dw_faces.append(bowl_f[fi])
    print(f"  {len(dw_faces)} rim faces (wall + rolled edge, trimmed at pad tops)")

    if not dw_faces:
        print("  No drum wall faces found!")
        return

    # Reindex to compact vertex array
    dw_v, dw_f = reindex_mesh(bowl_v, dw_faces)

    # Stats
    radii = np.sqrt(dw_v[:, 0]**2 + dw_v[:, 2]**2)
    y_vals = dw_v[:, 1]
    print(f"  {len(dw_v)} vertices, {len(dw_f)} faces")
    print(f"  R: [{radii.min():.1f}, {radii.max():.1f}] mm")
    print(f"  Y: [{y_vals.min():.1f}, {y_vals.max():.1f}] mm")

    # Phase 5: Export
    print("\nPhase 5: Export...")
    _write_obj("data/drum_rim.obj", dw_v, dw_f, "DrumRim")
    print(f"  data/drum_rim.obj ({len(dw_v)}v, {len(dw_f)}f)")

    _write_stl("data/drum_rim.stl", dw_v, dw_f)
    # Count triangles (faces may be quads)
    n_tri = sum(len(face) - 2 for face in dw_f)
    print(f"  data/drum_rim.stl ({n_tri} tri)")

    print("\nDone!")


if __name__ == "__main__":
    main()
