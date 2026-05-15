#!/usr/bin/env python3
"""
Thicken pantry.stl by offsetting the mesh inward by a uniform thickness.

The original pantry surface is the outer hull (untouched). A second
surface is created by displacing each vertex along its inward normal.
The two surfaces are stitched at boundary edges to form a closed shell.

No voxels involved — pure mesh operation.

Usage:
    python thicken_mesh.py [--thickness=10]
"""

import numpy as np
import struct
import sys
import math
from collections import defaultdict


def read_stl(path):
    """Read binary STL → (vertices Nx3, faces Mx3)."""
    with open(path, 'rb') as f:
        f.read(80)
        n_tri = struct.unpack('<I', f.read(4))[0]
        raw = np.empty((n_tri * 3, 3), dtype=np.float64)
        for i in range(n_tri):
            data = struct.unpack('<12fH', f.read(50))
            for vi in range(3):
                raw[i*3 + vi] = data[3 + vi*3: 6 + vi*3]
    # Deduplicate vertices
    unique, inverse = np.unique(np.round(raw, 4), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    return unique.astype(np.float64), faces


def compute_vertex_normals(verts, faces):
    """Compute area-weighted vertex normals."""
    normals = np.zeros_like(verts)
    for face in faces:
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        fn = np.cross(v1 - v0, v2 - v0)
        for vi in face:
            normals[vi] += fn
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-10] = 1.0
    return normals / lengths


def find_boundary_edges(faces):
    """Find edges that appear in only one face (mesh boundary)."""
    edge_count = defaultdict(int)
    edge_face = {}
    for fi, face in enumerate(faces):
        for i in range(len(face)):
            e = tuple(sorted([face[i], face[(i+1) % len(face)]]))
            edge_count[e] += 1
            edge_face[e] = fi
    return {e for e, c in edge_count.items() if c == 1}


def write_stl(path, verts, faces):
    """Write binary STL."""
    with open(path, 'wb') as f:
        f.write(b'Thickened mesh'.ljust(80, b'\0'))
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


def write_obj(path, verts, faces, name="Mesh"):
    with open(path, 'w') as f:
        f.write(f"# {name}\n# Units: mm\n\no {name}\n")
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        f.write("\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def main():
    thickness = 10.0
    for arg in sys.argv[1:]:
        if arg.startswith('--thickness='):
            thickness = float(arg.split('=')[1])

    print(f"Thicken pantry.stl by {thickness}mm (mesh offset, no voxels)")
    print("=" * 60)

    # Load
    print("\nLoading pantry.stl...")
    verts, faces = read_stl('data/pantry.stl')
    n_v = len(verts)
    n_f = len(faces)
    print(f"  {n_v} vertices, {n_f} faces")

    # Compute vertex normals (outward-pointing)
    print("Computing vertex normals...")
    vnormals = compute_vertex_normals(verts, faces)

    # For a closed STL, normals point outward by convention.
    # Offset inward = subtract normal * thickness.  No flip needed.
    print(f"Offsetting {thickness}mm inward (along -normal)...")
    inner_verts = verts - thickness * vnormals

    # Build combined mesh:
    # - Outer surface: original faces (outward normals)
    # - Inner surface: same faces but reversed winding + vertex offset
    # - Side walls: stitch boundary edges
    combined_verts = np.vstack([verts, inner_verts])  # 0..n_v-1 = outer, n_v..2*n_v-1 = inner

    outer_faces = faces.copy()
    # Inner faces: reversed winding, vertex indices shifted by n_v
    inner_faces = faces[:, ::-1] + n_v

    all_faces = list(outer_faces) + list(inner_faces)

    # Find boundary edges and stitch
    boundary = find_boundary_edges(faces)
    print(f"  {len(boundary)} boundary edges to stitch")
    for e in boundary:
        a, b = e
        # Quad: outer_a, outer_b, inner_b, inner_a → two triangles
        all_faces.append([a, b, b + n_v])
        all_faces.append([a, b + n_v, a + n_v])

    all_faces = np.array(all_faces)
    print(f"  Combined: {len(combined_verts)} verts, {len(all_faces)} faces")

    # Export
    out_stl = 'data/pantry_thick.stl'
    print(f"\nWriting {out_stl}...")
    write_stl(out_stl, combined_verts, all_faces)
    print(f"  {len(all_faces)} triangles")

    out_obj = 'data/pantry_thick.obj'
    write_obj(out_obj, combined_verts, all_faces, "PantryThick")
    print(f"  {out_obj}")

    print("\nDone!")


if __name__ == "__main__":
    main()
