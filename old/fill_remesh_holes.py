#!/usr/bin/env python3
"""
Find and fill all boundary-edge holes in pan_assembly_remesh.stl.

For each boundary loop, fan-triangulate from the loop centroid. Works well
for small near-planar holes (which is what the voxel remesh leaves behind).
Reports hole stats before and after.

Usage:
    python fill_remesh_holes.py [--in=PATH] [--out=PATH] [--max-perim=MM]
"""
from collections import defaultdict
import argparse
import struct
import numpy as np
import trimesh


def find_loops(faces):
    """Return list of boundary loops (each = ordered list of vertex indices)."""
    edge_count = defaultdict(int)
    directed = {}
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            e = (min(a, b), max(a, b))
            edge_count[e] += 1
            directed[(a, b)] = directed.get((a, b), 0) + 1
    boundary = [e for e, c in edge_count.items() if c == 1]
    adj = defaultdict(list)
    for a, b in boundary:
        adj[a].append(b)
        adj[b].append(a)
    visited = set()
    loops = []
    for start in list(adj):
        if start in visited:
            continue
        loop, v, prev = [], start, None
        while v is not None and v not in visited:
            visited.add(v)
            loop.append(v)
            nxt = None
            for n in adj[v]:
                if n != prev and n not in visited:
                    nxt = n
                    break
            prev, v = v, nxt
        loops.append(loop)
    return loops, directed


def write_stl_binary(path, verts, faces):
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    tri_v = verts[faces]  # (n_tri, 3, 3)
    n0 = np.cross(tri_v[:, 1] - tri_v[:, 0], tri_v[:, 2] - tri_v[:, 0])
    norms = np.linalg.norm(n0, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    n0 = (n0 / norms).astype(np.float32)
    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(faces)))
        for ni, tri in zip(n0, tri_v):
            f.write(struct.pack('<3f', *ni))
            f.write(struct.pack('<3f', *tri[0]))
            f.write(struct.pack('<3f', *tri[1]))
            f.write(struct.pack('<3f', *tri[2]))
            f.write(b'\0\0')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp",
                   default="data/quarters/pan_assembly_remesh.stl")
    p.add_argument("--out",
                   default="data/quarters/pan_assembly_remesh.stl")
    p.add_argument("--max-perim", type=float, default=200.0,
                   help="only fill holes with perimeter <= this (mm) "
                        "to avoid accidentally filling the outer rim of the pan")
    args = p.parse_args()

    print(f"Loading {args.inp} ...")
    m = trimesh.load(args.inp)  # merges duplicate verts
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int64)
    print(f"  Loaded: {len(V):,} verts, {len(F):,} faces  "
          f"watertight: {m.is_watertight}")

    loops, directed = find_loops(F)
    if not loops:
        print("No boundary loops — already closed.")
        return
    print(f"Found {len(loops)} boundary loop(s)")

    # Filter loops by size
    new_faces = list(F)
    n_filled = 0
    n_skipped = 0
    fill_log = []
    for li, loop in enumerate(loops):
        if len(loop) < 3:
            continue
        Vl = V[np.asarray(loop)]
        perim = float(sum(np.linalg.norm(Vl[(j + 1) % len(loop)] - Vl[j])
                          for j in range(len(loop))))
        if perim > args.max_perim:
            n_skipped += 1
            continue
        centroid = Vl.mean(axis=0)
        c_idx = len(V) + n_filled  # we'll append this centroid below
        # Determine which edge direction is missing (so the new triangles
        # have correct winding to make the surrounding faces consistent).
        # For each (loop[i], loop[i+1]), check whether directed[(a,b)] or
        # directed[(b,a)] already exists. The MISSING direction is what
        # the fill needs to provide.
        # Add triangles (a, b, centroid) if (a,b) is missing; else (b, a, centroid).
        loop_filled = []
        for j in range(len(loop)):
            a = loop[j]
            b = loop[(j + 1) % len(loop)]
            has_ab = directed.get((a, b), 0) > 0
            has_ba = directed.get((b, a), 0) > 0
            # boundary edge: exactly one of has_ab / has_ba is true
            if has_ab and not has_ba:
                # (a→b) exists in the body; fill triangle goes (b, a, centroid)
                loop_filled.append([b, a, c_idx])
            elif has_ba and not has_ab:
                loop_filled.append([a, b, c_idx])
            else:
                # both or neither — shouldn't happen on boundary, but be safe
                loop_filled.append([a, b, c_idx])
        new_faces.extend(loop_filled)
        # Append the centroid as a new vertex (after all loops we'll vstack)
        fill_log.append((perim, len(loop), centroid))
        n_filled += 1

    print(f"  Filled {n_filled} hole(s), skipped {n_skipped} "
          f"(perim > {args.max_perim}mm)")
    if fill_log:
        perims = np.asarray([p for p, _, _ in fill_log])
        print(f"  Filled hole perimeter: min={perims.min():.1f}mm  "
              f"max={perims.max():.1f}mm  total={perims.sum():.1f}mm")

    if n_filled == 0:
        print("Nothing to fill.")
        return

    # Append centroids in order
    new_verts = np.vstack([V] + [c[None, :] for _, _, c in fill_log])
    new_faces = np.asarray(new_faces, dtype=np.int64)

    print(f"  Saving {args.out} ...")
    write_stl_binary(args.out, new_verts, new_faces)
    print(f"  Wrote {len(new_verts):,} verts, {len(new_faces):,} faces")

    # Verify
    m2 = trimesh.load(args.out)
    print(f"\nVerify: watertight: {m2.is_watertight}   "
          f"Euler: {m2.euler_number}   volume: {m2.volume:.1f}")


if __name__ == "__main__":
    main()
