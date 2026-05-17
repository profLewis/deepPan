#!/usr/bin/env python3
"""
Check whether an STL is watertight (closed). Report boundary edges and the
boundary loops (= mesh holes) with their positions and sizes.

Usage:
    python check_holes.py <mesh.stl> [<mesh2.stl> ...]
"""
from collections import defaultdict
import sys
import numpy as np
import trimesh


def find_loops(boundary_edges):
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)
    visited = set()
    loops = []
    for start in list(adj):
        if start in visited:
            continue
        loop = []
        v = start
        prev = None
        while v is not None and v not in visited:
            visited.add(v)
            loop.append(v)
            nxt = None
            for n in adj[v]:
                if n != prev and n not in visited:
                    nxt = n
                    break
            prev, v = v, nxt
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def check(path):
    m = trimesh.load(path)  # process=True merges duplicate vertices (needed for STL)
    print(f"\n=== {path} ===")
    print(f"  verts: {len(m.vertices):,}    faces: {len(m.faces):,}")
    print(f"  trimesh watertight: {m.is_watertight}   winding consistent: {m.is_winding_consistent}")
    print(f"  Euler number: {m.euler_number}    (closed solid → 2)")

    edge_count = defaultdict(int)
    for f in m.faces:
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            e = (min(a, b), max(a, b))
            edge_count[e] += 1

    n_edges = len(edge_count)
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    nonmanifold = [e for e, c in edge_count.items() if c > 2]

    print(f"  unique edges: {n_edges:,}    boundary: {len(boundary_edges)}    nonmanifold(>2 faces): {len(nonmanifold)}")

    if not boundary_edges:
        print("  ✓ NO HOLES — closed mesh")
        return

    loops = find_loops(boundary_edges)
    print(f"  HOLE COUNT: {len(loops)} loop(s)")
    V = m.vertices
    for i, L in enumerate(sorted(loops, key=lambda x: -len(x))[:20]):
        Vl = V[L]
        centroid = Vl.mean(axis=0)
        perim = sum(np.linalg.norm(Vl[(j + 1) % len(L)] - Vl[j]) for j in range(len(L)))
        bbox = Vl.max(axis=0) - Vl.min(axis=0)
        print(f"    hole {i:3d}: {len(L):4d} verts  perim={perim:7.1f}mm  "
              f"center=({centroid[0]:+7.1f},{centroid[1]:+7.1f},{centroid[2]:+7.1f})  "
              f"bbox=({bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f})")
    if len(loops) > 20:
        print(f"    ... and {len(loops) - 20} more")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        check(p)
