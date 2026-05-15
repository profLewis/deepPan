#!/usr/bin/env python3
"""
Post-process pan_holes.obj: extend the drum surface inward at each pad hole
using a smoothed Laplacian displacement field.

Per-ring displacement direction (chosen by loop-centroid radius from drum axis):
  • Outer ring  (R >= R_BOUNDARY): displacement in the local tangent plane
                of the bowl (slides verts along the curved bowl wall).
  • Inner + central rings (R < R_BOUNDARY): displacement strictly in the
                horizontal XZ plane (drum surface plane). Verts shift sideways
                only; Y untouched. Avoids the curvature-related artifacts
                that the tangent-plane mode produces near the bowl floor.

Boundary verts get the full EXTENSION displacement; band verts within RINGS
edges get a Jacobi-smoothed falloff to zero. Final per-ring projection
re-enforces the displacement plane on the smoothed band.

Usage:
    python extend_pan_holes.py [--extension=1.0] [--rings=4] [--iters=60]
                               [--r-boundary=150]
                               [--in=PATH] [--out=PATH]
"""
from collections import defaultdict, deque
import argparse
import numpy as np


def load_obj(path):
    verts, faces, name = [], [], "PanHoles"
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == 'v':
                verts.append([float(t[1]), float(t[2]), float(t[3])])
            elif t[0] == 'f':
                faces.append([int(s.split('/')[0]) - 1 for s in t[1:]])
            elif t[0] == 'o':
                name = ' '.join(t[1:])
    return np.asarray(verts, dtype=np.float64), faces, name


def save_obj(path, verts, faces, name):
    with open(path, 'w') as f:
        f.write(f"# {name} - pad-hole boundaries extended (per-ring strategy)\n")
        f.write(f"o {name}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write("f " + " ".join(str(i + 1) for i in face) + "\n")


def boundary_loops(faces):
    edge_count = defaultdict(int)
    for face in faces:
        n = len(face)
        for i in range(n):
            a, b = face[i], face[(i + 1) % n]
            edge_count[(min(a, b), max(a, b))] += 1
    boundary = [e for e, c in edge_count.items() if c == 1]
    adj = defaultdict(list)
    for a, b in boundary:
        adj[a].append(b)
        adj[b].append(a)
    visited, loops = set(), []
    for start in adj:
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
    return loops


def vertex_normals(verts, faces):
    out = np.zeros_like(verts)
    for face in faces:
        if len(face) < 3:
            continue
        a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
        fn = np.cross(b - a, c - a)
        for vi in face:
            out[vi] += fn
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return out / norms


def vertex_neighbors(faces, n_v):
    nbrs = [set() for _ in range(n_v)]
    for face in faces:
        n = len(face)
        for i in range(n):
            a, b = face[i], face[(i + 1) % n]
            nbrs[a].add(b)
            nbrs[b].add(a)
    return [np.fromiter(s, dtype=np.int64) for s in nbrs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extension", type=float, default=1.0)
    p.add_argument("--rings", type=int, default=4)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--r-inner-max", type=float, default=60.0,
                   help="loop-centroid R < this -> INNER ring (skipped)")
    p.add_argument("--r-boundary", type=float, default=150.0,
                   help="loop-centroid R >= this -> OUTER ring (tangent+band); "
                        "R in [r_inner_max, r_boundary) -> CENTRAL (xz no-band)")
    p.add_argument("--center-extension", type=float, default=None,
                   help="extension magnitude for CENTRAL ring (defaults to --extension)")
    p.add_argument("--skip-inner", action="store_true",
                   help="(legacy) skip inner+central rings entirely")
    p.add_argument("--in", dest="inp", default="data/quarters/pan_holes.obj")
    p.add_argument("--out", default="data/quarters/pan_holes.obj")
    args = p.parse_args()

    verts, faces, name = load_obj(args.inp)
    n_v = len(verts)
    print(f"Loaded {args.inp}: {n_v} verts, {len(faces)} faces")

    vnorms = vertex_normals(verts, faces)
    nbrs = vertex_neighbors(faces, n_v)

    loops = boundary_loops(faces)
    print(f"Found {len(loops)} boundary loop(s)")
    outer_rim_idx = max(range(len(loops)), key=lambda i: len(loops[i]))
    print(f"Outer rim: loop {outer_rim_idx} ({len(loops[outer_rim_idx])} verts) — preserved")

    # Per-boundary-vert ring tag
    ring_tag = {}  # vert_idx -> 'tan' (outer) | 'xz_no_band' (central)
    disp = np.zeros((n_v, 3))
    boundary_verts = set()
    n_outer_loops = n_central_loops = n_inner_skipped = 0
    center_ext = args.extension if args.center_extension is None else args.center_extension
    for li, loop in enumerate(loops):
        if li == outer_rim_idx or len(loop) < 3:
            continue
        loop_arr = np.asarray(loop)
        centroid = verts[loop_arr].mean(axis=0)
        R = float(np.hypot(centroid[0], centroid[2]))

        if R >= args.r_boundary:
            tag = 'tan'  # outer: tangent-plane + smoothing band
            n_outer_loops += 1
            for vi in loop:
                v = verts[vi]
                n = vnorms[vi]
                d3 = centroid - v
                d_tan = d3 - np.dot(d3, n) * n
                mag = np.linalg.norm(d_tan)
                if mag > 1e-9:
                    disp[vi] = (d_tan / mag) * args.extension
                boundary_verts.add(vi)
                ring_tag[vi] = tag
        elif R >= args.r_inner_max and not args.skip_inner:
            # central: horizontal XZ at boundary only (no falloff band)
            tag = 'xz_no_band'
            n_central_loops += 1
            for vi in loop:
                v = verts[vi]
                d3 = centroid - v
                d_h = np.array([d3[0], 0.0, d3[2]])
                mag = np.linalg.norm(d_h)
                if mag > 1e-9:
                    disp[vi] = (d_h / mag) * center_ext
                boundary_verts.add(vi)
                ring_tag[vi] = tag
        else:
            # inner ring (or --skip-inner): no extension
            n_inner_skipped += 1
            continue

    print(f"Loops classified: outer (tangent+band, ext={args.extension}) = {n_outer_loops}   "
          f"central (xz no-band, ext={center_ext}) = {n_central_loops}   "
          f"inner skipped = {n_inner_skipped}")
    print(f"Set displacement on {len(boundary_verts)} boundary verts")

    # BFS only seeds from boundary verts that opted into the smoothing band
    # (i.e. tangent-plane outer ones). Inner+central boundaries are fixed
    # displacement with no falloff band — adjacent tris stretch a bit but
    # the body surface farther in is untouched.
    band_seed_verts = {v for v in boundary_verts if ring_tag[v] == 'tan'}
    dist = {v: 0 for v in band_seed_verts}
    parent = {v: v for v in band_seed_verts}
    q = deque(band_seed_verts)
    while q:
        v = q.popleft()
        if dist[v] >= args.rings:
            continue
        for nb in nbrs[v]:
            nbi = int(nb)
            if nbi not in dist:
                dist[nbi] = dist[v] + 1
                parent[nbi] = parent[v]
                q.append(nbi)
    band_verts = np.fromiter((v for v in dist if 0 < dist[v] < args.rings),
                             dtype=np.int64)
    print(f"Falloff band: {len(band_verts)} verts within {args.rings} rings")

    # Jacobi-smooth disp in the band (boundary fixed, far verts fixed at 0)
    for _ in range(args.iters):
        new_disp = disp.copy()
        for vi in band_verts:
            ns = nbrs[vi]
            new_disp[vi] = disp[ns].mean(axis=0)
        disp = new_disp

    # Final per-band-vert projection based on its parent boundary's ring tag
    for vi in band_verts:
        tag = ring_tag[parent[vi]]
        if tag == 'tan':
            n = vnorms[vi]
            disp[vi] = disp[vi] - np.dot(disp[vi], n) * n
        else:  # 'xz' — strictly horizontal
            disp[vi][1] = 0.0

    new_verts = verts + disp
    moved = np.linalg.norm(disp, axis=1)
    print(f"Displacement: max={moved.max():.3f}  "
          f"mean(band)={moved[band_verts].mean():.3f}  boundary={args.extension:.3f}")

    save_obj(args.out, new_verts, faces, name)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
