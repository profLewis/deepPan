#!/usr/bin/env python3
"""
Thicken each groove surface strictly downward (-Y) by THICKNESS mm.

For every groove object:
  - Top face: original surface vertices, original winding
  - Bottom face: same vertices shifted by -THICKNESS in Y, reversed winding
  - Side rim: a quad on each boundary edge connecting top to bottom

This guarantees grooves are only thickened downward — never upward — which
the merged-surface solidify pass cannot.

Output:
    data/grooves/grooves_outer_down.obj
    data/grooves/grooves_central_down.obj
    data/grooves/grooves_inner_down.obj
"""
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np


THICKNESS = 5.0  # mm, downward (overridable via --thickness)

SOURCES = [
    Path("data/grooves/grooves_outer.obj"),
    Path("data/grooves/grooves_central.obj"),
    Path("data/grooves/grooves_inner.obj"),
]

SUFFIX = "_down"  # output suffix appended to each source stem


def parse_obj_with_objects(path):
    """Yield (object_name, vertices Nx3, faces list-of-lists 0-based) for each `o` block."""
    cur_name = None
    cur_face_starts = []  # absolute face indices that begin a new object
    all_verts = []
    all_faces = []  # each face is a list of 0-based global vertex indices
    obj_face_ranges = []  # [(name, start, end)]
    obj_start = 0
    for line in open(path):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        tok = s.split()
        if tok[0] == 'v':
            all_verts.append((float(tok[1]), float(tok[2]), float(tok[3])))
        elif tok[0] == 'o':
            if cur_name is not None:
                obj_face_ranges.append((cur_name, obj_start, len(all_faces)))
            cur_name = ' '.join(tok[1:]) or path.stem
            obj_start = len(all_faces)
        elif tok[0] == 'f':
            idx = [int(t.split('/')[0]) - 1 for t in tok[1:]]
            all_faces.append(idx)
    if cur_name is None:
        cur_name = path.stem
    obj_face_ranges.append((cur_name, obj_start, len(all_faces)))

    V_all = np.asarray(all_verts, dtype=np.float64)
    out = []
    for name, fs, fe in obj_face_ranges:
        if fe <= fs:
            continue
        faces = all_faces[fs:fe]
        used_set = set()
        for f in faces:
            used_set.update(f)
        used = np.array(sorted(used_set), dtype=np.int64)
        remap = -np.ones(len(V_all), dtype=np.int64)
        remap[used] = np.arange(len(used))
        local_faces = [[int(remap[v]) for v in f] for f in faces]
        out.append((name, V_all[used], local_faces))
    return out


def directed_boundary_edges(faces):
    """Return ordered (a, b) edges that appear as directed edges in exactly one
    face (no reverse twin) — i.e., the actual boundary."""
    edge_count = defaultdict(int)
    for f in faces:
        n = len(f)
        for i in range(n):
            edge_count[(f[i], f[(i + 1) % n])] += 1
    boundary = []
    for (a, b), c in edge_count.items():
        if (b, a) not in edge_count:
            # Single-sided edge; emit `c` copies if it appears multiple times
            for _ in range(c):
                boundary.append((a, b))
    return boundary


def solidify_object(name, V, faces):
    """Build a closed-solid (V_out, faces_out) by extruding V downward by THICKNESS."""
    n = len(V)
    V_bot = V.copy()
    V_bot[:, 1] -= THICKNESS
    V_out = np.vstack([V, V_bot])

    out_faces = []
    # Top: original winding (normals point up/outward)
    for f in faces:
        out_faces.append(list(f))
    # Bottom: same verts shifted by n, reversed winding
    for f in faces:
        out_faces.append([v + n for v in reversed(f)])
    # Side rim
    for (a, b) in directed_boundary_edges(faces):
        # Quad a -> b -> b+n -> a+n (winding makes side normal point outward
        # if top winding makes the top normal point upward)
        out_faces.append([a, b, b + n, a + n])
    return V_out, out_faces


def write_obj(path, pieces):
    """pieces: list of (name, V, faces)."""
    v_offset = 0
    n_v = 0
    n_f = 0
    with open(path, 'w') as f:
        f.write("# Grooves extruded downward (-Y) by "
                f"{THICKNESS}mm — closed solids\n\n")
        for name, V, faces in pieces:
            f.write(f"o {name}\n")
            for v in V:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                idx = ' '.join(str(v + 1 + v_offset) for v in face)
                f.write(f"f {idx}\n")
            f.write("\n")
            v_offset += len(V)
            n_v += len(V)
            n_f += len(faces)
    print(f"Wrote {path}  ({n_v} verts, {n_f} faces)")


def main():
    global THICKNESS, SUFFIX
    p = argparse.ArgumentParser()
    p.add_argument("--thickness", type=float, default=THICKNESS,
                   help=f"downward Y extrusion in mm (default {THICKNESS})")
    p.add_argument("--suffix", default=SUFFIX,
                   help=f"output suffix appended to each source stem "
                        f"(default {SUFFIX!r})")
    args = p.parse_args()
    THICKNESS = args.thickness
    SUFFIX = args.suffix
    print(f"Thickness: {THICKNESS}mm   Suffix: {SUFFIX!r}")
    for src in SOURCES:
        if not src.exists():
            print(f"WARN: missing {src}, skipping")
            continue
        groups = parse_obj_with_objects(src)
        print(f"  {src.name}: {len(groups)} object(s)")
        solids = []
        for name, V, faces in groups:
            Vs, Fs = solidify_object(name, V, faces)
            solids.append((name, Vs, Fs))
        out_path = src.with_name(src.stem + SUFFIX + ".obj")
        write_obj(out_path, solids)


if __name__ == '__main__':
    main()
