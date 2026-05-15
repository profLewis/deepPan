#!/usr/bin/env python3
"""
Extract the raw groove surface meshes (just the depressed bands of the bowl
where the original pan grooves are) and save as STL for visual overlay.

These are NOT the solidified groove cutters — they're the un-thickened groove
face groups straight out of the bowl mesh. Useful to confirm whether visible
'holes' in the remesh line up with the groove perimeters.
"""
from pathlib import Path
import sys
import numpy as np
import trimesh

OUT_DIR = Path("pipeline_output")
OUT_DIR.mkdir(exist_ok=True)

SOURCES = {
    "groove_central_surface.stl": "data/grooves/grooves_central.obj",
    "groove_inner_surface.stl":   "data/grooves/grooves_inner.obj",
    "groove_outer_surface.stl":   "data/grooves/grooves_outer.obj",
}

for out_name, src in SOURCES.items():
    p = Path(src)
    if not p.exists():
        print(f"SKIP (missing): {src}")
        continue
    m = trimesh.load(src, process=False)
    out = OUT_DIR / out_name
    m.export(out)
    print(f"  {src} -> {out}  ({len(m.vertices):,} verts, {len(m.faces):,} faces)")

# Also write the perimeter loops of the central grooves as a line-style
# mesh (thin tube along each loop edge). Helps see "where the join is".
from collections import defaultdict
src = "data/grooves/grooves_central.obj"
print(f"\nExtracting central groove perimeter loops from {src} ...")

verts = []
faces = []
with open(src) as f:
    for line in f:
        t = line.split()
        if not t: continue
        if t[0] == 'v':
            verts.append([float(t[1]), float(t[2]), float(t[3])])
        elif t[0] == 'f':
            faces.append([int(s.split('/')[0]) - 1 for s in t[1:]])
V = np.asarray(verts)
print(f"  {len(V):,} verts, {len(faces):,} faces")

edge_count = defaultdict(int)
for f in faces:
    for i in range(len(f)):
        a, b = f[i], f[(i+1) % len(f)]
        edge_count[(min(a,b), max(a,b))] += 1
boundary = [e for e, c in edge_count.items() if c == 1]
print(f"  Boundary edges (groove perimeters): {len(boundary)}")

# Build thin tubes along each boundary edge by extruding each edge as a
# small XZ-aligned bar (cheaper than tubes; just enough to be visible).
TUBE_R = 0.3  # mm
new_v, new_f = [], []
for (a, b) in boundary:
    va, vb = V[a], V[b]
    edge = vb - va
    L = float(np.linalg.norm(edge))
    if L < 1e-9: continue
    # Perpendicular in XZ plane
    perp = np.array([-edge[2], 0, edge[0]])
    pn = np.linalg.norm(perp)
    if pn < 1e-9: continue
    perp = perp / pn * TUBE_R
    up = np.array([0.0, TUBE_R, 0.0])
    # 4 corners of a small box at va, 4 at vb
    base = len(new_v)
    for src_v, off_perp, off_up in [
        (va, -1, -1), (va, +1, -1), (va, +1, +1), (va, -1, +1),
        (vb, -1, -1), (vb, +1, -1), (vb, +1, +1), (vb, -1, +1),
    ]:
        new_v.append(src_v + perp * off_perp + up * off_up)
    # 6 faces of a box (12 triangles)
    quads = [
        (0,1,2,3),(7,6,5,4),  # ends
        (0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0),  # sides
    ]
    for q in quads:
        a0,b0,c0,d0 = (base+i for i in q)
        new_f.append([a0,b0,c0])
        new_f.append([a0,c0,d0])

if new_v:
    line_mesh = trimesh.Trimesh(vertices=np.asarray(new_v), faces=np.asarray(new_f))
    out = OUT_DIR / "groove_central_perimeter.stl"
    line_mesh.export(out)
    print(f"  Wrote perimeter tubes: {out}  "
          f"({len(line_mesh.vertices):,} verts, {len(line_mesh.faces):,} tris)")
