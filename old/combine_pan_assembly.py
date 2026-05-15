#!/usr/bin/env python3
"""
Combine pan_holes_solid + all_notepads + grooves into one OBJ/STL.

Output:
    data/quarters/pan_assembly.obj
    data/quarters/pan_assembly.stl
"""
import sys
from pathlib import Path
import numpy as np
import trimesh


THICK = "--thick" in sys.argv[1:]
if THICK:
    SOURCES = [
        Path("data/quarters/pan_holes_solid_thick.obj"),
        Path("data/notepads/all_notepads.obj"),
        Path("data/grooves/grooves_outer_down_thick.obj"),
        Path("data/grooves/grooves_central_down_thick.obj"),
        Path("data/grooves/grooves_inner_down_thick.obj"),
    ]
    OUT_OBJ = Path("data/quarters/pan_assembly_thick.obj")
    OUT_STL = Path("data/quarters/pan_assembly_thick.stl")
else:
    SOURCES = [
        Path("data/quarters/pan_holes_solid.obj"),
        Path("data/notepads/all_notepads.obj"),
        # Downward-only solidified grooves (closed solids extruded -Y by 5mm).
        # See solidify_grooves.py.
        Path("data/grooves/grooves_outer_down.obj"),
        Path("data/grooves/grooves_central_down.obj"),
        Path("data/grooves/grooves_inner_down.obj"),
    ]
    OUT_OBJ = Path("data/quarters/pan_assembly.obj")
    OUT_STL = Path("data/quarters/pan_assembly.stl")

# Per-object planimetric (XZ) widening of groove rings: split each ring by its
# median XZ radius (around the bbox centroid). Inner-half vertices scale by
# 1/(1+margin) around the centroid (moves toward pad → closes the pad/groove
# gap); outer-half vertices scale by (1+margin) (moves toward pan body).
# Y is untouched, so the downward extrusion stays vertical.
# Widen only the CENTRAL groove ring (the central pads' grooves), where the
# pad/body boundary leaves sub-voxel gaps. Outer and inner grooves are left
# alone — extending those caused other issues.
GROOVE_SCALE_SOURCES = {"data/grooves/grooves_central_down.obj"}
GROOVE_XZ_MARGIN = 0.30


def parse_obj_groups(path):
    """Yield (object_name, vertices Nx3, faces Mx3 0-based) per `o` block.

    Faces with >3 vertices are fan-triangulated. Only positional indices used.
    """
    name = path.stem
    verts = []
    groups = []
    cur_name = None
    cur_faces = []

    def flush():
        if cur_name is not None and cur_faces:
            groups.append((cur_name, list(cur_faces)))

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tok = line.split()
            if tok[0] == 'v':
                verts.append((float(tok[1]), float(tok[2]), float(tok[3])))
            elif tok[0] == 'o':
                flush()
                cur_name = ' '.join(tok[1:]) or name
                cur_faces = []
            elif tok[0] == 'f':
                # OBJ faces are 1-based; strip slash refs (vt/vn)
                idx = [int(t.split('/')[0]) - 1 for t in tok[1:]]
                if len(idx) < 3:
                    continue
                for i in range(1, len(idx) - 1):
                    cur_faces.append((idx[0], idx[i], idx[i + 1]))
    if cur_name is None:
        cur_name = name
    flush()
    if not groups:
        return []

    V = np.asarray(verts, dtype=np.float64)
    out = []
    for gname, faces in groups:
        F = np.asarray(faces, dtype=np.int64)
        used = np.unique(F)
        remap = -np.ones(len(V), dtype=np.int64)
        remap[used] = np.arange(len(used))
        out.append((gname, V[used], remap[F]))
    return out


def main():
    OUT_OBJ.parent.mkdir(parents=True, exist_ok=True)

    pieces = []
    for src in SOURCES:
        if not src.exists():
            print(f"WARN: missing {src}, skipping")
            continue
        groups = parse_obj_groups(src)
        if not groups:
            print(f"WARN: no geometry in {src}")
            continue
        scale_xz = (str(src) in GROOVE_SCALE_SOURCES)
        if scale_xz:
            scaled = []
            inner_scale = 1.0 / (1.0 + GROOVE_XZ_MARGIN)
            outer_scale = 1.0 + GROOVE_XZ_MARGIN
            eps = 1e-9
            for name, V, F in groups:
                # The _down OBJs from solidify_grooves.py have the top surface
                # vertices first, then the bottom (-Y extruded) vertices —
                # exactly 2N total. Recover that split here.
                n_total = len(V)
                n_top = n_total // 2
                V_top = V[:n_top]
                bot_offset = V[n_top:] - V_top  # per-vert -Y offset

                # Planimetric (XZ-only) widening from the groove's XZ centroid.
                # Y is untouched, so verts slide horizontally — no tilt drag
                # off the bowl curve and no chord-vs-arc Y artifacts.
                Cx = float(V_top[:, 0].mean())
                Cz = float(V_top[:, 2].mean())
                dx = V_top[:, 0] - Cx
                dz = V_top[:, 2] - Cz
                d = np.sqrt(dx * dx + dz * dz)
                d_min = float(d.min())
                d_max = float(d.max())
                if d_max - d_min > eps:
                    t = (d - d_min) / (d_max - d_min)
                else:
                    t = np.zeros_like(d)
                scale = inner_scale + (outer_scale - inner_scale) * t

                V_top_new = V_top.copy()
                V_top_new[:, 0] = Cx + dx * scale
                V_top_new[:, 2] = Cz + dz * scale
                # Y untouched: V_top_new[:, 1] == V_top[:, 1]
                V_bot_new = V_top_new + bot_offset
                V2 = np.vstack([V_top_new, V_bot_new])
                scaled.append((name, V2, F))
            groups = scaled
        tag = (f"  (planimetric widen XZ: {inner_scale:.3f}..{outer_scale:.3f}"
               f" inner→outer)" if scale_xz else "")
        print(f"  {src.name}: {len(groups)} object(s), "
              f"{sum(len(g[1]) for g in groups)} verts, "
              f"{sum(len(g[2]) for g in groups)} tris{tag}")
        pieces.extend(groups)

    print(f"Total: {len(pieces)} object groups")

    # Write combined OBJ
    total_v = 0
    total_f = 0
    with open(OUT_OBJ, 'w') as f:
        f.write("# Combined pan assembly: pan_holes_solid + all_notepads + grooves\n\n")
        v_offset = 0
        for name, V, F in pieces:
            f.write(f"o {name}\n")
            for v in V:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for tri in F:
                f.write(f"f {tri[0]+1+v_offset} {tri[1]+1+v_offset} {tri[2]+1+v_offset}\n")
            f.write("\n")
            v_offset += len(V)
            total_v += len(V)
            total_f += len(F)
    print(f"Wrote {OUT_OBJ}  ({total_v} verts, {total_f} tris)")

    # Build trimesh Scene -> STL (one combined mesh)
    all_V = np.vstack([V for _, V, _ in pieces])
    all_F = []
    v_offset = 0
    for _, V, F in pieces:
        all_F.append(F + v_offset)
        v_offset += len(V)
    all_F = np.vstack(all_F)
    mesh = trimesh.Trimesh(vertices=all_V, faces=all_F, process=False)
    mesh.export(OUT_STL)
    print(f"Wrote {OUT_STL}  ({len(mesh.vertices)} verts, {len(mesh.faces)} tris)")


if __name__ == '__main__':
    main()
