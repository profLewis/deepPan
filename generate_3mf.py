#!/usr/bin/env python3
"""
Generate Bambu P1S-ready .3mf files for each printable part.

Per the user's preference: one big piece per .3mf, all small parts share
a .3mf.

Output (under pipeline_output/3mf/):
  pan_central.3mf
  pan_outer_0.3mf  …  pan_outer_5.3mf            (6 outer pieces)
  bottom_plate_sector_0.3mf  …  _5.3mf           (6 plate sectors)
  small_parts_plugs.3mf                          (12 plugs grouped)

Each file can be opened directly in Bambu Studio.  Each big piece is laid
flat on the bed (Z=0 at the piece's lowest point) and centred on XY=0.
"""
from pathlib import Path
import math
import trimesh

OUT_DIR = Path("pipeline_output")
THREEMF_DIR = OUT_DIR / "3mf"
THREEMF_DIR.mkdir(exist_ok=True)

P1S = 256.0   # P1S build volume edge


def lay_flat_centered(mesh, rotate_y_deg=0):
    """Translate `mesh` so its bottom is at Z=0 and it is centred on XY=0.
    If rotate_y_deg is non-zero, first rotates around the Y axis (useful for
    standing the central-piece halves on their flat-cut face)."""
    import math
    m = mesh.copy()
    if rotate_y_deg != 0:
        Ry = trimesh.transformations.rotation_matrix(
            math.radians(rotate_y_deg), [0, 1, 0])
        m.apply_transform(Ry)
    minx, miny, minz = m.bounds[0]
    maxx, maxy, maxz = m.bounds[1]
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    m.apply_translation([-cx, -cy, -minz])
    return m


def export_3mf(mesh, path):
    scene = trimesh.Scene(mesh)
    scene.export(path)


def report(name, mesh):
    b = mesh.bounds[1] - mesh.bounds[0]
    fit = "OK" if max(b[0], b[1], b[2]) <= P1S else "TOO BIG"
    return f"{name}: {b[0]:.1f} × {b[1]:.1f} × {b[2]:.1f} mm  [{fit}]"


# ── Big pieces — one .3mf each ──
# Central piece is split into 2 semicircles; each printed with its FLAT CUT
# FACE on the bed (rotate the +X half by -90° around Y so its X=0 plane
# becomes Z=0; rotate the -X half by +90° around Y similarly).
big_pieces = (
    [("pan_central_half_0", "pan_piece_central_half_0.stl", -90)] +
    [("pan_central_half_1", "pan_piece_central_half_1.stl", +90)] +
    [(f"pan_outer_{i}", f"pan_piece_outer_{i}.stl", 0) for i in range(6)] +
    [(f"bottom_plate_sector_{i}", f"bottom_plate_sector_{i}.stl", 0)
     for i in range(6)]
)

for label, stl_name, rot_y in big_pieces:
    stl_path = OUT_DIR / stl_name
    if not stl_path.exists():
        print(f"  SKIP {label} — missing {stl_path}")
        continue
    m = trimesh.load(stl_path)
    m = lay_flat_centered(m, rotate_y_deg=rot_y)
    out = THREEMF_DIR / f"{label}.3mf"
    export_3mf(m, out)
    note = f" (rot Y {rot_y:+d}°)" if rot_y else ""
    print(f"  {report(label + note, m):65s} -> {out.name}")


# ── Small parts: 15 plugs on one plate ──
# 12 plugs for the strut-plane joints + 3 plugs for the central-piece dowel pins
# (same 4 × 18 mm geometry — reused).
plug_path = OUT_DIR / "plug.stl"
if plug_path.exists():
    plug = trimesh.load(plug_path)
    plug = lay_flat_centered(plug)
    plug_d = max(plug.bounds[1][0] - plug.bounds[0][0],
                 plug.bounds[1][1] - plug.bounds[0][1]) + 6.0
    N_PLUGS = 15  # 12 wedge plugs + 3 central-piece dowels
    plugs = []
    # 5 cols × 3 rows = 15
    for i in range(N_PLUGS):
        col = i % 5
        row = i // 5
        p = plug.copy()
        p.apply_translation([(col - 2.0) * plug_d, (row - 1.0) * plug_d, 0])
        plugs.append(p)
    combined = trimesh.util.concatenate(plugs)
    out = THREEMF_DIR / "small_parts_plugs.3mf"
    export_3mf(combined, out)
    print(f"  {report('plugs_x15', combined):65s} -> {out.name}")

print(f"\nWrote {len(list(THREEMF_DIR.glob('*.3mf')))} .3mf files in {THREEMF_DIR}")
