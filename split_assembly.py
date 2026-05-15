"""
Split a fused pan assembly STL into 7 printable pieces using boolean intersects.

  Central piece: R < CYL_SPLIT_R (110 mm)
  6 outer pieces: R >= CYL_SPLIT_R, each spanning 60 deg between adjacent
                  strut center angles (15/75/135/195/255/315 deg).

Cuts pass straight down the centerline of each strut wall (20 mm thick), so
each outer piece keeps a half-strut on either side as designed.

Run headless:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python split_assembly.py \
        -- [<input.stl>] [--solver=EXACT|FAST] [--out-dir=<dir>]

Default input: data/quarters/pan_assembly_remesh.stl
Default solver: EXACT
Default out-dir: pipeline_output (project root)
Output names: pan_piece_central.stl,
              pan_piece_outer_0.stl .. pan_piece_outer_5.stl
"""
import bpy
import bmesh
import math
import os
import sys
from mathutils import Vector

# Inlined from generate_cylinders.py (sys.path is awkward in Blender --background).
CYL_SPLIT_R = 110.0
STRUT_ANGLES_DEG = [15.0, 75.0, 135.0, 195.0, 255.0, 315.0]

argv = sys.argv
user_args = argv[argv.index("--") + 1:] if "--" in argv else []

SOLVER = 'EXACT'
OUT_DIR = None
INPUT = None
for a in user_args:
    if a.startswith("--solver="):
        SOLVER = a.split("=", 1)[1].upper()
    elif a.startswith("--out-dir="):
        OUT_DIR = a.split("=", 1)[1]
    elif not a.startswith("--"):
        INPUT = a

if INPUT is None:
    INPUT = "data/quarters/pan_assembly_remesh.stl"
INPUT = os.path.abspath(INPUT)
if OUT_DIR is None:
    OUT_DIR = "pipeline_output"
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Input:    {INPUT}")
print(f"Solver:   {SOLVER}")
print(f"Out dir:  {OUT_DIR}")
print(f"Split R:  {CYL_SPLIT_R} mm")
print(f"Strut angles: {STRUT_ANGLES_DEG}")

# Cutters need to fully encompass the assembly in Z; the pan is much shallower
# than this but the overshoot is harmless.
Z_HALF = 500.0
R_BIG = 600.0
ARC_SEGMENTS_PER_DEG = 1.0  # 60 segs per 60-deg sector arc; 220 around full inner cyl

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)

print("Importing STL...")
bpy.ops.wm.stl_import(filepath=INPUT)
src_meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not src_meshes:
    raise SystemExit("No mesh imported")
src = src_meshes[0]
src.name = "Assembly"
print(f"  Imported: verts={len(src.data.vertices)}  polys={len(src.data.polygons)}")


def make_central_cutter():
    """Solid cylinder R = CYL_SPLIT_R, full Z height."""
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=int(360 * ARC_SEGMENTS_PER_DEG),
        radius=CYL_SPLIT_R,
        depth=Z_HALF * 2,
        location=(0, 0, 0),
    )
    obj = bpy.context.active_object
    obj.name = "Cutter_Central"
    return obj


def make_wedge_cutter(name, a_deg_start, a_deg_end):
    """Annular wedge prism: R in [CYL_SPLIT_R, R_BIG], theta in [start, end]."""
    # Build a closed polygon footprint in XY: inner arc CW, then outer arc CCW.
    bm = bmesh.new()
    arc_span = a_deg_end - a_deg_start
    n_arc = max(2, int(round(arc_span * ARC_SEGMENTS_PER_DEG)))

    # Inner arc (R = CYL_SPLIT_R), traverse from start -> end
    inner_verts = []
    for i in range(n_arc + 1):
        t = a_deg_start + arc_span * i / n_arc
        r = CYL_SPLIT_R
        x = r * math.cos(math.radians(t))
        y = r * math.sin(math.radians(t))
        inner_verts.append(bm.verts.new((x, y, -Z_HALF)))

    # Outer arc (R = R_BIG), traverse from end -> start (reverse, to close polygon)
    outer_verts = []
    for i in range(n_arc + 1):
        t = a_deg_end - arc_span * i / n_arc
        r = R_BIG
        x = r * math.cos(math.radians(t))
        y = r * math.sin(math.radians(t))
        outer_verts.append(bm.verts.new((x, y, -Z_HALF)))

    bm.verts.ensure_lookup_table()
    # Bottom face = polygon in order: inner (start->end), then outer (end->start)
    bottom_loop = inner_verts + outer_verts
    bm.faces.new(bottom_loop)
    bm.normal_update()

    # Send to a fresh mesh object
    mesh = bpy.data.meshes.new(name + "_mesh")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # Select + activate, then extrude the bottom face up by 2*Z_HALF to form prism
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.extrude_region_move(
        TRANSFORM_OT_translate={"value": (0, 0, Z_HALF * 2)}
    )
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def extract_piece(name, cutter):
    """Duplicate the source, boolean-intersect with cutter, export STL."""
    print(f"\n=== {name} ===")
    bpy.ops.object.select_all(action='DESELECT')
    src.select_set(True)
    bpy.context.view_layer.objects.active = src
    bpy.ops.object.duplicate()
    dup = bpy.context.active_object
    dup.name = f"Piece_{name}"

    mod = dup.modifiers.new("Bool", type='BOOLEAN')
    mod.object = cutter
    mod.operation = 'INTERSECT'
    mod.solver = SOLVER
    print(f"  Applying boolean ({SOLVER})...")
    bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"  Result: verts={len(dup.data.vertices)}  polys={len(dup.data.polygons)}")

    out_stl = os.path.join(OUT_DIR, f"pan_piece_{name}.stl")
    bpy.ops.object.select_all(action='DESELECT')
    dup.select_set(True)
    bpy.context.view_layer.objects.active = dup
    bpy.ops.wm.stl_export(
        filepath=out_stl,
        export_selected_objects=True,
        apply_modifiers=True,
        ascii_format=False,
    )
    print(f"  Wrote {out_stl}")
    # Remove the duplicate now that it's exported, to free memory
    bpy.data.objects.remove(dup, do_unlink=True)


# ─────────────────────────────────────────────────────────────────────────────
# Build cutters
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding central cutter...")
central_cutter = make_central_cutter()

wedge_cutters = []
n = len(STRUT_ANGLES_DEG)
for i in range(n):
    a0 = STRUT_ANGLES_DEG[i]
    a1 = STRUT_ANGLES_DEG[(i + 1) % n]
    if a1 <= a0:
        a1 += 360.0
    print(f"Building wedge cutter {i}: {a0:.1f} -> {a1:.1f} deg")
    wedge_cutters.append(make_wedge_cutter(f"Cutter_Outer_{i}", a0, a1))

# ─────────────────────────────────────────────────────────────────────────────
# Extract pieces
# ─────────────────────────────────────────────────────────────────────────────
extract_piece("central", central_cutter)
for i, c in enumerate(wedge_cutters):
    extract_piece(f"outer_{i}", c)

print("\nDone.")
