"""
Drill assembly hardware holes into pan_printable.stl before splitting:

  • 12 piece-to-piece M3 through-bolt holes:
      - 6 at each strut-angle plane (15/75/135/195/255/315 deg), placed at
        two radii (R=150 and R=230) inboard of the skirt.
      - 6 at the R=110 cylindrical cut (central<->outer interfaces),
        placed at sector midpoint angles (45/105/165/225/285/345 deg).
      Bolts run HORIZONTALLY through the cut surface (perpendicular to it),
      so the cut plane naturally bisects each hole into matching halves.

  • 8 M4 countersunk bottom-plate holes from generate_bottom_plate.py's
      bottom_plate_screw_pattern.json. Drilled vertically (along +Y) up
      through the drum skirt.

Run headless:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python add_assembly_holes.py \
        -- [--input=PATH] [--output=PATH]

Defaults:
    --input=pipeline_output/pan_printable.stl
    --output=pipeline_output/pan_printable_drilled.stl
"""
import bpy
import json
import math
import os
import sys


argv = sys.argv
user_args = argv[argv.index("--") + 1:] if "--" in argv else []
INPUT = None
OUTPUT = None
for a in user_args:
    if a.startswith("--input="):
        INPUT = a.split("=", 1)[1]
    elif a.startswith("--output="):
        OUTPUT = a.split("=", 1)[1]

ROOT = os.path.dirname(os.path.abspath(__file__))
if INPUT is None:
    INPUT = os.path.join(ROOT, "pipeline_output/pan_printable.stl")
if OUTPUT is None:
    OUTPUT = os.path.join(ROOT, "pipeline_output/pan_printable_drilled.stl")
PATTERN = os.path.join(ROOT, "pipeline_output/bottom_plate_screw_pattern.json")

# Geometry constants (mirror generate_cylinders.py)
STRUT_ANGLES_DEG = [15.0, 75.0, 135.0, 195.0, 255.0, 315.0]
CYL_SPLIT_R = 110.0

# Piece-to-piece M3 hole parameters
M3_CLEAR_R = 1.7          # 3.4mm clearance hole radius
M3_BOLT_LENGTH = 50.0     # plenty long; cut plane bisects it
STRUT_HOLE_RADII = [150.0, 230.0]  # 2 bolts per strut plane: inner + outer
STRUT_HOLE_Y = -50.0      # vertical position inside the drum (below playing surface)
RADIAL_HOLE_ANGLES_DEG = [45.0, 105.0, 165.0, 225.0, 285.0, 345.0]  # sector midpoints
RADIAL_HOLE_Y = -50.0     # same Y as strut holes


def add_horizontal_bore(x, y, z, length, radius, axis_angle_deg):
    """Add a horizontal cylinder (along the XZ plane direction perpendicular
    to axis_angle_deg → so the bolt axis lies IN the XZ horizontal plane,
    perpendicular to the radial direction at axis_angle_deg).

    For a STRUT plane at angle theta (radial), the cut plane normal is
    the tangent direction: (-sin theta, 0, cos theta). The bolt must lie
    along this tangent so the cut bisects it.

    For a RADIAL (cylindrical) cut at angle theta, the bolt must lie
    along the radial direction (cos theta, 0, sin theta) so the cylindrical
    cut surface bisects it locally."""
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=length,
        location=(x, y, z),
        rotation=(math.pi / 2, 0, math.radians(axis_angle_deg)),
        vertices=24,
    )
    return bpy.context.active_object


def add_vertical_bore(x, z, y_top, y_bot, radius):
    """Add a Y-axis cylinder from y_bot to y_top."""
    h = y_top - y_bot
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=h,
        location=(x, (y_top + y_bot) / 2, z),
        rotation=(math.pi / 2, 0, 0),
        vertices=24,
    )
    return bpy.context.active_object


# ─────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────
print(f"Input:   {INPUT}")
print(f"Output:  {OUTPUT}")
print(f"Pattern: {PATTERN}")

with open(PATTERN) as f:
    pat = json.load(f)
print(f"  bottom plate: {pat['n_holes']} holes at R={pat['hole_circle_r']}mm")

bpy.ops.wm.read_factory_settings(use_empty=True)
print("\nImporting...")
bpy.ops.wm.stl_import(filepath=INPUT)
body = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
body.name = "Pan"
print(f"  {len(body.data.vertices):,} verts, {len(body.data.polygons):,} polys")

# Build cutter union: all bores stacked into one mesh, then one boolean DIFF.
bores = []

# Strut-plane horizontal bolts (perpendicular to the strut, i.e. tangent direction)
for ang_deg in STRUT_ANGLES_DEG:
    for R in STRUT_HOLE_RADII:
        ang = math.radians(ang_deg)
        # Bolt center on the strut plane at radial distance R
        cx = R * math.cos(ang)
        cz = R * math.sin(ang)
        # Bolt axis = tangent direction (perp to radial) so cut plane bisects it
        bore = add_horizontal_bore(cx, STRUT_HOLE_Y, cz, M3_BOLT_LENGTH,
                                    M3_CLEAR_R, ang_deg)
        bores.append(bore)
print(f"Strut M3 bolts: {len(STRUT_ANGLES_DEG)} planes x {len(STRUT_HOLE_RADII)} bolts = {len(bores)}")

# Radial bolts at R=110, perpendicular to the cylindrical cut
n_radial_start = len(bores)
for ang_deg in RADIAL_HOLE_ANGLES_DEG:
    ang = math.radians(ang_deg)
    # Bolt center on the cylindrical cut at this angle
    cx = CYL_SPLIT_R * math.cos(ang)
    cz = CYL_SPLIT_R * math.sin(ang)
    # Bolt axis = radial direction so the cylindrical cut bisects it locally
    bore = add_horizontal_bore(cx, RADIAL_HOLE_Y, cz, M3_BOLT_LENGTH,
                                M3_CLEAR_R, ang_deg + 90.0)
    bores.append(bore)
print(f"Radial (R={CYL_SPLIT_R}) M3 bolts: {len(RADIAL_HOLE_ANGLES_DEG)} (one per outer piece)")

# Bottom-plate M4 holes — vertical, through the skirt bottom
y_skirt = pat["skirt_y"]
for h in pat["holes"]:
    bore = add_vertical_bore(h["x"], h["z"],
                             y_top=y_skirt + 30.0,
                             y_bot=y_skirt - 5.0,
                             radius=h["clear_r"])
    bores.append(bore)
print(f"Bottom-plate M4 holes: {pat['n_holes']}")

# Join all bores into one cutter and DIFF the body
bpy.ops.object.select_all(action='DESELECT')
for b in bores:
    b.select_set(True)
bpy.context.view_layer.objects.active = bores[0]
bpy.ops.object.join()
cutter = bpy.context.view_layer.objects.active
cutter.name = "AllBores"
print(f"\nCutter: {len(cutter.data.vertices):,} verts, {len(cutter.data.polygons):,} polys")

bpy.context.view_layer.objects.active = body
body.select_set(True)
print("Boolean DIFFERENCE (EXACT) ...")
mod = body.modifiers.new(name="Drill", type='BOOLEAN')
mod.object = cutter
mod.operation = 'DIFFERENCE'
mod.solver = 'EXACT'
mod.use_self = False
mod.use_hole_tolerant = True
bpy.ops.object.modifier_apply(modifier=mod.name)
print(f"After drill: {len(body.data.vertices):,} verts, {len(body.data.polygons):,} polys")

# Cleanup
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.mesh.remove_doubles(threshold=0.001)
bpy.ops.mesh.delete_loose()
bpy.ops.object.mode_set(mode='OBJECT')
print(f"After cleanup: {len(body.data.vertices):,} verts, {len(body.data.polygons):,} polys")

bpy.data.objects.remove(cutter, do_unlink=True)
bpy.ops.object.select_all(action='DESELECT')
body.select_set(True)
bpy.context.view_layer.objects.active = body
bpy.ops.wm.stl_export(
    filepath=OUTPUT,
    export_selected_objects=True,
    apply_modifiers=True,
    ascii_format=False,
)
print(f"Wrote {OUTPUT}")
