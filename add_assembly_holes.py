"""
Add hidden, strong assembly hardware to pan_printable.stl before splitting.

All bolt heads visible only from inside the drum cavity (below the playing
surface) or from underneath the skirt — never on the player-facing top.

Joints:

  (A) STRUT joints — between adjacent outer pieces at strut angles
      15/75/135/195/255/315°.
      Each strut plane gets 3 reinforcement TABS hanging inside the cavity
      from the inner face of the skirt. Each tab straddles the cut plane;
      M4 bolt perpendicular to the cut. Bolt visible only from inside the
      cavity. (18 tabs total.)

  (B) RADIAL joint — between central and each outer piece, at R=110.
      6 horizontal DOWEL-PIN HOLES (4.3mm Ø, 20mm long, radial axis) at
      sector midpoints chosen to avoid central pad mount bosses.
      Cut bisects each hole → each piece has matching half-hole.
      A 4mm × 18mm dowel pin (separate STL, print 6) slips in for
      alignment; clamping is provided by the LOCKING RING (separate STL)
      that screws from below into thread bosses on the central + outer
      pieces. (See generate_locking_ring.py for the ring part.)

  (C) BASEPLATE — 8 M4 countersunk vertical holes through the skirt rim,
      matching bottom_plate_screw_pattern.json.

Run headless:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python add_assembly_holes.py \
        -- [--input=PATH] [--output=PATH] [--solver=FAST|EXACT]
"""
import bpy
import bmesh
import json
import math
import os
import sys


argv = sys.argv
user_args = argv[argv.index("--") + 1:] if "--" in argv else []
INPUT = None
OUTPUT = None
SOLVER = "FAST"
for a in user_args:
    if a.startswith("--input="):
        INPUT = a.split("=", 1)[1]
    elif a.startswith("--output="):
        OUTPUT = a.split("=", 1)[1]
    elif a.startswith("--solver="):
        SOLVER = a.split("=", 1)[1].upper()

ROOT = os.path.dirname(os.path.abspath(__file__))
if INPUT is None:
    INPUT = os.path.join(ROOT, "pipeline_output/pan_printable.stl")
if OUTPUT is None:
    OUTPUT = os.path.join(ROOT, "pipeline_output/pan_printable_drilled.stl")
PATTERN = os.path.join(ROOT, "pipeline_output/bottom_plate_screw_pattern.json")

# Geometry (Z-up: Z is the drum's vertical axis)
STRUT_ANGLES_DEG = [15.0, 75.0, 135.0, 195.0, 255.0, 315.0]
CYL_SPLIT_R = 110.0

# (A) Strut joints — internal tabs that BRIDGE FROM the skirt INTO the cavity.
# Skirt wall: outer ≈ R=246, inner ≈ R=241 (5mm thick). Tab spans R=234..246
# so it overlaps the skirt by 5mm AND protrudes 7mm into the cavity.
STRUT_TAB_R = 240.0                      # tab center radius
STRUT_TAB_RADIAL = 12.0                  # mm radial (so 234..246, overlaps skirt 5mm)
STRUT_TAB_TANGENTIAL = 22.0              # mm tangential (across cut plane)
STRUT_TAB_VERTICAL = 25.0                # mm vertical (Z)
STRUT_TAB_ZS = [-95.0, -65.0, -35.0]     # 3 tabs per strut plane
BOLT_CLEAR_R = 2.3                       # M4 clearance hole radius (4.6mm Ø)
BOLT_LENGTH = 50.0

# (B) Dowel-pin holes at R=110 sector midpoints (avoiding pad mount bosses)
SECTOR_MID_ANGLES_DEG = [74.5, 108.5, 167.0, 227.5, 259.5, 340.0]
RADIAL_PIN_HOLE_R = 2.15                 # 4.3mm Ø (= 4mm pin + 0.3mm clearance)
RADIAL_PIN_HOLE_LENGTH = 20.0            # 10mm each side of the cut
RADIAL_PIN_HOLE_Z = -30.0                # just inside body wall (between Z=-35 and -28)

# (D) Locking-ring mounting bosses — short cylinders on the body's underside
# that the locking ring screws into from below. Inner bosses on the central
# piece (R=100), outer bosses on the outer pieces (R=120). All boss bottoms
# end at the same Z so the flat ring sits flush.
BOSS_OD = 10.0                            # mm cylinder OD
BOSS_BOTTOM_Z = -50.0                     # common bottom (where ring sits)
INNER_BOSS_R = 100.0                      # central-piece bosses
INNER_BOSS_TOP_Z = -28.0                  # top into central-piece body wall
OUTER_BOSS_R = 120.0                      # outer-piece bosses (1 per outer piece)
OUTER_BOSS_TOP_Z = -18.0                  # top into outer-piece body wall
BOSS_THREAD_R = 1.6                       # mm M4 tap drill (3.2mm Ø) — user taps


def add_horizontal_bore(x, y, z, length, radius, axis_angle_deg):
    """Horizontal cylinder, axis in XY plane at axis_angle_deg from +X."""
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=length,
        location=(x, y, z),
        rotation=(math.pi / 2, 0, math.radians(axis_angle_deg)),
        vertices=24,
    )
    return bpy.context.active_object


def add_vertical_bore(x, y, z_top, z_bot, radius):
    """Vertical Z-axis cylinder from z_bot to z_top."""
    h = z_top - z_bot
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=h,
        location=(x, y, (z_top + z_bot) / 2),
        rotation=(0, 0, 0),
        vertices=24,
    )
    return bpy.context.active_object


def add_oriented_box(cx, cy, cz, dx, dy, dz, yaw_deg):
    """Box centered at (cx,cy,cz), local sizes (dx radial, dy tangential, dz vertical),
    rotated yaw_deg around Z axis. yaw_deg = angle from +X of the LOCAL X (radial) axis."""
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(cx, cy, cz))
    obj = bpy.context.active_object
    for v in obj.data.vertices:
        v.co.x *= dx
        v.co.y *= dy
        v.co.z *= dz
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    for v in obj.data.vertices:
        x, y, z = v.co
        v.co.x = x * c - y * s
        v.co.y = x * s + y * c
    obj.data.update()
    return obj


print(f"Input:  {INPUT}")
print(f"Output: {OUTPUT}")
print(f"Solver: {SOLVER}")

with open(PATTERN) as f:
    pat = json.load(f)
print(f"Baseplate pattern: {pat['n_holes']} holes at R={pat['hole_circle_r']}mm")

bpy.ops.wm.read_factory_settings(use_empty=True)
print("\nImporting body...")
bpy.ops.wm.stl_import(filepath=INPUT)
body = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
body.name = "Pan"
print(f"  verts={len(body.data.vertices):,} polys={len(body.data.polygons):,}")


# ─────────────────────────────────────────────────────────────────────────
# Stage 1: UNION strut tabs into the body (interior reinforcement)
# ─────────────────────────────────────────────────────────────────────────
print("\nBuilding strut reinforcement tabs (overlapping with skirt wall) ...")
tabs = []
for ang_deg in STRUT_ANGLES_DEG:
    ang = math.radians(ang_deg)
    cx = STRUT_TAB_R * math.cos(ang)
    cy = STRUT_TAB_R * math.sin(ang)
    for Z in STRUT_TAB_ZS:
        # Tab: local X = radial (12mm so R=234..246 — overlaps skirt by 5mm),
        # local Y = tangential (22mm across cut), Z = vertical (25mm).
        tab = add_oriented_box(
            cx, cy, Z,
            dx=STRUT_TAB_RADIAL,
            dy=STRUT_TAB_TANGENTIAL,
            dz=STRUT_TAB_VERTICAL,
            yaw_deg=ang_deg,
        )
        tabs.append(tab)
print(f"  {len(tabs)} strut tabs "
      f"({STRUT_TAB_RADIAL}x{STRUT_TAB_TANGENTIAL}x{STRUT_TAB_VERTICAL}mm)")

# Locking-ring mounting bosses
print("\nBuilding locking-ring mounting bosses ...")
bosses = []
for ang_deg in SECTOR_MID_ANGLES_DEG:
    ang = math.radians(ang_deg)
    # Inner boss (engages central piece): R=100
    ix = INNER_BOSS_R * math.cos(ang)
    iy = INNER_BOSS_R * math.sin(ang)
    inner_boss = add_vertical_bore(ix, iy,
                                    z_top=INNER_BOSS_TOP_Z,
                                    z_bot=BOSS_BOTTOM_Z,
                                    radius=BOSS_OD / 2)
    bosses.append(inner_boss)
    # Outer boss (engages outer piece): R=120
    ox = OUTER_BOSS_R * math.cos(ang)
    oy = OUTER_BOSS_R * math.sin(ang)
    outer_boss = add_vertical_bore(ox, oy,
                                    z_top=OUTER_BOSS_TOP_Z,
                                    z_bot=BOSS_BOTTOM_Z,
                                    radius=BOSS_OD / 2)
    bosses.append(outer_boss)
print(f"  {len(bosses)} bosses "
      f"(6 inner R={INNER_BOSS_R}, 6 outer R={OUTER_BOSS_R}; "
      f"all bottoms at Z={BOSS_BOTTOM_Z})")

# Join tabs + bosses into one cutter mesh and UNION into body
union_geometry = tabs + bosses
bpy.ops.object.select_all(action='DESELECT')
for o in union_geometry:
    o.select_set(True)
bpy.context.view_layer.objects.active = union_geometry[0]
bpy.ops.object.join()
add_obj = bpy.context.view_layer.objects.active
add_obj.name = "TabsAndBosses"

bpy.context.view_layer.objects.active = body
body.select_set(True)
print(f"\nUNION ({SOLVER}) tabs + bosses into body...")
mod = body.modifiers.new(name="UnionAdd", type='BOOLEAN')
mod.operation = 'UNION'
mod.object = add_obj
mod.solver = SOLVER
if SOLVER == 'EXACT':
    mod.use_self = False
    mod.use_hole_tolerant = True
bpy.ops.object.modifier_apply(modifier=mod.name)
print(f"  after union: verts={len(body.data.vertices):,} "
      f"polys={len(body.data.polygons):,}")
bpy.data.objects.remove(add_obj, do_unlink=True)


# ─────────────────────────────────────────────────────────────────────────
# Stage 2: DIFF all bolt + pin bores out of the body
# ─────────────────────────────────────────────────────────────────────────
print("\nBuilding bores ...")
bores = []

# (A) Strut M4 bolts — through the tabs we just unioned, tangent direction
n_strut = 0
for ang_deg in STRUT_ANGLES_DEG:
    ang = math.radians(ang_deg)
    cx = STRUT_TAB_R * math.cos(ang)
    cy = STRUT_TAB_R * math.sin(ang)
    for Z in STRUT_TAB_ZS:
        # Bolt axis = tangent direction (perpendicular to radial)
        bore = add_horizontal_bore(cx, cy, Z, BOLT_LENGTH,
                                    BOLT_CLEAR_R, ang_deg + 90.0)
        bores.append(bore)
        n_strut += 1
print(f"  (A) Strut M4 bolts through internal tabs: {n_strut}")

# (B) Dowel-pin holes at R=110 — radial axis, between pad mount bosses
n_pin = 0
for ang_deg in SECTOR_MID_ANGLES_DEG:
    ang = math.radians(ang_deg)
    cx = CYL_SPLIT_R * math.cos(ang)
    cy = CYL_SPLIT_R * math.sin(ang)
    bore = add_horizontal_bore(cx, cy, RADIAL_PIN_HOLE_Z,
                                RADIAL_PIN_HOLE_LENGTH,
                                RADIAL_PIN_HOLE_R, ang_deg)
    bores.append(bore)
    n_pin += 1
print(f"  (B) R=110 dowel-pin holes: {n_pin} (Ø {RADIAL_PIN_HOLE_R*2}mm)")

# (C) Baseplate M4 vertical bores. The pattern JSON uses 'y' key after the
# Z-up fix; fall back to legacy 'z' key for older JSONs.
n_base = 0
z_skirt = pat.get("skirt_z", pat.get("skirt_y"))
for h in pat["holes"]:
    hx = h["x"]
    hy = h.get("y", h.get("z"))  # accept either key
    bore = add_vertical_bore(hx, hy,
                             z_top=z_skirt + 30.0,
                             z_bot=z_skirt - 5.0,
                             radius=h["clear_r"])
    bores.append(bore)
    n_base += 1
print(f"  (C) Baseplate M4 vertical holes: {n_base}")

# (D) Locking-ring M4 tap-drill holes through each boss (vertical, from
# boss bottom going up into boss).
n_ring = 0
for ang_deg in SECTOR_MID_ANGLES_DEG:
    ang = math.radians(ang_deg)
    for R, top_z in [(INNER_BOSS_R, INNER_BOSS_TOP_Z),
                     (OUTER_BOSS_R, OUTER_BOSS_TOP_Z)]:
        bx = R * math.cos(ang)
        by = R * math.sin(ang)
        bore = add_vertical_bore(bx, by,
                                 z_top=top_z - 2.0,        # leave 2mm of solid above
                                 z_bot=BOSS_BOTTOM_Z - 1.0,
                                 radius=BOSS_THREAD_R)
        bores.append(bore)
        n_ring += 1
print(f"  (D) Locking-ring M4 tap-drill holes: {n_ring} (Ø {BOSS_THREAD_R*2}mm)")

bpy.ops.object.select_all(action='DESELECT')
for b in bores:
    b.select_set(True)
bpy.context.view_layer.objects.active = bores[0]
bpy.ops.object.join()
cutter = bpy.context.view_layer.objects.active
cutter.name = "AllBores"

bpy.context.view_layer.objects.active = body
body.select_set(True)
print(f"\n  DIFFERENCE ({SOLVER}) all bores out of body...")
mod = body.modifiers.new(name="Drill", type='BOOLEAN')
mod.operation = 'DIFFERENCE'
mod.object = cutter
mod.solver = SOLVER
if SOLVER == 'EXACT':
    mod.use_self = False
    mod.use_hole_tolerant = True
bpy.ops.object.modifier_apply(modifier=mod.name)
print(f"  after drill: verts={len(body.data.vertices):,} "
      f"polys={len(body.data.polygons):,}")
bpy.data.objects.remove(cutter, do_unlink=True)


# Cleanup
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=0.001)
bpy.ops.mesh.delete_loose()
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')
print(f"\nFinal: verts={len(body.data.vertices):,} polys={len(body.data.polygons):,}")

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
