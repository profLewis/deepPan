"""
Split pan_printable.stl into 7 printable parts with the strut wedges attached
to the OUTER pieces (so the wedges support the central piece from below).

  Central piece (1):  bowl at R < 110, unmodified.
  Outer pieces  (6):  60° sectors at R >= 110, each piece carries two
                       half-wedge tabs extending inward to R = 65 along its
                       two bounding strut planes (15/75/135/195/255/315°).
                       When two adjacent outer pieces are mated, their
                       half-wedges form the full wedge at the strut plane.

Wedges are rebuilt fresh here (NOT taken from pan_printable_features.stl) so
that we can vary the wedge-top overshoot along R:
    R <= 105 mm  : overshoot = -0.5 mm  → wedge tab sits just BELOW the
                                          central piece's bowl underside,
                                          giving a 0.5 mm assembly clearance.
                                          Each tab is the support shelf.
    R == 110     : transitions linearly
    R >= 115 mm  : overshoot = +4 mm    → wedge embeds INTO the outer-piece
                                          bowl shell (firm fusion via UNION).

Output (under pipeline_output/):
    pan_piece_central.stl
    pan_piece_outer_0.stl … pan_piece_outer_5.stl

Each piece is checked against the Bambu P1S build volume (256 × 256 × 256 mm).

Run headless:
    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python split_assembly.py -- [<input.stl>] [--solver=FAST|EXACT] [--out-dir=<dir>]

Defaults:
    input    pipeline_output/pan_printable.stl
    solver   FAST   (works on 17M-poly body in seconds; EXACT can be 30+ min)
    out-dir  pipeline_output
"""
import bpy
import bmesh
import math
import os
import sys
from mathutils import Vector
from mathutils.bvhtree import BVHTree


# ───── Geometry constants ─────
STRUT_ANGLES_DEG = [15.0, 75.0, 135.0, 195.0, 255.0, 315.0]
CYL_SPLIT_R = 110.0

# Wedge dimensions (must match build_assembly_features.py to keep visual
# consistency where possible — but here overshoot is variable, not +4 flat).
WALL_R_INNER  = 65.0
WALL_R_OUTER  = 246.0
N_RADIAL_RIBS = 64

W_INNER_BOT = 8.0
W_INNER_TOP = 4.0
W_OUTER_BOT = 36.0
W_OUTER_TOP = 16.0

WALL_Z_BOT = -122.0

OVERSHOOT_INNER  = -0.5
OVERSHOOT_OUTER  = +4.0
R_TRANSITION_LO  = 105.0
R_TRANSITION_HI  = 115.0

CABLE_HOLE_R     = 15.0
CABLE_HOLE_R_POS = 170.0
CABLE_HOLE_Z     = -75.0

# Joinery: HORIZONTAL alignment pegs, axis tangential to the strut plane
# (perpendicular to the strut radial). A peg passes through both half-wedges
# at the strut plane — the user pushes the peg in from the cavity side
# (perpendicular to the strut), and it locks adjacent half-wedges from
# sliding apart tangentially. 4 mm Ø × 18 mm long peg fits in a 4.2 mm × 20 mm
# channel (10 mm in each half-wedge after the strut-plane cut).
PEG_HOLE_DIA      = 4.2
PEG_HOLE_LENGTH   = 20.0
PEG_POSITIONS     = [                # (R, Z) — 2 peg channels per strut plane
    (130.0, -55.0),
    (200.0, -90.0),
]

# Joinery: M4 tap holes in each half-wedge base, for screws coming UP
# from below through the bottom plate. Offset 5 mm tangentially from the
# strut plane so each tap hole is fully inside one half-wedge after split.
TAP_HOLE_DIA          = 3.2   # M4 self-tap pilot
TAP_HOLE_DEPTH        = 15.0  # depth measured up from the wedge base
TAP_HOLE_R            = 170.0
TAP_TANGENT_OFFSET    = 5.0   # mm offset from strut plane, both ±

# Vertical peg holes drilled into each wedge tab top at R = 85 (inside the
# central-piece support shelf), receiving 5 mm pegs that stick down from the
# central piece's underside.  Centered ON the strut plane, so each half-wedge
# gets half the cylindrical hole — when adjacent half-wedges mate, the full
# hole appears, and the central piece's peg further locks them together.
CENTRAL_PEG_R       = 85.0
CENTRAL_PEG_HOLE_DIA = 4.2
CENTRAL_PEG_DEPTH    = 12.0   # depth of the receiving hole in the wedge tab
CENTRAL_PEG_PIN_DIA  = 4.0    # Ø of the actual peg baked on the central piece
CENTRAL_PEG_PIN_LEN  = 6.0    # how far the peg sticks down from the bowl underside

Z_HALF  = 500.0
R_BIG   = 600.0

P1S_SPACE = 256.0  # Bambu P1S build volume (cube edge)


# ───── Arg parsing ─────
argv = sys.argv
user_args = argv[argv.index("--") + 1:] if "--" in argv else []
SOLVER = 'EXACT'   # FAST is unreliable on 17M-poly meshes (gives 69 KB or 819 MB garbage)
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
    INPUT = "pipeline_output/pan_printable.stl"
if OUT_DIR is None:
    OUT_DIR = "pipeline_output"
INPUT = os.path.abspath(INPUT)
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Input:    {INPUT}")
print(f"Solver:   {SOLVER}")
print(f"Out dir:  {OUT_DIR}")
print(f"P1S fit:  {P1S_SPACE} mm cube\n")


# ───── Setup ─────
bpy.ops.wm.read_factory_settings(use_empty=True)

print("Importing bare body...")
bpy.ops.wm.stl_import(filepath=INPUT)
body = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
body.name = "BodyBare"
print(f"  verts={len(body.data.vertices):,} polys={len(body.data.polygons):,}")


# ── Body preparation: flatten drum bottom + solidify drum wall ──
# 1) Truncate the body at Z=-122 so the drum bottom is a clean flat plane
#    that the bottom plate can tightly mate against. WALL_Z_BOT also = -122
#    so wedges meet the plate flush.
# 2) UNION a solid annular ring at the drum-wall position (R=240..246,
#    Z=-122..+50), filling any internal voids in the drum wall.
print("\n── Body preparation ──")
print("Truncating body at Z=-122 (flat drum bottom)...")
# A huge box that occupies everything below Z=-122 (Z range -300..-122).
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
trunc = bpy.context.active_object
trunc.name = "TruncBelow122"
for v in trunc.data.vertices:
    v.co.x *= 1000.0   # X half-extent 500
    v.co.y *= 1000.0   # Y half-extent 500
    v.co.z *= 178.0    # Z half-extent 89  → after locating at z=-211 spans -300..-122
trunc.location = (0.0, 0.0, -211.0)
bpy.context.view_layer.objects.active = body
mod = body.modifiers.new("TruncBottom", type='BOOLEAN')
mod.operation = 'DIFFERENCE'
mod.object = trunc
mod.solver = SOLVER
bpy.ops.object.modifier_apply(modifier=mod.name)
bpy.data.objects.remove(trunc, do_unlink=True)
print(f"  after truncate: verts={len(body.data.vertices):,}")

print("Solidifying drum wall (UNION annular ring R=240..246, Z=-122..+50)...")
bpy.ops.mesh.primitive_cylinder_add(radius=246.0, depth=172.0,
                                    location=(0, 0, -36.0), vertices=128)
outer_cyl = bpy.context.active_object
outer_cyl.name = "DrumOuter"
bpy.ops.mesh.primitive_cylinder_add(radius=240.0, depth=180.0,
                                    location=(0, 0, -36.0), vertices=128)
inner_cyl = bpy.context.active_object
inner_cyl.name = "DrumInner"
bpy.context.view_layer.objects.active = outer_cyl
mod = outer_cyl.modifiers.new("MakeRing", type='BOOLEAN')
mod.operation = 'DIFFERENCE'
mod.object = inner_cyl
mod.solver = 'EXACT'
bpy.ops.object.modifier_apply(modifier=mod.name)
bpy.data.objects.remove(inner_cyl, do_unlink=True)
# UNION the solid drum-wall ring into the body
bpy.context.view_layer.objects.active = body
mod = body.modifiers.new("UnionDrumRing", type='BOOLEAN')
mod.operation = 'UNION'
mod.object = outer_cyl
mod.solver = SOLVER
bpy.ops.object.modifier_apply(modifier=mod.name)
bpy.data.objects.remove(outer_cyl, do_unlink=True)
print(f"  after drum solidify: verts={len(body.data.vertices):,}")

print("\nBuilding BVH for bowl-underside sampling...")
bm_bvh = bmesh.new()
bm_bvh.from_mesh(body.data)
bvh = BVHTree.FromBMesh(bm_bvh)
bm_bvh.free()


# ───── Wedge mesh builder ─────
def overshoot_for_R(R):
    if R <= R_TRANSITION_LO:
        return OVERSHOOT_INNER
    if R >= R_TRANSITION_HI:
        return OVERSHOOT_OUTER
    t = (R - R_TRANSITION_LO) / (R_TRANSITION_HI - R_TRANSITION_LO)
    return OVERSHOOT_INNER + t * (OVERSHOOT_OUTER - OVERSHOOT_INNER)


def find_wedge_top_Z(R, ang_rad):
    x, y = R * math.cos(ang_rad), R * math.sin(ang_rad)
    hits_z = []
    cur_z = +200.0
    for _ in range(8):
        hit = bvh.ray_cast(Vector((x, y, cur_z)), Vector((0, 0, -1)))
        if hit[0] is None:
            break
        z = hit[0].z
        hits_z.append(z)
        cur_z = z - 0.1
    ov = overshoot_for_R(R)
    if len(hits_z) >= 2:
        top, second = hits_z[0], hits_z[1]
        if second < top - 30.0:
            return top - 1.0  # drum-wall column
        return second + ov
    elif len(hits_z) == 1:
        return hits_z[0] - 1.0
    else:
        return WALL_Z_BOT + 10.0


def build_wedge(yaw_deg, name):
    bm = bmesh.new()
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    tx, ty = -s, c

    ribs = []
    for i in range(N_RADIAL_RIBS + 1):
        t = i / N_RADIAL_RIBS
        R = WALL_R_INNER + t * (WALL_R_OUTER - WALL_R_INNER)
        w_bot = W_INNER_BOT + t * (W_OUTER_BOT - W_INNER_BOT)
        w_top = W_INNER_TOP + t * (W_OUTER_TOP - W_INNER_TOP)
        Z_top = find_wedge_top_Z(R, yaw)
        Z_bot = WALL_Z_BOT
        if Z_top < Z_bot + 2.0:
            Z_top = Z_bot + 2.0
        rx, ry = R * c, R * s
        v_bn = bm.verts.new((rx + tx * (-w_bot / 2), ry + ty * (-w_bot / 2), Z_bot))
        v_bp = bm.verts.new((rx + tx * (+w_bot / 2), ry + ty * (+w_bot / 2), Z_bot))
        v_tp = bm.verts.new((rx + tx * (+w_top / 2), ry + ty * (+w_top / 2), Z_top))
        v_tn = bm.verts.new((rx + tx * (-w_top / 2), ry + ty * (-w_top / 2), Z_top))
        ribs.append([v_bn, v_bp, v_tp, v_tn])

    for i in range(N_RADIAL_RIBS):
        r0, r1 = ribs[i], ribs[i + 1]
        bm.faces.new([r0[0], r1[0], r1[1], r0[1]])
        bm.faces.new([r0[3], r0[2], r1[2], r1[3]])
        bm.faces.new([r0[0], r0[3], r1[3], r1[0]])
        bm.faces.new([r0[1], r1[1], r1[2], r0[2]])
    bm.faces.new([ribs[0][0], ribs[0][1], ribs[0][2], ribs[0][3]])
    bm.faces.new([ribs[-1][3], ribs[-1][2], ribs[-1][1], ribs[-1][0]])

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def make_cable_hole(angle_deg):
    ang = math.radians(angle_deg)
    cx, cy = CABLE_HOLE_R_POS * math.cos(ang), CABLE_HOLE_R_POS * math.sin(ang)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=CABLE_HOLE_R, depth=W_OUTER_BOT + 40.0,
        location=(cx, cy, CABLE_HOLE_Z),
        rotation=(-math.pi / 2, 0.0, ang),
        vertices=32,
    )
    return bpy.context.active_object


def make_peg_hole_horizontal(angle_deg, R_pos, Z_pos):
    """HORIZONTAL peg-channel cylinder, axis tangential at angle_deg, centered
    ON the strut plane. When the wedge is split at the strut plane, each
    half-wedge gets half the cylindrical channel; a peg pushed through from
    the cavity side locks the two halves together."""
    ang = math.radians(angle_deg)
    cx, cy = R_pos * math.cos(ang), R_pos * math.sin(ang)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=PEG_HOLE_DIA / 2.0, depth=PEG_HOLE_LENGTH,
        location=(cx, cy, Z_pos),
        rotation=(-math.pi / 2, 0.0, ang),
        vertices=32,
    )
    return bpy.context.active_object


def make_central_peg_receiver(angle_deg):
    """Vertical hole drilled DOWN into the wedge top at (R=85, strut_angle),
    receiving a peg that sticks down from the central piece's underside."""
    ang = math.radians(angle_deg)
    cx = CENTRAL_PEG_R * math.cos(ang)
    cy = CENTRAL_PEG_R * math.sin(ang)
    # Start the cylinder above any plausible wedge top (≈ Z=-30) and extend it
    # well into the wedge body; the part outside the wedge does nothing.
    top_z = -28.0
    bottom_z = top_z - 18.0  # 18 mm long; ≈ 12 mm of it ends up inside the wedge
    depth = top_z - bottom_z
    cz = (top_z + bottom_z) / 2
    bpy.ops.mesh.primitive_cylinder_add(
        radius=CENTRAL_PEG_HOLE_DIA / 2.0, depth=depth,
        location=(cx, cy, cz), vertices=32,
    )
    return bpy.context.active_object


def make_tap_hole(angle_deg, side):
    """Vertical M4 tap-pilot cylinder OFFSET tangentially from the strut plane,
    so the tap hole is fully inside one half-wedge after the cut. side = +1 or -1."""
    ang = math.radians(angle_deg)
    # Tangential unit vector (perpendicular to radial)
    tx, ty = -math.sin(ang), math.cos(ang)
    cx = TAP_HOLE_R * math.cos(ang) + tx * side * TAP_TANGENT_OFFSET
    cy = TAP_HOLE_R * math.sin(ang) + ty * side * TAP_TANGENT_OFFSET
    # Cylinder extends from below the wedge bottom up to TAP_HOLE_DEPTH inside
    bottom_z = WALL_Z_BOT - 5.0
    top_z = WALL_Z_BOT + TAP_HOLE_DEPTH
    depth = top_z - bottom_z
    cz = (top_z + bottom_z) / 2
    bpy.ops.mesh.primitive_cylinder_add(
        radius=TAP_HOLE_DIA / 2.0, depth=depth,
        location=(cx, cy, cz), vertices=24,
    )
    return bpy.context.active_object


def boolean_subtract(target, cutter, op_name):
    bpy.context.view_layer.objects.active = target
    mod = target.modifiers.new(op_name, type='BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.object = cutter
    mod.solver = SOLVER
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


print("\nBuilding wedges with all joinery features...")
wedges = []
for ang_deg in STRUT_ANGLES_DEG:
    w = build_wedge(ang_deg, f"Wedge_{int(ang_deg)}")
    # 30 mm cable hole through wedge (cavity routing)
    boolean_subtract(w, make_cable_hole(ang_deg), "Cable")
    # 2 HORIZONTAL alignment-peg channels (tangential axis, centered on strut plane)
    for (R_p, Z_p) in PEG_POSITIONS:
        boolean_subtract(w, make_peg_hole_horizontal(ang_deg, R_p, Z_p),
                         f"Peg_R{int(R_p)}")
    # Vertical receiver hole at R=85 — central piece's peg drops in here
    boolean_subtract(w, make_central_peg_receiver(ang_deg), "CentralPegRx")
    # 2 vertical M4 tap holes (one per half-wedge), offset from strut plane
    for side in (+1, -1):
        boolean_subtract(w, make_tap_hole(ang_deg, side),
                         f"Tap_{'p' if side > 0 else 'n'}")
    wedges.append(w)
    print(f"  wedge at {ang_deg}°  (cable + 2 horizontal pegs + 1 central-peg "
          f"receiver + 2 base-screw tap holes)")


# ───── Cutters ─────
def _extrude_z(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.extrude_region_move(
        TRANSFORM_OT_translate={"value": (0, 0, Z_HALF * 2)}
    )
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')


def make_annular_sector(a_start, a_end, r_inner, r_outer, name):
    """Annular sector solid = pie(a_start..a_end, R<=r_outer) − cylinder(R<=r_inner).
    Built via EXACT boolean on small cutter geometry (a few hundred verts), which
    is reliable, then used as a single-shot cutter against the 17M-poly body."""
    pie = make_pie_cutter(a_start, a_end, name + "_pie")
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=360, radius=r_inner,
        depth=Z_HALF * 2, location=(0, 0, 0),
    )
    disk = bpy.context.active_object
    disk.name = name + "_disk"
    bpy.context.view_layer.objects.active = pie
    mod = pie.modifiers.new("SubDisk", type='BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.object = disk
    mod.solver = 'EXACT'
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(disk, do_unlink=True)
    pie.name = name
    return pie


def make_pie_cutter(a_start, a_end, name):
    """Filled pie sector (origin → outer arc), used as the BOUNDS for slicing
    each wedge into its two half-wedges."""
    arc_span = a_end - a_start
    n_arc = max(4, int(arc_span))
    bm = bmesh.new()
    verts = [bm.verts.new((0, 0, -Z_HALF))]
    for i in range(n_arc + 1):
        t = a_start + arc_span * i / n_arc
        x, y = R_BIG * math.cos(math.radians(t)), R_BIG * math.sin(math.radians(t))
        verts.append(bm.verts.new((x, y, -Z_HALF)))
    bm.faces.new(verts)
    mesh = bpy.data.meshes.new(name + "_mesh")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    _extrude_z(obj)
    return obj


def make_disk_cutter(R, name):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=360, radius=R,
        depth=Z_HALF * 2, location=(0, 0, 0),
    )
    obj = bpy.context.active_object
    obj.name = name
    return obj


print("\nBuilding cutters (pie cutters + inner disk)...")
n = len(STRUT_ANGLES_DEG)
pie_cutters = []
for i in range(n):
    a0 = STRUT_ANGLES_DEG[i]
    a1 = STRUT_ANGLES_DEG[(i + 1) % n]
    if a1 <= a0:
        a1 += 360.0
    pie_cutters.append(make_pie_cutter(a0, a1, f"Pie_{i}"))
inner_disk = make_disk_cutter(CYL_SPLIT_R, "InnerDisk")


# ───── Helpers for export & fit check ─────
def aabb(obj):
    if not obj.data.vertices:
        return (0, 0, 0)
    xs = [v.co.x for v in obj.data.vertices]
    ys = [v.co.y for v in obj.data.vertices]
    zs = [v.co.z for v in obj.data.vertices]
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


def report_fit(name, obj):
    dx, dy, dz = aabb(obj)
    fit = "OK" if max(dx, dy, dz) <= P1S_SPACE else "TOO BIG for P1S"
    print(f"  {name}: bbox {dx:.1f} × {dy:.1f} × {dz:.1f} mm  [{fit}]")


def export_obj(obj, path):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.stl_export(
        filepath=path,
        export_selected_objects=True,
        ascii_format=False,
    )


# ───── Extract outer pieces ─────
print("\n=== Extracting outer pieces ===")
for i in range(n):
    a0 = STRUT_ANGLES_DEG[i]
    a1 = STRUT_ANGLES_DEG[(i + 1) % n]
    a1_show = a1 if a1 > a0 else a1 + 360
    print(f"\n— Outer piece {i} (sector {a0:.0f}..{a1_show:.0f}°) —")

    # Two-step body cut: first intersect with the pie (60° sector), then
    # subtract the inner disk. Two simple booleans against the 17M-poly
    # body, each with a small primitive cutter — reliable with EXACT.
    bpy.ops.object.select_all(action='DESELECT')
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.duplicate()
    bowl = bpy.context.active_object
    bowl.name = f"Bowl_{i}"

    # Step 1a: body ∩ pie_i  (slow)
    mod = bowl.modifiers.new("PieCut", type='BOOLEAN')
    mod.operation = 'INTERSECT'
    mod.object = pie_cutters[i]
    mod.solver = SOLVER
    bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"  after pie cut: verts={len(bowl.data.vertices):,}")

    # Step 1b: bowl − inner_disk  (slow)
    mod = bowl.modifiers.new("InnerSub", type='BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.object = inner_disk
    mod.solver = SOLVER
    bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"  after inner-disk subtract: verts={len(bowl.data.vertices):,}")

    # Step 2: half-wedge left = wedge_i ∩ pie_i  (small mesh, fast op)
    bpy.ops.object.select_all(action='DESELECT')
    wedges[i].select_set(True)
    bpy.context.view_layer.objects.active = wedges[i]
    bpy.ops.object.duplicate()
    hl = bpy.context.active_object
    hl.name = f"HalfL_{i}"
    mod = hl.modifiers.new("HLCut", type='BOOLEAN')
    mod.operation = 'INTERSECT'
    mod.object = pie_cutters[i]
    mod.solver = SOLVER
    bpy.ops.object.modifier_apply(modifier=mod.name)

    # Step 3: half-wedge right = wedge_{i+1} ∩ pie_i
    j = (i + 1) % n
    bpy.ops.object.select_all(action='DESELECT')
    wedges[j].select_set(True)
    bpy.context.view_layer.objects.active = wedges[j]
    bpy.ops.object.duplicate()
    hr = bpy.context.active_object
    hr.name = f"HalfR_{i}"
    mod = hr.modifiers.new("HRCut", type='BOOLEAN')
    mod.operation = 'INTERSECT'
    mod.object = pie_cutters[i]
    mod.solver = SOLVER
    bpy.ops.object.modifier_apply(modifier=mod.name)

    # Step 4: UNION half-wedges into bowl
    bpy.context.view_layer.objects.active = bowl
    mod = bowl.modifiers.new("UnionHL", type='BOOLEAN')
    mod.operation = 'UNION'
    mod.object = hl
    mod.solver = SOLVER
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(hl, do_unlink=True)

    mod = bowl.modifiers.new("UnionHR", type='BOOLEAN')
    mod.operation = 'UNION'
    mod.object = hr
    mod.solver = SOLVER
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(hr, do_unlink=True)

    bowl.name = f"OuterPiece_{i}"
    print(f"  after wedge union: verts={len(bowl.data.vertices):,}  polys={len(bowl.data.polygons):,}")
    report_fit(f"outer_{i}", bowl)

    out_stl = os.path.join(OUT_DIR, f"pan_piece_outer_{i}.stl")
    export_obj(bowl, out_stl)
    print(f"  -> {out_stl}")
    bpy.data.objects.remove(bowl, do_unlink=True)


# ───── Extract central piece (with 6 pegs baked on underside) ─────
print("\n=== Central piece ===")
bpy.ops.object.select_all(action='DESELECT')
body.select_set(True)
bpy.context.view_layer.objects.active = body
bpy.ops.object.duplicate()
central = bpy.context.active_object
central.name = "Central"

mod = central.modifiers.new("InnerCut", type='BOOLEAN')
mod.operation = 'INTERSECT'
mod.object = inner_disk
mod.solver = SOLVER
bpy.ops.object.modifier_apply(modifier=mod.name)
print(f"  bowl-only verts={len(central.data.vertices):,}")

# Add 6 vertical pegs on the underside at each strut angle, R=CENTRAL_PEG_R.
# Pegs slot into the wedge-top receiver holes — locking the central piece in
# place tangentially & radially.
print("  Adding 6 pegs to central piece underside...")
for ang_deg in STRUT_ANGLES_DEG:
    ang = math.radians(ang_deg)
    cx = CENTRAL_PEG_R * math.cos(ang)
    cy = CENTRAL_PEG_R * math.sin(ang)
    # Sample bowl underside at this exact point to set the peg's TOP face
    # flush with the bowl underside (peg's BOTTOM is then PIN_LEN below it).
    bowl_z = find_wedge_top_Z(CENTRAL_PEG_R, ang) - OVERSHOOT_INNER  # remove the -0.5 overshoot
    peg_top = bowl_z + 1.0          # embed 1 mm into the bowl shell
    peg_bot = bowl_z - CENTRAL_PEG_PIN_LEN
    peg_cz = (peg_top + peg_bot) / 2
    peg_h  = peg_top - peg_bot
    bpy.ops.mesh.primitive_cylinder_add(
        radius=CENTRAL_PEG_PIN_DIA / 2.0, depth=peg_h,
        location=(cx, cy, peg_cz), vertices=32,
    )
    peg = bpy.context.active_object
    peg.name = f"CentralPeg_{int(ang_deg)}"
    bpy.context.view_layer.objects.active = central
    mod = central.modifiers.new(f"UnionPeg_{int(ang_deg)}", type='BOOLEAN')
    mod.operation = 'UNION'
    mod.object = peg
    mod.solver = SOLVER
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(peg, do_unlink=True)
print(f"  after pegs: verts={len(central.data.vertices):,}")

report_fit("central", central)
out_stl = os.path.join(OUT_DIR, "pan_piece_central.stl")
export_obj(central, out_stl)
print(f"  -> {out_stl}")

print("\nDone.")
