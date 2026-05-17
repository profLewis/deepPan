"""
Build assembly features into pan_printable.stl BEFORE splitting.

  * 6 radial strut WEDGES (one per strut plane: 15/75/135/195/255/315°)
    – Custom bmesh: top surface follows the ACTUAL bowl underside,
      sampled by ray-casting against the body STL per strut. Wedges
      therefore meet the bowl underside exactly (no gap, no protrusion).
    – Wedge is double-tapered: wider at base than at top, and wider at the
      outer (drum-wall) end than at the inner (centre) end. Inner end is
      very thin (4–8 mm) so it clears the central / inner pad mounts.
    – Wedge base sits at the skirt-bottom Z, so nothing protrudes below.
    – Wedge outer end is flush with R = drum-wall outer (246); the body of
      the wedge overlaps the drum wall thickness so the UNION fuses them
      into the drum wall (no gap at the rim).
  * 30 mm cable hole through each strut wedge (tangential axis).

  --stage=1  walls + cable holes (default)
  --stage=2  walls + bolt holes + dowel pegs                [later stage]
  --stage=3  + locking ring bosses + baseplate holes        [later stage]

Output:
  pipeline_output/pan_printable_features.stl
"""
import bpy
import bmesh
import math
import os
import sys
from mathutils import Vector
from mathutils.bvhtree import BVHTree


argv = sys.argv
user_args = argv[argv.index("--") + 1:] if "--" in argv else []
STAGE = 1
SOLVER = "FAST"
INPUT = None
OUTPUT = None
for a in user_args:
    if a.startswith("--stage="):
        STAGE = int(a.split("=", 1)[1])
    elif a.startswith("--solver="):
        SOLVER = a.split("=", 1)[1].upper()
    elif a.startswith("--input="):
        INPUT = a.split("=", 1)[1]
    elif a.startswith("--output="):
        OUTPUT = a.split("=", 1)[1]

ROOT = os.path.dirname(os.path.abspath(__file__))
if INPUT is None:
    INPUT = os.path.join(ROOT, "pipeline_output/pan_printable.stl")
if OUTPUT is None:
    OUTPUT = os.path.join(ROOT, "pipeline_output/pan_printable_features.stl")

# Geometry constants (Z-up)
STRUT_ANGLES_DEG = [15.0, 75.0, 135.0, 195.0, 255.0, 315.0]

# Wedge radial extent
WALL_R_INNER = 65.0      # well inside the central piece, narrow enough to clear mounts
WALL_R_OUTER = 246.0     # flush with drum-wall outer surface
N_RADIAL_RIBS = 64       # samples along the radial direction (curve resolution)

# Tangential widths at the 4 (R, Z) corners (TOTAL — each half is post-split width)
W_INNER_BOT = 8.0        # at (R_inner, base) — very narrow to clear central/inner mounts
W_INNER_TOP = 4.0        # at (R_inner, top)  — even narrower at the playing surface
W_OUTER_BOT = 36.0       # at (R_outer, base) — wide flare for drum-wall connection
W_OUTER_TOP = 16.0       # at (R_outer, top)  — solid embed into drum-wall thickness

# Vertical extent
WALL_Z_BOT = -122.0      # matches drum-skirt bottom rim — no protrusion below
TOP_OVERSHOOT = 4.0      # mm: wedge top extends past the sampled bowl underside,
                         #     embedding into the shell rather than only touching.

# Cable holes through each strut wedge
CABLE_HOLE_R     = 15.0  # 30 mm diameter
CABLE_HOLE_R_POS = 170.0 # radial position
CABLE_HOLE_Z     = -75.0 # safely below any sampled bowl underside

# Bolt holes (for later stages)
M4_CLEAR_R = 2.3
BOLT_LENGTH = 80.0


def build_curved_wedge(yaw_deg, top_Zs, name):
    """Build a wedge mesh whose top follows the sampled bowl-underside heights."""
    bm = bmesh.new()
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    tx, ty = -s, c   # tangential unit vector

    ribs = []
    for i in range(N_RADIAL_RIBS + 1):
        t = i / N_RADIAL_RIBS
        R = WALL_R_INNER + t * (WALL_R_OUTER - WALL_R_INNER)
        w_bot = W_INNER_BOT + t * (W_OUTER_BOT - W_INNER_BOT)
        w_top = W_INNER_TOP + t * (W_OUTER_TOP - W_INNER_TOP)
        Z_top = top_Zs[i]
        Z_bot = WALL_Z_BOT
        if Z_top < Z_bot + 2.0:
            Z_top = Z_bot + 2.0  # keep rib non-degenerate

        rx, ry = R * c, R * s
        v_bn = bm.verts.new((rx + tx * (-w_bot / 2), ry + ty * (-w_bot / 2), Z_bot))
        v_bp = bm.verts.new((rx + tx * (+w_bot / 2), ry + ty * (+w_bot / 2), Z_bot))
        v_tp = bm.verts.new((rx + tx * (+w_top / 2), ry + ty * (+w_top / 2), Z_top))
        v_tn = bm.verts.new((rx + tx * (-w_top / 2), ry + ty * (-w_top / 2), Z_top))
        ribs.append([v_bn, v_bp, v_tp, v_tn])

    for i in range(N_RADIAL_RIBS):
        r0, r1 = ribs[i], ribs[i + 1]
        bm.faces.new([r0[0], r1[0], r1[1], r0[1]])  # bottom
        bm.faces.new([r0[3], r0[2], r1[2], r1[3]])  # top (curved)
        bm.faces.new([r0[0], r0[3], r1[3], r1[0]])  # -y side
        bm.faces.new([r0[1], r1[1], r1[2], r0[2]])  # +y side
    bm.faces.new([ribs[0][0], ribs[0][1], ribs[0][2], ribs[0][3]])
    bm.faces.new([ribs[-1][3], ribs[-1][2], ribs[-1][1], ribs[-1][0]])

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def add_tangential_cylinder(angle_deg, R_pos, Z_pos, radius, length, name):
    ang = math.radians(angle_deg)
    cx = R_pos * math.cos(ang)
    cy = R_pos * math.sin(ang)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=length,
        location=(cx, cy, Z_pos),
        rotation=(-math.pi / 2, 0.0, ang),
        vertices=32,
    )
    obj = bpy.context.active_object
    obj.name = name
    return obj


print(f"Input:   {INPUT}")
print(f"Output:  {OUTPUT}")
print(f"Stage:   {STAGE}")
print(f"Solver:  {SOLVER}")

bpy.ops.wm.read_factory_settings(use_empty=True)

print("\nImporting body...")
bpy.ops.wm.stl_import(filepath=INPUT)
body = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
body.name = "Pan"
print(f"  verts={len(body.data.vertices):,} polys={len(body.data.polygons):,}")


# ─────────────────────────────────────────────────────────────────────────
# Build BVH tree for fast ray casting against the body
# ─────────────────────────────────────────────────────────────────────────
print("\nBuilding BVH tree for bowl-underside sampling...")
bm_bvh = bmesh.new()
bm_bvh.from_mesh(body.data)
bvh = BVHTree.FromBMesh(bm_bvh)
bm_bvh.free()
print("  BVH ready")


def find_wedge_top_Z(R, ang_rad, overshoot=TOP_OVERSHOOT):
    """For (R, ang), cast a ray downward and find the bowl underside.
    Returns the desired wedge-top Z = bowl_underside + overshoot.
    Handles drum-wall region (R > ~240) by using top-of-drum minus 1 mm
    so the wedge is solidly inside the drum-wall thickness."""
    x = R * math.cos(ang_rad)
    y = R * math.sin(ang_rad)

    hits_z = []
    cur_z = +200.0
    for _ in range(8):
        hit = bvh.ray_cast(Vector((x, y, cur_z)), Vector((0, 0, -1)))
        if hit[0] is None:
            break
        z = hit[0].z
        hits_z.append(z)
        cur_z = z - 0.1

    if len(hits_z) >= 2:
        top = hits_z[0]
        second = hits_z[1]
        # In the drum-wall column the second hit is at the drum-wall bottom
        # (Z ≈ −122). Detect by checking it's far below the body top.
        if second < top - 30.0:
            # Drum-wall column: place wedge top 1 mm below drum-wall top so it
            # sits cleanly inside the drum-wall thickness after UNION.
            return top - 1.0
        return second + overshoot
    elif len(hits_z) == 1:
        # Only one hit — treat as drum wall, return top - 1.
        return hits_z[0] - 1.0
    else:
        # No hit (point is outside the body laterally) — short stub
        return WALL_Z_BOT + 10.0


# ─────────────────────────────────────────────────────────────────────────
# Sample bowl underside per strut angle, per R
# ─────────────────────────────────────────────────────────────────────────
print("\nSampling bowl underside per strut angle (ray casting)...")
bowl_profiles = {}
for ang_deg in STRUT_ANGLES_DEG:
    ang_rad = math.radians(ang_deg)
    profile = []
    for i in range(N_RADIAL_RIBS + 1):
        t = i / N_RADIAL_RIBS
        R = WALL_R_INNER + t * (WALL_R_OUTER - WALL_R_INNER)
        Z = find_wedge_top_Z(R, ang_rad)
        profile.append(Z)
    bowl_profiles[ang_deg] = profile
    samples_show = [0, 8, 16, 24, 32, 40, 48, 56, 64]
    bits = []
    for s in samples_show:
        if s >= len(profile):
            continue
        R_at = WALL_R_INNER + (s / N_RADIAL_RIBS) * (WALL_R_OUTER - WALL_R_INNER)
        bits.append(f"R{int(R_at)}:{profile[s]:+.1f}")
    print(f"  {ang_deg:>5.1f}°  " + "  ".join(bits))


# ─────────────────────────────────────────────────────────────────────────
# STAGE 1: build wedges using the sampled profiles
# ─────────────────────────────────────────────────────────────────────────
print("\nBuilding curved wedges from sampled bowl profiles...")
walls = []
for ang_deg in STRUT_ANGLES_DEG:
    w = build_curved_wedge(
        ang_deg, bowl_profiles[ang_deg],
        name=f"StrutWedge_{int(ang_deg)}",
    )
    walls.append(w)

# Join into one mesh
bpy.ops.object.select_all(action='DESELECT')
for w in walls:
    w.select_set(True)
bpy.context.view_layer.objects.active = walls[0]
bpy.ops.object.join()
wall_obj = bpy.context.view_layer.objects.active
wall_obj.name = "AllWedges"
print(f"  joined wedges: verts={len(wall_obj.data.vertices):,} "
      f"polys={len(wall_obj.data.polygons):,}")

# Drill cable holes
print("\nDrilling cable holes through strut wedges...")
for ang_deg in STRUT_ANGLES_DEG:
    cyl = add_tangential_cylinder(
        ang_deg, CABLE_HOLE_R_POS, CABLE_HOLE_Z,
        CABLE_HOLE_R, W_OUTER_BOT + 40.0,
        name=f"CableHole_{int(ang_deg)}",
    )
    bpy.context.view_layer.objects.active = wall_obj
    mod = wall_obj.modifiers.new("Cable", type='BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.object = cyl
    mod.solver = SOLVER
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cyl, do_unlink=True)
    print(f"  cable hole at {ang_deg}° (R={CABLE_HOLE_R_POS}, Z={CABLE_HOLE_Z})")

# UNION wedges into body
bpy.context.view_layer.objects.active = body
body.select_set(True)
print(f"\nUNION ({SOLVER}) wedges into body...")
mod = body.modifiers.new("UnionWedges", type='BOOLEAN')
mod.operation = 'UNION'
mod.object = wall_obj
mod.solver = SOLVER
if SOLVER == 'EXACT':
    mod.use_self = False
    mod.use_hole_tolerant = True
bpy.ops.object.modifier_apply(modifier=mod.name)
print(f"  after union: verts={len(body.data.vertices):,} "
      f"polys={len(body.data.polygons):,}")
bpy.data.objects.remove(wall_obj, do_unlink=True)


print(f"\nWriting {OUTPUT}...")
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
