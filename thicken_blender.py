"""
Thicken an OBJ into a solid body using Blender's Solidify modifier.

Run headless:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python thicken_blender.py \
        -- <input.obj> [<output_basename>]

Default input: data/pan.obj  -> data/pan_solid.{obj,stl}
"""
import bpy
import bmesh
import math
import os
import sys
from mathutils import Vector

# Parse args after the "--" separator
argv = sys.argv
if "--" in argv:
    user_args = argv[argv.index("--") + 1:]
else:
    user_args = []

USE_CORRECTIVE_SMOOTH = True
if "--no-smooth" in user_args:
    USE_CORRECTIVE_SMOOTH = False
    user_args = [a for a in user_args if a != "--no-smooth"]

WELD_THRESHOLD = 0.001
SUBDIV_OVERRIDE = None
OFFSET_OVERRIDE = None
THICKNESS_OVERRIDE = None
_rest = []
for a in user_args:
    if a.startswith("--weld="):
        WELD_THRESHOLD = float(a.split("=", 1)[1])
    elif a.startswith("--subdiv="):
        SUBDIV_OVERRIDE = int(a.split("=", 1)[1])
    elif a.startswith("--offset="):
        OFFSET_OVERRIDE = float(a.split("=", 1)[1])
    elif a.startswith("--thickness="):
        THICKNESS_OVERRIDE = float(a.split("=", 1)[1])
    else:
        _rest.append(a)
user_args = _rest

INPUT = os.path.abspath(user_args[0]) if len(user_args) >= 1 else os.path.abspath("data/pan.obj")
if len(user_args) >= 2:
    base = user_args[1]
else:
    suffix = "_solid" if USE_CORRECTIVE_SMOOTH else "_solid_raw"
    base = os.path.splitext(INPUT)[0] + suffix
OUT_OBJ = os.path.abspath(base + ".obj")
OUT_STL = os.path.abspath(base + ".stl")
print(f"Input:  {INPUT}")
print(f"Output: {OUT_OBJ}\n        {OUT_STL}")
THICKNESS = 5.0 if THICKNESS_OVERRIDE is None else THICKNESS_OVERRIDE
# Solidify OFFSET (Blender convention):
#   +1 → new surface on +normal side (outward); original stays on inside
#   -1 → new surface on -normal side (inward); original stays on outside
# Default -1: keep the original (player-facing) surface; thicken into the body.
OFFSET = -1.0 if OFFSET_OVERRIDE is None else OFFSET_OVERRIDE
SUBDIV_LEVELS = 2 if SUBDIV_OVERRIDE is None else SUBDIV_OVERRIDE
CS_FACTOR = 1.0
CS_ITERATIONS = 5
CS_SCALE = 0.5

bpy.ops.wm.read_factory_settings(use_empty=True)

bpy.ops.wm.obj_import(filepath=INPUT)

objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not objs:
    raise SystemExit("No mesh imported")

obj = objs[0]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
print(f"Imported: {obj.name}  verts={len(obj.data.vertices)}  polys={len(obj.data.polygons)}")

def diagnose_mesh(label):
    """Report manifoldness + neighbour-orientation consistency.

    An edge is 'flipped' if its two adjacent faces traverse it in the SAME
    direction (correct manifold orientation requires opposite directions).
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.normal_update()
    n_nonmanifold = sum(1 for e in bm.edges if not e.is_manifold)
    n_boundary = sum(1 for e in bm.edges if e.is_boundary)
    n_flipped = 0
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        v1, v2 = e.verts
        dirs = []
        for f in e.link_faces:
            for loop in f.loops:
                if loop.vert == v1 and loop.link_loop_next.vert == v2:
                    dirs.append(0)
                    break
                if loop.vert == v2 and loop.link_loop_next.vert == v1:
                    dirs.append(1)
                    break
        if len(dirs) == 2 and dirs[0] == dirs[1]:
            n_flipped += 1
    bm.free()
    print(f"  [{label}] non-manifold={n_nonmanifold}  boundary={n_boundary}  "
          f"flipped-neighbour edges={n_flipped}")


bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=WELD_THRESHOLD)
print(f"  Weld threshold: {WELD_THRESHOLD}mm")
bpy.ops.mesh.delete_loose()
bpy.ops.mesh.dissolve_degenerate(threshold=0.001)
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)  # outward
bpy.ops.object.mode_set(mode='OBJECT')
diagnose_mesh("after clean")

if SUBDIV_LEVELS > 0:
    sub = obj.modifiers.new(name="Subsurf", type='SUBSURF')
    sub.subdivision_type = 'SIMPLE'
    sub.levels = SUBDIV_LEVELS
    sub.render_levels = SUBDIV_LEVELS
    bpy.ops.object.modifier_apply(modifier=sub.name)
    print(f"After simple subdivide x{SUBDIV_LEVELS}: verts={len(obj.data.vertices)}  polys={len(obj.data.polygons)}")
else:
    print("Skipping subdivision (--subdiv=0)")

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
bpy.ops.mesh.beautify_fill(angle_limit=3.14159)  # re-triangulate to reduce slivers
bpy.ops.object.mode_set(mode='OBJECT')
print(f"After beauty triangulate: verts={len(obj.data.vertices)}  polys={len(obj.data.polygons)}")

mod = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
mod.thickness = THICKNESS
mod.offset = OFFSET
mod.use_even_offset = False  # True can explode at sharp angles
mod.use_quality_normals = True
mod.use_rim = True
mod.use_rim_only = False

bpy.ops.object.modifier_apply(modifier=mod.name)
print(f"After solidify: verts={len(obj.data.vertices)}  polys={len(obj.data.polygons)}")

if USE_CORRECTIVE_SMOOTH:
    cs = obj.modifiers.new(name="CorrectiveSmooth", type='CORRECTIVE_SMOOTH')
    cs.factor = CS_FACTOR
    cs.iterations = CS_ITERATIONS
    cs.scale = CS_SCALE
    cs.smooth_type = 'LENGTH_WEIGHTED'
    cs.use_only_smooth = True
    bpy.ops.object.modifier_apply(modifier=cs.name)
    print(f"After CorrectiveSmooth (length-weighted, only-smooth, "
          f"factor={CS_FACTOR}, iters={CS_ITERATIONS}, scale={CS_SCALE}): "
          f"verts={len(obj.data.vertices)}  polys={len(obj.data.polygons)}")
else:
    print("Skipping CorrectiveSmooth (--no-smooth)")

# Final orientation pass: recalc normals on the closed solid and shade smooth.
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)  # outward
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.shade_smooth()
diagnose_mesh("final")

bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

bpy.ops.wm.obj_export(
    filepath=OUT_OBJ,
    export_selected_objects=True,
    apply_modifiers=True,
    export_materials=False,
    export_triangulated_mesh=True,
)
print(f"Wrote {OUT_OBJ}")

bpy.ops.wm.stl_export(
    filepath=OUT_STL,
    export_selected_objects=True,
    apply_modifiers=True,
    ascii_format=False,
)
print(f"Wrote {OUT_STL}")
