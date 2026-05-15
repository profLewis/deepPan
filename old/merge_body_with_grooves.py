"""
Boolean-union pan_holes_solid with the three groove _down solids to make a
single closed body where the groove cavities have been filled in by the
grove geometry. Cleans up the non-manifold edges at the groove/pan junctions
that come from duplicated verts in the source bowl mesh.

Run headless:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python merge_body_with_grooves.py \
        -- [--solver=EXACT|FAST]

Outputs:
    pipeline_output/pan_holes_solid_merged.obj
    pipeline_output/pan_holes_solid_merged.stl
"""
import bpy
import bmesh
import os
import sys

argv = sys.argv
user_args = argv[argv.index("--") + 1:] if "--" in argv else []
SOLVER = "EXACT"
for a in user_args:
    if a.startswith("--solver="):
        SOLVER = a.split("=", 1)[1].upper()

ROOT = os.path.dirname(os.path.abspath(__file__))
BODY = os.path.join(ROOT, "data/quarters/pan_holes_solid.obj")
GROOVES = [
    os.path.join(ROOT, "data/grooves/grooves_central_down.obj"),
    os.path.join(ROOT, "data/grooves/grooves_inner_down.obj"),
    os.path.join(ROOT, "data/grooves/grooves_outer_down.obj"),
]
OUT_DIR = os.path.join(ROOT, "pipeline_output")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_OBJ = os.path.join(OUT_DIR, "pan_holes_solid_merged.obj")
OUT_STL = os.path.join(OUT_DIR, "pan_holes_solid_merged.stl")

print(f"Body:    {BODY}")
for g in GROOVES:
    print(f"Groove:  {g}")
print(f"Output:  {OUT_OBJ}\n         {OUT_STL}")
print(f"Solver:  {SOLVER}")


def diag(obj, label):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    n_boundary = sum(1 for e in bm.edges if e.is_boundary)
    n_nm = sum(1 for e in bm.edges if not e.is_manifold and not e.is_boundary)
    bm.free()
    print(f"  [{label}] verts={len(obj.data.vertices):,} polys={len(obj.data.polygons):,} "
          f"boundary={n_boundary} non-manifold={n_nm}")


bpy.ops.wm.read_factory_settings(use_empty=True)

print("\nImporting body...")
bpy.ops.wm.obj_import(filepath=BODY)
body = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
body.name = "Body"
diag(body, "body imported")

# Merge near-coincident verts in the body to collapse non-manifold edges.
bpy.context.view_layer.objects.active = body
body.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=0.001)
bpy.ops.mesh.delete_loose()
bpy.ops.mesh.dissolve_degenerate(threshold=0.001)
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')
diag(body, "body cleaned")

groove_objs = []
for i, g in enumerate(GROOVES):
    print(f"\nImporting groove {i}: {os.path.basename(g)}")
    before = {o.name for o in bpy.context.scene.objects}
    bpy.ops.wm.obj_import(filepath=g)
    new = [o for o in bpy.context.scene.objects if o.name not in before and o.type == 'MESH']
    print(f"  imported {len(new)} object(s) from this OBJ")
    # Join all imported groove objects from this file into one
    if len(new) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for o in new:
            o.select_set(True)
        bpy.context.view_layer.objects.active = new[0]
        bpy.ops.object.join()
        joined = bpy.context.view_layer.objects.active
    else:
        joined = new[0]
    joined.name = f"Groove_{i}"
    bpy.context.view_layer.objects.active = joined
    joined.select_set(True)
    # Clean up: merge duplicates, recalc normals
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.001)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    diag(joined, f"groove {i} cleaned")
    groove_objs.append(joined)

# Apply boolean unions in sequence on the body
bpy.context.view_layer.objects.active = body
body.select_set(True)
for i, g in enumerate(groove_objs):
    print(f"\nBoolean union with groove {i} ({SOLVER})...")
    mod = body.modifiers.new(name=f"Union_{i}", type='BOOLEAN')
    mod.operation = 'UNION'
    mod.object = g
    mod.solver = SOLVER
    if SOLVER == 'EXACT':
        mod.use_self = False
        mod.use_hole_tolerant = True
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.modifier_apply(modifier=mod.name)
    diag(body, f"after union {i}")
    bpy.data.objects.remove(g, do_unlink=True)

# Final cleanup pass
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=0.001)
bpy.ops.mesh.delete_loose()
bpy.ops.mesh.dissolve_degenerate(threshold=0.001)
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')
diag(body, "FINAL")

bpy.ops.object.select_all(action='DESELECT')
body.select_set(True)
bpy.context.view_layer.objects.active = body
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
