"""
Boolean-union pan_holes_solid with the per-pad groove ellipsoids (generated
by make_groove_ellipsoids.py) to produce a fixed body whose surface has no
holes at the groove/pan junctions.

Run headless:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python merge_body_with_ellipsoids.py \
        -- [--rings=central[,inner][,outer]] [--solver=EXACT|FAST]

Default: --rings=central  (only central ellipsoids — where the holes are)
Default solver: EXACT
Output: pipeline_output/pan_holes_solid_merged.{obj,stl}
"""
import bpy
import os
import sys

argv = sys.argv
user_args = argv[argv.index("--") + 1:] if "--" in argv else []
RINGS = ["central"]
SOLVER = "EXACT"
for a in user_args:
    if a.startswith("--rings="):
        RINGS = [r.strip() for r in a.split("=", 1)[1].split(",") if r.strip()]
    elif a.startswith("--solver="):
        SOLVER = a.split("=", 1)[1].upper()

ROOT = os.path.dirname(os.path.abspath(__file__))
BODY = os.path.join(ROOT, "data/quarters/pan_holes_solid.obj")
OUT_DIR = os.path.join(ROOT, "pipeline_output")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_OBJ = os.path.join(OUT_DIR, "pan_holes_solid_merged.obj")
OUT_STL = os.path.join(OUT_DIR, "pan_holes_solid_merged.stl")
ELLIPSOIDS = [os.path.join(OUT_DIR, f"groove_ellipsoids_{r}.obj") for r in RINGS]

print(f"Body:        {BODY}")
print(f"Ellipsoids:  {ELLIPSOIDS}")
print(f"Output:      {OUT_OBJ}\n             {OUT_STL}")
print(f"Solver:      {SOLVER}")

bpy.ops.wm.read_factory_settings(use_empty=True)
print("\nImporting body...")
bpy.ops.wm.obj_import(filepath=BODY)
body = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
body.name = "Body"
print(f"  verts={len(body.data.vertices):,} polys={len(body.data.polygons):,}")

# Pre-clean body
bpy.context.view_layer.objects.active = body
body.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=0.001)
bpy.ops.mesh.delete_loose()
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')
print(f"  after clean: verts={len(body.data.vertices):,} polys={len(body.data.polygons):,}")

# Import each ellipsoid set as one object, boolean-union into body
for ei, ell_path in enumerate(ELLIPSOIDS):
    if not os.path.exists(ell_path):
        print(f"  WARN: missing {ell_path}, skipping")
        continue
    print(f"\nImporting {ell_path} ...")
    before = {o.name for o in bpy.context.scene.objects}
    bpy.ops.wm.obj_import(filepath=ell_path)
    new = [o for o in bpy.context.scene.objects
           if o.name not in before and o.type == 'MESH']
    if len(new) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for o in new:
            o.select_set(True)
        bpy.context.view_layer.objects.active = new[0]
        bpy.ops.object.join()
        ell = bpy.context.view_layer.objects.active
    else:
        ell = new[0]
    ell.name = f"Ellipsoids_{ei}"
    print(f"  verts={len(ell.data.vertices):,} polys={len(ell.data.polygons):,}")

    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    print(f"  Boolean UNION ({SOLVER})...")
    mod = body.modifiers.new(name=f"Union_{ei}", type='BOOLEAN')
    mod.operation = 'UNION'
    mod.object = ell
    mod.solver = SOLVER
    if SOLVER == 'EXACT':
        mod.use_self = False
        mod.use_hole_tolerant = True
    bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"  after union: verts={len(body.data.vertices):,} polys={len(body.data.polygons):,}")
    bpy.data.objects.remove(ell, do_unlink=True)

# Final cleanup
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
print(f"\nFINAL: verts={len(body.data.vertices):,} polys={len(body.data.polygons):,}")

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
