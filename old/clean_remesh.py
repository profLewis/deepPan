"""
Clean a remesh STL in Blender. Two passes:
  1. Global gentle merge (e.g. 0.01mm) to fuse near-coincident verts.
  2. Optional masked merge in the central ring (XZ radius 60..150mm) at a
     larger tolerance to close visible micro-cracks at pad/body junctions
     without losing detail elsewhere.

Run headless:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python clean_remesh.py \
        -- [<input.stl>] [--merge-dist=0.01] [--center-merge=0.2]
                       [--r-inner=60] [--r-outer=150]
"""
import bpy
import bmesh
import math
import os
import sys

argv = sys.argv
user_args = argv[argv.index("--") + 1:] if "--" in argv else []
MERGE_DIST = 0.01
CENTER_MERGE = 0.2
R_INNER = 60.0
R_OUTER = 150.0
INPUT = None
for a in user_args:
    if a.startswith("--merge-dist="):
        MERGE_DIST = float(a.split("=", 1)[1])
    elif a.startswith("--center-merge="):
        CENTER_MERGE = float(a.split("=", 1)[1])
    elif a.startswith("--r-inner="):
        R_INNER = float(a.split("=", 1)[1])
    elif a.startswith("--r-outer="):
        R_OUTER = float(a.split("=", 1)[1])
    elif not a.startswith("--"):
        INPUT = a
if INPUT is None:
    INPUT = "data/quarters/pan_assembly_remesh.stl"
INPUT = os.path.abspath(INPUT)
OUT_STL = INPUT
print(f"Input:        {INPUT}")
print(f"Global merge: {MERGE_DIST} mm")
print(f"Central merge: {CENTER_MERGE} mm  (R in [{R_INNER}, {R_OUTER}] mm; "
      f"drum axis = Y)")

bpy.ops.wm.read_factory_settings(use_empty=True)
print("Importing STL...")
bpy.ops.wm.stl_import(filepath=INPUT)
mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not mesh_objs:
    raise SystemExit("No mesh imported")
obj = mesh_objs[0]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
print(f"  {len(obj.data.vertices):,} verts, {len(obj.data.polygons):,} polys")


def diag(label):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    n_boundary = sum(1 for e in bm.edges if e.is_boundary)
    n_nm_edge = sum(1 for e in bm.edges if not e.is_manifold and not e.is_boundary)
    n_wire = sum(1 for e in bm.edges if e.is_wire)
    bm.free()
    print(f"  [{label}] boundary={n_boundary}  non-manifold(>2 faces)={n_nm_edge}  wire={n_wire}")


diag("after import")

# Pass 1: global gentle merge + cleanup
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
print(f"\nPass 1: global merge {MERGE_DIST}mm...")
bpy.ops.mesh.remove_doubles(threshold=MERGE_DIST)
bpy.ops.mesh.delete_loose()
bpy.ops.mesh.dissolve_degenerate(threshold=0.001)
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')
print(f"  After pass 1: {len(obj.data.vertices):,} verts, {len(obj.data.polygons):,} polys")
diag("after pass 1")

# Pass 2: central-ring masked merge
if CENTER_MERGE > 0:
    print(f"\nPass 2: central-ring merge {CENTER_MERGE}mm "
          f"(R = sqrt(X^2 + Z^2) in [{R_INNER}, {R_OUTER}])...")
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    n_sel = 0
    for v in bm.verts:
        r = math.hypot(v.co.x, v.co.z)
        if R_INNER <= r <= R_OUTER:
            v.select = True
            n_sel += 1
        else:
            v.select = False
    # Sync selection mode and visualization
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)
    print(f"  Selected {n_sel:,} verts in central ring band")
    bpy.ops.mesh.remove_doubles(threshold=CENTER_MERGE)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"  After pass 2: {len(obj.data.vertices):,} verts, {len(obj.data.polygons):,} polys")
    diag("after pass 2")

bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

bpy.ops.wm.stl_export(
    filepath=OUT_STL,
    export_selected_objects=True,
    apply_modifiers=True,
    ascii_format=False,
)
print(f"\nWrote {OUT_STL}")
