"""
Build a printable pan from:
  - pipeline_output/pan_holes_solid_merged.obj  (body with central+inner ellipsoids merged in)
  - data/notepads/all_notepads.obj              (all 29 pads with bosses, screw holes)
  - data/grooves/grooves_outer_down.obj         (outer-ring grooves only;
                                                 central+inner are inside the merged body)

Joins them, voxel-remeshes at 0.5mm to fuse all parts and close any small
gaps, runs Blender cleanup, exports OBJ+STL.

Output:
    pipeline_output/pan_printable.obj
    pipeline_output/pan_printable.stl

Followed by strip_debris on the STL (Python step) to remove any tiny
disconnected fragments left by the voxel remesh.

Run headless:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python make_printable.py \
        -- [--voxel=0.5]
"""
import bpy
import bmesh
import os
import sys

argv = sys.argv
user_args = argv[argv.index("--") + 1:] if "--" in argv else []
VOXEL_SIZE = 0.5
for a in user_args:
    if a.startswith("--voxel="):
        VOXEL_SIZE = float(a.split("=", 1)[1])

ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCES = [
    os.path.join(ROOT, "pipeline_output/pan_holes_solid_merged.obj"),
    os.path.join(ROOT, "data/notepads/all_notepads.obj"),
    os.path.join(ROOT, "data/grooves/grooves_outer_down.obj"),
]
OUT_DIR = os.path.join(ROOT, "pipeline_output")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_OBJ = os.path.join(OUT_DIR, "pan_printable.obj")
OUT_STL = os.path.join(OUT_DIR, "pan_printable.stl")

print(f"Sources:")
for s in SOURCES:
    print(f"  {s}  (exists={os.path.exists(s)})")
print(f"Voxel size: {VOXEL_SIZE} mm")
print(f"Output:     {OUT_OBJ}\n            {OUT_STL}")

bpy.ops.wm.read_factory_settings(use_empty=True)

print("\nImporting sources...")
for src in SOURCES:
    if not os.path.exists(src):
        print(f"  SKIP missing: {src}")
        continue
    bpy.ops.wm.obj_import(filepath=src)
    print(f"  imported {src}")

meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
print(f"\nImported {len(meshes)} mesh object(s)")
total_v = sum(len(m.data.vertices) for m in meshes)
total_p = sum(len(m.data.polygons) for m in meshes)
print(f"Pre-join total: verts={total_v:,}  polys={total_p:,}")

# Select all + join
for o in bpy.context.scene.objects:
    o.select_set(False)
for m in meshes:
    m.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
print(f"After join: verts={len(obj.data.vertices):,}  polys={len(obj.data.polygons):,}")


def diag(label):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    n_b = sum(1 for e in bm.edges if e.is_boundary)
    n_nm = sum(1 for e in bm.edges if not e.is_manifold and not e.is_boundary)
    bm.free()
    print(f"  [{label}] boundary={n_b}  non-manifold={n_nm}")


# Voxel remesh: fuses everything into a single watertight shell
print(f"\nVoxel remesh ({VOXEL_SIZE}mm)...")
mod = obj.modifiers.new(name="VoxelRemesh", type='REMESH')
mod.mode = 'VOXEL'
mod.voxel_size = VOXEL_SIZE
mod.adaptivity = 0.0
mod.use_smooth_shade = True
bpy.ops.object.modifier_apply(modifier=mod.name)
print(f"After remesh: verts={len(obj.data.vertices):,}  polys={len(obj.data.polygons):,}")

# Final cleanup pass
print("\nCleanup...")
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.mesh.remove_doubles(threshold=0.001)
bpy.ops.mesh.delete_loose()
bpy.ops.mesh.dissolve_degenerate(threshold=0.001)
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')
print(f"After cleanup: verts={len(obj.data.vertices):,}  polys={len(obj.data.polygons):,}")
diag("final")

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
