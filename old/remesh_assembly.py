"""
Join all the parts in data/quarters/pan_assembly.obj into ONE watertight mesh
by voxel-remeshing. Eliminates any holes/gaps between adjoining parts.

Run headless:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python remesh_assembly.py \
        -- [<input.obj>] [--voxel=<size_mm>]

Default input: data/quarters/pan_assembly.obj
Default voxel size: 0.5 mm (smaller → more detail + slower + more memory).
Output: <input_stem>_remesh.{obj,stl}
"""
import bpy
import os
import sys

argv = sys.argv
user_args = argv[argv.index("--") + 1:] if "--" in argv else []

VOXEL_SIZE = 0.5  # mm
INPUT = None
for a in user_args:
    if a.startswith("--voxel="):
        VOXEL_SIZE = float(a.split("=", 1)[1])
    elif not a.startswith("--"):
        INPUT = a

if INPUT is None:
    INPUT = "data/quarters/pan_assembly.obj"
INPUT = os.path.abspath(INPUT)

base = os.path.splitext(INPUT)[0] + "_remesh"
OUT_OBJ = base + ".obj"
OUT_STL = base + ".stl"
print(f"Input:      {INPUT}")
print(f"Voxel size: {VOXEL_SIZE} mm")
print(f"Output:     {OUT_OBJ}\n            {OUT_STL}")

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=INPUT)

meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not meshes:
    raise SystemExit("No mesh imported")
print(f"Imported {len(meshes)} mesh object(s)")

# Join everything into one mesh
for o in bpy.context.scene.objects:
    o.select_set(False)
for m in meshes:
    m.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
print(f"After join: verts={len(obj.data.vertices)}  polys={len(obj.data.polygons)}")

# Voxel remesh — re-samples surface, guarantees watertight output, fuses
# overlapping parts into a single shell.
mod = obj.modifiers.new(name="VoxelRemesh", type='REMESH')
mod.mode = 'VOXEL'
mod.voxel_size = VOXEL_SIZE
mod.adaptivity = 0.0
mod.use_smooth_shade = True
bpy.ops.object.modifier_apply(modifier=mod.name)
print(f"After voxel remesh ({VOXEL_SIZE}mm): "
      f"verts={len(obj.data.vertices)}  polys={len(obj.data.polygons)}")

# Re-orient normals outward on the final closed solid
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

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
