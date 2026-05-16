"""
Decimate an STL via Blender's Decimate modifier (collapse mode) to a target
triangle count. Use this when the voxel-remesh output (~17M tris @ 0.5mm)
is too heavy for downstream tools.

Run headless:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python decimate.py \
        -- <input.stl> <output.stl> [--ratio=0.18]
"""
import bpy
import os
import sys

argv = sys.argv
user_args = argv[argv.index("--") + 1:] if "--" in argv else []
positional = [a for a in user_args if not a.startswith("--")]
RATIO = 0.18  # 17M -> ~3M
for a in user_args:
    if a.startswith("--ratio="):
        RATIO = float(a.split("=", 1)[1])

if len(positional) < 2:
    raise SystemExit("usage: decimate.py <input.stl> <output.stl> [--ratio=0.18]")

INPUT = os.path.abspath(positional[0])
OUTPUT = os.path.abspath(positional[1])
print(f"Input:  {INPUT}")
print(f"Output: {OUTPUT}")
print(f"Ratio:  {RATIO}")

bpy.ops.wm.read_factory_settings(use_empty=True)
print("Importing...")
bpy.ops.wm.stl_import(filepath=INPUT)
obj = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
print(f"  before: {len(obj.data.vertices):,} verts, {len(obj.data.polygons):,} polys")

mod = obj.modifiers.new("Dec", type='DECIMATE')
mod.decimate_type = 'COLLAPSE'
mod.ratio = RATIO
mod.use_collapse_triangulate = True
bpy.ops.object.modifier_apply(modifier=mod.name)
print(f"  after:  {len(obj.data.vertices):,} verts, {len(obj.data.polygons):,} polys")

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.wm.stl_export(
    filepath=OUTPUT,
    export_selected_objects=True,
    apply_modifiers=True,
    ascii_format=False,
)
print(f"Wrote {OUTPUT}")
