"""
Blender script to show the oriented pan surface.
Drum wall is vertical along +Z.
Run with: blender --python view_oriented_pan.py
"""
import bpy
import os

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

base_dir = os.path.dirname(bpy.data.filepath) or os.getcwd()
obj_path = os.path.join(base_dir, "data/quarters/pan_oriented.obj")

if not os.path.exists(obj_path):
    print(f"ERROR: {obj_path} not found")
    raise SystemExit(1)

bpy.ops.wm.obj_import(filepath=obj_path)

# Apply material
mat = bpy.data.materials.new(name="PanSurface")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (0.7, 0.7, 0.75, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.8
    bsdf.inputs["Roughness"].default_value = 0.3

for obj in bpy.context.selected_objects:
    obj.name = "OrientedPan"
    obj.data.materials.clear()
    obj.data.materials.append(mat)

# Frame view
bpy.ops.object.select_all(action='SELECT')
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        with bpy.context.temp_override(area=area):
            bpy.ops.view3d.view_selected()
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
        break

print("\n=== Oriented pan loaded ===")
print("Drum wall vertical along +Z")
print(f"Loaded from: {obj_path}")
