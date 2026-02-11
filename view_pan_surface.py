"""
Blender script to show just the extracted Pan surface.
Run with: blender --python view_pan_surface.py
"""
import bpy
import os

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

base_dir = os.path.dirname(bpy.data.filepath) or os.getcwd()

path = os.path.join(base_dir, "data/pan_surface.obj")
if os.path.exists(path):
    bpy.ops.wm.obj_import(filepath=path)
    for obj in bpy.context.selected_objects:
        mat = bpy.data.materials.new(name=f"Mat_{obj.name}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.75, 0.75, 0.78, 1.0)
            bsdf.inputs["Metallic"].default_value = 0.9
            bsdf.inputs["Roughness"].default_value = 0.25
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    print(f"Imported pan surface: {len(bpy.context.selected_objects)} objects")
else:
    print(f"ERROR: {path} not found")

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
