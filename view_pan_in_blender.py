"""
Blender script to import the full tenor pan OBJ and mounting components.
Run with: blender --python view_pan_in_blender.py
"""
import bpy
import os

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

base_dir = os.path.dirname(bpy.data.filepath) or os.getcwd()

# Import the full tenor pan OBJ
pan_path = os.path.join(base_dir, "data/Tenor Pan only.obj")
if os.path.exists(pan_path):
    bpy.ops.wm.obj_import(filepath=pan_path)
    # Color all imported pan objects
    for obj in bpy.context.selected_objects:
        mat = bpy.data.materials.new(name=f"Mat_{obj.name}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.75, 0.75, 0.75, 1.0)
            bsdf.inputs["Metallic"].default_value = 0.8
            bsdf.inputs["Roughness"].default_value = 0.3
        obj.data.materials.append(mat)
    print(f"Imported tenor pan from: {pan_path}")
else:
    print(f"WARNING: {pan_path} not found")

# Import mounting components offset to the side
mount_files = [
    ("data/notepads/notepad_O0.stl", "Notepad_O0", (0.2, 0.6, 0.9, 1.0)),
    ("data/mounts/mount_base.stl", "Mount_Base", (0.9, 0.5, 0.2, 1.0)),
    ("data/mounts/outer_sleeve.stl", "Outer_Sleeve", (0.5, 0.8, 0.3, 1.0)),
]

for rel_path, name, color in mount_files:
    path = os.path.join(base_dir, rel_path)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping")
        continue
    bpy.ops.wm.stl_import(filepath=path)
    obj = bpy.context.active_object
    obj.name = name
    # Offset mounting components to the right
    obj.location = (100, 0, 0)

    mat = bpy.data.materials.new(name=f"Mat_{name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.4
    obj.data.materials.append(mat)
    print(f"Imported: {name}")

# Frame all objects
bpy.ops.object.select_all(action='SELECT')
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        with bpy.context.temp_override(area=area):
            bpy.ops.view3d.view_selected()
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
        break

print("\n=== All loaded ===")
print("Silver: Full Tenor Pan")
print("Blue/Orange/Green: Notepad, Mount Base, Outer Sleeve (offset to right)")
