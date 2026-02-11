"""
Blender script to import and display the mounting system components.
Run with: blender --python view_in_blender.py
"""
import bpy
import os
import math

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

base_dir = os.path.dirname(bpy.data.filepath) or os.getcwd()

files = [
    ("data/notepads/notepad_O0.stl", "Notepad_O0", (0.2, 0.6, 0.9, 1.0)),
    ("data/mounts/mount_base.stl", "Mount_Base", (0.9, 0.5, 0.2, 1.0)),
    ("data/mounts/outer_sleeve.stl", "Outer_Sleeve", (0.5, 0.8, 0.3, 1.0)),
]

imported = []
for rel_path, name, color in files:
    path = os.path.join(base_dir, rel_path)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping")
        continue
    bpy.ops.wm.stl_import(filepath=path)
    obj = bpy.context.active_object
    obj.name = name

    # Assign a material with color
    mat = bpy.data.materials.new(name=f"Mat_{name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.4
    obj.data.materials.append(mat)

    imported.append(obj)
    print(f"Imported: {name} from {rel_path}")

# Spread objects out so they're visible
if len(imported) >= 3:
    imported[0].location = (-80, 0, 0)   # Notepad on left
    imported[1].location = (0, 0, 0)     # Mount base center
    imported[2].location = (80, 0, 0)    # Outer sleeve on right

# Set up camera to see all objects
bpy.ops.object.select_all(action='SELECT')
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        with bpy.context.temp_override(area=area):
            bpy.ops.view3d.view_selected()
            # Set shading to Material Preview
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
        break

print("\n=== All components loaded ===")
print("Blue: Notepad O0 (with screw bosses + anti-rotation rib)")
print("Orange: Mount Base (helical external threads + groove)")
print("Green: Outer Sleeve (helical internal threads, deeper)")
