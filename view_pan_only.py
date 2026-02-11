"""
Blender script to import only the Pan material objects from the tenor pan OBJ.
Run with: blender --python view_pan_only.py
"""
import bpy
import os

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

base_dir = os.path.dirname(bpy.data.filepath) or os.getcwd()

# Import the full OBJ
pan_path = os.path.join(base_dir, "data/Tenor Pan only.obj")
if os.path.exists(pan_path):
    bpy.ops.wm.obj_import(filepath=pan_path)
    print(f"Imported: {pan_path}")
else:
    print(f"ERROR: {pan_path} not found")

# Delete all objects that don't have the "Pan" material
to_delete = []
pan_objects = []
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        to_delete.append(obj)
        continue
    has_pan = False
    for mat_slot in obj.material_slots:
        if mat_slot.material and 'Pan' in mat_slot.material.name:
            has_pan = True
            break
    if not has_pan:
        to_delete.append(obj)
    else:
        pan_objects.append(obj)

# Delete non-Pan objects
bpy.ops.object.select_all(action='DESELECT')
for obj in to_delete:
    obj.select_set(True)
if to_delete:
    bpy.ops.object.delete()

# Scale to match notepad dimensions (OBJ is in cm, notepads use cm*10*2 = 20x)
SCALE = 20.0
for obj in pan_objects:
    obj.scale = (SCALE, SCALE, SCALE)

# Apply metallic material to Pan objects
for obj in pan_objects:
    if obj.name in [o.name for o in bpy.data.objects]:
        for mat_slot in obj.material_slots:
            if mat_slot.material:
                mat = mat_slot.material
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    bsdf.inputs["Base Color"].default_value = (0.7, 0.72, 0.75, 1.0)
                    bsdf.inputs["Metallic"].default_value = 0.9
                    bsdf.inputs["Roughness"].default_value = 0.25

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

print(f"\n=== Pan objects only: {len(pan_objects)} objects loaded ===")
