"""
Blender script to show extracted Pan surface + mounting components.
Run with: blender --python view_all.py
"""
import bpy
import os

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

base_dir = os.path.dirname(bpy.data.filepath) or os.getcwd()


def import_obj(rel_path, name, color, metallic=0.0, roughness=0.4, location=None):
    path = os.path.join(base_dir, rel_path)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return None
    bpy.ops.wm.obj_import(filepath=path)
    imported = bpy.context.selected_objects
    for obj in imported:
        mat = bpy.data.materials.new(name=f"Mat_{obj.name}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = color
            bsdf.inputs["Metallic"].default_value = metallic
            bsdf.inputs["Roughness"].default_value = roughness
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        if location:
            obj.location = location
    print(f"Imported: {name} ({len(imported)} objects)")
    return imported


def import_stl(rel_path, name, color, roughness=0.4, location=None):
    path = os.path.join(base_dir, rel_path)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return None
    bpy.ops.wm.stl_import(filepath=path)
    obj = bpy.context.active_object
    obj.name = name
    mat = bpy.data.materials.new(name=f"Mat_{name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
    obj.data.materials.append(mat)
    if location:
        obj.location = location
    print(f"Imported: {name}")
    return obj


# Pan surface (already scaled to mm in extract_pan_surface.py)
import_obj("data/pan_surface.obj", "Pan_Surface",
           color=(0.75, 0.75, 0.78, 1.0), metallic=0.9, roughness=0.25)

# Mounting components (offset to the side for visibility)
offset = (200, 0, 0)
import_stl("data/notepads/notepad_O0.stl", "Notepad_O0",
           color=(0.2, 0.6, 0.9, 1.0), location=offset)
import_stl("data/mounts/mount_base.stl", "Mount_Base",
           color=(0.9, 0.5, 0.2, 1.0), location=(offset[0] + 80, 0, 0))
import_stl("data/mounts/outer_sleeve.stl", "Outer_Sleeve",
           color=(0.5, 0.8, 0.3, 1.0), location=(offset[0] + 160, 0, 0))

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

print("\n=== All loaded ===")
print("Silver: Pan surface (33 note pads, scaled to mm)")
print("Blue: Notepad O0    Orange: Mount Base    Green: Outer Sleeve")
