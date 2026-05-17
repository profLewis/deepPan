"""
Blender script to show a pan section (sixth) with its notepad pieces.
Run with: blender --python view_quarter.py

Shows:
- The section shell (blue) with support walls
- The 2 notepad playing surfaces (red) positioned correctly
"""
import bpy
import os
import json
import numpy as np
from mathutils import Matrix

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

base_dir = os.path.dirname(bpy.data.filepath) or os.getcwd()

# ============================================================
# Load properties
# ============================================================
section_props_path = os.path.join(base_dir, "data/quarters/section_properties.json")
notepad_props_path = os.path.join(base_dir, "data/notepads/notepad_properties.json")
offset_path = os.path.join(base_dir, "data/pan_centroid_offset.json")

if not os.path.exists(section_props_path):
    print(f"ERROR: {section_props_path} not found. Run 'python generate_quarter.py' first.")
    raise SystemExit(1)

with open(section_props_path) as f:
    sdata = json.load(f)

R = np.array(sdata['rotation_matrix'])
sections = sdata['sections']

with open(notepad_props_path) as f:
    notepad_props = {p['index']: p for p in json.load(f)}

with open(offset_path) as f:
    offset_data = json.load(f)
    pan_centroid_offset = np.array(offset_data['centroid_offset_mm'])

# Convert rotation to Blender 4x4 matrix
R4 = Matrix([
    [R[0][0], R[0][1], R[0][2], 0],
    [R[1][0], R[1][1], R[1][2], 0],
    [R[2][0], R[2][1], R[2][2], 0],
    [0, 0, 0, 1],
])

# ============================================================
# Colors
# ============================================================
SECTION_COLOR = (0.2, 0.5, 0.9, 1.0)    # Blue
NOTEPAD_COLOR = (0.9, 0.2, 0.2, 1.0)    # Red

def make_material(name, color, metallic=0.3, roughness=0.5):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
    return mat

# ============================================================
# Load each section and its notepads
# ============================================================
for s in sections:
    section_id = s['section']
    note_names = s['notes']
    obj_path = os.path.join(base_dir, s['obj_path'])

    if not os.path.exists(obj_path):
        print(f"WARNING: {obj_path} not found")
        continue

    # Import section mesh (already in rotated coordinate system)
    bpy.ops.wm.obj_import(filepath=obj_path)
    section_mat = make_material(f"Mat_{section_id}", SECTION_COLOR)
    for obj in bpy.context.selected_objects:
        obj.name = f"Section_{section_id}"
        obj.data.materials.clear()
        obj.data.materials.append(section_mat)

    print(f"Imported section {section_id}")

    # Import notepads for this section
    for nname in note_names:
        if nname not in notepad_props:
            print(f"  WARNING: notepad {nname} not in properties")
            continue

        np_props = notepad_props[nname]
        np_obj_path = os.path.join(base_dir, np_props['obj_path'])

        if not os.path.exists(np_obj_path):
            print(f"  WARNING: {np_obj_path} not found")
            continue

        bpy.ops.wm.obj_import(filepath=np_obj_path)

        # Position notepad on the section surface
        # The notepad OBJ is centered at its mesh geometric mean (origin)
        # offset = centroid_original - centroid_centered
        # Pan coords = offset - pan_centroid_offset
        # Section uses rotated coords, so also rotate the position
        centroid_original = np.array(np_props['centroid_original'])
        centroid_centered = np.array(np_props['centroid'])
        pan_pos = centroid_original - centroid_centered - pan_centroid_offset
        # Apply the same rotation used for the section
        rotated_pos = R @ pan_pos

        for obj in bpy.context.selected_objects:
            obj.name = f"Notepad_{nname}"
            obj.location = (rotated_pos[0], rotated_pos[1], rotated_pos[2])
            # Also rotate the notepad mesh to match the section orientation
            obj.matrix_world = R4 @ obj.matrix_world

            np_mat = make_material(f"Mat_NP_{nname}", NOTEPAD_COLOR, metallic=0.1, roughness=0.7)
            obj.data.materials.clear()
            obj.data.materials.append(np_mat)

        print(f"  Imported notepad {nname}")

# ============================================================
# Frame view
# ============================================================
bpy.ops.object.select_all(action='SELECT')
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        with bpy.context.temp_override(area=area):
            bpy.ops.view3d.view_selected()
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
        break

print(f"\n=== Section view loaded ===")
print(f"Blue: Section shell    Red: Notepad playing surfaces")
