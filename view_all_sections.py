"""
Blender script to show all 6 pan sections assembled.
Verifies they tile into a complete circle with star-pattern walls.
Run with: blender --python view_all_sections.py
"""
import bpy
import os
import json

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

base_dir = os.path.dirname(bpy.data.filepath) or os.getcwd()
props_path = os.path.join(base_dir, "data/quarters/section_properties.json")

if not os.path.exists(props_path):
    print(f"ERROR: {props_path} not found. Run 'python generate_quarter.py --all' first.")
    raise SystemExit(1)

with open(props_path) as f:
    sdata = json.load(f)

# Colors: one per section for visual distinction
COLORS = [
    (0.2, 0.5, 0.9, 1.0),   # Blue
    (0.9, 0.3, 0.2, 1.0),   # Red
    (0.2, 0.8, 0.3, 1.0),   # Green
    (0.9, 0.7, 0.1, 1.0),   # Yellow
    (0.7, 0.2, 0.8, 1.0),   # Purple
    (0.1, 0.8, 0.8, 1.0),   # Cyan
]

for i, s in enumerate(sdata['sections']):
    section_id = s['section']
    obj_path = os.path.join(base_dir, s['obj_path'])

    if not os.path.exists(obj_path):
        print(f"WARNING: {obj_path} not found")
        continue

    bpy.ops.wm.obj_import(filepath=obj_path)

    mat = bpy.data.materials.new(name=f"Mat_{section_id}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = COLORS[i % len(COLORS)]
        bsdf.inputs["Metallic"].default_value = 0.4
        bsdf.inputs["Roughness"].default_value = 0.4

    for obj in bpy.context.selected_objects:
        obj.name = f"Section_{section_id}"
        obj.data.materials.clear()
        obj.data.materials.append(mat)

    print(f"Imported {section_id}: {s['bbox_size'][0]:.0f}x{s['bbox_size'][1]:.0f}x{s['bbox_size'][2]:.0f}mm  fits={s['fits_p1s']}")

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

print(f"\n=== All {len(sdata['sections'])} sections loaded ===")
print("Each section shown in a different color")
print("Verify: complete circle, star-pattern walls, cable hole at center")
