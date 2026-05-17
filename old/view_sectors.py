"""
Blender script to show all 29 pan shell sectors assembled.
Run with: blender --python view_sectors.py
"""
import bpy
import os
import json

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

base_dir = os.path.dirname(bpy.data.filepath) or os.getcwd()

# Ring-based colors
RING_COLORS = {
    'outer': (0.2, 0.5, 0.9, 1.0),    # Blue
    'central': (0.9, 0.6, 0.2, 1.0),   # Orange
    'inner': (0.3, 0.8, 0.4, 1.0),     # Green
}

# Load sector properties
props_path = os.path.join(base_dir, "data/sectors/sector_properties.json")
if not os.path.exists(props_path):
    print(f"ERROR: {props_path} not found. Run 'python generate_sector.py --all' first.")
else:
    with open(props_path) as f:
        sectors = json.load(f)

    for sector in sectors:
        idx = sector['index']
        ring = sector['ring']
        obj_path = os.path.join(base_dir, sector['obj_path'])

        if not os.path.exists(obj_path):
            print(f"WARNING: {obj_path} not found")
            continue

        bpy.ops.wm.obj_import(filepath=obj_path)

        for obj in bpy.context.selected_objects:
            mat = bpy.data.materials.new(name=f"Mat_{idx}")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                color = RING_COLORS.get(ring, (0.5, 0.5, 0.5, 1.0))
                bsdf.inputs["Base Color"].default_value = color
                bsdf.inputs["Metallic"].default_value = 0.3
                bsdf.inputs["Roughness"].default_value = 0.5
            obj.data.materials.clear()
            obj.data.materials.append(mat)

        print(f"Imported sector {idx} ({ring})")

    print(f"\nLoaded {len(sectors)} sectors")

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

print("\n=== Sectors loaded ===")
print("Blue: Outer ring (12)  Orange: Central ring (12)  Green: Inner ring (5)")
