#!/usr/bin/env python3
"""
Extract only 'Pan' material objects from the tenor pan OBJ file.
Applies the same CM_TO_MM * PAN_SCALE transform used by generate_notepad.py.
"""

CM_TO_MM = 10.0
PAN_SCALE = 2.0
TOTAL_SCALE = CM_TO_MM * PAN_SCALE

def extract_pan_objects(input_path, output_path):
    """Extract objects with 'Pan' material, rescaled to match notepad geometry."""
    all_vertices = []
    pan_groups = {}  # group_name -> list of face lines
    current_group = None
    current_mat = None
    is_pan = False

    with open(input_path, 'r') as f:
        content = f.read().replace('\\\r\n', ' ').replace('\\\n', ' ')

    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('v '):
            parts = line.split()
            all_vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith('g '):
            current_group = line[2:]
        elif line.startswith('usemtl '):
            current_mat = line[7:]
            is_pan = (current_mat == 'Pan')
            if is_pan and current_group:
                pan_groups[current_group] = []
        elif line.startswith('f ') and is_pan and current_group in pan_groups:
            pan_groups[current_group].append(line)

    # Collect all vertex indices used by pan faces
    used_indices = set()
    for group, face_lines in pan_groups.items():
        for fl in face_lines:
            for p in fl.split()[1:]:
                try:
                    idx = int(p.split('/')[0]) - 1
                    used_indices.add(idx)
                except ValueError:
                    pass

    # Create re-indexed vertex mapping
    sorted_indices = sorted(used_indices)
    old_to_new = {old: new for new, old in enumerate(sorted_indices)}

    # Compute centroid of used vertices for centering at origin
    cx = sum(all_vertices[i][0] for i in sorted_indices) / len(sorted_indices)
    cy = sum(all_vertices[i][1] for i in sorted_indices) / len(sorted_indices)
    cz = sum(all_vertices[i][2] for i in sorted_indices) / len(sorted_indices)

    # Write output OBJ with scaled and centered vertices
    with open(output_path, 'w') as f:
        f.write("# Pan surface objects extracted from Tenor Pan only.obj\n")
        f.write(f"# Scaled by {TOTAL_SCALE}x (CM_TO_MM={CM_TO_MM} * PAN_SCALE={PAN_SCALE})\n")
        f.write(f"# Centered at origin\n")
        f.write(f"# Units: mm\n\n")

        # Write vertices (scaled and centered)
        for old_idx in sorted_indices:
            x, y, z = all_vertices[old_idx]
            f.write(f"v {(x - cx) * TOTAL_SCALE:.6f} {(y - cy) * TOTAL_SCALE:.6f} {(z - cz) * TOTAL_SCALE:.6f}\n")

        f.write("\n")

        # Write groups and faces
        for group, face_lines in pan_groups.items():
            f.write(f"g {group}\n")
            f.write("usemtl Pan\n")
            for fl in face_lines:
                parts = fl.split()[1:]
                new_parts = []
                for p in parts:
                    try:
                        old_idx = int(p.split('/')[0]) - 1
                        new_parts.append(str(old_to_new[old_idx] + 1))
                    except (ValueError, KeyError):
                        continue
                if len(new_parts) >= 3:
                    f.write("f " + " ".join(new_parts) + "\n")

    print(f"Extracted {len(pan_groups)} Pan objects ({len(sorted_indices)} vertices)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    extract_pan_objects("data/Tenor Pan only.obj", "data/pan_surface.obj")
