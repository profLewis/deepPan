#!/usr/bin/env python3
"""
Generate a labeled diagram of the tenor pan from OBJ file data.
Projects the actual 3D geometry onto the rim plane (top-down view).
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
from matplotlib.path import Path
import matplotlib.patheffects as pe

# Fixed note mapping based on 3D object names
# Maps (grove_object, pan_object) -> (index, note, ring_type)
# Outer: O0 at top (~90°), Central: C0 at top (~92°), Inner: I0 at top (~110°)
NOTE_MAPPING = {
    # Outer Ring (4ths) - unchanged
    ('object_58', 'object_62'): ('O0', 'Bb', 'outer'),
    ('object_57', 'object_63'): ('O1', 'F', 'outer'),
    ('object_56', 'object_64'): ('O2', 'C', 'outer'),
    ('object_55', 'object_90'): ('O3', 'G', 'outer'),
    ('object_54', 'object_65'): ('O4', 'D', 'outer'),
    ('object_53', 'object_66'): ('O5', 'A', 'outer'),
    ('object_59', 'object_60'): ('O6', 'E', 'outer'),
    ('object_52', 'object_61'): ('O7', 'B', 'outer'),
    ('object_51', 'object_67'): ('O8', 'F#', 'outer'),
    ('object_50', 'object_88'): ('O9', 'C#', 'outer'),
    ('object_49', 'object_68'): ('O10', 'Ab', 'outer'),
    ('object_48', 'object_69'): ('O11', 'Eb', 'outer'),
    # Central Ring (5ths) - rotated 1 anti-clockwise
    ('object_25', 'object_45'): ('C0', 'C#', 'central'),
    ('object_24', 'object_46'): ('C1', 'Ab', 'central'),
    ('object_23', 'object_47'): ('C2', 'Eb', 'central'),
    ('object_22', 'object_37'): ('C3', 'Bb', 'central'),
    ('object_21', 'object_38'): ('C4', 'F', 'central'),
    ('object_20', 'object_39'): ('C5', 'C', 'central'),
    ('object_73', 'object_31'): ('C6', 'G', 'central'),
    ('object_30', 'object_40'): ('C7', 'D', 'central'),
    ('object_29', 'object_41'): ('C8', 'A', 'central'),
    ('object_28', 'object_42'): ('C9', 'E', 'central'),
    ('object_27', 'object_43'): ('C10', 'B', 'central'),
    ('object_26', 'object_44'): ('C11', 'F#', 'central'),
    # Inner Ring (6ths) - rotated 1 anti-clockwise
    ('object_72', 'object_36'): ('I0', 'Eb', 'inner'),
    ('object_71', 'object_32'): ('I1', 'C', 'inner'),
    ('object_19', 'object_33'): ('I2', 'E', 'inner'),
    ('object_70', 'object_34'): ('I3', 'D', 'inner'),
    ('object_18', 'object_35'): ('I4', 'C#', 'inner'),
}


def parse_obj_file(filepath):
    """Parse OBJ file and extract geometry."""
    objects = {}
    current_object = None
    current_material = None
    all_vertices = []

    with open(filepath, 'r') as f:
        content = f.read()
        content = content.replace('\\\r\n', ' ').replace('\\\n', ' ')
        lines = content.split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith('g '):
            current_object = line[2:]
        elif line.startswith('usemtl '):
            current_material = line[7:]
            if current_object:
                objects[current_object] = {
                    'material': current_material,
                    'face_vertices': set(),
                    'faces': []
                }
        elif line.startswith('v '):
            parts = line.split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            all_vertices.append((x, y, z))
        elif line.startswith('f ') and current_object in objects:
            parts = line.split()[1:]
            face_verts = []
            for p in parts:
                try:
                    v_idx = int(p.split('/')[0]) - 1
                    objects[current_object]['face_vertices'].add(v_idx)
                    face_verts.append(v_idx)
                except ValueError:
                    continue
            if face_verts:
                objects[current_object]['faces'].append(face_verts)

    return objects, np.array(all_vertices)


def get_object_boundary(objects, obj_name, all_vertices):
    """Get the 2D boundary of an object projected onto X-Z plane."""
    obj = objects[obj_name]
    indices = list(obj['face_vertices'])
    if not indices:
        return None, None

    verts = all_vertices[indices]
    # Project to X-Z plane (top-down view, Y is vertical)
    points_2d = verts[:, [0, 2]]  # X and Z coordinates

    # Calculate centroid
    centroid = points_2d.mean(axis=0)

    return points_2d, centroid


def get_object_faces_2d(objects, obj_name, all_vertices):
    """Get all faces of an object projected to 2D (X-Z plane)."""
    obj = objects[obj_name]
    faces_2d = []

    for face in obj['faces']:
        if len(face) >= 3:
            face_verts = all_vertices[face]
            # Project to X-Z plane
            face_2d = face_verts[:, [0, 2]]
            faces_2d.append(face_2d)

    return faces_2d


def get_convex_hull(points):
    """Get convex hull of 2D points."""
    from scipy.spatial import ConvexHull
    if len(points) < 3:
        return points
    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except:
        return points


def distance_3d(c1, c2):
    """Calculate 3D distance."""
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)


def get_centroid_3d(objects, obj_name, all_vertices):
    """Get 3D centroid of an object."""
    obj = objects[obj_name]
    indices = list(obj['face_vertices'])
    if not indices:
        return None
    verts = all_vertices[indices]
    return verts.mean(axis=0)


def create_note_pads(objects, all_vertices):
    """Match Groves and Pan objects to create note pads."""
    groves = []
    pans = []

    for name, obj in objects.items():
        centroid = get_centroid_3d(objects, name, all_vertices)
        if centroid is not None:
            if obj['material'] == 'Groves':
                groves.append((name, centroid))
            elif obj['material'] == 'Pan':
                pans.append((name, centroid))

    note_pads = []
    used_pans = set()

    for grove_name, grove_centroid in groves:
        best_pan = None
        best_dist = float('inf')
        for pan_name, pan_centroid in pans:
            if pan_name in used_pans:
                continue
            d = distance_3d(grove_centroid, pan_centroid)
            if d < best_dist:
                best_dist = d
                best_pan = (pan_name, pan_centroid)
        if best_pan and best_dist < 15:
            # Get 2D projection centroid (X-Z plane)
            combined_centroid_2d = np.array([
                (grove_centroid[0] + best_pan[1][0]) / 2,
                (grove_centroid[2] + best_pan[1][2]) / 2  # Z becomes Y in 2D
            ])
            note_pads.append({
                'grove': grove_name,
                'pan': best_pan[0],
                'centroid_2d': combined_centroid_2d,
                'centroid_3d': (grove_centroid + best_pan[1]) / 2
            })
            used_pans.add(best_pan[0])

    return note_pads


def classify_rings(note_pads):
    """Classify note pads into inner, central, and outer rings."""
    # Calculate pan center in 2D
    centroids = np.array([pad['centroid_2d'] for pad in note_pads])
    center = centroids.mean(axis=0)

    for pad in note_pads:
        dx = pad['centroid_2d'][0] - center[0]
        dy = pad['centroid_2d'][1] - center[1]
        pad['radius'] = math.sqrt(dx**2 + dy**2)
        pad['angle'] = math.degrees(math.atan2(dy, dx))

    # Sort by radius and classify
    note_pads.sort(key=lambda x: x['radius'])

    inner_ring = note_pads[:5]
    central_ring = note_pads[5:17]
    outer_ring = note_pads[17:]

    return inner_ring, central_ring, outer_ring, center


def assign_notes_from_mapping(note_pads):
    """Assign note names to pads using the fixed NOTE_MAPPING dictionary.

    This ensures the diagram exactly matches the documented object-to-note mapping.
    """
    for pad in note_pads:
        key = (pad['grove'], pad['pan'])
        if key in NOTE_MAPPING:
            index, note, ring_type = NOTE_MAPPING[key]
            pad['index'] = index
            pad['note'] = note
            pad['ring_type'] = ring_type
        else:
            # Fallback for any unmatched pads
            pad['index'] = '??'
            pad['note'] = '?'
            pad['ring_type'] = 'unknown'

    return note_pads


def generate_3d_diagram(objects, all_vertices, inner_ring, central_ring, outer_ring, output_path):
    """Generate a top-down diagram using actual 3D geometry."""

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))

    # Dark background
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#0d0d0d')

    # Color schemes
    colors = {
        'outer': '#c0392b',     # Dark red
        'central': '#2471a3',   # Dark blue
        'inner': '#1e8449',     # Dark green
        'grove': '#2c3e50',     # Dark gray for grooves
        'rim': '#7f8c8d'        # Gray for rim
    }

    highlight_colors = {
        'outer': '#e74c3c',     # Bright red
        'central': '#3498db',   # Bright blue
        'inner': '#2ecc71',     # Bright green
    }

    # Draw the rim circle
    # Find rim by getting max radius of pan/groves vertices
    pan_grove_verts = []
    for name, obj in objects.items():
        if obj['material'] in ['Pan', 'Groves']:
            for idx in obj['face_vertices']:
                pan_grove_verts.append(all_vertices[idx])
    pan_grove_verts = np.array(pan_grove_verts)

    center_x = pan_grove_verts[:, 0].mean()
    center_z = pan_grove_verts[:, 2].mean()

    radii = np.sqrt((pan_grove_verts[:,0] - center_x)**2 + (pan_grove_verts[:,2] - center_z)**2)
    rim_radius = radii.max() * 1.05  # Slightly larger than max

    rim_circle = plt.Circle((center_x, center_z), rim_radius,
                             fill=False, color=colors['rim'], linewidth=3)
    ax.add_patch(rim_circle)

    # Draw each note pad using actual geometry
    def draw_note_pad_geometry(pad, ring_type):
        pan_name = pad['pan']
        grove_name = pad['grove']

        # Get faces for the pan object
        pan_faces = get_object_faces_2d(objects, pan_name, all_vertices)
        grove_faces = get_object_faces_2d(objects, grove_name, all_vertices)

        # Draw grove faces (darker)
        for face in grove_faces:
            if len(face) >= 3:
                poly = plt.Polygon(face, facecolor=colors['grove'],
                                  edgecolor='none', alpha=0.7)
                ax.add_patch(poly)

        # Draw pan faces (colored by ring)
        for face in pan_faces:
            if len(face) >= 3:
                poly = plt.Polygon(face, facecolor=colors[ring_type],
                                  edgecolor=highlight_colors[ring_type],
                                  linewidth=0.5, alpha=0.85)
                ax.add_patch(poly)

        # Add label at centroid
        cx, cy = pad['centroid_2d']
        note = pad['note']
        idx = pad['index']  # Already full index like 'O0', 'C5', 'I3'

        # Note name with outline for visibility
        ax.text(cx, cy + 0.3, note,
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='white',
                path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        # Index below
        ax.text(cx, cy - 0.5, idx,
                ha='center', va='center',
                fontsize=7,
                color='#cccccc',
                path_effects=[pe.withStroke(linewidth=1, foreground='black')])

    # Draw all note pads
    for pad in outer_ring:
        draw_note_pad_geometry(pad, 'outer')
    for pad in central_ring:
        draw_note_pad_geometry(pad, 'central')
    for pad in inner_ring:
        draw_note_pad_geometry(pad, 'inner')

    # Set axis limits with padding
    padding = 2
    ax.set_xlim(center_x - rim_radius - padding, center_x + rim_radius + padding)
    ax.set_ylim(center_z - rim_radius - padding, center_z + rim_radius + padding)
    ax.set_aspect('equal')

    # Title
    ax.set_title('Tenor Pan Note Layout\n(Top-Down View from 3D Model)',
                 fontsize=18, fontweight='bold', color='white', pad=20)

    # Legend
    legend_elements = [
        patches.Patch(facecolor=highlight_colors['outer'], edgecolor='white',
                     label='Outer Ring (4ths) - 12 notes'),
        patches.Patch(facecolor=highlight_colors['central'], edgecolor='white',
                     label='Central Ring (5ths) - 12 notes'),
        patches.Patch(facecolor=highlight_colors['inner'], edgecolor='white',
                     label='Inner Ring (6ths) - 5 notes'),
    ]
    legend = ax.legend(handles=legend_elements, loc='upper left',
                       facecolor='#333333', edgecolor='white', labelcolor='white',
                       fontsize=10)
    legend.get_frame().set_alpha(0.9)

    # Index format note
    ax.text(center_x, center_z - rim_radius - 1,
            'Index: O=Outer, C=Central, I=Inner',
            ha='center', va='center', fontsize=10, color='#888888')

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    plt.close()

    print(f"Diagram saved to: {output_path}")


def generate_note_mapping(inner_ring, central_ring, outer_ring):
    """Generate a mapping dictionary for all notes."""
    mapping = {'inner': [], 'central': [], 'outer': []}

    for ring_name, ring in [('inner', inner_ring), ('central', central_ring), ('outer', outer_ring)]:
        prefix = {'inner': 'I', 'central': 'C', 'outer': 'O'}[ring_name]
        for pad in ring:
            mapping[ring_name].append({
                'index': f"{prefix}{pad['index']}",
                'note': pad['note'],
                'grove_obj': pad['grove'],
                'pan_obj': pad['pan'],
                'angle': round(pad['angle'], 1),
                'radius': round(pad['radius'], 2)
            })

    return mapping


def main():
    import os

    obj_path = "data/Tenor Pan only.obj"
    output_image = "docs/tenor_pan_layout.png"

    print("Parsing OBJ file...")
    objects, all_vertices = parse_obj_file(obj_path)

    print("Creating note pads...")
    note_pads = create_note_pads(objects, all_vertices)
    print(f"  Found {len(note_pads)} note pads")

    print("Assigning notes from fixed mapping...")
    note_pads = assign_notes_from_mapping(note_pads)

    # Separate into rings based on the assigned ring_type
    inner_ring = [p for p in note_pads if p.get('ring_type') == 'inner']
    central_ring = [p for p in note_pads if p.get('ring_type') == 'central']
    outer_ring = [p for p in note_pads if p.get('ring_type') == 'outer']

    print(f"  Inner: {len(inner_ring)}, Central: {len(central_ring)}, Outer: {len(outer_ring)}")

    print("Generating 3D-based diagram...")
    os.makedirs("docs", exist_ok=True)
    generate_3d_diagram(objects, all_vertices, inner_ring, central_ring, outer_ring, output_image)

    # Print mapping
    print("\n" + "="*60)
    print("NOTE PAD MAPPING")
    print("="*60)

    for ring_name, ring in [('outer', outer_ring), ('central', central_ring), ('inner', inner_ring)]:
        ring_label = {'outer': 'OUTER (4ths)', 'central': 'CENTRAL (5ths)', 'inner': 'INNER (6ths)'}
        print(f"\n{ring_label[ring_name]}:")
        for pad in sorted(ring, key=lambda x: x['index']):
            print(f"  {pad['index']:4s} | {pad['note']:3s} | Grove: {pad['grove']:12s} | Pan: {pad['pan']:12s}")

    return note_pads


if __name__ == "__main__":
    main()
