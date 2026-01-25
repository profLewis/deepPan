#!/usr/bin/env python3
"""
Generate a labeled diagram of the tenor pan from OBJ file data.
"""

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import numpy as np

# Note assignments for each ring
OUTER_NOTES = ['F#', 'B', 'E', 'A', 'D', 'G', 'C', 'F', 'Bb', 'Eb', 'Ab', 'C#']  # 4ths
CENTRAL_NOTES = ['F#', 'B', 'E', 'A', 'D', 'G', 'C', 'F', 'Bb', 'Eb', 'Ab', 'C#']  # 5ths
INNER_NOTES = ['C#', 'E', 'D', 'C', 'Eb']  # 6ths

def parse_obj_file(filepath):
    """Parse OBJ file and extract note pad positions."""
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
            if current_object and current_material in ['Groves', 'Pan']:
                objects[current_object] = {'material': current_material, 'face_vertices': set()}
        elif line.startswith('v '):
            parts = line.split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            all_vertices.append((x, y, z))
        elif line.startswith('f ') and current_object in objects:
            parts = line.split()[1:]
            for p in parts:
                try:
                    v_idx = int(p.split('/')[0]) - 1
                    objects[current_object]['face_vertices'].add(v_idx)
                except ValueError:
                    continue

    return objects, all_vertices


def get_centroid(objects, obj_name, all_vertices):
    """Calculate centroid of an object."""
    obj = objects[obj_name]
    face_vertices = obj['face_vertices']
    if not face_vertices:
        return None
    xs, ys, zs = [], [], []
    for idx in face_vertices:
        if 0 <= idx < len(all_vertices):
            x, y, z = all_vertices[idx]
            xs.append(x)
            ys.append(y)
            zs.append(z)
    if not xs:
        return None
    return (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))


def distance(c1, c2):
    """Calculate 3D distance between two points."""
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)


def create_note_pads(objects, all_vertices):
    """Match Groves and Pan objects to create note pads."""
    groves = [(name, get_centroid(objects, name, all_vertices))
              for name, obj in objects.items()
              if obj['material'] == 'Groves' and get_centroid(objects, name, all_vertices)]
    pans = [(name, get_centroid(objects, name, all_vertices))
            for name, obj in objects.items()
            if obj['material'] == 'Pan' and get_centroid(objects, name, all_vertices)]

    note_pads = []
    used_pans = set()

    for grove_name, grove_centroid in groves:
        best_pan = None
        best_dist = float('inf')
        for pan_name, pan_centroid in pans:
            if pan_name in used_pans:
                continue
            d = distance(grove_centroid, pan_centroid)
            if d < best_dist:
                best_dist = d
                best_pan = (pan_name, pan_centroid)
        if best_pan and best_dist < 15:
            combined_centroid = (
                (grove_centroid[0] + best_pan[1][0]) / 2,
                (grove_centroid[1] + best_pan[1][1]) / 2,
                (grove_centroid[2] + best_pan[1][2]) / 2
            )
            note_pads.append({
                'grove': grove_name,
                'pan': best_pan[0],
                'centroid': combined_centroid
            })
            used_pans.add(best_pan[0])

    return note_pads


def classify_rings(note_pads):
    """Classify note pads into inner, central, and outer rings."""
    # Calculate pan center
    center_x = sum(np['centroid'][0] for np in note_pads) / len(note_pads)
    center_y = sum(np['centroid'][1] for np in note_pads) / len(note_pads)

    # Calculate radius and angle for each note pad
    for np in note_pads:
        dx = np['centroid'][0] - center_x
        dy = np['centroid'][1] - center_y
        np['radius'] = math.sqrt(dx**2 + dy**2)
        np['angle'] = math.degrees(math.atan2(dy, dx))
        np['x'] = dx  # Relative to center
        np['y'] = dy

    # Sort by radius and classify
    note_pads.sort(key=lambda x: x['radius'])

    inner_ring = note_pads[:5]
    central_ring = note_pads[5:17]
    outer_ring = note_pads[17:]

    return inner_ring, central_ring, outer_ring, (center_x, center_y)


def assign_notes(ring, note_names):
    """Assign note names to pads in a ring based on angular position."""
    # Sort by angle (starting from a reference point)
    sorted_ring = sorted(ring, key=lambda x: x['angle'])

    for i, pad in enumerate(sorted_ring):
        pad['note'] = note_names[i % len(note_names)]
        pad['index'] = i

    return sorted_ring


def generate_diagram(inner_ring, central_ring, outer_ring, output_path):
    """Generate a top-down diagram of the pan."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))

    # Set up the plot
    ax.set_aspect('equal')
    ax.set_xlim(-14, 14)
    ax.set_ylim(-14, 14)
    ax.set_facecolor('#2a2a2a')
    fig.patch.set_facecolor('#1a1a1a')

    # Draw pan outline (outer circle)
    pan_circle = plt.Circle((0, 0), 12, fill=False, color='#888888', linewidth=3)
    ax.add_patch(pan_circle)

    # Color schemes for each ring
    colors = {
        'outer': '#e74c3c',    # Red
        'central': '#3498db',  # Blue
        'inner': '#2ecc71'     # Green
    }

    def draw_note_pad(pad, ring_type, ring_index):
        """Draw a single note pad."""
        x, y = pad['x'], pad['y']
        note = pad['note']
        idx = pad['index']

        # Pad size varies by ring
        if ring_type == 'inner':
            size = 1.8
            prefix = 'I'
        elif ring_type == 'central':
            size = 1.5
            prefix = 'C'
        else:
            size = 1.3
            prefix = 'O'

        # Draw ellipse for the note pad
        ellipse = Ellipse((x, y), size * 1.2, size,
                         facecolor=colors[ring_type],
                         edgecolor='white',
                         linewidth=2,
                         alpha=0.8)
        ax.add_patch(ellipse)

        # Add note name
        ax.text(x, y + 0.15, note,
                ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white')

        # Add index identifier
        index_label = f"{prefix}{idx}"
        ax.text(x, y - 0.35, index_label,
                ha='center', va='center',
                fontsize=8,
                color='#cccccc')

    # Draw all note pads
    for pad in outer_ring:
        draw_note_pad(pad, 'outer', pad['index'])

    for pad in central_ring:
        draw_note_pad(pad, 'central', pad['index'])

    for pad in inner_ring:
        draw_note_pad(pad, 'inner', pad['index'])

    # Add title
    ax.set_title('Tenor Pan Note Layout\n(View from Above)',
                 fontsize=18, fontweight='bold', color='white', pad=20)

    # Add legend
    legend_elements = [
        patches.Patch(facecolor=colors['outer'], edgecolor='white', label=f'Outer Ring (4ths) - 12 notes'),
        patches.Patch(facecolor=colors['central'], edgecolor='white', label=f'Central Ring (5ths) - 12 notes'),
        patches.Patch(facecolor=colors['inner'], edgecolor='white', label=f'Inner Ring (6ths) - 5 notes'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              facecolor='#333333', edgecolor='white', labelcolor='white',
              fontsize=10)

    # Add ring labels
    ax.text(0, -13, 'Index format: O=Outer, C=Central, I=Inner',
            ha='center', va='center', fontsize=10, color='#888888')

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    plt.close()

    print(f"Diagram saved to: {output_path}")


def generate_note_mapping(inner_ring, central_ring, outer_ring):
    """Generate a mapping dictionary for all notes."""
    mapping = {
        'inner': [],
        'central': [],
        'outer': []
    }

    for pad in inner_ring:
        mapping['inner'].append({
            'index': f"I{pad['index']}",
            'note': pad['note'],
            'grove_obj': pad['grove'],
            'pan_obj': pad['pan'],
            'angle': round(pad['angle'], 1),
            'radius': round(pad['radius'], 2)
        })

    for pad in central_ring:
        mapping['central'].append({
            'index': f"C{pad['index']}",
            'note': pad['note'],
            'grove_obj': pad['grove'],
            'pan_obj': pad['pan'],
            'angle': round(pad['angle'], 1),
            'radius': round(pad['radius'], 2)
        })

    for pad in outer_ring:
        mapping['outer'].append({
            'index': f"O{pad['index']}",
            'note': pad['note'],
            'grove_obj': pad['grove'],
            'pan_obj': pad['pan'],
            'angle': round(pad['angle'], 1),
            'radius': round(pad['radius'], 2)
        })

    return mapping


def main():
    obj_path = "data/Tenor Pan only.obj"
    output_image = "docs/tenor_pan_layout.png"

    print("Parsing OBJ file...")
    objects, all_vertices = parse_obj_file(obj_path)

    print("Creating note pads...")
    note_pads = create_note_pads(objects, all_vertices)

    print("Classifying rings...")
    inner_ring, central_ring, outer_ring, center = classify_rings(note_pads)

    print("Assigning notes...")
    inner_ring = assign_notes(inner_ring, INNER_NOTES)
    central_ring = assign_notes(central_ring, CENTRAL_NOTES)
    outer_ring = assign_notes(outer_ring, OUTER_NOTES)

    print("Generating diagram...")
    import os
    os.makedirs("docs", exist_ok=True)
    generate_diagram(inner_ring, central_ring, outer_ring, output_image)

    # Print mapping
    mapping = generate_note_mapping(inner_ring, central_ring, outer_ring)

    print("\n" + "="*60)
    print("NOTE PAD MAPPING")
    print("="*60)

    for ring_name in ['outer', 'central', 'inner']:
        ring_label = {'outer': 'OUTER (4ths)', 'central': 'CENTRAL (5ths)', 'inner': 'INNER (6ths)'}
        print(f"\n{ring_label[ring_name]}:")
        for note in mapping[ring_name]:
            print(f"  {note['index']:4s} | {note['note']:3s} | Grove: {note['grove_obj']:12s} | Pan: {note['pan_obj']:12s} | θ={note['angle']:6.1f}°")

    return mapping


if __name__ == "__main__":
    main()
