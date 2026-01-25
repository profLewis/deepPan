#!/usr/bin/env python3
"""
Generate an interactive HTML page for the tenor pan.
Uses the 3D geometry from the OBJ file to create clickable note pads.
"""

import json
import numpy as np
from scipy.spatial import ConvexHull

# Fixed note mapping based on 3D object names
NOTE_MAPPING = {
    # Outer Ring (4ths): F#, B, E, A, D, G, C, F, Bb, Eb, Ab, C#
    ('object_58', 'object_62'): ('O0', 'F#', 'outer', 4),
    ('object_57', 'object_63'): ('O1', 'B', 'outer', 4),
    ('object_56', 'object_64'): ('O2', 'E', 'outer', 4),
    ('object_55', 'object_90'): ('O3', 'A', 'outer', 4),
    ('object_54', 'object_65'): ('O4', 'D', 'outer', 4),
    ('object_53', 'object_66'): ('O5', 'G', 'outer', 4),
    ('object_59', 'object_60'): ('O6', 'C', 'outer', 4),
    ('object_52', 'object_61'): ('O7', 'F', 'outer', 4),
    ('object_51', 'object_67'): ('O8', 'Bb', 'outer', 4),
    ('object_50', 'object_88'): ('O9', 'Eb', 'outer', 4),
    ('object_49', 'object_68'): ('O10', 'Ab', 'outer', 4),
    ('object_48', 'object_69'): ('O11', 'C#', 'outer', 4),
    # Central Ring (5ths): F#, B, E, A, D, G, C, F, Bb, Eb, Ab, C#
    ('object_25', 'object_45'): ('C0', 'F#', 'central', 5),
    ('object_24', 'object_46'): ('C1', 'B', 'central', 5),
    ('object_23', 'object_47'): ('C2', 'E', 'central', 5),
    ('object_22', 'object_37'): ('C3', 'A', 'central', 5),
    ('object_21', 'object_38'): ('C4', 'D', 'central', 5),
    ('object_20', 'object_39'): ('C5', 'G', 'central', 5),
    ('object_73', 'object_31'): ('C6', 'C', 'central', 5),
    ('object_30', 'object_40'): ('C7', 'F', 'central', 5),
    ('object_29', 'object_41'): ('C8', 'Bb', 'central', 5),
    ('object_28', 'object_42'): ('C9', 'Eb', 'central', 5),
    ('object_27', 'object_43'): ('C10', 'Ab', 'central', 5),
    ('object_26', 'object_44'): ('C11', 'C#', 'central', 5),
    # Inner Ring (6ths): C#, E, D, C, Eb
    ('object_72', 'object_36'): ('I0', 'C#', 'inner', 6),
    ('object_71', 'object_32'): ('I1', 'E', 'inner', 6),
    ('object_19', 'object_33'): ('I2', 'D', 'inner', 6),
    ('object_70', 'object_34'): ('I3', 'C', 'inner', 6),
    ('object_18', 'object_35'): ('I4', 'Eb', 'inner', 6),
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


def get_centroid_3d(objects, obj_name, all_vertices):
    """Get 3D centroid of an object."""
    obj = objects[obj_name]
    indices = list(obj['face_vertices'])
    if not indices:
        return None
    verts = all_vertices[indices]
    return verts.mean(axis=0)


def distance_3d(c1, c2):
    """Calculate 3D distance."""
    return np.sqrt(np.sum((c1 - c2)**2))


def get_object_hull_2d(objects, obj_name, all_vertices):
    """Get the convex hull of an object projected to X-Z plane."""
    obj = objects[obj_name]
    indices = list(obj['face_vertices'])
    if not indices:
        return []

    verts = all_vertices[indices]
    points_2d = verts[:, [0, 2]]  # X and Z coordinates

    if len(points_2d) < 3:
        return points_2d.tolist()

    try:
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
        return hull_points.tolist()
    except:
        return points_2d.tolist()


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
            # Look up note info from mapping
            key = (grove_name, best_pan[0])
            if key in NOTE_MAPPING:
                idx, note, ring, octave = NOTE_MAPPING[key]

                # Get hull points for grove and pan
                grove_hull = get_object_hull_2d(objects, grove_name, all_vertices)
                pan_hull = get_object_hull_2d(objects, best_pan[0], all_vertices)

                # Calculate 2D centroid
                combined_centroid_2d = np.array([
                    (grove_centroid[0] + best_pan[1][0]) / 2,
                    (grove_centroid[2] + best_pan[1][2]) / 2
                ])

                note_pads.append({
                    'idx': idx,
                    'name': note,
                    'ring': ring,
                    'octave': octave,
                    'grove_path': grove_hull,
                    'pan_path': pan_hull,
                    'centroid': combined_centroid_2d.tolist()
                })
            used_pans.add(best_pan[0])

    return note_pads


def generate_html(note_pads, output_path):
    """Generate interactive HTML with embedded note shapes."""

    # Find center and scale
    all_points = []
    for pad in note_pads:
        all_points.extend(pad['grove_path'])
        all_points.extend(pad['pan_path'])
    all_points = np.array(all_points)

    center_x = all_points[:, 0].mean()
    center_y = all_points[:, 1].mean()

    # Normalize coordinates to center at origin
    for pad in note_pads:
        pad['grove_path'] = [[p[0] - center_x, p[1] - center_y] for p in pad['grove_path']]
        pad['pan_path'] = [[p[0] - center_x, p[1] - center_y] for p in pad['pan_path']]
        pad['centroid'] = [pad['centroid'][0] - center_x, pad['centroid'][1] - center_y]

    # Calculate view box
    all_centered = []
    for pad in note_pads:
        all_centered.extend(pad['grove_path'])
        all_centered.extend(pad['pan_path'])
    all_centered = np.array(all_centered)

    max_extent = max(abs(all_centered).max() * 1.1, 20)
    viewbox = f"{-max_extent} {-max_extent} {max_extent * 2} {max_extent * 2}"

    # Convert note shapes to JSON
    note_shapes_json = json.dumps(note_pads)

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Tenor Steel Pan - deepPan</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 50%, #1a1a1a 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            color: white;
        }}
        h1 {{
            margin-bottom: 5px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
            background: linear-gradient(90deg, #C0C0C0, #FFFFFF, #C0C0C0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .subtitle {{ color: #999; margin-bottom: 20px; font-size: 14px; }}
        .pan-container {{
            position: relative;
            width: min(650px, 90vw);
            height: min(650px, 90vw);
        }}
        svg {{ width: 100%; height: 100%; }}
        .pan-bowl {{
            fill: url(#bowlGradient);
            stroke: #666;
            stroke-width: 0.5;
        }}
        .note-shape {{
            cursor: pointer;
            transition: all 0.1s ease;
        }}
        .note-shape:hover {{ filter: brightness(1.3) drop-shadow(0 0 3px #FFD700); }}
        .note-shape.playing {{
            filter: brightness(1.5);
            animation: pulse 0.2s ease-out;
        }}
        @keyframes pulse {{
            0% {{ filter: brightness(2) drop-shadow(0 0 10px #FFD700); }}
            100% {{ filter: brightness(1.3); }}
        }}
        .grove-shape {{ fill: #1a1a1a; stroke: #333; stroke-width: 0.1; }}
        .pan-shape.outer {{ fill: #c0392b; stroke: #e74c3c; stroke-width: 0.05; }}
        .pan-shape.central {{ fill: #2471a3; stroke: #3498db; stroke-width: 0.05; }}
        .pan-shape.inner {{ fill: #1e8449; stroke: #2ecc71; stroke-width: 0.05; }}
        .note-label {{
            font-size: 0.9px;
            font-weight: bold;
            fill: white;
            text-anchor: middle;
            pointer-events: none;
            text-shadow: 0 0 2px black;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin: 15px 0;
            flex-wrap: wrap;
            justify-content: center;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 1px solid #666;
        }}
        .legend-color.outer {{ background: #c0392b; }}
        .legend-color.central {{ background: #2471a3; }}
        .legend-color.inner {{ background: #1e8449; }}
        .now-playing {{
            margin-top: 10px;
            font-size: 18px;
            min-height: 30px;
            color: #FFD700;
        }}
        .mode-toggle {{
            margin: 15px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .mode-toggle input {{
            width: 18px;
            height: 18px;
            cursor: pointer;
        }}
        .instructions {{
            color: #888;
            font-size: 14px;
            margin-top: 10px;
        }}
        .credit {{
            color: #666;
            font-size: 12px;
            margin-top: 20px;
        }}
        .credit a {{ color: #888; }}
    </style>
</head>
<body>
    <h1>Tenor Steel Pan</h1>
    <p class="subtitle">deepPan Interactive</p>

    <div class="pan-container">
        <svg viewBox="{viewbox}" id="panSvg">
            <defs>
                <radialGradient id="bowlGradient" cx="30%" cy="30%">
                    <stop offset="0%" stop-color="#4a4a4a"/>
                    <stop offset="50%" stop-color="#2a2a2a"/>
                    <stop offset="100%" stop-color="#1a1a1a"/>
                </radialGradient>
            </defs>
            <circle class="pan-bowl" cx="0" cy="0" r="{max_extent * 0.95:.1f}"/>
        </svg>
    </div>

    <div class="legend">
        <div class="legend-item"><div class="legend-color outer"></div><span>Outer (4ths)</span></div>
        <div class="legend-item"><div class="legend-color central"></div><span>Central (5ths)</span></div>
        <div class="legend-item"><div class="legend-color inner"></div><span>Inner (6ths)</span></div>
    </div>

    <div class="now-playing" id="nowPlaying">Loading sounds...</div>
    <div class="mode-toggle">
        <input type="checkbox" id="clickMode" onchange="toggleMode()">
        <label for="clickMode">Click mode (uncheck for hover)</label>
    </div>
    <p class="instructions" id="instructions">Hover over any note to hear its sound</p>
    <p class="credit">Geometry from 3D model | <a href="https://github.com/profLewis/deepPan" target="_blank">GitHub</a></p>

    <script>
        const noteShapes = {note_shapes_json};

        const audioCache = {{}};
        let currentAudio = null;
        let clickMode = false;

        function toggleMode() {{
            clickMode = document.getElementById('clickMode').checked;
            document.getElementById('instructions').textContent =
                clickMode ? 'Click any note to hear its sound' : 'Hover over any note to hear its sound';
            renderNotes();
        }}

        // Preload audio files
        function preloadAudio() {{
            let loadedCount = 0;
            const total = noteShapes.length;

            noteShapes.forEach(note => {{
                const audio = new Audio();
                const safeName = note.name.replace('#', 's');
                audio.src = `sounds/${{note.idx}}_${{safeName}}${{note.octave}}.wav`;
                audio.preload = 'auto';

                audio.addEventListener('canplaythrough', () => {{
                    loadedCount++;
                    document.getElementById('nowPlaying').textContent =
                        `Loading sounds... ${{loadedCount}}/${{total}}`;
                    if (loadedCount === total) {{
                        document.getElementById('nowPlaying').textContent = 'Ready - hover or click a note!';
                    }}
                }}, {{ once: true }});

                audio.addEventListener('error', () => {{
                    loadedCount++;
                    console.warn(`Failed to load: ${{audio.src}}`);
                }}, {{ once: true }});

                audioCache[note.idx] = audio;
            }});
        }}

        function playNote(idx, name, octave, element) {{
            if (currentAudio) {{
                currentAudio.pause();
                currentAudio.currentTime = 0;
            }}

            const audio = audioCache[idx];
            if (audio) {{
                currentAudio = audio;
                audio.currentTime = 0;
                audio.play().then(() => {{
                    document.getElementById('nowPlaying').textContent = `${{name}}${{octave}} (${{idx}})`;
                }}).catch(e => console.warn('Playback failed:', e));
            }}

            // Visual feedback
            document.querySelectorAll('.note-shape').forEach(el => el.classList.remove('playing'));
            if (element) element.classList.add('playing');
        }}

        function renderNotes() {{
            const svg = document.getElementById('panSvg');

            // Remove existing note shapes
            svg.querySelectorAll('.note-shape').forEach(el => el.remove());

            noteShapes.forEach(note => {{
                const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                g.classList.add('note-shape');

                if (clickMode) {{
                    g.onclick = () => playNote(note.idx, note.name, note.octave, g);
                }} else {{
                    g.onmouseenter = () => playNote(note.idx, note.name, note.octave, g);
                }}

                // Draw grove (groove)
                if (note.grove_path.length > 2) {{
                    const grovePath = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                    grovePath.classList.add('grove-shape');
                    grovePath.setAttribute('points', note.grove_path.map(p => p.join(',')).join(' '));
                    g.appendChild(grovePath);
                }}

                // Draw pan surface
                if (note.pan_path.length > 2) {{
                    const panPath = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                    panPath.classList.add('pan-shape', note.ring);
                    panPath.setAttribute('points', note.pan_path.map(p => p.join(',')).join(' '));
                    g.appendChild(panPath);
                }}

                // Add label
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.classList.add('note-label');
                label.setAttribute('x', note.centroid[0]);
                label.setAttribute('y', note.centroid[1] + 0.3);
                label.textContent = `${{note.idx}} ${{note.name}}`;
                g.appendChild(label);

                svg.appendChild(g);
            }});
        }}

        // Initialize
        preloadAudio();
        renderNotes();
    </script>
</body>
</html>
'''

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Interactive HTML saved to: {output_path}")


def main():
    obj_path = "data/Tenor Pan only.obj"
    output_html = "index.html"

    print("Parsing OBJ file...")
    objects, all_vertices = parse_obj_file(obj_path)

    print("Creating note pads...")
    note_pads = create_note_pads(objects, all_vertices)
    print(f"  Found {len(note_pads)} note pads")

    print("Generating interactive HTML...")
    generate_html(note_pads, output_html)

    print("\nNote pads generated:")
    for ring in ['outer', 'central', 'inner']:
        pads = [p for p in note_pads if p['ring'] == ring]
        print(f"  {ring.upper()}: {', '.join(p['idx'] for p in sorted(pads, key=lambda x: x['idx']))}")


if __name__ == "__main__":
    main()
