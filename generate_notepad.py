#!/usr/bin/env python3
"""
Generate printable note pad geometry from the tenor pan OBJ file.

Extracts note pad (Pan) and groove (Groves) geometry, converts from cm to mm,
and thickens into a solid object suitable for 3D printing.
"""

import numpy as np
from pathlib import Path
import math

# Conversion factor: OBJ file is in cm, output in mm
CM_TO_MM = 10.0

# Thickness for the solids (mm)
PAN_THICKNESS = 1.5       # Thickness of the pan playing surface (downward)
GROVE_DEPTH = 1.5         # Groove thickness downward (same as pan)
GROVE_PROTRUSION = 0.8    # Groove protrusion upward (ridge above surface)

# Mounting cylinder parameters (mm)
MOUNT_INNER_DIAMETER = 25.0   # Internal diameter
MOUNT_DEPTH = 9.0             # Cylinder depth
MOUNT_WALL_THICKNESS = 2.5    # Wall thickness
MOUNT_THREAD_PITCH = 2.0      # Thread pitch
MOUNT_THREAD_DEPTH = 1.0      # Thread depth (outward from wall)
MOUNT_NOTCH_WIDTH = 1.5       # Wire notch width
MOUNT_SEGMENTS = 48           # Resolution for cylinder

# Note mapping: (grove_object, pan_object) -> (index, note, ring, octave)
NOTE_MAPPING = {
    # Outer Ring (4ths)
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
    # Central Ring (5ths)
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
    # Inner Ring (6ths)
    ('object_72', 'object_36'): ('I0', 'C#', 'inner', 6),
    ('object_71', 'object_32'): ('I1', 'E', 'inner', 6),
    ('object_19', 'object_33'): ('I2', 'D', 'inner', 6),
    ('object_70', 'object_34'): ('I3', 'C', 'inner', 6),
    ('object_18', 'object_35'): ('I4', 'Eb', 'inner', 6),
}

# Reverse lookup: index -> (grove_object, pan_object, note, ring, octave)
NOTE_BY_INDEX = {}
for (grove, pan), (idx, note, ring, octave) in NOTE_MAPPING.items():
    NOTE_BY_INDEX[idx] = {
        'grove_object': grove,
        'pan_object': pan,
        'note': note,
        'ring': ring,
        'octave': octave
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


def extract_object_mesh(objects, obj_name, all_vertices):
    """Extract vertices and faces for an object, re-indexed."""
    obj = objects[obj_name]
    old_indices = sorted(list(obj['face_vertices']))

    # Create mapping from old to new indices
    index_map = {old: new for new, old in enumerate(old_indices)}

    # Extract vertices (convert cm to mm)
    vertices = all_vertices[old_indices] * CM_TO_MM

    # Re-index faces
    faces = []
    for face in obj['faces']:
        new_face = [index_map[idx] for idx in face if idx in index_map]
        if len(new_face) >= 3:
            faces.append(new_face)

    return vertices, faces


def compute_vertex_normals(vertices, faces):
    """Compute per-vertex normals by averaging face normals."""
    vertex_normals = np.zeros_like(vertices)

    for face in faces:
        if len(face) < 3:
            continue
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(face_normal)
        if norm > 0:
            face_normal = face_normal / norm

        # Add to all vertices in the face
        for idx in face:
            vertex_normals[idx] += face_normal

    # Normalize
    for i in range(len(vertex_normals)):
        norm = np.linalg.norm(vertex_normals[i])
        if norm > 0:
            vertex_normals[i] = vertex_normals[i] / norm
        else:
            vertex_normals[i] = np.array([0, -1, 0])  # Default downward

    return vertex_normals


def compute_surface_normal(vertices, faces):
    """Compute the average surface normal weighted by face area."""
    total_normal = np.zeros(3)
    total_area = 0

    for face in faces:
        if len(face) < 3:
            continue
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        area = np.linalg.norm(cross) / 2
        if area > 0:
            normal = cross / (2 * area)
            total_normal += normal * area
            total_area += area

    if total_area > 0:
        total_normal = total_normal / np.linalg.norm(total_normal)
    else:
        total_normal = np.array([0, -1, 0])

    return total_normal


def compute_interior_centroid(vertices, faces, normal, thickness_down, thickness_up=0):
    """
    Compute a centroid guaranteed to be inside the thickened volume.

    Takes the surface centroid and offsets it to the middle of the thickness.
    """
    # Surface centroid (average of all vertices)
    surface_centroid = vertices.mean(axis=0)

    # The volume extends from (surface - thickness_down*normal) to (surface + thickness_up*normal)
    # The middle of this range is: surface + (thickness_up - thickness_down)/2 * normal
    offset = (thickness_up - thickness_down) / 2.0
    interior_centroid = surface_centroid + normal * offset

    return interior_centroid, surface_centroid


def generate_threaded_mount_cylinder(inner_diameter, depth, wall_thickness, thread_pitch,
                                      thread_depth, notch_width, segments=MOUNT_SEGMENTS):
    """
    Generate a threaded mounting cylinder with a wire notch.

    The cylinder is centered at origin, extending downward along -Z axis.
    External threads on the outside, notch cut from top to bottom.

    Returns vertices and faces for the mesh.
    """
    inner_r = inner_diameter / 2
    outer_r = inner_r + wall_thickness
    thread_outer_r = outer_r + thread_depth

    vertices = []
    faces = []

    # Calculate notch angle (for removing vertices in that region)
    notch_half_angle = math.asin(notch_width / 2 / outer_r) if notch_width < outer_r * 2 else math.pi / 4

    # Generate rings at top and bottom
    # We'll create: inner top, outer top (with threads), inner bottom, outer bottom (with threads)

    def is_in_notch(angle):
        """Check if angle is within the notch region (centered at angle=0)."""
        # Normalize angle to -pi to pi
        a = angle % (2 * math.pi)
        if a > math.pi:
            a -= 2 * math.pi
        return abs(a) < notch_half_angle

    # Generate vertex rings
    # Top inner ring
    top_inner_start = len(vertices)
    top_inner_indices = []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        if not is_in_notch(angle):
            x = inner_r * math.cos(angle)
            y = inner_r * math.sin(angle)
            top_inner_indices.append(len(vertices))
            vertices.append([x, y, 0])

    # Bottom inner ring
    bot_inner_start = len(vertices)
    bot_inner_indices = []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        if not is_in_notch(angle):
            x = inner_r * math.cos(angle)
            y = inner_r * math.sin(angle)
            bot_inner_indices.append(len(vertices))
            vertices.append([x, y, -depth])

    # Thread profile along the outer surface
    num_thread_turns = depth / thread_pitch
    thread_steps_per_turn = segments
    total_thread_steps = int(num_thread_turns * thread_steps_per_turn) + 1

    # Generate thread helix vertices (outer surface with thread profile)
    thread_rings = []
    for t in range(total_thread_steps):
        z = -t * thread_pitch / thread_steps_per_turn
        if z < -depth:
            z = -depth

        ring_indices = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments

            if is_in_notch(angle):
                continue

            # Thread profile: sinusoidal variation
            thread_phase = (t + i) / thread_steps_per_turn
            thread_offset = thread_depth * 0.5 * (1 + math.sin(2 * math.pi * thread_phase))
            r = outer_r + thread_offset

            x = r * math.cos(angle)
            y = r * math.sin(angle)
            ring_indices.append(len(vertices))
            vertices.append([x, y, z])

        if ring_indices:
            thread_rings.append(ring_indices)

    # Notch edge vertices (vertical edges where notch cuts through)
    # Left edge of notch (at -notch_half_angle)
    notch_left_angle = -notch_half_angle
    notch_right_angle = notch_half_angle

    notch_verts = {
        'top_inner_left': len(vertices),
        'top_inner_right': None,
        'top_outer_left': None,
        'top_outer_right': None,
        'bot_inner_left': None,
        'bot_inner_right': None,
        'bot_outer_left': None,
        'bot_outer_right': None,
    }

    # Inner left
    vertices.append([inner_r * math.cos(notch_left_angle), inner_r * math.sin(notch_left_angle), 0])
    notch_verts['top_inner_right'] = len(vertices)
    vertices.append([inner_r * math.cos(notch_right_angle), inner_r * math.sin(notch_right_angle), 0])

    notch_verts['top_outer_left'] = len(vertices)
    vertices.append([outer_r * math.cos(notch_left_angle), outer_r * math.sin(notch_left_angle), 0])
    notch_verts['top_outer_right'] = len(vertices)
    vertices.append([outer_r * math.cos(notch_right_angle), outer_r * math.sin(notch_right_angle), 0])

    notch_verts['bot_inner_left'] = len(vertices)
    vertices.append([inner_r * math.cos(notch_left_angle), inner_r * math.sin(notch_left_angle), -depth])
    notch_verts['bot_inner_right'] = len(vertices)
    vertices.append([inner_r * math.cos(notch_right_angle), inner_r * math.sin(notch_right_angle), -depth])

    notch_verts['bot_outer_left'] = len(vertices)
    vertices.append([outer_r * math.cos(notch_left_angle), outer_r * math.sin(notch_left_angle), -depth])
    notch_verts['bot_outer_right'] = len(vertices)
    vertices.append([outer_r * math.cos(notch_right_angle), outer_r * math.sin(notch_right_angle), -depth])

    # Build faces

    # Top annular face (between inner and first thread ring, excluding notch)
    if thread_rings and top_inner_indices:
        first_ring = thread_rings[0]
        # Connect top inner to first outer ring
        n_inner = len(top_inner_indices)
        n_outer = len(first_ring)
        # Simple approach: fan triangulation from inner to outer
        for i in range(min(n_inner, n_outer) - 1):
            faces.append([top_inner_indices[i], first_ring[i], first_ring[i + 1]])
            faces.append([top_inner_indices[i], first_ring[i + 1], top_inner_indices[i + 1]])

    # Bottom annular face
    if thread_rings and bot_inner_indices:
        last_ring = thread_rings[-1]
        n_inner = len(bot_inner_indices)
        n_outer = len(last_ring)
        for i in range(min(n_inner, n_outer) - 1):
            faces.append([bot_inner_indices[i], bot_inner_indices[i + 1], last_ring[i + 1]])
            faces.append([bot_inner_indices[i], last_ring[i + 1], last_ring[i]])

    # Inner wall
    for i in range(len(top_inner_indices) - 1):
        faces.append([top_inner_indices[i], top_inner_indices[i + 1],
                     bot_inner_indices[i + 1], bot_inner_indices[i]])

    # Outer wall with threads (connect thread rings)
    for r in range(len(thread_rings) - 1):
        ring1 = thread_rings[r]
        ring2 = thread_rings[r + 1]
        n = min(len(ring1), len(ring2))
        for i in range(n - 1):
            faces.append([ring1[i], ring1[i + 1], ring2[i + 1], ring2[i]])

    # Notch faces (walls of the notch)
    # Left wall of notch
    faces.append([notch_verts['top_inner_left'], notch_verts['top_outer_left'],
                  notch_verts['bot_outer_left'], notch_verts['bot_inner_left']])
    # Right wall of notch
    faces.append([notch_verts['top_outer_right'], notch_verts['top_inner_right'],
                  notch_verts['bot_inner_right'], notch_verts['bot_outer_right']])
    # Bottom of notch (connects left and right at outer radius)
    # This is the "floor" of the notch channel
    faces.append([notch_verts['top_outer_left'], notch_verts['top_outer_right'],
                  notch_verts['bot_outer_right'], notch_verts['bot_outer_left']])

    return np.array(vertices), faces


def transform_cylinder_to_normal(cylinder_verts, centroid, normal):
    """
    Transform cylinder from Z-down orientation to align with given normal.
    Cylinder is moved so its top center is at the centroid.
    """
    # Default cylinder points down along -Z
    # We need to rotate so -Z aligns with -normal (cylinder extends away from surface)

    z_axis = np.array([0, 0, -1])
    target = -normal / np.linalg.norm(normal)

    # Rotation matrix from z_axis to target
    v = np.cross(z_axis, target)
    c = np.dot(z_axis, target)

    if np.linalg.norm(v) < 1e-10:
        if c > 0:
            rot_matrix = np.eye(3)
        else:
            # 180 degree rotation around X axis
            rot_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))

    # Apply rotation then translation
    rotated = cylinder_verts @ rot_matrix.T
    translated = rotated + centroid

    return translated


def find_boundary_edges(faces):
    """Find boundary edges (edges that belong to only one face)."""
    edge_count = {}
    edge_face_order = {}  # Track the order vertices appear in faces

    for face in faces:
        n = len(face)
        for i in range(n):
            v1, v2 = face[i], face[(i + 1) % n]
            edge = tuple(sorted([v1, v2]))
            edge_count[edge] = edge_count.get(edge, 0) + 1
            # Store the directed edge for winding order
            if edge not in edge_face_order:
                edge_face_order[edge] = (v1, v2)

    # Boundary edges appear exactly once
    boundary_edges = [(edge, edge_face_order[edge]) for edge, count in edge_count.items() if count == 1]
    return boundary_edges


def find_all_boundary_loops(edges_with_order):
    """Find all boundary loops (handles multiple disconnected boundaries)."""
    if not edges_with_order:
        return []

    # Build adjacency from directed edges
    adj = {}
    for edge, (v1, v2) in edges_with_order:
        if v1 not in adj:
            adj[v1] = []
        adj[v1].append(v2)
        if v2 not in adj:
            adj[v2] = []
        adj[v2].append(v1)

    all_loops = []
    visited_vertices = set()

    # Find all connected boundary loops
    all_boundary_verts = set(adj.keys())

    while all_boundary_verts - visited_vertices:
        # Start a new loop from an unvisited boundary vertex
        start = next(iter(all_boundary_verts - visited_vertices))
        loop = [start]
        visited_vertices.add(start)
        current = start

        while True:
            neighbors = adj.get(current, [])
            next_vertex = None
            for n in neighbors:
                if n not in visited_vertices:
                    next_vertex = n
                    break

            if next_vertex is None:
                # Check if we can close the loop
                if start in neighbors and len(loop) > 2:
                    pass  # Loop is closed
                break

            loop.append(next_vertex)
            visited_vertices.add(next_vertex)
            current = next_vertex

        if len(loop) >= 3:
            all_loops.append(loop)

    return all_loops


def thicken_surface(vertices, faces, thickness_down, thickness_up=0):
    """
    Thicken a surface mesh into a solid by:
    1. Creating offset vertices along normals (top and bottom surfaces)
    2. Creating side walls between all boundary edges

    Args:
        vertices: Original surface vertices
        faces: Original surface faces
        thickness_down: How much to extrude downward (along -normal)
        thickness_up: How much to extrude upward (along +normal), default 0
    """
    # Compute vertex normals
    normals = compute_vertex_normals(vertices, faces)

    # Create top vertices (offset upward if thickness_up > 0, else same as original)
    if thickness_up > 0:
        top_vertices = vertices + normals * thickness_up
    else:
        top_vertices = vertices.copy()

    # Create bottom vertices (offset downward)
    bottom_vertices = vertices - normals * thickness_down

    # Combine vertices: [top_vertices, bottom_vertices]
    n_verts = len(vertices)
    all_vertices = np.vstack([top_vertices, bottom_vertices])

    # Top faces (original, keep winding)
    top_faces = [list(face) for face in faces]

    # Bottom faces (offset indices, reverse winding for correct normals)
    bottom_faces = [[idx + n_verts for idx in reversed(face)] for face in faces]

    # Find all boundary loops and create side walls
    boundary_edges = find_boundary_edges(faces)
    boundary_loops = find_all_boundary_loops(boundary_edges)

    side_faces = []
    for loop in boundary_loops:
        for i in range(len(loop)):
            v1 = loop[i]
            v2 = loop[(i + 1) % len(loop)]
            # Create quad: top_v1, top_v2, bottom_v2, bottom_v1
            side_faces.append([v1, v2, v2 + n_verts, v1 + n_verts])

    all_faces = top_faces + bottom_faces + side_faces

    return all_vertices, all_faces


def center_mesh(vertices):
    """Center mesh at origin and return the offset used."""
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    return centered, centroid


def write_obj(filepath, vertices, faces, object_name="NotePad"):
    """Write mesh to OBJ file."""
    with open(filepath, 'w') as f:
        f.write(f"# Note Pad Geometry for 3D Printing\n")
        f.write(f"# Generated by generate_notepad.py\n")
        f.write(f"# Units: mm\n\n")

        f.write(f"o {object_name}\n")

        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        # Write faces (1-indexed)
        for face in faces:
            face_str = " ".join(str(idx + 1) for idx in face)
            f.write(f"f {face_str}\n")

    print(f"Saved: {filepath}")


def write_stl(filepath, vertices, faces, object_name="NotePad"):
    """Write mesh to binary STL file for 3D printing."""
    import struct

    def compute_face_normal(v0, v1, v2):
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        return normal

    # Triangulate faces (split quads into triangles)
    triangles = []
    for face in faces:
        if len(face) == 3:
            triangles.append(face)
        elif len(face) == 4:
            # Split quad into two triangles
            triangles.append([face[0], face[1], face[2]])
            triangles.append([face[0], face[2], face[3]])
        elif len(face) > 4:
            # Fan triangulation
            for i in range(1, len(face) - 1):
                triangles.append([face[0], face[i], face[i + 1]])

    with open(filepath, 'wb') as f:
        # Header (80 bytes)
        header = f"STL generated by generate_notepad.py - {object_name}".encode('ascii')
        header = header[:80].ljust(80, b'\0')
        f.write(header)

        # Number of triangles
        f.write(struct.pack('<I', len(triangles)))

        # Write triangles
        for tri in triangles:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            normal = compute_face_normal(v0, v1, v2)

            # Normal
            f.write(struct.pack('<3f', normal[0], normal[1], normal[2]))
            # Vertices
            f.write(struct.pack('<3f', v0[0], v0[1], v0[2]))
            f.write(struct.pack('<3f', v1[0], v1[1], v1[2]))
            f.write(struct.pack('<3f', v2[0], v2[1], v2[2]))
            # Attribute byte count
            f.write(struct.pack('<H', 0))

    print(f"Saved: {filepath}")


def generate_notepad(note_index, obj_path, output_dir,
                     pan_thickness=PAN_THICKNESS,
                     grove_depth=GROVE_DEPTH,
                     grove_protrusion=GROVE_PROTRUSION):
    """Generate a printable note pad for the given note index."""

    if note_index not in NOTE_BY_INDEX:
        print(f"Error: Unknown note index '{note_index}'")
        print(f"Valid indices: {sorted(NOTE_BY_INDEX.keys())}")
        return None

    note_info = NOTE_BY_INDEX[note_index]
    grove_obj = note_info['grove_object']
    pan_obj = note_info['pan_object']
    note_name = note_info['note']
    octave = note_info['octave']
    ring = note_info['ring']

    print(f"\n{'='*60}")
    print(f"Generating Note Pad: {note_index} ({note_name}{octave}) - {ring} ring")
    print(f"{'='*60}")
    print(f"  Grove object: {grove_obj}")
    print(f"  Pan object: {pan_obj}")

    # Parse OBJ file
    print(f"\nParsing {obj_path}...")
    objects, all_vertices = parse_obj_file(obj_path)

    # Extract pan surface
    print(f"Extracting pan surface ({pan_obj})...")
    pan_verts, pan_faces = extract_object_mesh(objects, pan_obj, all_vertices)
    print(f"  Pan: {len(pan_verts)} vertices, {len(pan_faces)} faces")

    # Extract grove
    print(f"Extracting grove ({grove_obj})...")
    grove_verts, grove_faces = extract_object_mesh(objects, grove_obj, all_vertices)
    print(f"  Grove: {len(grove_verts)} vertices, {len(grove_faces)} faces")

    # Compute surface normal from pan (the playing surface defines the orientation)
    pan_normal = compute_surface_normal(pan_verts, pan_faces)
    print(f"  Pan surface normal: ({pan_normal[0]:.4f}, {pan_normal[1]:.4f}, {pan_normal[2]:.4f})")

    # Thicken pan surface (downward only)
    print(f"Thickening pan surface (down: {pan_thickness}mm)...")
    pan_solid_verts, pan_solid_faces = thicken_surface(pan_verts, pan_faces,
                                                        thickness_down=pan_thickness,
                                                        thickness_up=0)
    print(f"  Pan solid: {len(pan_solid_verts)} vertices, {len(pan_solid_faces)} faces")
    print(f"  Pan boundary loops: {len(find_all_boundary_loops(find_boundary_edges(pan_faces)))}")

    # Compute interior centroid for pan (must lie within the solid volume)
    pan_interior_centroid, pan_surface_centroid = compute_interior_centroid(
        pan_verts, pan_faces, pan_normal, pan_thickness, 0)
    print(f"  Pan surface centroid: ({pan_surface_centroid[0]:.2f}, {pan_surface_centroid[1]:.2f}, {pan_surface_centroid[2]:.2f})")
    print(f"  Pan interior centroid: ({pan_interior_centroid[0]:.2f}, {pan_interior_centroid[1]:.2f}, {pan_interior_centroid[2]:.2f})")

    # Thicken grove (downward + slight protrusion upward)
    print(f"Thickening grove (down: {grove_depth}mm, up: {grove_protrusion}mm)...")
    grove_solid_verts, grove_solid_faces = thicken_surface(grove_verts, grove_faces,
                                                            thickness_down=grove_depth,
                                                            thickness_up=grove_protrusion)
    print(f"  Grove solid: {len(grove_solid_verts)} vertices, {len(grove_solid_faces)} faces")
    print(f"  Grove boundary loops: {len(find_all_boundary_loops(find_boundary_edges(grove_faces)))}")

    # Combine pan and groove solids
    print(f"Combining pan and groove...")
    n_pan_solid_verts = len(pan_solid_verts)
    solid_verts = np.vstack([pan_solid_verts, grove_solid_verts])
    solid_faces = pan_solid_faces + [[idx + n_pan_solid_verts for idx in face] for face in grove_solid_faces]
    print(f"  Pan+Grove solid: {len(solid_verts)} vertices, {len(solid_faces)} faces")

    # Combined note pad properties (use pan's normal and centroid as reference)
    notepad_normal = pan_normal
    notepad_centroid = pan_interior_centroid

    # Generate mounting cylinder
    print(f"Generating mounting cylinder...")
    print(f"  Inner diameter: {MOUNT_INNER_DIAMETER}mm, Depth: {MOUNT_DEPTH}mm")
    print(f"  Thread pitch: {MOUNT_THREAD_PITCH}mm, Notch width: {MOUNT_NOTCH_WIDTH}mm")

    cylinder_verts, cylinder_faces = generate_threaded_mount_cylinder(
        inner_diameter=MOUNT_INNER_DIAMETER,
        depth=MOUNT_DEPTH,
        wall_thickness=MOUNT_WALL_THICKNESS,
        thread_pitch=MOUNT_THREAD_PITCH,
        thread_depth=MOUNT_THREAD_DEPTH,
        notch_width=MOUNT_NOTCH_WIDTH
    )
    print(f"  Cylinder: {len(cylinder_verts)} vertices, {len(cylinder_faces)} faces")

    # Position cylinder at pan surface centroid, oriented along normal
    # The cylinder top should be at the surface, extending downward (into the pan)
    cylinder_verts_transformed = transform_cylinder_to_normal(
        cylinder_verts, pan_surface_centroid, notepad_normal)

    # Add cylinder to combined mesh
    n_solid_verts = len(solid_verts)
    solid_verts = np.vstack([solid_verts, cylinder_verts_transformed])
    solid_faces = solid_faces + [[idx + n_solid_verts for idx in face] for face in cylinder_faces]
    print(f"  Combined with cylinder: {len(solid_verts)} vertices, {len(solid_faces)} faces")

    print(f"\nNote pad properties (before centering):")
    print(f"  Normal vector: ({notepad_normal[0]:.6f}, {notepad_normal[1]:.6f}, {notepad_normal[2]:.6f})")
    print(f"  Surface centroid: ({pan_surface_centroid[0]:.2f}, {pan_surface_centroid[1]:.2f}, {pan_surface_centroid[2]:.2f}) mm")

    # Center at origin
    solid_verts, offset = center_mesh(solid_verts)
    print(f"  Centered (offset: {offset[0]:.1f}, {offset[1]:.1f}, {offset[2]:.1f})")

    # Adjust centroid to centered coordinate system
    notepad_centroid_centered = notepad_centroid - offset
    print(f"  Interior centroid (centered): ({notepad_centroid_centered[0]:.2f}, {notepad_centroid_centered[1]:.2f}, {notepad_centroid_centered[2]:.2f}) mm")

    # Calculate bounding box
    bbox_min = solid_verts.min(axis=0)
    bbox_max = solid_verts.max(axis=0)
    bbox_size = bbox_max - bbox_min
    print(f"  Bounding box: {bbox_size[0]:.1f} x {bbox_size[1]:.1f} x {bbox_size[2]:.1f} mm")

    # Output paths
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    safe_note = note_name.replace('#', 's')
    obj_name = f"NotePad_{note_index}_{safe_note}{octave}"

    obj_path = output_dir / f"notepad_{note_index}.obj"
    stl_path = output_dir / f"notepad_{note_index}.stl"

    # Write files
    print(f"\nWriting output files...")
    write_obj(obj_path, solid_verts, solid_faces, obj_name)
    write_stl(stl_path, solid_verts, solid_faces, obj_name)

    print(f"\n{'='*60}")
    print(f"Generated printable note pad: {note_index} ({note_name}{octave})")
    print(f"  OBJ: {obj_path}")
    print(f"  STL: {stl_path}")
    print(f"  Size: {bbox_size[0]:.1f} x {bbox_size[1]:.1f} x {bbox_size[2]:.1f} mm")
    print(f"  Pan: {pan_thickness}mm down")
    print(f"  Grove: {grove_depth}mm down, {grove_protrusion}mm up (ridge)")
    print(f"{'='*60}")

    return {
        'index': note_index,
        'note': note_name,
        'octave': octave,
        'ring': ring,
        'normal': notepad_normal.tolist(),  # Unit normal vector of playing surface
        'centroid': notepad_centroid_centered.tolist(),  # Interior centroid (centered coords)
        'centroid_original': notepad_centroid.tolist(),  # Interior centroid (original coords, mm)
        'vertices': solid_verts,
        'faces': solid_faces,
        'bbox_size': bbox_size.tolist(),
        'obj_path': str(obj_path),
        'stl_path': str(stl_path)
    }


def save_notepad_properties(results, output_path):
    """Save note pad properties to JSON file."""
    import json

    # Extract properties (exclude large vertex/face data)
    properties = []
    for r in results:
        props = {
            'index': r['index'],
            'note': r['note'],
            'octave': r['octave'],
            'ring': r['ring'],
            'normal': r['normal'],
            'centroid': r['centroid'],
            'centroid_original': r['centroid_original'],
            'bbox_size': r['bbox_size'],
            'obj_path': r['obj_path'],
            'stl_path': r['stl_path']
        }
        properties.append(props)

    with open(output_path, 'w') as f:
        json.dump(properties, f, indent=2)

    print(f"Saved properties to: {output_path}")


def main():
    import sys
    obj_path = "data/Tenor Pan only.obj"
    output_dir = "data/notepads"

    # Check for command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            # Generate all 29 note pads
            print("Generating all 29 note pads...")
            results = []
            for note_index in sorted(NOTE_BY_INDEX.keys()):
                result = generate_notepad(note_index, obj_path, output_dir)
                if result:
                    results.append(result)

            if results:
                # Save all properties to JSON
                save_notepad_properties(results, Path(output_dir) / "notepad_properties.json")
                print(f"\n{'='*60}")
                print(f"Generated {len(results)} note pads")
                print(f"Properties saved to: {output_dir}/notepad_properties.json")
                print(f"{'='*60}")
            return
        else:
            # Generate specific note
            note_index = sys.argv[1]
            result = generate_notepad(note_index, obj_path, output_dir)
            if result:
                save_notepad_properties([result], Path(output_dir) / f"notepad_{note_index}_properties.json")
            return

    # Default: generate test note O0
    result = generate_notepad("O0", obj_path, output_dir)

    if result:
        print(f"\nTest note pad generated successfully!")
        print(f"\nNote pad properties:")
        print(f"  Normal: ({result['normal'][0]:.6f}, {result['normal'][1]:.6f}, {result['normal'][2]:.6f})")
        print(f"  Centroid (centered): ({result['centroid'][0]:.2f}, {result['centroid'][1]:.2f}, {result['centroid'][2]:.2f}) mm")
        print(f"  Centroid (original): ({result['centroid_original'][0]:.2f}, {result['centroid_original'][1]:.2f}, {result['centroid_original'][2]:.2f}) mm")
        print(f"\nUsage:")
        print(f"  python generate_notepad.py          # Generate O0 (test)")
        print(f"  python generate_notepad.py C5       # Generate specific note")
        print(f"  python generate_notepad.py --all    # Generate all 29 notes")

        # Save properties to JSON
        save_notepad_properties([result], Path(output_dir) / "notepad_properties.json")


if __name__ == "__main__":
    main()
