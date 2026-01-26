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

# Scale factor for the pan geometry (2x to make it bigger)
PAN_SCALE = 2.0

# Thickness for the solids (mm)
PAN_THICKNESS = 1.5       # Thickness of the pan playing surface (downward)
GROVE_DEPTH = 1.5         # Groove thickness downward (same as pan)
GROVE_PROTRUSION = 0.0    # Groove protrusion upward (0 = no lip)

# Mounting cylinder parameters (mm)
MOUNT_INNER_DIAMETER = 23.5   # Internal diameter
MOUNT_DEPTH = 9.0             # Cylinder depth
MOUNT_WALL_THICKNESS = 2.5    # Wall thickness
MOUNT_THREAD_PITCH = 2.0      # Thread pitch
MOUNT_THREAD_DEPTH = 1.0      # Thread depth (outward from wall)
MOUNT_NOTCH_WIDTH = 2.0       # Wire notch width
MOUNT_SEGMENTS = 48           # Resolution for cylinder

# Minimum pad size to accommodate cylinder (with margin)
MOUNT_OUTER_DIAMETER = MOUNT_INNER_DIAMETER + 2 * MOUNT_WALL_THICKNESS + 2 * MOUNT_THREAD_DEPTH
MIN_PAD_SIZE = MOUNT_OUTER_DIAMETER + 0.5  # Add 0.5mm margin

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

    # Extract vertices (convert cm to mm and apply scale factor)
    vertices = all_vertices[old_indices] * CM_TO_MM * PAN_SCALE

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
    Generate a solid threaded mounting cylinder with a wire notch properly cut out.

    The cylinder is centered at origin, extending downward along -Z axis.
    The notch is cut through the wall at angle=0 (positive X direction).

    Returns vertices and faces for a watertight mesh.
    """
    inner_r = inner_diameter / 2
    outer_r = inner_r + wall_thickness
    thread_r = outer_r + thread_depth

    vertices = []
    faces = []

    notch_half_width = notch_width / 2

    # Calculate notch angle (use the angle that gives notch_half_width at inner radius)
    notch_half_angle = math.atan2(notch_half_width, inner_r)

    # Number of segments to skip for notch (at least 2)
    notch_segments = max(2, int(notch_half_angle * 2 * segments / (2 * math.pi)) + 1)

    # Make notch symmetric around segment 0
    # Skip segments from (segments - notch_segments//2) to (notch_segments//2)
    notch_start = segments - notch_segments // 2
    notch_end = notch_segments - notch_segments // 2

    def in_notch(seg_idx):
        return seg_idx >= notch_start or seg_idx < notch_end

    # Number of Z levels (16 per pitch for smooth helical thread)
    z_levels = max(int(depth / thread_pitch * 16), 32)

    # Generate cylinder vertices (excluding notch segments)
    inner_rings = []
    outer_rings = []

    for z_idx in range(z_levels + 1):
        z = -z_idx * depth / z_levels

        inner_ring = {}
        outer_ring = {}

        # Thread phase depends only on Z (not angle) for non-helical push-fit threads
        thread_phase = (z_idx / z_levels * depth / thread_pitch) % 1.0
        thread_h = thread_depth * (1 - abs(2 * thread_phase - 1))
        thread_r = outer_r + thread_h

        for i in range(segments):
            if in_notch(i):
                continue

            angle = 2 * math.pi * i / segments

            # Inner vertex
            inner_ring[i] = len(vertices)
            vertices.append([inner_r * math.cos(angle), inner_r * math.sin(angle), z])

            # Outer vertex with thread profile (same radius for all segments at this Z)
            outer_ring[i] = len(vertices)
            vertices.append([thread_r * math.cos(angle), thread_r * math.sin(angle), z])

        inner_rings.append(inner_ring)
        outer_rings.append(outer_ring)

    # Notch boundary angles (first and last non-notch segments)
    valid_segs = sorted([i for i in range(segments) if not in_notch(i)])
    first_valid = valid_segs[0]
    last_valid = valid_segs[-1]

    first_angle = 2 * math.pi * first_valid / segments
    last_angle = 2 * math.pi * last_valid / segments

    # Notch wall vertices (at the edges of the notch)
    # Left wall at last_valid angle, right wall at first_valid angle
    notch_inner_left = []  # at last_angle
    notch_outer_left = []
    notch_inner_right = []  # at first_angle
    notch_outer_right = []

    for z_idx in range(z_levels + 1):
        z = -z_idx * depth / z_levels

        # Left edge (at last_angle, which is just before the notch going clockwise)
        notch_inner_left.append(len(vertices))
        vertices.append([inner_r * math.cos(last_angle), inner_r * math.sin(last_angle), z])
        notch_outer_left.append(len(vertices))
        vertices.append([thread_r * math.cos(last_angle), thread_r * math.sin(last_angle), z])

        # Right edge (at first_angle, which is just after the notch going clockwise)
        notch_inner_right.append(len(vertices))
        vertices.append([inner_r * math.cos(first_angle), inner_r * math.sin(first_angle), z])
        notch_outer_right.append(len(vertices))
        vertices.append([thread_r * math.cos(first_angle), thread_r * math.sin(first_angle), z])

    # === BUILD FACES ===

    # Top cap - connect inner to outer for each segment pair
    for idx in range(len(valid_segs) - 1):
        i = valid_segs[idx]
        i_next = valid_segs[idx + 1]
        faces.append([inner_rings[0][i], outer_rings[0][i],
                     outer_rings[0][i_next], inner_rings[0][i_next]])

    # Top cap notch closure (connects main cap to notch edges)
    faces.append([notch_inner_left[0], notch_outer_left[0],
                 outer_rings[0][last_valid], inner_rings[0][last_valid]])
    faces.append([inner_rings[0][first_valid], outer_rings[0][first_valid],
                 notch_outer_right[0], notch_inner_right[0]])
    # NOTE: No face across notch opening - it's an open slot for wires

    # Bottom cap
    for idx in range(len(valid_segs) - 1):
        i = valid_segs[idx]
        i_next = valid_segs[idx + 1]
        faces.append([inner_rings[-1][i], inner_rings[-1][i_next],
                     outer_rings[-1][i_next], outer_rings[-1][i]])

    # Bottom cap notch closure (connects main cap to notch edges)
    faces.append([notch_inner_left[-1], inner_rings[-1][last_valid],
                 outer_rings[-1][last_valid], notch_outer_left[-1]])
    faces.append([inner_rings[-1][first_valid], notch_inner_right[-1],
                 notch_outer_right[-1], outer_rings[-1][first_valid]])
    # NOTE: No face across notch opening - it's an open slot for wires

    # Inner wall (cylinder bore)
    for z_idx in range(z_levels):
        for idx in range(len(valid_segs) - 1):
            i = valid_segs[idx]
            i_next = valid_segs[idx + 1]
            faces.append([inner_rings[z_idx][i], inner_rings[z_idx][i_next],
                         inner_rings[z_idx+1][i_next], inner_rings[z_idx+1][i]])

    # Outer wall (threaded surface)
    for z_idx in range(z_levels):
        for idx in range(len(valid_segs) - 1):
            i = valid_segs[idx]
            i_next = valid_segs[idx + 1]
            faces.append([outer_rings[z_idx][i], outer_rings[z_idx+1][i],
                         outer_rings[z_idx+1][i_next], outer_rings[z_idx][i_next]])

    # Notch left wall (radial face at last_angle)
    for z_idx in range(z_levels):
        faces.append([notch_inner_left[z_idx], inner_rings[z_idx][last_valid],
                     inner_rings[z_idx+1][last_valid], notch_inner_left[z_idx+1]])
        faces.append([outer_rings[z_idx][last_valid], notch_outer_left[z_idx],
                     notch_outer_left[z_idx+1], outer_rings[z_idx+1][last_valid]])
        faces.append([notch_inner_left[z_idx], notch_outer_left[z_idx],
                     notch_outer_left[z_idx+1], notch_inner_left[z_idx+1]])

    # Notch right wall (radial face at first_angle)
    for z_idx in range(z_levels):
        faces.append([inner_rings[z_idx][first_valid], notch_inner_right[z_idx],
                     notch_inner_right[z_idx+1], inner_rings[z_idx+1][first_valid]])
        faces.append([notch_outer_right[z_idx], outer_rings[z_idx][first_valid],
                     outer_rings[z_idx+1][first_valid], notch_outer_right[z_idx+1]])
        faces.append([notch_outer_right[z_idx], notch_inner_right[z_idx],
                     notch_inner_right[z_idx+1], notch_outer_right[z_idx+1]])

    return np.array(vertices), faces


def check_and_scale_pad(pan_verts, grove_verts, min_size=MIN_PAD_SIZE):
    """
    Check if the pad is large enough for the mounting cylinder.
    If not, scale it up uniformly in the XZ plane (preserving Y/thickness).

    The cylinder will be placed at the centroid, so we need the pad to be
    at least min_size in both X and Z dimensions around that center.

    Returns:
        scaled_pan_verts, scaled_grove_verts, scale_factor, was_scaled
    """
    # Use pan vertices to determine size (pan is the primary surface)
    # The cylinder is placed at the pan centroid

    x_extent = pan_verts[:, 0].max() - pan_verts[:, 0].min()
    z_extent = pan_verts[:, 2].max() - pan_verts[:, 2].min()
    min_extent = min(x_extent, z_extent)

    if min_extent >= min_size:
        # No scaling needed
        return pan_verts, grove_verts, 1.0, False

    # Calculate scale factor needed
    scale_factor = min_size / min_extent

    # Find centroid for scaling (scale from pan center)
    center_x = (pan_verts[:, 0].max() + pan_verts[:, 0].min()) / 2
    center_z = (pan_verts[:, 2].max() + pan_verts[:, 2].min()) / 2

    # Scale pan vertices (XZ only, preserve Y)
    scaled_pan = pan_verts.copy()
    scaled_pan[:, 0] = center_x + (pan_verts[:, 0] - center_x) * scale_factor
    scaled_pan[:, 2] = center_z + (pan_verts[:, 2] - center_z) * scale_factor

    # Scale grove vertices from same center
    scaled_grove = grove_verts.copy()
    scaled_grove[:, 0] = center_x + (grove_verts[:, 0] - center_x) * scale_factor
    scaled_grove[:, 2] = center_z + (grove_verts[:, 2] - center_z) * scale_factor

    return scaled_pan, scaled_grove, scale_factor, True


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


def compute_cylinder_surface_offset(pan_verts, centroid, normal, cylinder_radius):
    """
    Compute how much to lower the cylinder to avoid protrusions through the pan surface.

    The pan surface is curved, so the surface height at the cylinder footprint edge
    may be lower than at the centroid. We find the minimum surface height within
    the cylinder's footprint and return the offset needed to lower the cylinder.

    Returns: offset amount (negative means lower the cylinder)
    """
    # Create local coordinate system with normal as Z
    z_local = normal / np.linalg.norm(normal)

    # Find perpendicular axes
    if abs(z_local[0]) < 0.9:
        x_local = np.cross(z_local, np.array([1, 0, 0]))
    else:
        x_local = np.cross(z_local, np.array([0, 1, 0]))
    x_local = x_local / np.linalg.norm(x_local)
    y_local = np.cross(z_local, x_local)

    # Transform pan vertices to local coordinates (relative to centroid)
    # In local coords: centroid is at origin, normal points along +Z
    rel_verts = pan_verts - centroid
    local_x = rel_verts @ x_local
    local_y = rel_verts @ y_local
    local_z = rel_verts @ z_local

    # Find vertices within cylinder radius (with small margin)
    radial_dist = np.sqrt(local_x**2 + local_y**2)
    within_radius = radial_dist <= cylinder_radius * 1.1  # 10% margin

    if not np.any(within_radius):
        return 0.0  # No adjustment needed

    # Find minimum Z (lowest surface point within cylinder footprint)
    min_z = local_z[within_radius].min()

    # If min_z is negative, the surface dips below the centroid plane
    # We should lower the cylinder top to this level (plus small margin)
    if min_z < 0:
        return min_z - 0.5  # Extra 0.5mm margin

    return 0.0


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

    # Check if pad needs scaling to accommodate mounting cylinder
    pan_verts, grove_verts, scale_factor, was_scaled = check_and_scale_pad(pan_verts, grove_verts)
    if was_scaled:
        print(f"  ** PAD SCALED by {scale_factor:.3f}x to fit mounting cylinder (min size: {MIN_PAD_SIZE}mm)")
    else:
        scale_factor = 1.0

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

    # Check if cylinder protrudes through curved surface and lower if needed
    cylinder_outer_radius = MOUNT_INNER_DIAMETER/2 + MOUNT_WALL_THICKNESS + MOUNT_THREAD_DEPTH
    surface_offset = compute_cylinder_surface_offset(
        pan_verts, pan_surface_centroid, notepad_normal, cylinder_outer_radius)
    if surface_offset < 0:
        # Lower the cylinder along the normal direction
        cylinder_verts_transformed = cylinder_verts_transformed + notepad_normal * surface_offset
        print(f"  Cylinder lowered by {-surface_offset:.2f}mm to avoid surface protrusion")

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
        'scale_factor': scale_factor,  # 1.0 if not scaled, >1.0 if enlarged
        'was_scaled': was_scaled,  # True if pad was enlarged to fit cylinder
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
            'scale_factor': r['scale_factor'],
            'was_scaled': r['was_scaled'],
            'bbox_size': r['bbox_size'],
            'obj_path': r['obj_path'],
            'stl_path': r['stl_path']
        }
        properties.append(props)

    with open(output_path, 'w') as f:
        json.dump(properties, f, indent=2)

    # Report scaled pads
    scaled = [p for p in properties if p['was_scaled']]
    if scaled:
        print(f"\nScaled pads ({len(scaled)}):")
        for p in scaled:
            print(f"  {p['index']}: scaled {p['scale_factor']:.3f}x")

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
