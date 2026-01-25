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
PAN_THICKNESS = 3.0       # Thickness of the pan playing surface (downward)
GROVE_DEPTH = 3.0         # Groove thickness downward (same as pan)
GROVE_PROTRUSION = 1.5    # Groove protrusion upward (ridge above surface)

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

    # Thicken pan surface (downward only)
    print(f"Thickening pan surface (down: {pan_thickness}mm)...")
    pan_solid_verts, pan_solid_faces = thicken_surface(pan_verts, pan_faces,
                                                        thickness_down=pan_thickness,
                                                        thickness_up=0)
    print(f"  Pan solid: {len(pan_solid_verts)} vertices, {len(pan_solid_faces)} faces")
    print(f"  Pan boundary loops: {len(find_all_boundary_loops(find_boundary_edges(pan_faces)))}")

    # Thicken grove (downward + slight protrusion upward)
    print(f"Thickening grove (down: {grove_depth}mm, up: {grove_protrusion}mm)...")
    grove_solid_verts, grove_solid_faces = thicken_surface(grove_verts, grove_faces,
                                                            thickness_down=grove_depth,
                                                            thickness_up=grove_protrusion)
    print(f"  Grove solid: {len(grove_solid_verts)} vertices, {len(grove_solid_faces)} faces")
    print(f"  Grove boundary loops: {len(find_all_boundary_loops(find_boundary_edges(grove_faces)))}")

    # Combine the two solids
    print(f"Combining solids...")
    n_pan_solid_verts = len(pan_solid_verts)
    solid_verts = np.vstack([pan_solid_verts, grove_solid_verts])
    solid_faces = pan_solid_faces + [[idx + n_pan_solid_verts for idx in face] for face in grove_solid_faces]
    print(f"  Combined solid: {len(solid_verts)} vertices, {len(solid_faces)} faces")

    # Center at origin
    solid_verts, offset = center_mesh(solid_verts)
    print(f"  Centered (offset: {offset[0]:.1f}, {offset[1]:.1f}, {offset[2]:.1f})")

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
        'vertices': solid_verts,
        'faces': solid_faces,
        'bbox_size': bbox_size,
        'obj_path': str(obj_path),
        'stl_path': str(stl_path)
    }


def main():
    obj_path = "data/Tenor Pan only.obj"
    output_dir = "data/notepads"

    # Generate test note: O0 (F#4) - outer ring
    result = generate_notepad("O0", obj_path, output_dir)

    if result:
        print(f"\nTest note pad generated successfully!")
        print(f"View the OBJ file in any 3D viewer, or import the STL into a slicer for printing.")


if __name__ == "__main__":
    main()
