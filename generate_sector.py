#!/usr/bin/env python3
"""
Generate 3D-printable sector pieces that tile together to form the pan shell.

Each sector:
- Follows the pan's curved bowl surface in its region
- Has a pocket where the notepad sits flush (1.5mm deep)
- Has a clearance hole for the mount assembly (50mm)
- Has 4x M4 through-holes at boss positions
- Has overlap tabs at edges for bolting to adjacent sectors

Usage:
    python generate_sector.py              # Generate O0 (test)
    python generate_sector.py C5           # Generate specific sector
    python generate_sector.py --all        # Generate all 29 sectors
    python generate_sector.py --verify     # Verify tiling completeness
"""

import numpy as np
from pathlib import Path
import math
import json
import sys
import struct

# Import shared constants and functions from generate_notepad
from generate_notepad import (
    CM_TO_MM, PAN_SCALE, PAN_THICKNESS, GROVE_DEPTH,
    NOTE_MAPPING, NOTE_BY_INDEX,
    parse_obj_file, extract_object_mesh,
    compute_vertex_normals, compute_surface_normal,
    thicken_surface, find_boundary_edges, find_all_boundary_loops,
    transform_cylinder_to_normal,
    BOSS_HOLE_DIAMETER, BOSS_HEIGHT, BOSS_OUTER_DIAMETER
)

TOTAL_SCALE = CM_TO_MM * PAN_SCALE  # 20x

# ============================================================
# Sector-specific constants
# ============================================================

SECTOR_THICKNESS = 4.0          # mm, shell wall thickness
POCKET_DEPTH = PAN_THICKNESS    # 1.5mm, flush with notepad surface
MOUNT_CLEARANCE_HOLE = 50.0     # mm diameter, clears outer sleeve (48.5mm + tolerance)
M4_THROUGH_HOLE = 4.3           # mm, M4 clearance for notepad screws
OVERLAP_WIDTH = 8.0             # mm, overlap tab extension beyond Voronoi boundary
M3_BOLT_HOLE = 3.3              # mm, M3 clearance for sector-to-sector bolts
BOLT_SPACING = 30.0             # mm, spacing between bolt holes along boundary
HOLE_SEGMENTS = 32              # resolution for circular holes

# Mounting standoff parameters (on underside of sector at boss positions)
STANDOFF_OUTER_DIAMETER = 14.0  # mm, larger than notepad boss (10mm) for strength
STANDOFF_HOLE_DIAMETER = 4.3    # mm, M4 clearance
STANDOFF_SEGMENTS = 24          # resolution for standoff cylinders
STANDOFF_FILLET_HEIGHT = 3.0    # mm, gradual fillet transition to shell surface
# Height computed dynamically: BOSS_HEIGHT + POCKET_DEPTH - SECTOR_THICKNESS

# Interlocking peg-and-socket parameters
PEG_DIAMETER = 4.0              # mm, peg outer diameter
PEG_HEIGHT = 3.0                # mm, protrusion from tab bottom
SOCKET_DIAMETER = 4.3           # mm, slightly larger for clearance
SOCKET_DEPTH = 3.5              # mm, slightly deeper for tolerance
PEG_SPACING = 35.0              # mm, spacing between pegs along boundary
PEG_SEGMENTS = 12               # resolution for peg/socket cylinders

# Printability parameters
POCKET_CHAMFER = 0.5            # mm, chamfer width at pocket-to-rim edge
RIB_WIDTH = 2.0                 # mm, reinforcement rib cross-section width
RIB_DEPTH = 2.0                 # mm, rib protrusion below shell
RIB_SIZE_THRESHOLD = 150.0      # mm, add ribs if sector bbox exceeds this


# ============================================================
# Phase A: Bowl Surface Extraction
# ============================================================

def extract_bowl_surface(obj_path):
    """
    Extract the complete bowl surface (Pan + Groves materials) from the OBJ file.

    Returns:
        vertices: np.ndarray (N, 3) in centered 20x-scaled mm (matching pan_surface.obj coords)
        faces: list of face index lists
        face_material: list of material name per face ('Pan' or 'Groves')
        face_group: list of group name per face
        pan_centroid_offset: np.ndarray (3,) the centering offset in scaled mm
    """
    # Parse the OBJ to get all geometry
    objects, all_vertices_cm = parse_obj_file(obj_path)

    # Identify Pan and Groves material objects
    bowl_objects = {}
    for obj_name, obj_data in objects.items():
        if obj_data['material'] in ('Pan', 'Groves'):
            bowl_objects[obj_name] = obj_data

    print(f"Bowl objects: {len(bowl_objects)} ({sum(1 for o in bowl_objects.values() if o['material'] == 'Pan')} Pan, "
          f"{sum(1 for o in bowl_objects.values() if o['material'] == 'Groves')} Groves)")

    # Collect all vertex indices used by bowl faces
    all_used_indices = set()
    for obj_data in bowl_objects.values():
        all_used_indices.update(obj_data['face_vertices'])

    sorted_indices = sorted(all_used_indices)
    old_to_new = {old: new for new, old in enumerate(sorted_indices)}

    # Compute centroid in cm (same as extract_pan_surface.py uses for Pan-only objects)
    # Load the stored centroid offset instead to ensure exact match
    offset_path = Path(obj_path).parent / "pan_centroid_offset.json"
    if offset_path.exists():
        with open(offset_path) as f:
            offset_data = json.load(f)
        pan_centroid_offset = np.array(offset_data["centroid_offset_mm"])
        raw_centroid_cm = np.array(offset_data["centroid_raw_cm"])
        print(f"Loaded pan centroid offset: [{pan_centroid_offset[0]:.2f}, {pan_centroid_offset[1]:.2f}, {pan_centroid_offset[2]:.2f}]")
    else:
        # Fallback: compute from Pan-only vertices (matching extract_pan_surface.py)
        pan_indices = set()
        for obj_data in bowl_objects.values():
            if obj_data['material'] == 'Pan':
                pan_indices.update(obj_data['face_vertices'])
        pan_sorted = sorted(pan_indices)
        cx = sum(all_vertices_cm[i][0] for i in pan_sorted) / len(pan_sorted)
        cy = sum(all_vertices_cm[i][1] for i in pan_sorted) / len(pan_sorted)
        cz = sum(all_vertices_cm[i][2] for i in pan_sorted) / len(pan_sorted)
        raw_centroid_cm = np.array([cx, cy, cz])
        pan_centroid_offset = raw_centroid_cm * TOTAL_SCALE
        print(f"Computed pan centroid offset: [{pan_centroid_offset[0]:.2f}, {pan_centroid_offset[1]:.2f}, {pan_centroid_offset[2]:.2f}]")

    # Create vertices: scaled and centered (matching pan_surface.obj coordinate system)
    vertices = np.zeros((len(sorted_indices), 3))
    for new_idx, old_idx in enumerate(sorted_indices):
        v_cm = all_vertices_cm[old_idx]
        vertices[new_idx] = (v_cm - raw_centroid_cm) * TOTAL_SCALE

    # Merge duplicate vertices (within tolerance) that come from different objects
    vertices, old_to_merged, merge_count = merge_close_vertices(vertices, tolerance=0.01)
    # Update old_to_new to go through merge mapping
    old_to_final = {old: old_to_merged[new] for old, new in old_to_new.items()}

    print(f"Merged {merge_count} duplicate vertices ({len(sorted_indices)} -> {len(vertices)})")

    # Build faces with new indices, tracking material and group
    faces = []
    face_material = []
    face_group = []

    for obj_name, obj_data in bowl_objects.items():
        for face in obj_data['faces']:
            new_face = []
            for idx in face:
                if idx in old_to_final:
                    new_face.append(old_to_final[idx])
            if len(new_face) >= 3:
                # Remove degenerate faces (duplicate vertices)
                if len(set(new_face)) >= 3:
                    faces.append(new_face)
                    face_material.append(obj_data['material'])
                    face_group.append(obj_name)

    print(f"Bowl surface: {len(vertices)} vertices, {len(faces)} faces")
    return vertices, faces, face_material, face_group, pan_centroid_offset


def merge_close_vertices(vertices, tolerance=0.01):
    """Merge vertices that are within tolerance distance of each other."""
    n = len(vertices)
    merged_to = list(range(n))  # mapping: old index -> merged index

    # Sort vertices by x coordinate for efficient neighbor search
    sorted_by_x = np.argsort(vertices[:, 0])

    merge_count = 0
    for ii in range(n):
        i = sorted_by_x[ii]
        if merged_to[i] != i:
            continue  # already merged

        for jj in range(ii + 1, n):
            j = sorted_by_x[jj]
            if merged_to[j] != j:
                continue

            # Early exit: if x difference exceeds tolerance, no more matches
            if vertices[j, 0] - vertices[i, 0] > tolerance:
                break

            dist = np.linalg.norm(vertices[i] - vertices[j])
            if dist <= tolerance:
                merged_to[j] = i
                merge_count += 1

    # Build compact mapping: remove gaps
    unique_indices = sorted(set(merged_to))
    compact_map = {old: new for new, old in enumerate(unique_indices)}

    # Final mapping: old index -> compact new index
    final_map = np.array([compact_map[merged_to[i]] for i in range(n)])

    new_vertices = vertices[unique_indices]
    return new_vertices, final_map, merge_count


# ============================================================
# Phase B.5: Boundary Smoothing
# ============================================================

def find_triple_points(boundary_edges_dict):
    """
    Find vertices where 3 or more sector boundaries meet.
    These vertices are pinned during smoothing (must not move).

    Returns: set of vertex indices that are triple-points.
    """
    vertex_boundary_pairs = {}  # vertex -> set of boundary pair keys
    for key, edges in boundary_edges_dict.items():
        for v1, v2 in edges:
            vertex_boundary_pairs.setdefault(v1, set()).add(key)
            vertex_boundary_pairs.setdefault(v2, set()).add(key)
    return {v for v, pairs in vertex_boundary_pairs.items() if len(pairs) >= 2}


def order_boundary_chain(edges):
    """
    Order boundary edges into a connected chain of vertex indices.
    Returns list of vertex chains (may be multiple disconnected segments).
    """
    adj = {}
    for v1, v2 in edges:
        adj.setdefault(v1, []).append(v2)
        adj.setdefault(v2, []).append(v1)

    visited = set()
    chains = []

    for start_v in sorted(adj.keys()):
        if start_v in visited:
            continue
        chain = [start_v]
        visited.add(start_v)
        current = start_v
        while True:
            neighbors = [n for n in adj.get(current, []) if n not in visited]
            if not neighbors:
                break
            current = neighbors[0]
            chain.append(current)
            visited.add(current)
        if len(chain) >= 3:
            chains.append(chain)

    return chains


def smooth_boundary_chains(vertices, boundary_edges_dict, iterations=8, alpha=0.4):
    """
    Smooth Voronoi boundary vertex chains using constrained Laplacian smoothing.

    For each boundary between sectors, extracts the chain of vertices and
    iteratively moves each interior vertex toward the average of its neighbors,
    constrained to the bowl surface (preserves radial distance from center).

    Modifies vertices in-place.
    """
    triple_points = find_triple_points(boundary_edges_dict)
    bowl_center = vertices.mean(axis=0)

    # Collect all boundary chains and their movable vertices
    all_chains = []
    for key, edges in boundary_edges_dict.items():
        chains = order_boundary_chain(edges)
        for chain in chains:
            all_chains.append(chain)

    if not all_chains:
        return

    # Build set of all boundary vertices and their chain neighbors
    # A vertex can appear in multiple chains; we average all contributions
    smoothed = 0
    for iteration in range(iterations):
        new_positions = {}

        for chain in all_chains:
            for i in range(len(chain)):
                vi = chain[i]
                if vi in triple_points:
                    continue  # pinned
                # Endpoints of chain are also pinned
                if i == 0 or i == len(chain) - 1:
                    continue

                prev_v = chain[i - 1]
                next_v = chain[i + 1]
                avg_pos = (vertices[prev_v] + vertices[next_v]) / 2.0
                smoothed_pos = (1 - alpha) * vertices[vi] + alpha * avg_pos

                if vi in new_positions:
                    # Average multiple chain contributions
                    new_positions[vi] = (new_positions[vi] + smoothed_pos) / 2.0
                else:
                    new_positions[vi] = smoothed_pos

        # Apply positions, projecting back to original radial distance
        for vi, new_pos in new_positions.items():
            # Preserve radial distance from bowl center (stay on bowl surface)
            old_radial = np.linalg.norm(vertices[vi] - bowl_center)
            new_dir = new_pos - bowl_center
            new_radial = np.linalg.norm(new_dir)
            if new_radial > 0:
                vertices[vi] = bowl_center + new_dir * (old_radial / new_radial)
            smoothed += 1

    n_verts = len(set(v for chain in all_chains for v in chain)) - len(triple_points)
    print(f"  Smoothed {n_verts} boundary vertices over {iterations} iterations "
          f"({len(triple_points)} triple-points pinned)")


# ============================================================
# Phase B: Voronoi Partitioning
# ============================================================

def load_notepad_properties(props_path, pan_centroid_offset):
    """
    Load notepad properties and convert coordinates to pan_surface space.

    Returns dict: note_index -> {
        'centroid_pan': np.ndarray(3,),
        'normal': np.ndarray(3,),
        'boss_positions_pan': list of np.ndarray(3,),
        'ring': str,
        'pan_object': str,
        'grove_object': str,
        ...
    }
    """
    with open(props_path) as f:
        props_list = json.load(f)

    note_props = {}
    for p in props_list:
        idx = p['index']
        centroid_orig = np.array(p['centroid_original'])
        centroid_pan = centroid_orig - pan_centroid_offset

        boss_pan = []
        if 'boss_positions' in p:
            for bp in p['boss_positions']:
                boss_pan.append(np.array(bp) - pan_centroid_offset)

        note_info = NOTE_BY_INDEX.get(idx, {})

        note_props[idx] = {
            'centroid_pan': centroid_pan,
            'normal': np.array(p['normal']),
            'boss_positions_pan': boss_pan,
            'ring': p['ring'],
            'bbox_size': p.get('bbox_size', [0, 0, 0]),
            'pan_object': note_info.get('pan_object', ''),
            'grove_object': note_info.get('grove_object', ''),
        }

    return note_props


def assign_faces_to_sectors(vertices, faces, face_group, note_props):
    """
    Assign each bowl face to the nearest note (Voronoi partitioning).

    Uses face centroid distance to note centroids.
    Also tracks which faces belong to each note's own Pan/Grove objects.

    Returns:
        face_assignments: list of note_index strings (one per face)
        note_own_faces: dict mapping note_index -> set of face indices that are
                        this note's own Pan or Grove objects
        adjacency: dict mapping note_index -> set of adjacent note indices
    """
    # Build lookup: object_name -> note_index
    obj_to_note = {}
    for idx, props in note_props.items():
        if props['pan_object']:
            obj_to_note[props['pan_object']] = idx
        if props['grove_object']:
            obj_to_note[props['grove_object']] = idx

    # Compute face centroids
    centroids_list = list(note_props.keys())
    centroid_positions = np.array([note_props[idx]['centroid_pan'] for idx in centroids_list])

    face_assignments = []
    note_own_faces = {idx: set() for idx in note_props}

    for fi, face in enumerate(faces):
        # Compute face centroid
        face_verts = vertices[face]
        face_center = face_verts.mean(axis=0)

        # Find nearest note centroid
        dists = np.linalg.norm(centroid_positions - face_center, axis=1)
        nearest_idx = centroids_list[np.argmin(dists)]
        face_assignments.append(nearest_idx)

        # Track own Pan/Grove faces
        group = face_group[fi]
        if group in obj_to_note:
            note_own_faces[obj_to_note[group]].add(fi)

    # Compute adjacency and boundary edges between sectors
    adjacency = {idx: set() for idx in note_props}
    boundary_edges = {}  # (note_a, note_b) -> list of (v1, v2) vertex index pairs
    edge_to_faces = {}

    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            v1, v2 = face[i], face[(i + 1) % n]
            edge = tuple(sorted([v1, v2]))
            if edge in edge_to_faces:
                other_fi = edge_to_faces[edge]
                a = face_assignments[fi]
                b = face_assignments[other_fi]
                if a != b:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
                    key = tuple(sorted([a, b]))
                    if key not in boundary_edges:
                        boundary_edges[key] = []
                    boundary_edges[key].append((v1, v2))
            else:
                edge_to_faces[edge] = fi

    return face_assignments, note_own_faces, adjacency, boundary_edges


# ============================================================
# Phase C: Sector Geometry Generation
# ============================================================

def extract_sector_submesh(vertices, faces, face_indices):
    """
    Extract a submesh for the given face indices, re-indexing vertices.

    Returns:
        sector_verts: np.ndarray of unique vertices
        sector_faces: list of faces with new indices
        old_to_new: dict mapping original vertex index -> new index
    """
    used_verts = set()
    for fi in face_indices:
        for vi in faces[fi]:
            used_verts.add(vi)

    sorted_verts = sorted(used_verts)
    old_to_new = {old: new for new, old in enumerate(sorted_verts)}

    sector_verts = vertices[sorted_verts]
    sector_faces = []
    for fi in face_indices:
        new_face = [old_to_new[vi] for vi in faces[fi]]
        sector_faces.append(new_face)

    return sector_verts, sector_faces, old_to_new


def thicken_with_pocket(vertices, faces, is_pocket_face, sector_thickness, pocket_depth):
    """
    Thicken a surface with differential thickness to create a pocket.

    Pocket faces: thicken only downward by sector_thickness.
    Non-pocket faces: thicken downward by sector_thickness AND upward by pocket_depth.

    This creates a raised rim around the pocket so the notepad sits flush.

    Returns:
        all_vertices, all_faces
    """
    normals = compute_vertex_normals(vertices, faces)
    n_verts = len(vertices)

    # Determine which vertices are pocket-only, rim-only, or shared
    # A vertex is "pocket" if ALL its faces are pocket faces
    # A vertex is "rim" if ALL its faces are non-pocket faces
    # A vertex on the boundary between pocket and rim gets rim treatment
    vertex_pocket_count = np.zeros(n_verts, dtype=int)
    vertex_face_count = np.zeros(n_verts, dtype=int)

    for fi, face in enumerate(faces):
        for vi in face:
            vertex_face_count[vi] += 1
            if is_pocket_face[fi]:
                vertex_pocket_count[vi] += 1

    # Vertex is pocket if ALL faces are pocket, otherwise rim (raised)
    vertex_is_pocket = vertex_pocket_count == vertex_face_count

    # Create top vertices
    top_vertices = np.copy(vertices)
    for i in range(n_verts):
        if not vertex_is_pocket[i]:
            top_vertices[i] = vertices[i] + normals[i] * pocket_depth

    # Create bottom vertices (all go down by sector_thickness)
    bottom_vertices = vertices - normals * sector_thickness

    # Combine: [top_vertices, bottom_vertices]
    all_vertices = np.vstack([top_vertices, bottom_vertices])

    # Top faces (keep winding)
    top_faces = [list(face) for face in faces]

    # Bottom faces (offset + reverse winding)
    bottom_faces = [[vi + n_verts for vi in reversed(face)] for face in faces]

    # Side walls at boundary loops
    boundary_edges = find_boundary_edges(faces)
    boundary_loops = find_all_boundary_loops(boundary_edges)

    side_faces = []
    for loop in boundary_loops:
        for i in range(len(loop)):
            v1 = loop[i]
            v2 = loop[(i + 1) % len(loop)]
            side_faces.append([v1, v2, v2 + n_verts, v1 + n_verts])

    all_faces = top_faces + bottom_faces + side_faces
    return all_vertices, all_faces


def remove_faces_in_circle(vertices, faces, is_pocket_face, center, normal, diameter):
    """
    Remove faces from a surface mesh whose centroid falls within a circular region.

    This is applied to the SURFACE mesh before thickening, so thicken_with_pocket
    will automatically create clean walls around the hole boundary.

    Returns:
        filtered_faces: list of faces (indices unchanged)
        filtered_pocket: list of booleans matching filtered_faces
    """
    radius = diameter / 2.0
    z_local = normal / np.linalg.norm(normal)
    if abs(z_local[0]) < 0.9:
        x_local = np.cross(z_local, np.array([1, 0, 0]))
    else:
        x_local = np.cross(z_local, np.array([0, 1, 0]))
    x_local = x_local / np.linalg.norm(x_local)
    y_local = np.cross(z_local, x_local)

    filtered_faces = []
    filtered_pocket = []

    for fi, face in enumerate(faces):
        face_center = vertices[face].mean(axis=0)
        rel = face_center - center
        proj_x = np.dot(rel, x_local)
        proj_y = np.dot(rel, y_local)
        radial_dist = math.sqrt(proj_x**2 + proj_y**2)

        if radial_dist >= radius:
            filtered_faces.append(face)
            filtered_pocket.append(is_pocket_face[fi])

    return filtered_faces, filtered_pocket


def generate_standoff(height, outer_diameter, hole_diameter, fillet_height,
                      segments=STANDOFF_SEGMENTS):
    """
    Generate a hollow cylindrical standoff with a fillet at the top.

    The standoff extends downward along -Z from z=0.
    The fillet creates a smooth transition at z=0 where it meets the shell surface,
    widening from outer_diameter/2 to outer_diameter/2 + fillet_height over fillet_height mm.

    Returns vertices (np.ndarray) and faces (list of lists).
    """
    outer_r = outer_diameter / 2
    inner_r = hole_diameter / 2
    fillet_r = outer_r + fillet_height  # max radius at the fillet base (z=0)
    vertices = []
    faces = []

    # Fillet profile: quarter-circle from (r=fillet_r, z=0) to (r=outer_r, z=-fillet_height)
    fillet_steps = 4
    fillet_rings = []
    for step in range(fillet_steps + 1):
        t = step / fillet_steps  # 0 at top (z=0), 1 at bottom of fillet
        angle_t = t * math.pi / 2
        r = outer_r + fillet_height * math.cos(angle_t)
        z = -fillet_height * math.sin(angle_t)
        ring = []
        for i in range(segments):
            a = 2 * math.pi * i / segments
            ring.append(len(vertices))
            vertices.append([r * math.cos(a), r * math.sin(a), z])
        fillet_rings.append(ring)

    # Inner top ring at z=0 (for top annulus)
    inner_top_ring = []
    for i in range(segments):
        a = 2 * math.pi * i / segments
        inner_top_ring.append(len(vertices))
        vertices.append([inner_r * math.cos(a), inner_r * math.sin(a), 0])

    # Main cylinder body: from bottom of fillet to bottom of standoff
    body_z = -(fillet_height)  # already covered by fillet bottom
    bottom_z = -height
    outer_bottom_ring = []
    for i in range(segments):
        a = 2 * math.pi * i / segments
        outer_bottom_ring.append(len(vertices))
        vertices.append([outer_r * math.cos(a), outer_r * math.sin(a), bottom_z])

    inner_bottom_ring = []
    for i in range(segments):
        a = 2 * math.pi * i / segments
        inner_bottom_ring.append(len(vertices))
        vertices.append([inner_r * math.cos(a), inner_r * math.sin(a), bottom_z])

    # --- Faces ---

    # Fillet outer wall (connect fillet rings top to bottom)
    for ri in range(len(fillet_rings) - 1):
        ring_a = fillet_rings[ri]
        ring_b = fillet_rings[ri + 1]
        for i in range(segments):
            i_next = (i + 1) % segments
            faces.append([ring_a[i], ring_b[i], ring_b[i_next], ring_a[i_next]])

    # Outer wall from fillet bottom to standoff bottom
    fillet_bottom = fillet_rings[-1]
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([fillet_bottom[i], outer_bottom_ring[i],
                     outer_bottom_ring[i_next], fillet_bottom[i_next]])

    # Inner wall (top to bottom, inward-facing)
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([inner_top_ring[i], inner_top_ring[i_next],
                     inner_bottom_ring[i_next], inner_bottom_ring[i]])

    # Top annulus (fillet top ring to inner top ring, upward-facing)
    fillet_top = fillet_rings[0]
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([fillet_top[i], fillet_top[i_next],
                     inner_top_ring[i_next], inner_top_ring[i]])

    # Bottom annulus (outer bottom to inner bottom, downward-facing)
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([outer_bottom_ring[i], inner_bottom_ring[i],
                     inner_bottom_ring[i_next], outer_bottom_ring[i_next]])

    return np.array(vertices), faces


def generate_peg(diameter, height, segments=PEG_SEGMENTS):
    """
    Generate a solid cylindrical peg centered at origin, extending along -Z.
    Returns vertices and faces for a watertight closed cylinder.
    """
    r = diameter / 2
    vertices = []
    faces = []

    # Top ring at z=0
    top_ring = []
    for i in range(segments):
        a = 2 * math.pi * i / segments
        top_ring.append(len(vertices))
        vertices.append([r * math.cos(a), r * math.sin(a), 0])

    # Bottom ring at z=-height
    bottom_ring = []
    for i in range(segments):
        a = 2 * math.pi * i / segments
        bottom_ring.append(len(vertices))
        vertices.append([r * math.cos(a), r * math.sin(a), -height])

    # Center points for caps
    top_center = len(vertices)
    vertices.append([0, 0, 0])
    bottom_center = len(vertices)
    vertices.append([0, 0, -height])

    # Side wall
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([top_ring[i], bottom_ring[i],
                     bottom_ring[i_next], top_ring[i_next]])

    # Top cap (upward-facing fan)
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([top_center, top_ring[i_next], top_ring[i]])

    # Bottom cap (downward-facing fan)
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([bottom_center, bottom_ring[i], bottom_ring[i_next]])

    return np.array(vertices, dtype=float), faces


def generate_socket(diameter, depth, segments=PEG_SEGMENTS):
    """
    Generate a cylindrical socket (blind hole) as geometry to merge into a surface.

    Creates a cylinder wall + bottom cap centered at origin, extending along -Z.
    The top is open (merges with the host surface that has had faces removed).
    """
    r = diameter / 2
    vertices = []
    faces = []

    # Top ring at z=0 (open - connects to host surface boundary)
    top_ring = []
    for i in range(segments):
        a = 2 * math.pi * i / segments
        top_ring.append(len(vertices))
        vertices.append([r * math.cos(a), r * math.sin(a), 0])

    # Bottom ring at z=-depth
    bottom_ring = []
    for i in range(segments):
        a = 2 * math.pi * i / segments
        bottom_ring.append(len(vertices))
        vertices.append([r * math.cos(a), r * math.sin(a), -depth])

    # Bottom center
    bottom_center = len(vertices)
    vertices.append([0, 0, -depth])

    # Inner wall (inward-facing, viewed from inside the socket)
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([top_ring[i], top_ring[i_next],
                     bottom_ring[i_next], bottom_ring[i]])

    # Bottom cap (upward-facing, closes the socket)
    for i in range(segments):
        i_next = (i + 1) % segments
        faces.append([bottom_center, bottom_ring[i], bottom_ring[i_next]])

    return np.array(vertices, dtype=float), faces


def determine_overlap_side(note_a, note_b):
    """
    Determine which sector gets the overlap tab at a shared boundary.
    The lexicographically smaller index gets the 'over' tab.
    """
    return note_a if note_a < note_b else note_b


def generate_overlap_tab(vertices, boundary_edges, avg_normal, overlap_width, thickness):
    """
    Generate an overlap tab extending outward from boundary edges.

    For each boundary edge chain, extend outward by overlap_width to create
    a thickened tab with bolt holes and interlock positions.

    Returns:
        tab_verts: np.ndarray
        tab_faces: list
        bolt_positions: list of np.ndarray(3,) bolt hole center positions
        interlock_positions: list of (position, normal) for peg/socket placement
    """
    if not boundary_edges:
        return np.zeros((0, 3)), [], [], []

    # Collect unique vertices and build adjacency for chain ordering
    edge_verts = set()
    adj = {}
    for v1, v2 in boundary_edges:
        edge_verts.add(v1)
        edge_verts.add(v2)
        adj.setdefault(v1, []).append(v2)
        adj.setdefault(v2, []).append(v1)

    if len(edge_verts) < 2:
        return np.zeros((0, 3)), [], []

    z_local = avg_normal / np.linalg.norm(avg_normal)

    # Walk all chains (may be multiple disconnected boundary segments)
    visited = set()
    chains = []

    for start_v in sorted(edge_verts):
        if start_v in visited:
            continue
        chain = [start_v]
        visited.add(start_v)
        current = start_v
        while True:
            neighbors = [n for n in adj.get(current, []) if n not in visited]
            if not neighbors:
                break
            current = neighbors[0]
            chain.append(current)
            visited.add(current)
        if len(chain) >= 2:
            chains.append(chain)

    all_tab_verts = []
    all_tab_faces = []
    bolt_positions = []
    interlock_positions = []

    for chain in chains:
        chain_verts = vertices[chain]
        n_chain = len(chain)

        # Compute outward direction for each vertex
        outer_verts = []
        for i in range(n_chain):
            # Edge tangent (average of adjacent edges)
            if i == 0:
                tangent = chain_verts[1] - chain_verts[0]
            elif i == n_chain - 1:
                tangent = chain_verts[-1] - chain_verts[-2]
            else:
                tangent = chain_verts[i + 1] - chain_verts[i - 1]

            tangent_len = np.linalg.norm(tangent)
            if tangent_len > 0:
                tangent = tangent / tangent_len

            # Outward = cross(normal, tangent) — perpendicular to edge, in surface plane
            outward = np.cross(z_local, tangent)
            outward_len = np.linalg.norm(outward)
            if outward_len > 0:
                outward = outward / outward_len

            outer_verts.append(chain_verts[i] + outward * overlap_width)

        outer_verts = np.array(outer_verts)

        # Build the tab mesh: top + bottom surfaces + side walls
        base_idx = len(all_tab_verts)
        n_cv = n_chain

        # Top surface: inner (chain) + outer vertices
        top_inner = chain_verts.copy()
        top_outer = outer_verts.copy()

        # Bottom surface: offset downward by thickness
        bot_inner = chain_verts - z_local * thickness
        bot_outer = outer_verts - z_local * thickness

        # Vertex layout: [top_inner, top_outer, bot_inner, bot_outer]
        tab_v = np.vstack([top_inner, top_outer, bot_inner, bot_outer])
        ti_start = base_idx
        to_start = base_idx + n_cv
        bi_start = base_idx + 2 * n_cv
        bo_start = base_idx + 3 * n_cv

        tab_f = []
        for i in range(n_cv - 1):
            # Top surface quad
            tab_f.append([ti_start + i, ti_start + i + 1, to_start + i + 1, to_start + i])
            # Bottom surface quad (reversed winding)
            tab_f.append([bi_start + i, bo_start + i, bo_start + i + 1, bi_start + i + 1])
            # Outer wall
            tab_f.append([to_start + i, to_start + i + 1, bo_start + i + 1, bo_start + i])
            # Inner wall
            tab_f.append([ti_start + i, bi_start + i, bi_start + i + 1, ti_start + i + 1])

        # End caps
        if n_cv >= 2:
            # Start cap
            tab_f.append([ti_start, to_start, bo_start, bi_start])
            # End cap
            tab_f.append([ti_start + n_cv - 1, bi_start + n_cv - 1,
                         bo_start + n_cv - 1, to_start + n_cv - 1])

        all_tab_verts.extend(tab_v)
        all_tab_faces.extend(tab_f)

        # Place bolt holes along the chain at regular intervals
        chain_length = 0
        cumulative = [0]
        for i in range(1, n_chain):
            seg_len = np.linalg.norm(chain_verts[i] - chain_verts[i - 1])
            chain_length += seg_len
            cumulative.append(chain_length)

        if chain_length > BOLT_SPACING * 0.5:
            n_bolts = max(1, int(chain_length / BOLT_SPACING))
            for bi in range(n_bolts):
                t = (bi + 0.5) / n_bolts
                target_len = t * chain_length

                # Find position along chain at this length
                for si in range(1, len(cumulative)):
                    if cumulative[si] >= target_len:
                        frac = (target_len - cumulative[si - 1]) / max(0.01, cumulative[si] - cumulative[si - 1])
                        inner_pt = chain_verts[si - 1] + frac * (chain_verts[si] - chain_verts[si - 1])
                        outer_pt = outer_verts[si - 1] + frac * (outer_verts[si] - outer_verts[si - 1])
                        bolt_pos = (inner_pt + outer_pt) / 2  # center of tab
                        bolt_positions.append(bolt_pos)
                        break

        # Place interlock pegs/sockets at PEG_SPACING intervals along boundary
        if chain_length > PEG_SPACING * 0.5:
            n_pegs = max(1, int(chain_length / PEG_SPACING))
            for pi in range(n_pegs):
                t = (pi + 0.5) / n_pegs
                target_len = t * chain_length

                for si in range(1, len(cumulative)):
                    if cumulative[si] >= target_len:
                        frac = (target_len - cumulative[si - 1]) / max(0.01, cumulative[si] - cumulative[si - 1])
                        inner_pt = chain_verts[si - 1] + frac * (chain_verts[si] - chain_verts[si - 1])
                        outer_pt = outer_verts[si - 1] + frac * (outer_verts[si] - outer_verts[si - 1])
                        peg_pos = (inner_pt + outer_pt) / 2
                        interlock_positions.append((peg_pos, z_local))
                        break

    if not all_tab_verts:
        return np.zeros((0, 3)), [], [], []

    return np.array(all_tab_verts), all_tab_faces, bolt_positions, interlock_positions


def add_pocket_chamfer(vertices, faces, is_pocket_face, normal, chamfer_width=POCKET_CHAMFER):
    """
    Add a small chamfer at the pocket-to-rim transition.

    Vertices on the boundary between pocket and rim faces are offset slightly
    upward along the normal, creating a 45-degree ramp that eases notepad insertion
    and reduces stress concentration.

    Modifies vertices in-place. Applied BEFORE thickening.
    """
    n_verts = len(vertices)
    vertex_pocket_count = np.zeros(n_verts, dtype=int)
    vertex_face_count = np.zeros(n_verts, dtype=int)

    for fi, face in enumerate(faces):
        for vi in face:
            vertex_face_count[vi] += 1
            if is_pocket_face[fi]:
                vertex_pocket_count[vi] += 1

    # Boundary vertices: have both pocket and non-pocket faces
    normals = compute_vertex_normals(vertices, faces)
    chamfered = 0
    for i in range(n_verts):
        if vertex_face_count[i] > 0 and vertex_pocket_count[i] > 0:
            pocket_ratio = vertex_pocket_count[i] / vertex_face_count[i]
            if 0 < pocket_ratio < 1:
                # Offset upward proportional to how much it's a rim vertex
                rim_ratio = 1.0 - pocket_ratio
                vertices[i] = vertices[i] + normals[i] * chamfer_width * rim_ratio * 0.5
                chamfered += 1

    return chamfered


def add_reinforcement_ribs(shell_verts, shell_faces, centroid, normal,
                           boss_positions, sector_thickness):
    """
    Add reinforcement ribs on the underside of the sector.

    Ribs run from the central area outward toward each boss position,
    providing stiffness to large/thin sectors. Each rib is a rectangular
    cross-section extrusion on the bottom surface.

    Returns updated shell_verts, shell_faces.
    """
    if not boss_positions:
        return shell_verts, shell_faces

    z_local = normal / np.linalg.norm(normal)
    if abs(z_local[0]) < 0.9:
        x_local = np.cross(z_local, np.array([1, 0, 0]))
    else:
        x_local = np.cross(z_local, np.array([0, 1, 0]))
    x_local = x_local / np.linalg.norm(x_local)
    y_local = np.cross(z_local, x_local)

    rib_half_w = RIB_WIDTH / 2

    for bp in boss_positions:
        # Rib runs from centroid toward boss position on bottom surface
        start = centroid - z_local * sector_thickness
        end = bp - z_local * sector_thickness

        direction = end - start
        length = np.linalg.norm(direction)
        if length < 10.0:  # skip very short ribs
            continue

        # Shorten rib: start 30mm from centroid (past mount hole), end 10mm before boss
        if length < 50.0:
            continue
        rib_start = start + direction * (30.0 / length)
        rib_end = start + direction * ((length - 10.0) / length)

        # Cross direction perpendicular to rib direction in bottom surface plane
        rib_dir = direction / length
        rib_cross = np.cross(z_local, rib_dir)
        rib_cross_len = np.linalg.norm(rib_cross)
        if rib_cross_len < 0.01:
            continue
        rib_cross = rib_cross / rib_cross_len

        # 8 vertices: 4 at bottom surface, 4 at bottom-RIB_DEPTH
        p1 = rib_start + rib_cross * rib_half_w
        p2 = rib_start - rib_cross * rib_half_w
        p3 = rib_end - rib_cross * rib_half_w
        p4 = rib_end + rib_cross * rib_half_w

        p5 = p1 - z_local * RIB_DEPTH
        p6 = p2 - z_local * RIB_DEPTH
        p7 = p3 - z_local * RIB_DEPTH
        p8 = p4 - z_local * RIB_DEPTH

        base = len(shell_verts)
        rib_verts = np.array([p1, p2, p3, p4, p5, p6, p7, p8])
        shell_verts = np.vstack([shell_verts, rib_verts])

        # Faces (outward-facing normals)
        rib_faces = [
            [base+0, base+3, base+2, base+1],  # top (at shell bottom)
            [base+4, base+5, base+6, base+7],  # bottom
            [base+0, base+1, base+5, base+4],  # start cap
            [base+2, base+3, base+7, base+6],  # end cap
            [base+0, base+4, base+7, base+3],  # side 1
            [base+1, base+2, base+6, base+5],  # side 2
        ]
        shell_faces = shell_faces + rib_faces

    return shell_verts, shell_faces


def generate_sector(note_index, bowl_vertices, bowl_faces, face_assignments,
                    face_material, face_group, note_own_faces, note_props,
                    adjacency, sector_boundary_edges, output_dir):
    """Generate the complete 3D-printable sector for one note."""
    props = note_props[note_index]
    centroid = props['centroid_pan']
    normal = props['normal']
    boss_positions = props['boss_positions_pan']

    print(f"\n{'='*60}")
    print(f"Generating Sector: {note_index} ({props['ring']} ring)")
    print(f"{'='*60}")

    # Step 1: Collect faces for this sector
    sector_face_indices = [fi for fi, a in enumerate(face_assignments) if a == note_index]
    own_face_set = note_own_faces[note_index]

    print(f"  Sector faces: {len(sector_face_indices)} (own Pan/Grove: {len(own_face_set)})")

    # Determine which faces are "pocket" (notepad area) vs "rim" (surrounding shell)
    is_pocket = set()
    for fi in sector_face_indices:
        if fi in own_face_set:
            is_pocket.add(fi)

    # Step 2: Extract submesh
    sector_verts, sector_faces, old_to_new = extract_sector_submesh(
        bowl_vertices, bowl_faces, sector_face_indices)

    # Build pocket flag per sector face
    is_pocket_face = []
    for i, fi in enumerate(sector_face_indices):
        is_pocket_face.append(fi in is_pocket)

    print(f"  Submesh: {len(sector_verts)} vertices, {len(sector_faces)} faces")
    print(f"  Pocket faces: {sum(is_pocket_face)}, Rim faces: {sum(not p for p in is_pocket_face)}")

    # Step 3: Remove hole regions from surface BEFORE thickening
    print(f"  Removing mount clearance ({MOUNT_CLEARANCE_HOLE}mm) and M4 hole regions...")
    sector_faces, is_pocket_face = remove_faces_in_circle(
        sector_verts, sector_faces, is_pocket_face,
        centroid, normal, MOUNT_CLEARANCE_HOLE)

    if boss_positions:
        for bp in boss_positions:
            sector_faces, is_pocket_face = remove_faces_in_circle(
                sector_verts, sector_faces, is_pocket_face,
                bp, normal, M4_THROUGH_HOLE)

    print(f"  After hole removal: {len(sector_faces)} faces")

    # Step 4: Re-index to remove unused vertices
    used_verts = set()
    for face in sector_faces:
        for vi in face:
            used_verts.add(vi)
    sorted_used = sorted(used_verts)
    reindex = {old: new for new, old in enumerate(sorted_used)}
    sector_verts = sector_verts[sorted_used]
    sector_faces = [[reindex[vi] for vi in face] for face in sector_faces]

    # Step 4.5: Apply pocket chamfer before thickening
    n_chamfered = add_pocket_chamfer(sector_verts, sector_faces, is_pocket_face, normal)
    if n_chamfered > 0:
        print(f"  Chamfered {n_chamfered} pocket boundary vertices")

    # Step 5: Differential thickening (pocket + raised rim)
    shell_verts, shell_faces = thicken_with_pocket(
        sector_verts, sector_faces, is_pocket_face,
        SECTOR_THICKNESS, POCKET_DEPTH)

    print(f"  Thickened shell: {len(shell_verts)} vertices, {len(shell_faces)} faces")

    # Step 5.5: Add mounting standoffs at boss positions
    standoff_height = BOSS_HEIGHT + POCKET_DEPTH - SECTOR_THICKNESS
    standoff_count = 0

    if boss_positions:
        for bp in boss_positions:
            so_verts, so_faces = generate_standoff(
                height=standoff_height,
                outer_diameter=STANDOFF_OUTER_DIAMETER,
                hole_diameter=STANDOFF_HOLE_DIAMETER,
                fillet_height=STANDOFF_FILLET_HEIGHT
            )
            # Position: top of standoff at sector bottom surface at boss position
            standoff_origin = bp - normal * SECTOR_THICKNESS
            so_verts_xf = transform_cylinder_to_normal(so_verts, standoff_origin, normal)

            n_existing = len(shell_verts)
            shell_verts = np.vstack([shell_verts, so_verts_xf])
            shell_faces = shell_faces + [[vi + n_existing for vi in f] for f in so_faces]
            standoff_count += 1

    if standoff_count > 0:
        print(f"  Added {standoff_count} mounting standoffs "
              f"(h={standoff_height:.1f}mm, OD={STANDOFF_OUTER_DIAMETER}mm)")

    # Step 6: Add overlap tabs at sector boundaries with interlocking pegs
    neighbors = adjacency.get(note_index, set())
    tab_count = 0
    peg_count = 0
    bolt_positions_all = []

    for neighbor in sorted(neighbors):
        # Only add tab if this sector is the "over" side
        if determine_overlap_side(note_index, neighbor) != note_index:
            continue

        key = tuple(sorted([note_index, neighbor]))
        edges = sector_boundary_edges.get(key, [])
        if not edges:
            continue

        neighbor_normal = note_props[neighbor]['normal']
        avg_normal = (normal + neighbor_normal)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)

        tab_verts, tab_faces, bolt_positions, interlock_positions = generate_overlap_tab(
            bowl_vertices, edges, avg_normal, OVERLAP_WIDTH, SECTOR_THICKNESS)

        if len(tab_verts) > 0 and len(tab_faces) > 0:
            n_existing = len(shell_verts)
            shell_verts = np.vstack([shell_verts, tab_verts])
            shell_faces = shell_faces + [[vi + n_existing for vi in f] for f in tab_faces]
            tab_count += 1
            bolt_positions_all.extend(bolt_positions)

            # Add pegs on "over" side tabs (protrude from bottom of tab)
            for peg_pos, peg_normal in interlock_positions:
                peg_v, peg_f = generate_peg(PEG_DIAMETER, PEG_HEIGHT)
                # Position peg at bottom of tab, extending further downward
                peg_origin = peg_pos - avg_normal * SECTOR_THICKNESS
                peg_v_xf = transform_cylinder_to_normal(peg_v, peg_origin, avg_normal)
                n_existing = len(shell_verts)
                shell_verts = np.vstack([shell_verts, peg_v_xf])
                shell_faces = shell_faces + [[vi + n_existing for vi in f] for f in peg_f]
                peg_count += 1

    # Step 6.5: Add sockets on "under" side (where neighbor has the tab)
    socket_count = 0
    for neighbor in sorted(neighbors):
        if determine_overlap_side(note_index, neighbor) == note_index:
            continue  # this side has the tab, not the socket

        key = tuple(sorted([note_index, neighbor]))
        edges = sector_boundary_edges.get(key, [])
        if not edges:
            continue

        neighbor_normal = note_props[neighbor]['normal']
        avg_normal = (normal + neighbor_normal)
        avg_norm_len = np.linalg.norm(avg_normal)
        if avg_norm_len > 0:
            avg_normal = avg_normal / avg_norm_len

        # Compute interlock positions (same algorithm as tab generation)
        # to get matching socket positions on the under-side
        _, _, _, interlock_positions = generate_overlap_tab(
            bowl_vertices, edges, avg_normal, OVERLAP_WIDTH, SECTOR_THICKNESS)

        for peg_pos, peg_normal in interlock_positions:
            sock_v, sock_f = generate_socket(SOCKET_DIAMETER, SOCKET_DEPTH)
            # Socket on the top surface at the boundary edge
            sock_v_xf = transform_cylinder_to_normal(sock_v, peg_pos, avg_normal)
            n_existing = len(shell_verts)
            shell_verts = np.vstack([shell_verts, sock_v_xf])
            shell_faces = shell_faces + [[vi + n_existing for vi in f] for f in sock_f]
            socket_count += 1

    if tab_count > 0 or peg_count > 0 or socket_count > 0:
        print(f"  Added {tab_count} overlap tabs, {len(bolt_positions_all)} bolt holes, "
              f"{peg_count} pegs, {socket_count} sockets")

    # Step 7: Printability - reinforcement ribs for large sectors
    bbox_min = shell_verts.min(axis=0)
    bbox_max = shell_verts.max(axis=0)
    bbox_size = bbox_max - bbox_min
    max_extent = max(bbox_size)

    if max_extent > RIB_SIZE_THRESHOLD and boss_positions:
        pre_rib = len(shell_faces)
        shell_verts, shell_faces = add_reinforcement_ribs(
            shell_verts, shell_faces, centroid, normal,
            boss_positions, SECTOR_THICKNESS)
        n_ribs = len(shell_faces) - pre_rib
        if n_ribs > 0:
            print(f"  Added reinforcement ribs ({n_ribs} faces, sector extent {max_extent:.0f}mm)")

    print(f"  Final mesh: {len(shell_verts)} vertices, {len(shell_faces)} faces")

    # Output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stl_path = output_dir / f"sector_{note_index}.stl"
    obj_path = output_dir / f"sector_{note_index}.obj"

    write_sector_obj(obj_path, shell_verts, shell_faces, f"Sector_{note_index}")
    write_sector_stl(stl_path, shell_verts, shell_faces, f"Sector_{note_index}")

    bbox_min = shell_verts.min(axis=0)
    bbox_max = shell_verts.max(axis=0)
    bbox_size = bbox_max - bbox_min

    print(f"  Bounding box: {bbox_size[0]:.1f} x {bbox_size[1]:.1f} x {bbox_size[2]:.1f} mm")
    print(f"  Saved: {stl_path}")

    return {
        'index': note_index,
        'ring': props['ring'],
        'bbox_size': bbox_size.tolist(),
        'neighbors': sorted(list(neighbors)),
        'stl_path': str(stl_path),
        'obj_path': str(obj_path),
        'centroid_pan': centroid.tolist(),
        'normal': normal.tolist(),
        'boss_positions_pan': [bp.tolist() for bp in boss_positions],
        'standoff_height': standoff_height,
        'n_vertices': len(shell_verts),
        'n_faces': len(shell_faces),
    }


# ============================================================
# Output Functions
# ============================================================

def write_sector_obj(filepath, vertices, faces, object_name="Sector"):
    """Write sector mesh to OBJ file."""
    with open(filepath, 'w') as f:
        f.write(f"# Sector Geometry for Pan Shell\n")
        f.write(f"# Generated by generate_sector.py\n")
        f.write(f"# Units: mm\n\n")
        f.write(f"o {object_name}\n")

        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        for face in faces:
            face_str = " ".join(str(idx + 1) for idx in face)
            f.write(f"f {face_str}\n")

    print(f"  Saved: {filepath}")


def write_sector_stl(filepath, vertices, faces, object_name="Sector"):
    """Write sector mesh to binary STL file."""
    def compute_face_normal(v0, v1, v2):
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        return normal

    # Triangulate faces
    triangles = []
    for face in faces:
        if len(face) == 3:
            triangles.append(face)
        elif len(face) == 4:
            triangles.append([face[0], face[1], face[2]])
            triangles.append([face[0], face[2], face[3]])
        elif len(face) > 4:
            for i in range(1, len(face) - 1):
                triangles.append([face[0], face[i], face[i + 1]])

    with open(filepath, 'wb') as f:
        header = f"STL generated by generate_sector.py - {object_name}".encode('ascii')
        header = header[:80].ljust(80, b'\0')
        f.write(header)
        f.write(struct.pack('<I', len(triangles)))

        for tri in triangles:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            normal = compute_face_normal(v0, v1, v2)
            f.write(struct.pack('<3f', normal[0], normal[1], normal[2]))
            f.write(struct.pack('<3f', v0[0], v0[1], v0[2]))
            f.write(struct.pack('<3f', v1[0], v1[1], v1[2]))
            f.write(struct.pack('<3f', v2[0], v2[1], v2[2]))
            f.write(struct.pack('<H', 0))


def save_sector_properties(results, output_path):
    """Save sector properties to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved sector properties to: {output_path}")


# ============================================================
# Verification
# ============================================================

def verify_tiling(face_assignments, n_faces):
    """Verify all bowl faces are assigned to exactly one sector."""
    assigned = len(face_assignments)
    if assigned != n_faces:
        print(f"WARNING: {assigned} assignments for {n_faces} faces")
        return False

    from collections import Counter
    counts = Counter(face_assignments)
    print(f"\nTiling verification:")
    print(f"  Total faces: {n_faces}")
    print(f"  Assigned faces: {assigned}")
    print(f"  Sectors: {len(counts)}")
    for idx in sorted(counts.keys()):
        print(f"    {idx}: {counts[idx]} faces")

    return True


# ============================================================
# Main
# ============================================================

def main():
    OBJ_PATH = "data/Tenor Pan only.obj"
    PROPS_PATH = "data/notepads/notepad_properties.json"
    OUTPUT_DIR = "data/sectors"

    args = sys.argv[1:]
    generate_all = '--all' in args
    verify_only = '--verify' in args
    specific_notes = [a for a in args if not a.startswith('--')]

    if not specific_notes and not generate_all and not verify_only:
        specific_notes = ['O0']  # default test

    print("=" * 60)
    print("Pan Shell Sector Generator")
    print("=" * 60)

    # Step 1: Extract full bowl surface
    print("\nPhase A: Extracting bowl surface (Pan + Groves)...")
    bowl_verts, bowl_faces, face_material, face_group, pan_offset = extract_bowl_surface(OBJ_PATH)

    # Save bowl surface for reference
    bowl_path = Path(OUTPUT_DIR)
    bowl_path.mkdir(parents=True, exist_ok=True)
    write_sector_obj(bowl_path / "bowl_surface.obj", bowl_verts, bowl_faces, "BowlSurface")

    # Step 2: Load notepad properties
    print("\nLoading notepad properties...")
    note_props = load_notepad_properties(PROPS_PATH, pan_offset)
    print(f"  Loaded {len(note_props)} notes")

    # Check boss_positions availability
    no_boss = [idx for idx, p in note_props.items() if not p['boss_positions_pan']]
    if no_boss:
        print(f"  WARNING: {len(no_boss)} notes missing boss_positions: {no_boss}")
        print(f"  Run 'python generate_notepad.py --all' first")

    # Step 3: Voronoi partitioning
    print("\nPhase B: Voronoi partitioning...")
    face_assignments, note_own_faces, adjacency, sector_boundary_edges = assign_faces_to_sectors(
        bowl_verts, bowl_faces, face_group, note_props)

    print(f"  Adjacency map:")
    for idx in sorted(adjacency.keys()):
        print(f"    {idx}: {sorted(adjacency[idx])}")

    if verify_only:
        verify_tiling(face_assignments, len(bowl_faces))
        return

    # Step 3.5: Smooth sector boundaries
    print("\nPhase B.5: Smoothing sector boundaries...")
    smooth_boundary_chains(bowl_verts, sector_boundary_edges, iterations=8, alpha=0.4)

    # Step 4: Generate sectors
    notes_to_generate = sorted(note_props.keys()) if generate_all else specific_notes

    print(f"\nPhase C: Generating {len(notes_to_generate)} sector(s)...")
    results = []
    for note_index in notes_to_generate:
        if note_index not in note_props:
            print(f"ERROR: Unknown note index '{note_index}'")
            continue
        result = generate_sector(
            note_index, bowl_verts, bowl_faces, face_assignments,
            face_material, face_group, note_own_faces, note_props,
            adjacency, sector_boundary_edges, OUTPUT_DIR)
        results.append(result)

    # Save properties
    if results:
        save_sector_properties(results, Path(OUTPUT_DIR) / "sector_properties.json")

    print(f"\n{'='*60}")
    print(f"Generated {len(results)} sector(s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
