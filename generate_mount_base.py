#!/usr/bin/env python3
"""
Generate a screw-on mount base that screws onto the notepad.

Features:
- Internal helical threads matching the notepad's external threads
- External helical threads for the outer sleeve
- Wire notch through the wall (aligned with notepad notch)
- Solid closed base with PCB mounting bosses
"""

import numpy as np
from pathlib import Path
import struct

# Import matching parameters from notepad generator
NOTEPAD_INNER_DIAMETER = 23.5
NOTEPAD_WALL_THICKNESS = 2.5
NOTEPAD_THREAD_DEPTH = 1.0
NOTEPAD_THREAD_PITCH = 2.0

# Derived notepad dimensions
NOTEPAD_WALL_OUTER = NOTEPAD_INNER_DIAMETER + 2 * NOTEPAD_WALL_THICKNESS
NOTEPAD_THREAD_OUTER = NOTEPAD_WALL_OUTER + 2 * NOTEPAD_THREAD_DEPTH

# Mount base parameters
THREAD_CLEARANCE = 0.5
BASE_INNER_DIAMETER = NOTEPAD_THREAD_OUTER + THREAD_CLEARANCE  # Internal thread root
BASE_THREAD_DEPTH = 1.0  # Internal thread depth
BASE_WALL_THICKNESS = 3.0
THREAD_HEIGHT = 9.0  # Depth of internal threaded cavity
FLOOR_THICKNESS = 6.0  # Solid floor below the threaded cavity (extended for longer external threads)
BASE_HEIGHT = THREAD_HEIGHT + FLOOR_THICKNESS  # Total height

# External thread parameters (for outer sleeve to grip)
EXT_THREAD_DEPTH = 1.0
EXT_THREAD_PITCH = 2.0

# Notch parameters (wire routing, aligned at angle=0)
NOTCH_WIDTH = 5.0  # Width of notch at inner radius

# Anti-rotation groove parameters (matches rib on notepad mount)
GROOVE_WIDTH = 2.5   # Slightly wider than notepad rib (2.0mm) for clearance
GROOVE_DEPTH = 1.0   # Depth inward from interior thread root
GROOVE_ANGLE = np.pi  # Position: 180 degrees from notch

# PCB mount parameters
PCB_HOLE_GRID = 16.0        # M2 holes on 16mm grid
PCB_HOLE_DIAMETER = 2.2     # M2 clearance hole
PCB_BOSS_DIAMETER = 5.0     # Boss around screw hole
PCB_BOSS_HEIGHT = 6.0       # Height of screw bosses (for M2*8 screws)

# Resolution
SEGMENTS = 48
BOSS_SEGMENTS = 16


def helical_thread_profile(phase, depth, crest_fraction=0.25):
    """
    Trapezoidal thread profile — sharp raised ridge with wide groove.

    crest_fraction controls what fraction of the pitch is the raised crest.
    Smaller values = narrower, more visible thread ridges.
    """
    phase = phase % 1.0
    if phase < crest_fraction:
        # Rising edge
        return depth * (phase / crest_fraction)
    elif phase < 0.5:
        # Crest (full height)
        return depth
    elif phase < 0.5 + crest_fraction:
        # Falling edge
        return depth * (1.0 - (phase - 0.5) / crest_fraction)
    else:
        # Groove (root)
        return 0.0


def generate_cylinder():
    """Generate mount base with internal/external threads and wire notch."""
    vertices = []
    faces = []

    inner_r = BASE_INNER_DIAMETER / 2
    outer_r = inner_r + BASE_WALL_THICKNESS
    ext_thread_r = outer_r + EXT_THREAD_DEPTH

    z_top = 0
    z_floor = -THREAD_HEIGHT
    z_bottom = -BASE_HEIGHT

    # Notch angles (centered at angle=0)
    notch_half_angle = np.arctan2(NOTCH_WIDTH / 2, inner_r)
    notch_segs = max(2, int(notch_half_angle * 2 * SEGMENTS / (2 * np.pi)) + 1)
    notch_start = SEGMENTS - notch_segs // 2
    notch_end = (notch_segs + 1) // 2

    def in_notch(seg):
        return seg >= notch_start or seg < notch_end

    valid_segs = [i for i in range(SEGMENTS) if not in_notch(i)]
    first_valid = valid_segs[0]
    last_valid = valid_segs[-1]

    # Notch edge angles
    first_angle = 2 * np.pi * first_valid / SEGMENTS
    last_angle = 2 * np.pi * last_valid / SEGMENTS

    z_levels_thread = max(int(THREAD_HEIGHT / NOTEPAD_THREAD_PITCH * 16), 32)
    z_levels_base = 4
    total_z_levels = z_levels_thread + z_levels_base

    # Anti-rotation groove: which segments are in the groove region
    groove_half_angle = np.arctan2(GROOVE_WIDTH / 2, inner_r)

    def in_groove(seg):
        angle = 2 * np.pi * seg / SEGMENTS
        angle_diff = abs(angle - GROOVE_ANGLE)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        return angle_diff <= groove_half_angle

    # ========== CREATE ALL RING VERTICES ==========
    # Interior threaded rings (valid_segs only) — with anti-rotation groove
    interior_rings = []
    for z_idx in range(z_levels_thread + 1):
        z = z_top - (z_idx / z_levels_thread) * THREAD_HEIGHT
        thread_phase = (z_idx / z_levels_thread * THREAD_HEIGHT / NOTEPAD_THREAD_PITCH) % 1.0
        thread_h = BASE_THREAD_DEPTH * (1 - abs(2 * thread_phase - 1))
        r = inner_r - thread_h

        int_ring = {}
        for seg in valid_segs:
            angle = 2 * np.pi * seg / SEGMENTS
            # Add groove: increase radius (cut outward into wall) for groove segments
            seg_r = r + GROOVE_DEPTH if in_groove(seg) else r
            int_ring[seg] = len(vertices)
            vertices.append([seg_r * np.cos(angle), seg_r * np.sin(angle), z])
        interior_rings.append(int_ring)

    # Exterior rings (valid_segs only) — helical screw threads
    exterior_rings = []
    for z_idx in range(total_z_levels + 1):
        if z_idx <= z_levels_thread:
            z = z_top - (z_idx / z_levels_thread) * THREAD_HEIGHT
        else:
            base_idx = z_idx - z_levels_thread
            z = z_floor - (base_idx / z_levels_base) * FLOOR_THICKNESS

        ext_ring = {}
        for seg in valid_segs:
            angle = 2 * np.pi * seg / SEGMENTS
            # Helical thread: phase depends on both Z and angle
            thread_phase = (z_idx / total_z_levels * BASE_HEIGHT / EXT_THREAD_PITCH + angle / (2 * np.pi)) % 1.0
            thread_h = helical_thread_profile(thread_phase, EXT_THREAD_DEPTH)
            ext_r = outer_r + thread_h
            ext_ring[seg] = len(vertices)
            vertices.append([ext_r * np.cos(angle), ext_r * np.sin(angle), z])
        exterior_rings.append(ext_ring)

    # Floor ring (valid_segs only)
    floor_ring = {}
    for seg in valid_segs:
        angle = 2 * np.pi * seg / SEGMENTS
        floor_ring[seg] = len(vertices)
        vertices.append([inner_r * np.cos(angle), inner_r * np.sin(angle), z_floor])

    # Bottom ring (valid_segs only)
    bottom_ring = {}
    for seg in valid_segs:
        angle = 2 * np.pi * seg / SEGMENTS
        bottom_ring[seg] = len(vertices)
        vertices.append([outer_r * np.cos(angle), outer_r * np.sin(angle), z_bottom])

    # Center vertices
    floor_center = len(vertices)
    vertices.append([0, 0, z_floor])
    bottom_center = len(vertices)
    vertices.append([0, 0, z_bottom])

    # Notch wall vertices (separate from main rings)
    # Left wall at last_angle, right wall at first_angle
    notch_int_left = []  # interior at last_angle
    notch_ext_left = []  # exterior at last_angle
    notch_int_right = []  # interior at first_angle
    notch_ext_right = []  # exterior at first_angle

    for z_idx in range(z_levels_thread + 1):
        z = z_top - (z_idx / z_levels_thread) * THREAD_HEIGHT
        thread_phase = (z_idx / z_levels_thread * THREAD_HEIGHT / NOTEPAD_THREAD_PITCH) % 1.0
        int_thread_h = BASE_THREAD_DEPTH * (1 - abs(2 * thread_phase - 1))
        int_r = inner_r - int_thread_h

        # Helical external thread at notch edge angles
        ext_phase_left = (z_idx / total_z_levels * BASE_HEIGHT / EXT_THREAD_PITCH + last_angle / (2 * np.pi)) % 1.0
        ext_r_left = outer_r + helical_thread_profile(ext_phase_left, EXT_THREAD_DEPTH)
        ext_phase_right = (z_idx / total_z_levels * BASE_HEIGHT / EXT_THREAD_PITCH + first_angle / (2 * np.pi)) % 1.0
        ext_r_right = outer_r + helical_thread_profile(ext_phase_right, EXT_THREAD_DEPTH)

        notch_int_left.append(len(vertices))
        vertices.append([int_r * np.cos(last_angle), int_r * np.sin(last_angle), z])
        notch_ext_left.append(len(vertices))
        vertices.append([ext_r_left * np.cos(last_angle), ext_r_left * np.sin(last_angle), z])
        notch_int_right.append(len(vertices))
        vertices.append([int_r * np.cos(first_angle), int_r * np.sin(first_angle), z])
        notch_ext_right.append(len(vertices))
        vertices.append([ext_r_right * np.cos(first_angle), ext_r_right * np.sin(first_angle), z])

    # Notch wall vertices for base section (exterior only, below floor)
    notch_ext_left_base = [notch_ext_left[-1]]  # Start from floor level
    notch_ext_right_base = [notch_ext_right[-1]]
    for z_idx in range(z_levels_thread + 1, total_z_levels + 1):
        base_idx = z_idx - z_levels_thread
        z = z_floor - (base_idx / z_levels_base) * FLOOR_THICKNESS

        # Helical external thread at notch edge angles
        ext_phase_left = (z_idx / total_z_levels * BASE_HEIGHT / EXT_THREAD_PITCH + last_angle / (2 * np.pi)) % 1.0
        ext_r_left = outer_r + helical_thread_profile(ext_phase_left, EXT_THREAD_DEPTH)
        ext_phase_right = (z_idx / total_z_levels * BASE_HEIGHT / EXT_THREAD_PITCH + first_angle / (2 * np.pi)) % 1.0
        ext_r_right = outer_r + helical_thread_profile(ext_phase_right, EXT_THREAD_DEPTH)

        notch_ext_left_base.append(len(vertices))
        vertices.append([ext_r_left * np.cos(last_angle), ext_r_left * np.sin(last_angle), z])
        notch_ext_right_base.append(len(vertices))
        vertices.append([ext_r_right * np.cos(first_angle), ext_r_right * np.sin(first_angle), z])

    # Floor and bottom notch edge vertices
    notch_floor_left = len(vertices)
    vertices.append([inner_r * np.cos(last_angle), inner_r * np.sin(last_angle), z_floor])
    notch_floor_right = len(vertices)
    vertices.append([inner_r * np.cos(first_angle), inner_r * np.sin(first_angle), z_floor])
    notch_bottom_left = len(vertices)
    vertices.append([outer_r * np.cos(last_angle), outer_r * np.sin(last_angle), z_bottom])
    notch_bottom_right = len(vertices)
    vertices.append([outer_r * np.cos(first_angle), outer_r * np.sin(first_angle), z_bottom])

    # ========== BUILD FACES ==========
    # Interior surface
    for z_idx in range(z_levels_thread):
        for j in range(len(valid_segs) - 1):
            seg, seg_next = valid_segs[j], valid_segs[j + 1]
            faces.append([interior_rings[z_idx][seg], interior_rings[z_idx][seg_next],
                         interior_rings[z_idx + 1][seg_next], interior_rings[z_idx + 1][seg]])

    # Exterior surface
    for z_idx in range(total_z_levels):
        for j in range(len(valid_segs) - 1):
            seg, seg_next = valid_segs[j], valid_segs[j + 1]
            faces.append([exterior_rings[z_idx][seg], exterior_rings[z_idx + 1][seg],
                         exterior_rings[z_idx + 1][seg_next], exterior_rings[z_idx][seg_next]])

    # Top annulus
    for j in range(len(valid_segs) - 1):
        seg, seg_next = valid_segs[j], valid_segs[j + 1]
        faces.append([exterior_rings[0][seg], interior_rings[0][seg],
                     interior_rings[0][seg_next], exterior_rings[0][seg_next]])
    # Top annulus notch closures
    faces.append([notch_ext_left[0], notch_int_left[0],
                 interior_rings[0][last_valid], exterior_rings[0][last_valid]])
    faces.append([exterior_rings[0][first_valid], interior_rings[0][first_valid],
                 notch_int_right[0], notch_ext_right[0]])

    # Interior to floor
    for j in range(len(valid_segs) - 1):
        seg, seg_next = valid_segs[j], valid_segs[j + 1]
        faces.append([interior_rings[-1][seg], interior_rings[-1][seg_next],
                     floor_ring[seg_next], floor_ring[seg]])
    # Floor notch closures
    faces.append([notch_int_left[-1], interior_rings[-1][last_valid],
                 floor_ring[last_valid], notch_floor_left])
    faces.append([interior_rings[-1][first_valid], notch_int_right[-1],
                 notch_floor_right, floor_ring[first_valid]])

    # Floor disk
    for j in range(len(valid_segs) - 1):
        seg, seg_next = valid_segs[j], valid_segs[j + 1]
        faces.append([floor_center, floor_ring[seg], floor_ring[seg_next]])
    # Floor notch opening (triangles from center to notch edges)
    faces.append([floor_center, notch_floor_left, floor_ring[last_valid]])
    faces.append([floor_center, floor_ring[first_valid], notch_floor_right])

    # Exterior to bottom
    for j in range(len(valid_segs) - 1):
        seg, seg_next = valid_segs[j], valid_segs[j + 1]
        faces.append([exterior_rings[-1][seg], exterior_rings[-1][seg_next],
                     bottom_ring[seg_next], bottom_ring[seg]])
    # Bottom notch closures
    faces.append([notch_ext_left_base[-1], exterior_rings[-1][last_valid],
                 bottom_ring[last_valid], notch_bottom_left])
    faces.append([exterior_rings[-1][first_valid], notch_ext_right_base[-1],
                 notch_bottom_right, bottom_ring[first_valid]])

    # Bottom disk
    for j in range(len(valid_segs) - 1):
        seg, seg_next = valid_segs[j], valid_segs[j + 1]
        faces.append([bottom_center, bottom_ring[seg_next], bottom_ring[seg]])
    # Bottom notch opening
    faces.append([bottom_center, bottom_ring[last_valid], notch_bottom_left])
    faces.append([bottom_center, notch_bottom_right, bottom_ring[first_valid]])

    # Inner notch walls (from center to notch edges, closing the solid base)
    # Left inner wall: floor_center → notch_floor_left → notch_bottom_left → bottom_center
    faces.append([floor_center, notch_floor_left, notch_bottom_left, bottom_center])
    # Right inner wall: floor_center → bottom_center → notch_bottom_right → notch_floor_right
    faces.append([floor_center, bottom_center, notch_bottom_right, notch_floor_right])

    # LEFT NOTCH WALL (at last_angle)
    # Interior connection
    for z_idx in range(z_levels_thread):
        faces.append([notch_int_left[z_idx], interior_rings[z_idx][last_valid],
                     interior_rings[z_idx + 1][last_valid], notch_int_left[z_idx + 1]])
    # Exterior connection (upper)
    for z_idx in range(z_levels_thread):
        faces.append([exterior_rings[z_idx][last_valid], notch_ext_left[z_idx],
                     notch_ext_left[z_idx + 1], exterior_rings[z_idx + 1][last_valid]])
    # Exterior connection (base)
    for z_idx in range(z_levels_base):
        faces.append([exterior_rings[z_levels_thread + z_idx][last_valid], notch_ext_left_base[z_idx],
                     notch_ext_left_base[z_idx + 1], exterior_rings[z_levels_thread + z_idx + 1][last_valid]])
    # Radial wall (inner to outer) - threaded section
    for z_idx in range(z_levels_thread):
        faces.append([notch_int_left[z_idx], notch_ext_left[z_idx],
                     notch_ext_left[z_idx + 1], notch_int_left[z_idx + 1]])
    # Radial wall at floor level (triangle connecting threaded section to floor edge)
    faces.append([notch_int_left[-1], notch_ext_left[-1], notch_floor_left])
    # Radial wall (base section) - triangle fan from floor edge to exterior and bottom
    for z_idx in range(len(notch_ext_left_base) - 1):
        faces.append([notch_floor_left, notch_ext_left_base[z_idx], notch_ext_left_base[z_idx + 1]])
    # Final triangle connecting floor edge to exterior bottom to bottom edge
    faces.append([notch_floor_left, notch_ext_left_base[-1], notch_bottom_left])

    # RIGHT NOTCH WALL (at first_angle)
    # Interior connection
    for z_idx in range(z_levels_thread):
        faces.append([interior_rings[z_idx][first_valid], notch_int_right[z_idx],
                     notch_int_right[z_idx + 1], interior_rings[z_idx + 1][first_valid]])
    # Exterior connection (upper)
    for z_idx in range(z_levels_thread):
        faces.append([notch_ext_right[z_idx], exterior_rings[z_idx][first_valid],
                     exterior_rings[z_idx + 1][first_valid], notch_ext_right[z_idx + 1]])
    # Exterior connection (base)
    for z_idx in range(z_levels_base):
        faces.append([notch_ext_right_base[z_idx], exterior_rings[z_levels_thread + z_idx][first_valid],
                     exterior_rings[z_levels_thread + z_idx + 1][first_valid], notch_ext_right_base[z_idx + 1]])
    # Radial wall (inner to outer) - threaded section
    for z_idx in range(z_levels_thread):
        faces.append([notch_ext_right[z_idx], notch_int_right[z_idx],
                     notch_int_right[z_idx + 1], notch_ext_right[z_idx + 1]])
    # Radial wall at floor level (triangle connecting threaded section to floor edge)
    faces.append([notch_ext_right[-1], notch_int_right[-1], notch_floor_right])
    # Radial wall (base section) - triangle fan from floor edge to exterior and bottom
    for z_idx in range(len(notch_ext_right_base) - 1):
        faces.append([notch_ext_right_base[z_idx], notch_floor_right, notch_ext_right_base[z_idx + 1]])
    # Final triangle connecting floor edge to exterior bottom to bottom edge
    faces.append([notch_ext_right_base[-1], notch_floor_right, notch_bottom_right])

    return np.array(vertices), faces


def generate_pcb_mount():
    """Generate PCB mount with M2 screw bosses on the bottom."""
    vertices = []
    faces = []

    z_bottom = -BASE_HEIGHT
    z_boss_bottom = z_bottom - PCB_BOSS_HEIGHT

    half_grid = PCB_HOLE_GRID / 2
    boss_r = PCB_BOSS_DIAMETER / 2
    hole_r = PCB_HOLE_DIAMETER / 2

    def add_ring(cx, cy, radius, z):
        start = len(vertices)
        for i in range(BOSS_SEGMENTS):
            angle = 2 * np.pi * i / BOSS_SEGMENTS
            vertices.append([cx + radius * np.cos(angle), cy + radius * np.sin(angle), z])
        return start

    def connect_rings(ring1_start, ring2_start, reverse=False):
        for i in range(BOSS_SEGMENTS):
            i0 = ring1_start + i
            i1 = ring1_start + (i + 1) % BOSS_SEGMENTS
            j0 = ring2_start + i
            j1 = ring2_start + (i + 1) % BOSS_SEGMENTS

            if reverse:
                faces.append([i0, i1, j1, j0])
            else:
                faces.append([i0, j0, j1, i1])

    def connect_annulus(outer_start, inner_start, flip=False):
        for i in range(BOSS_SEGMENTS):
            o0 = outer_start + i
            o1 = outer_start + (i + 1) % BOSS_SEGMENTS
            i0 = inner_start + i
            i1 = inner_start + (i + 1) % BOSS_SEGMENTS

            if flip:
                faces.append([o0, o1, i1, i0])
            else:
                faces.append([o0, i0, i1, o1])

    # Screw bosses at corners of 16mm grid
    boss_positions = [
        (-half_grid, -half_grid),
        (half_grid, -half_grid),
        (half_grid, half_grid),
        (-half_grid, half_grid),
    ]

    for cx, cy in boss_positions:
        boss_outer_top = add_ring(cx, cy, boss_r, z_bottom)
        boss_outer_bottom = add_ring(cx, cy, boss_r, z_boss_bottom)
        boss_inner_top = add_ring(cx, cy, hole_r, z_bottom)
        boss_inner_bottom = add_ring(cx, cy, hole_r, z_boss_bottom)

        connect_rings(boss_outer_top, boss_outer_bottom)
        connect_rings(boss_inner_top, boss_inner_bottom, reverse=True)
        connect_annulus(boss_outer_top, boss_inner_top, flip=True)
        connect_annulus(boss_outer_bottom, boss_inner_bottom)

    return np.array(vertices), faces


def check_manifold(vertices, faces):
    """Check mesh for non-manifold edges and holes."""
    from collections import defaultdict
    edge_count = defaultdict(int)

    for face in faces:
        n = len(face)
        for i in range(n):
            v1, v2 = face[i], face[(i + 1) % n]
            edge = tuple(sorted([v1, v2]))
            edge_count[edge] += 1

    non_manifold = [(e, c) for e, c in edge_count.items() if c > 2]
    boundary = [e for e, c in edge_count.items() if c == 1]

    return len(non_manifold) == 0 and len(boundary) == 0, non_manifold, boundary


def write_combined_obj(filepath, objects):
    """Write multiple objects to a single OBJ file."""
    with open(filepath, 'w') as f:
        f.write(f"# Mount Base - {len(objects)} objects\n")

        vertex_offset = 0
        for name, verts, fcs in objects:
            f.write(f"o {name}\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in fcs:
                f.write("f " + " ".join(str(idx + 1 + vertex_offset) for idx in face) + "\n")
            vertex_offset += len(verts)

    print(f"Saved: {filepath}")


def write_combined_stl(filepath, objects):
    """Write multiple objects to a single STL file."""
    all_triangles = []

    for name, verts, fcs in objects:
        verts = np.array(verts)
        for face in fcs:
            if len(face) == 4:
                all_triangles.append((verts[face[0]], verts[face[1]], verts[face[2]]))
                all_triangles.append((verts[face[0]], verts[face[2]], verts[face[3]]))
            else:
                all_triangles.append((verts[face[0]], verts[face[1]], verts[face[2]]))

    with open(filepath, 'wb') as f:
        header = b'Binary STL - Mount Base' + b'\0' * 57
        f.write(header[:80])

        f.write(struct.pack('<I', len(all_triangles)))

        for v0, v1, v2 in all_triangles:
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm

            f.write(struct.pack('<3f', *normal))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))

    print(f"Saved: {filepath}")


def main():
    print("=" * 60)
    print("Generating Mount Base with Internal & External Threads")
    print("=" * 60)

    inner_r = BASE_INNER_DIAMETER / 2
    thread_inner_r = inner_r - BASE_THREAD_DEPTH
    outer_r = inner_r + BASE_WALL_THICKNESS
    ext_thread_r = outer_r + EXT_THREAD_DEPTH

    print(f"Mount base dimensions:")
    print(f"  Internal thread root diameter: {BASE_INNER_DIAMETER:.1f}mm")
    print(f"  Internal thread crest diameter: {thread_inner_r * 2:.1f}mm")
    print(f"  Wall outer diameter: {outer_r * 2:.1f}mm")
    print(f"  External thread crest diameter: {ext_thread_r * 2:.1f}mm")
    print(f"  Thread depth (cavity): {THREAD_HEIGHT:.1f}mm")
    print(f"  Floor thickness: {FLOOR_THICKNESS:.1f}mm")
    print(f"  Total height: {BASE_HEIGHT:.1f}mm")
    print(f"  PCB mount: M2 holes on {PCB_HOLE_GRID:.1f}mm grid")

    # Generate objects
    cylinder_verts, cylinder_faces = generate_cylinder()
    pcb_verts, pcb_faces = generate_pcb_mount()

    print(f"\nCylinder: {len(cylinder_verts)} vertices, {len(cylinder_faces)} faces")
    print(f"PCB mount: {len(pcb_verts)} vertices, {len(pcb_faces)} faces")

    # Check manifold
    print("\nChecking mesh integrity...")
    is_cyl_manifold, cyl_nm, cyl_boundary = check_manifold(cylinder_verts, cylinder_faces)
    if is_cyl_manifold:
        print("  Cylinder: watertight and manifold")
    else:
        print(f"  Cylinder: {len(cyl_nm)} non-manifold, {len(cyl_boundary)} boundary")

    is_pcb_manifold, pcb_nm, pcb_boundary = check_manifold(pcb_verts, pcb_faces)
    if is_pcb_manifold:
        print("  PCB mount: watertight and manifold")
    else:
        print(f"  PCB mount: {len(pcb_nm)} non-manifold, {len(pcb_boundary)} boundary")

    # Bounding box
    all_verts = np.vstack([cylinder_verts, pcb_verts])
    bbox_min = all_verts.min(axis=0)
    bbox_max = all_verts.max(axis=0)
    bbox_size = bbox_max - bbox_min
    print(f"\nBounding box: {bbox_size[0]:.1f} x {bbox_size[1]:.1f} x {bbox_size[2]:.1f} mm")

    # Output
    output_dir = Path("data/mounts")
    output_dir.mkdir(exist_ok=True)

    obj_path = output_dir / "mount_base.obj"
    stl_path = output_dir / "mount_base.stl"

    objects = [
        ("Cylinder", cylinder_verts, cylinder_faces),
        ("PCB_Mount", pcb_verts, pcb_faces),
    ]

    write_combined_obj(obj_path, objects)
    write_combined_stl(stl_path, objects)

    print(f"\n{'=' * 60}")
    print(f"Generated: {obj_path}")
    print(f"Generated: {stl_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
