#!/usr/bin/env python3
"""
Generate an outer sleeve that screws onto the mount base.

Features:
- Internal thread at top to fit mount base external thread
- Grip ridges on exterior
- Wire slit (rectangular hole at edge)
- Access hole in floor for easier dismantling
- Solid bottom cap
"""

import numpy as np
from pathlib import Path
import struct
import math

# Match mount_base external thread dimensions
MOUNT_BASE_WALL_OUTER = 36.8
MOUNT_BASE_EXT_THREAD_DEPTH = 1.0
MOUNT_BASE_HEIGHT = 15.0  # Lengthened by 3mm for extended external threads
PCB_BOSS_HEIGHT = 6.0  # Lengthened for M2*8 screws

# Sleeve parameters
THREAD_CLEARANCE = 0.5
SLEEVE_INNER_DIAMETER = MOUNT_BASE_WALL_OUTER + 2 * MOUNT_BASE_EXT_THREAD_DEPTH + THREAD_CLEARANCE
SLEEVE_THREAD_DEPTH = 1.0
SLEEVE_THREAD_PITCH = 2.0
SLEEVE_WALL_THICKNESS = 3.0
FLOOR_THICKNESS = 2.0

THREAD_REGION_HEIGHT = MOUNT_BASE_HEIGHT  # 12mm

# Floor position
FLOOR_Z = -MOUNT_BASE_HEIGHT - PCB_BOSS_HEIGHT - 10.0  # -28mm (5mm deeper than before)
SLEEVE_HEIGHT = -FLOOR_Z + FLOOR_THICKNESS  # 22mm

# Grip parameters
GRIP_RIDGES = 12
GRIP_RIDGE_DEPTH = 1.5
GRIP_START_Z = -2.0
GRIP_END_Z = -18.0

# Slit parameters (rectangular hole at edge)
SLIT_WIDTH = 12.0  # Width in mm
SLIT_TOP_Z = -12.0
SLIT_BOTTOM_Z = FLOOR_Z

# Access hole in floor
FLOOR_HOLE_DIAMETER = 10.0

SEGMENTS = 48
HOLE_SEGMENTS = 24


def helical_thread_profile(phase, depth, crest_fraction=0.25):
    """
    Trapezoidal thread profile — sharp raised ridge with wide groove.
    Must match the profile in generate_mount_base.py.
    """
    phase = phase % 1.0
    if phase < crest_fraction:
        return depth * (phase / crest_fraction)
    elif phase < 0.5:
        return depth
    elif phase < 0.5 + crest_fraction:
        return depth * (1.0 - (phase - 0.5) / crest_fraction)
    else:
        return 0.0


def generate_sleeve():
    """Generate sleeve with thread, grip, slit, and floor hole."""
    vertices = []
    faces = []

    inner_r = SLEEVE_INNER_DIAMETER / 2
    outer_r = inner_r + SLEEVE_WALL_THICKNESS
    hole_r = FLOOR_HOLE_DIAMETER / 2

    z_top = 0
    z_thread_bottom = -THREAD_REGION_HEIGHT
    z_floor = FLOOR_Z
    z_bottom = FLOOR_Z - FLOOR_THICKNESS

    z_levels_thread = max(int(THREAD_REGION_HEIGHT / SLEEVE_THREAD_PITCH * 16), 32)

    # Calculate how many segments the slit spans
    slit_half_angle = math.atan2(SLIT_WIDTH / 2, inner_r)
    slit_half_segs = max(1, int(math.ceil(slit_half_angle / (2 * math.pi / SEGMENTS))))

    # Slit centered at angle=0, spanning multiple segments
    slit_left_seg = SEGMENTS - slit_half_segs  # Left edge
    slit_right_seg = slit_half_segs  # Right edge (exclusive in loops)

    def in_slit_seg(seg):
        """Check if segment is within slit region."""
        return seg >= slit_left_seg or seg < slit_right_seg

    def grip_radius(angle, z):
        if z > GRIP_START_Z or z < GRIP_END_Z:
            return outer_r
        ridge_angle = 2 * math.pi / GRIP_RIDGES
        phase = (angle % ridge_angle) / ridge_angle
        ridge_height = GRIP_RIDGE_DEPTH * 0.5 * (1 + math.cos(2 * math.pi * phase))
        return outer_r + ridge_height

    # Collect all Z levels
    z_levels = [z_top]
    for z_idx in range(1, z_levels_thread + 1):
        z_levels.append(z_top - (z_idx / z_levels_thread) * THREAD_REGION_HEIGHT)
    z_levels.append(z_floor)
    z_levels.append(z_bottom)
    grip_z_steps = 8
    for i in range(1, grip_z_steps):
        z = GRIP_START_Z + (GRIP_END_Z - GRIP_START_Z) * i / grip_z_steps
        z_levels.append(z)
    z_levels.append(SLIT_TOP_Z)
    z_levels.append(SLIT_BOTTOM_Z)
    z_levels = sorted(set(z_levels), reverse=True)

    slit_top_idx = z_levels.index(SLIT_TOP_Z)
    slit_bottom_idx = z_levels.index(SLIT_BOTTOM_Z)

    def in_slit_z(z_idx):
        return slit_top_idx <= z_idx <= slit_bottom_idx

    # Create interior rings (full circles) — helical screw threads
    interior_rings = []
    for z in z_levels:
        if z < z_floor:
            interior_rings.append(None)
            continue

        ring = []
        for seg in range(SEGMENTS):
            angle = 2 * math.pi * seg / SEGMENTS
            if z >= z_thread_bottom:
                t = (z_top - z) / THREAD_REGION_HEIGHT if THREAD_REGION_HEIGHT > 0 else 0
                # Helical thread: phase depends on both Z and angle
                thread_phase = (t * THREAD_REGION_HEIGHT / SLEEVE_THREAD_PITCH + angle / (2 * math.pi)) % 1.0
                thread_h = helical_thread_profile(thread_phase, SLEEVE_THREAD_DEPTH)
                r = inner_r - thread_h
            else:
                r = inner_r
            ring.append(len(vertices))
            vertices.append([r * math.cos(angle), r * math.sin(angle), z])
        interior_rings.append(ring)

    # Create exterior rings (full circles)
    exterior_rings = []
    for z in z_levels:
        ring = []
        for seg in range(SEGMENTS):
            angle = 2 * math.pi * seg / SEGMENTS
            r = grip_radius(angle, z)
            ring.append(len(vertices))
            vertices.append([r * math.cos(angle), r * math.sin(angle), z])
        exterior_rings.append(ring)

    # Floor ring (at inner_r)
    floor_ring = []
    for seg in range(SEGMENTS):
        angle = 2 * math.pi * seg / SEGMENTS
        floor_ring.append(len(vertices))
        vertices.append([inner_r * math.cos(angle), inner_r * math.sin(angle), z_floor])

    # Floor hole ring
    floor_hole_ring = []
    for seg in range(HOLE_SEGMENTS):
        angle = 2 * math.pi * seg / HOLE_SEGMENTS
        floor_hole_ring.append(len(vertices))
        vertices.append([hole_r * math.cos(angle), hole_r * math.sin(angle), z_floor])

    # Bottom ring (at outer_r)
    bottom_ring = []
    for seg in range(SEGMENTS):
        angle = 2 * math.pi * seg / SEGMENTS
        bottom_ring.append(len(vertices))
        vertices.append([outer_r * math.cos(angle), outer_r * math.sin(angle), z_bottom])

    # Bottom hole ring
    bottom_hole_ring = []
    for seg in range(HOLE_SEGMENTS):
        angle = 2 * math.pi * seg / HOLE_SEGMENTS
        bottom_hole_ring.append(len(vertices))
        vertices.append([hole_r * math.cos(angle), hole_r * math.sin(angle), z_bottom])

    # ===== BUILD FACES =====

    # Interior surface
    for i in range(len(z_levels) - 1):
        if interior_rings[i] is None or interior_rings[i + 1] is None:
            continue
        for seg in range(SEGMENTS):
            seg_next = (seg + 1) % SEGMENTS
            # Skip slit segments in slit z region
            if in_slit_seg(seg) and in_slit_z(i) and in_slit_z(i + 1):
                continue
            faces.append([
                interior_rings[i][seg],
                interior_rings[i][seg_next],
                interior_rings[i + 1][seg_next],
                interior_rings[i + 1][seg]
            ])

    # Connect interior to floor
    last_int_idx = None
    for i, ring in enumerate(interior_rings):
        if ring is not None:
            last_int_idx = i
    if last_int_idx is not None:
        for seg in range(SEGMENTS):
            seg_next = (seg + 1) % SEGMENTS
            faces.append([
                interior_rings[last_int_idx][seg],
                interior_rings[last_int_idx][seg_next],
                floor_ring[seg_next],
                floor_ring[seg]
            ])

    # Floor annulus (from floor_ring to floor_hole_ring)
    # Use triangles to connect different segment counts
    for seg in range(SEGMENTS):
        seg_next = (seg + 1) % SEGMENTS
        # Map to hole segments
        hole_seg = int(seg * HOLE_SEGMENTS / SEGMENTS)
        hole_seg_next = int(seg_next * HOLE_SEGMENTS / SEGMENTS) if seg_next > 0 else 0
        if hole_seg_next == 0 and seg_next == 0:
            hole_seg_next = 0

        # Create triangles
        faces.append([floor_ring[seg], floor_ring[seg_next], floor_hole_ring[hole_seg]])
        if hole_seg != hole_seg_next:
            faces.append([floor_ring[seg_next], floor_hole_ring[hole_seg_next], floor_hole_ring[hole_seg]])

    # Exterior surface
    for i in range(len(z_levels) - 1):
        for seg in range(SEGMENTS):
            seg_next = (seg + 1) % SEGMENTS
            if in_slit_seg(seg) and in_slit_z(i) and in_slit_z(i + 1):
                continue
            faces.append([
                exterior_rings[i][seg],
                exterior_rings[i + 1][seg],
                exterior_rings[i + 1][seg_next],
                exterior_rings[i][seg_next]
            ])

    # Connect exterior to bottom
    for seg in range(SEGMENTS):
        seg_next = (seg + 1) % SEGMENTS
        faces.append([
            exterior_rings[-1][seg],
            exterior_rings[-1][seg_next],
            bottom_ring[seg_next],
            bottom_ring[seg]
        ])

    # Bottom annulus (from bottom_ring to bottom_hole_ring)
    for seg in range(SEGMENTS):
        seg_next = (seg + 1) % SEGMENTS
        hole_seg = int(seg * HOLE_SEGMENTS / SEGMENTS)
        hole_seg_next = int(seg_next * HOLE_SEGMENTS / SEGMENTS) if seg_next > 0 else 0

        faces.append([bottom_ring[seg_next], bottom_ring[seg], bottom_hole_ring[hole_seg]])
        if hole_seg != hole_seg_next:
            faces.append([bottom_hole_ring[hole_seg], bottom_hole_ring[hole_seg_next], bottom_ring[seg_next]])

    # Hole inner wall (connects floor hole to bottom hole)
    for seg in range(HOLE_SEGMENTS):
        seg_next = (seg + 1) % HOLE_SEGMENTS
        faces.append([
            floor_hole_ring[seg],
            floor_hole_ring[seg_next],
            bottom_hole_ring[seg_next],
            bottom_hole_ring[seg]
        ])

    # Top annulus
    for seg in range(SEGMENTS):
        seg_next = (seg + 1) % SEGMENTS
        faces.append([
            exterior_rings[0][seg],
            interior_rings[0][seg],
            interior_rings[0][seg_next],
            exterior_rings[0][seg_next]
        ])

    # ===== SLIT FACES =====
    # Slit top face (closes top of the rectangular hole) - may span multiple segments
    for seg in range(slit_left_seg, SEGMENTS):
        faces.append([
            interior_rings[slit_top_idx][seg],
            exterior_rings[slit_top_idx][seg],
            exterior_rings[slit_top_idx][(seg + 1) % SEGMENTS],
            interior_rings[slit_top_idx][(seg + 1) % SEGMENTS]
        ])
    for seg in range(0, slit_right_seg):
        faces.append([
            interior_rings[slit_top_idx][seg],
            exterior_rings[slit_top_idx][seg],
            exterior_rings[slit_top_idx][seg + 1],
            interior_rings[slit_top_idx][seg + 1]
        ])

    # Slit bottom face (at floor level)
    for seg in range(slit_left_seg, SEGMENTS):
        faces.append([
            exterior_rings[slit_bottom_idx][seg],
            interior_rings[slit_bottom_idx][seg],
            interior_rings[slit_bottom_idx][(seg + 1) % SEGMENTS],
            exterior_rings[slit_bottom_idx][(seg + 1) % SEGMENTS]
        ])
    for seg in range(0, slit_right_seg):
        faces.append([
            exterior_rings[slit_bottom_idx][seg],
            interior_rings[slit_bottom_idx][seg],
            interior_rings[slit_bottom_idx][seg + 1],
            exterior_rings[slit_bottom_idx][seg + 1]
        ])

    # Slit left wall (interior to exterior at left edge)
    for i in range(slit_top_idx, slit_bottom_idx):
        faces.append([
            exterior_rings[i][slit_left_seg],
            interior_rings[i][slit_left_seg],
            interior_rings[i + 1][slit_left_seg],
            exterior_rings[i + 1][slit_left_seg]
        ])

    # Slit right wall (exterior to interior at right edge)
    for i in range(slit_top_idx, slit_bottom_idx):
        faces.append([
            interior_rings[i][slit_right_seg],
            exterior_rings[i][slit_right_seg],
            exterior_rings[i + 1][slit_right_seg],
            interior_rings[i + 1][slit_right_seg]
        ])

    return np.array(vertices), faces


def check_manifold(vertices, faces):
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


def write_obj(filepath, vertices, faces, name="OuterSleeve"):
    with open(filepath, 'w') as f:
        f.write(f"# Outer Sleeve\no {name}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")
    print(f"Saved: {filepath}")


def write_stl(filepath, vertices, faces):
    vertices = np.array(vertices)
    triangles = []
    for face in faces:
        if len(face) == 4:
            triangles.append([face[0], face[1], face[2]])
            triangles.append([face[0], face[2], face[3]])
        else:
            triangles.append(face)
    with open(filepath, 'wb') as f:
        f.write((b'Binary STL - Outer Sleeve' + b'\0' * 55)[:80])
        f.write(struct.pack('<I', len(triangles)))
        for tri in triangles:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            edge1, edge2 = v1 - v0, v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            f.write(struct.pack('<3f', *normal))
            for v in [v0, v1, v2]:
                f.write(struct.pack('<3f', *v))
            f.write(struct.pack('<H', 0))
    print(f"Saved: {filepath}")


def main():
    print("=" * 60)
    print("Generating Outer Sleeve with Grip, Slit, and Floor Hole")
    print("=" * 60)

    inner_r = SLEEVE_INNER_DIAMETER / 2
    outer_r = inner_r + SLEEVE_WALL_THICKNESS

    print(f"Dimensions:")
    print(f"  Inner thread root: {SLEEVE_INNER_DIAMETER:.1f}mm")
    print(f"  Outer diameter: {outer_r * 2:.1f}mm")
    print(f"  Grip outer diameter: {(outer_r + GRIP_RIDGE_DEPTH) * 2:.1f}mm")
    print(f"  Total height: {SLEEVE_HEIGHT:.1f}mm")
    print(f"  Floor at z={FLOOR_Z:.1f}mm")
    print(f"  Grip: {GRIP_RIDGES} ridges, {GRIP_RIDGE_DEPTH:.1f}mm deep")
    print(f"  Slit: z={SLIT_TOP_Z} to {SLIT_BOTTOM_Z}")
    print(f"  Floor hole: {FLOOR_HOLE_DIAMETER:.1f}mm diameter")

    verts, faces = generate_sleeve()
    print(f"\nMesh: {len(verts)} vertices, {len(faces)} faces")

    is_ok, nm, bd = check_manifold(verts, faces)
    if is_ok:
        print("Mesh is watertight and manifold")
    else:
        print(f"WARNING: {len(nm)} non-manifold, {len(bd)} boundary edges")
        if bd:
            print("Boundary edges:")
            for e in bd[:10]:
                v1, v2 = verts[e[0]], verts[e[1]]
                print(f"  ({v1[0]:.1f}, {v1[1]:.1f}, {v1[2]:.1f}) - ({v2[0]:.1f}, {v2[1]:.1f}, {v2[2]:.1f})")

    bbox = verts.max(axis=0) - verts.min(axis=0)
    print(f"Bounding box: {bbox[0]:.1f} x {bbox[1]:.1f} x {bbox[2]:.1f} mm")

    output_dir = Path("data/mounts")
    output_dir.mkdir(exist_ok=True)
    write_obj(output_dir / "outer_sleeve.obj", verts, faces)
    write_stl(output_dir / "outer_sleeve.stl", verts, faces)


if __name__ == "__main__":
    main()
