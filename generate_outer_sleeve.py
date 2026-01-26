#!/usr/bin/env python3
"""
Generate an outer sleeve that screws onto the mount base.

Features:
- Internal thread at top to fit mount base external thread
- Grip ridges on exterior
- Wire slit through the wall (aligned with inner pieces)
- Solid floor at bottom
"""

import numpy as np
from pathlib import Path
import struct
import math

# Match mount_base external thread dimensions
MOUNT_BASE_WALL_OUTER = 36.8
MOUNT_BASE_EXT_THREAD_DEPTH = 1.0
MOUNT_BASE_HEIGHT = 12.0
PCB_BOSS_HEIGHT = 3.0

# Sleeve parameters
THREAD_CLEARANCE = 0.3
SLEEVE_INNER_DIAMETER = MOUNT_BASE_WALL_OUTER + 2 * MOUNT_BASE_EXT_THREAD_DEPTH + THREAD_CLEARANCE
SLEEVE_THREAD_DEPTH = 1.0
SLEEVE_THREAD_PITCH = 2.0
SLEEVE_WALL_THICKNESS = 3.0
FLOOR_THICKNESS = 2.0

THREAD_REGION_HEIGHT = MOUNT_BASE_HEIGHT  # 12mm

# Floor position - must be below where PCB sits
FLOOR_Z = -MOUNT_BASE_HEIGHT - PCB_BOSS_HEIGHT - 5.0  # -20mm
SLEEVE_HEIGHT = -FLOOR_Z + FLOOR_THICKNESS  # 22mm

# Grip parameters
GRIP_RIDGES = 12
GRIP_RIDGE_DEPTH = 1.5
GRIP_START_Z = -2.0
GRIP_END_Z = -18.0

# Slit parameters
SLIT_WIDTH = 5.0

SEGMENTS = 48


def generate_sleeve():
    """Generate sleeve with internal thread, grip ridges, and wire slit."""
    vertices = []
    faces = []

    inner_r = SLEEVE_INNER_DIAMETER / 2
    outer_r = inner_r + SLEEVE_WALL_THICKNESS

    z_top = 0
    z_thread_bottom = -THREAD_REGION_HEIGHT
    z_floor = FLOOR_Z
    z_bottom = FLOOR_Z - FLOOR_THICKNESS

    z_levels_thread = max(int(THREAD_REGION_HEIGHT / SLEEVE_THREAD_PITCH * 16), 32)

    # Slit setup
    slit_half_angle = math.atan2(SLIT_WIDTH / 2, inner_r)
    slit_segs = max(2, int(slit_half_angle * 2 * SEGMENTS / (2 * math.pi)) + 1)
    slit_start = SEGMENTS - slit_segs // 2
    slit_end = (slit_segs + 1) // 2

    def in_slit(seg):
        return seg >= slit_start or seg < slit_end

    valid_segs = [i for i in range(SEGMENTS) if not in_slit(i)]
    first_valid = valid_segs[0]
    last_valid = valid_segs[-1]
    first_angle = 2 * math.pi * first_valid / SEGMENTS
    last_angle = 2 * math.pi * last_valid / SEGMENTS

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
    z_levels.append(z_thread_bottom)
    z_levels.append(z_floor)
    z_levels.append(z_bottom)
    # Add grip intermediate levels
    grip_z_steps = 8
    for i in range(1, grip_z_steps):
        z = GRIP_START_Z + (GRIP_END_Z - GRIP_START_Z) * i / grip_z_steps
        z_levels.append(z)
    z_levels = sorted(set(z_levels), reverse=True)

    # Create interior rings (only in threaded and plain regions)
    interior_rings = []
    for z in z_levels:
        if z < z_floor:
            interior_rings.append(None)
            continue
        if z >= z_thread_bottom:
            # Threaded region
            t = (z_top - z) / THREAD_REGION_HEIGHT if THREAD_REGION_HEIGHT > 0 else 0
            thread_phase = (t * THREAD_REGION_HEIGHT / SLEEVE_THREAD_PITCH) % 1.0
            thread_h = SLEEVE_THREAD_DEPTH * (1 - abs(2 * thread_phase - 1))
            r = inner_r - thread_h
        else:
            # Plain region
            r = inner_r

        ring = {}
        for seg in valid_segs:
            angle = 2 * math.pi * seg / SEGMENTS
            ring[seg] = len(vertices)
            vertices.append([r * math.cos(angle), r * math.sin(angle), z])
        interior_rings.append(ring)

    # Create exterior rings
    exterior_rings = []
    for z in z_levels:
        ring = {}
        for seg in valid_segs:
            angle = 2 * math.pi * seg / SEGMENTS
            r = grip_radius(angle, z)
            ring[seg] = len(vertices)
            vertices.append([r * math.cos(angle), r * math.sin(angle), z])
        exterior_rings.append(ring)

    # Floor and bottom rings at inner_r and outer_r
    z_floor_idx = z_levels.index(z_floor)
    z_bottom_idx = z_levels.index(z_bottom)

    floor_ring = {}
    for seg in valid_segs:
        angle = 2 * math.pi * seg / SEGMENTS
        floor_ring[seg] = len(vertices)
        vertices.append([inner_r * math.cos(angle), inner_r * math.sin(angle), z_floor])

    bottom_ring = {}
    for seg in valid_segs:
        angle = 2 * math.pi * seg / SEGMENTS
        bottom_ring[seg] = len(vertices)
        vertices.append([outer_r * math.cos(angle), outer_r * math.sin(angle), z_bottom])

    floor_center = len(vertices)
    vertices.append([0, 0, z_floor])
    bottom_center = len(vertices)
    vertices.append([0, 0, z_bottom])

    # Slit edge vertices
    slit_floor_left = len(vertices)
    vertices.append([inner_r * math.cos(last_angle), inner_r * math.sin(last_angle), z_floor])
    slit_floor_right = len(vertices)
    vertices.append([inner_r * math.cos(first_angle), inner_r * math.sin(first_angle), z_floor])
    slit_bottom_left = len(vertices)
    vertices.append([outer_r * math.cos(last_angle), outer_r * math.sin(last_angle), z_bottom])
    slit_bottom_right = len(vertices)
    vertices.append([outer_r * math.cos(first_angle), outer_r * math.sin(first_angle), z_bottom])

    # Slit wall vertices at each z level
    slit_int_left = []
    slit_ext_left = []
    slit_int_right = []
    slit_ext_right = []

    for i, z in enumerate(z_levels):
        if z >= z_floor:
            # Interior exists
            if z >= z_thread_bottom:
                t = (z_top - z) / THREAD_REGION_HEIGHT if THREAD_REGION_HEIGHT > 0 else 0
                thread_phase = (t * THREAD_REGION_HEIGHT / SLEEVE_THREAD_PITCH) % 1.0
                thread_h = SLEEVE_THREAD_DEPTH * (1 - abs(2 * thread_phase - 1))
                int_r = inner_r - thread_h
            else:
                int_r = inner_r

            slit_int_left.append(len(vertices))
            vertices.append([int_r * math.cos(last_angle), int_r * math.sin(last_angle), z])
            slit_int_right.append(len(vertices))
            vertices.append([int_r * math.cos(first_angle), int_r * math.sin(first_angle), z])
        else:
            slit_int_left.append(slit_floor_left)
            slit_int_right.append(slit_floor_right)

        ext_r_left = grip_radius(last_angle, z)
        ext_r_right = grip_radius(first_angle, z)

        if z == z_bottom:
            slit_ext_left.append(slit_bottom_left)
            slit_ext_right.append(slit_bottom_right)
        else:
            slit_ext_left.append(len(vertices))
            vertices.append([ext_r_left * math.cos(last_angle), ext_r_left * math.sin(last_angle), z])
            slit_ext_right.append(len(vertices))
            vertices.append([ext_r_right * math.cos(first_angle), ext_r_right * math.sin(first_angle), z])

    # ===== BUILD FACES =====

    # Interior surface
    for i in range(len(z_levels) - 1):
        if interior_rings[i] is None or interior_rings[i + 1] is None:
            continue
        for j in range(len(valid_segs) - 1):
            seg, seg_next = valid_segs[j], valid_segs[j + 1]
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
        for j in range(len(valid_segs) - 1):
            seg, seg_next = valid_segs[j], valid_segs[j + 1]
            faces.append([
                interior_rings[last_int_idx][seg],
                interior_rings[last_int_idx][seg_next],
                floor_ring[seg_next],
                floor_ring[seg]
            ])

    # Floor disk
    for j in range(len(valid_segs) - 1):
        seg, seg_next = valid_segs[j], valid_segs[j + 1]
        faces.append([floor_center, floor_ring[seg], floor_ring[seg_next]])
    faces.append([floor_center, slit_floor_left, floor_ring[last_valid]])
    faces.append([floor_center, floor_ring[first_valid], slit_floor_right])

    # Exterior surface
    for i in range(len(z_levels) - 1):
        for j in range(len(valid_segs) - 1):
            seg, seg_next = valid_segs[j], valid_segs[j + 1]
            faces.append([
                exterior_rings[i][seg],
                exterior_rings[i + 1][seg],
                exterior_rings[i + 1][seg_next],
                exterior_rings[i][seg_next]
            ])

    # Connect exterior to bottom
    ext_floor_idx = z_floor_idx
    for j in range(len(valid_segs) - 1):
        seg, seg_next = valid_segs[j], valid_segs[j + 1]
        faces.append([
            exterior_rings[-1][seg],
            exterior_rings[-1][seg_next],
            bottom_ring[seg_next],
            bottom_ring[seg]
        ])

    # Bottom disk
    for j in range(len(valid_segs) - 1):
        seg, seg_next = valid_segs[j], valid_segs[j + 1]
        faces.append([bottom_center, bottom_ring[seg_next], bottom_ring[seg]])
    faces.append([bottom_center, bottom_ring[last_valid], slit_bottom_left])
    faces.append([bottom_center, slit_bottom_right, bottom_ring[first_valid]])

    # Top annulus
    for j in range(len(valid_segs) - 1):
        seg, seg_next = valid_segs[j], valid_segs[j + 1]
        faces.append([
            exterior_rings[0][seg],
            interior_rings[0][seg],
            interior_rings[0][seg_next],
            exterior_rings[0][seg_next]
        ])
    # Top annulus slit closures
    faces.append([exterior_rings[0][last_valid], interior_rings[0][last_valid],
                 slit_int_left[0], slit_ext_left[0]])
    faces.append([slit_ext_right[0], slit_int_right[0],
                 interior_rings[0][first_valid], exterior_rings[0][first_valid]])

    # LEFT SLIT WALL
    # Interior edge connection
    for i in range(len(z_levels) - 1):
        if interior_rings[i] is None or interior_rings[i + 1] is None:
            # Below floor - no interior
            if z_levels[i] == z_floor:
                # Connect floor to bottom via slit wall
                pass
            continue
        faces.append([slit_int_left[i], interior_rings[i][last_valid],
                     interior_rings[i + 1][last_valid], slit_int_left[i + 1]])

    # Connect last interior to floor
    if last_int_idx is not None:
        faces.append([slit_int_left[last_int_idx], interior_rings[last_int_idx][last_valid],
                     floor_ring[last_valid], slit_floor_left])

    # Exterior edge connection
    for i in range(len(z_levels) - 1):
        faces.append([exterior_rings[i][last_valid], slit_ext_left[i],
                     slit_ext_left[i + 1], exterior_rings[i + 1][last_valid]])

    # Connect exterior to bottom ring (slit_ext_left[-1] == slit_bottom_left, so use triangle)
    faces.append([exterior_rings[-1][last_valid], slit_bottom_left, bottom_ring[last_valid]])

    # Radial wall (interior to exterior)
    for i in range(len(z_levels) - 1):
        if z_levels[i + 1] < z_floor:
            # Below floor - interior edge is at floor
            if z_levels[i] >= z_floor:
                faces.append([slit_int_left[i], slit_ext_left[i], slit_ext_left[i + 1], slit_floor_left])
            else:
                faces.append([slit_floor_left, slit_ext_left[i], slit_ext_left[i + 1]])
        else:
            faces.append([slit_int_left[i], slit_ext_left[i], slit_ext_left[i + 1], slit_int_left[i + 1]])

    # Inner wall (floor center to bottom center through slit)
    faces.append([floor_center, slit_floor_left, slit_bottom_left, bottom_center])

    # RIGHT SLIT WALL (opposite winding)
    # Interior edge connection
    for i in range(len(z_levels) - 1):
        if interior_rings[i] is None or interior_rings[i + 1] is None:
            continue
        faces.append([interior_rings[i][first_valid], slit_int_right[i],
                     slit_int_right[i + 1], interior_rings[i + 1][first_valid]])

    # Connect last interior to floor
    if last_int_idx is not None:
        faces.append([interior_rings[last_int_idx][first_valid], slit_int_right[last_int_idx],
                     slit_floor_right, floor_ring[first_valid]])

    # Exterior edge connection
    for i in range(len(z_levels) - 1):
        faces.append([slit_ext_right[i], exterior_rings[i][first_valid],
                     exterior_rings[i + 1][first_valid], slit_ext_right[i + 1]])

    # Connect exterior to bottom ring (slit_ext_right[-1] == slit_bottom_right, so use triangle)
    faces.append([slit_bottom_right, exterior_rings[-1][first_valid], bottom_ring[first_valid]])

    # Radial wall (exterior to interior)
    for i in range(len(z_levels) - 1):
        if z_levels[i + 1] < z_floor:
            if z_levels[i] >= z_floor:
                faces.append([slit_ext_right[i], slit_int_right[i], slit_floor_right, slit_ext_right[i + 1]])
            else:
                faces.append([slit_ext_right[i], slit_floor_right, slit_ext_right[i + 1]])
        else:
            faces.append([slit_ext_right[i], slit_int_right[i], slit_int_right[i + 1], slit_ext_right[i + 1]])

    # Inner wall (floor center to bottom center through slit)
    faces.append([floor_center, bottom_center, slit_bottom_right, slit_floor_right])

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
    print("Generating Outer Sleeve with Grip and Slit")
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
    print(f"  Slit width: {SLIT_WIDTH:.1f}mm")

    verts, faces = generate_sleeve()
    print(f"\nMesh: {len(verts)} vertices, {len(faces)} faces")

    is_ok, nm, bd = check_manifold(verts, faces)
    if is_ok:
        print("Mesh is watertight and manifold")
    else:
        print(f"WARNING: {len(nm)} non-manifold, {len(bd)} boundary edges")
        if bd:
            print("Boundary edges:")
            for e in bd[:5]:
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
