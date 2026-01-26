#!/usr/bin/env python3
"""
Generate an outer sleeve that screws onto the mount base.

Features:
- Internal thread at top to fit mount base external thread
- Simple solid walls (NO notch - notch is in inner pieces)
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
# Mount base is 12mm, PCB bosses add 3mm, PCB + clearance add ~3mm more
FLOOR_Z = -MOUNT_BASE_HEIGHT - PCB_BOSS_HEIGHT - 5.0  # -20mm
SLEEVE_HEIGHT = -FLOOR_Z + FLOOR_THICKNESS  # 22mm

SEGMENTS = 48


def generate_sleeve():
    """Generate simple sleeve - cylinder with internal thread, no notch."""
    vertices = []
    faces = []

    inner_r = SLEEVE_INNER_DIAMETER / 2
    outer_r = inner_r + SLEEVE_WALL_THICKNESS

    z_top = 0
    z_thread_bottom = -THREAD_REGION_HEIGHT
    z_floor = FLOOR_Z
    z_bottom = FLOOR_Z - FLOOR_THICKNESS

    z_levels_thread = max(int(THREAD_REGION_HEIGHT / SLEEVE_THREAD_PITCH * 16), 32)

    # ===== INTERIOR THREADED (z_top to z_thread_bottom) =====
    int_thread_rings = []
    for z_idx in range(z_levels_thread + 1):
        z = z_top - (z_idx / z_levels_thread) * THREAD_REGION_HEIGHT
        thread_phase = (z_idx / z_levels_thread * THREAD_REGION_HEIGHT / SLEEVE_THREAD_PITCH) % 1.0
        thread_h = SLEEVE_THREAD_DEPTH * (1 - abs(2 * thread_phase - 1))
        r = inner_r - thread_h

        ring = []
        for seg in range(SEGMENTS):
            angle = 2 * math.pi * seg / SEGMENTS
            ring.append(len(vertices))
            vertices.append([r * math.cos(angle), r * math.sin(angle), z])
        int_thread_rings.append(ring)

    for z_idx in range(z_levels_thread):
        for seg in range(SEGMENTS):
            seg_next = (seg + 1) % SEGMENTS
            faces.append([
                int_thread_rings[z_idx][seg],
                int_thread_rings[z_idx][seg_next],
                int_thread_rings[z_idx + 1][seg_next],
                int_thread_rings[z_idx + 1][seg]
            ])

    # ===== INTERIOR PLAIN (z_thread_bottom to z_floor) =====
    int_plain_top = []
    int_plain_floor = []
    for seg in range(SEGMENTS):
        angle = 2 * math.pi * seg / SEGMENTS
        int_plain_top.append(len(vertices))
        vertices.append([inner_r * math.cos(angle), inner_r * math.sin(angle), z_thread_bottom])
        int_plain_floor.append(len(vertices))
        vertices.append([inner_r * math.cos(angle), inner_r * math.sin(angle), z_floor])

    # Connect thread bottom to plain top
    for seg in range(SEGMENTS):
        seg_next = (seg + 1) % SEGMENTS
        faces.append([
            int_thread_rings[-1][seg],
            int_thread_rings[-1][seg_next],
            int_plain_top[seg_next],
            int_plain_top[seg]
        ])

    # Connect plain top to floor
    for seg in range(SEGMENTS):
        seg_next = (seg + 1) % SEGMENTS
        faces.append([
            int_plain_top[seg],
            int_plain_top[seg_next],
            int_plain_floor[seg_next],
            int_plain_floor[seg]
        ])

    # Floor disk
    floor_center = len(vertices)
    vertices.append([0, 0, z_floor])
    for seg in range(SEGMENTS):
        seg_next = (seg + 1) % SEGMENTS
        faces.append([floor_center, int_plain_floor[seg], int_plain_floor[seg_next]])

    # ===== EXTERIOR (simple cylinder, z_top to z_bottom) =====
    ext_top = []
    ext_floor = []
    ext_bottom = []
    for seg in range(SEGMENTS):
        angle = 2 * math.pi * seg / SEGMENTS
        ext_top.append(len(vertices))
        vertices.append([outer_r * math.cos(angle), outer_r * math.sin(angle), z_top])
        ext_floor.append(len(vertices))
        vertices.append([outer_r * math.cos(angle), outer_r * math.sin(angle), z_floor])
        ext_bottom.append(len(vertices))
        vertices.append([outer_r * math.cos(angle), outer_r * math.sin(angle), z_bottom])

    # Connect exterior top to floor
    for seg in range(SEGMENTS):
        seg_next = (seg + 1) % SEGMENTS
        faces.append([ext_top[seg], ext_floor[seg], ext_floor[seg_next], ext_top[seg_next]])

    # Connect exterior floor to bottom
    for seg in range(SEGMENTS):
        seg_next = (seg + 1) % SEGMENTS
        faces.append([ext_floor[seg], ext_bottom[seg], ext_bottom[seg_next], ext_floor[seg_next]])

    # ===== TOP ANNULUS =====
    for seg in range(SEGMENTS):
        seg_next = (seg + 1) % SEGMENTS
        faces.append([
            ext_top[seg],
            int_thread_rings[0][seg],
            int_thread_rings[0][seg_next],
            ext_top[seg_next]
        ])

    # ===== BOTTOM CAP =====
    bot_center = len(vertices)
    vertices.append([0, 0, z_bottom])
    for seg in range(SEGMENTS):
        seg_next = (seg + 1) % SEGMENTS
        faces.append([bot_center, ext_bottom[seg_next], ext_bottom[seg]])

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
    print("Generating Outer Sleeve (Simple Cylinder)")
    print("=" * 60)

    inner_r = SLEEVE_INNER_DIAMETER / 2
    outer_r = inner_r + SLEEVE_WALL_THICKNESS

    print(f"Dimensions:")
    print(f"  Inner thread root: {SLEEVE_INNER_DIAMETER:.1f}mm")
    print(f"  Outer diameter: {outer_r * 2:.1f}mm")
    print(f"  Total height: {SLEEVE_HEIGHT:.1f}mm")
    print(f"  Floor at z={FLOOR_Z:.1f}mm")
    print(f"  (No notch - notch is in inner pieces)")

    verts, faces = generate_sleeve()
    print(f"\nMesh: {len(verts)} vertices, {len(faces)} faces")

    is_ok, nm, bd = check_manifold(verts, faces)
    if is_ok:
        print("Mesh is watertight and manifold")
    else:
        print(f"WARNING: {len(nm)} non-manifold, {len(bd)} boundary edges")

    bbox = verts.max(axis=0) - verts.min(axis=0)
    print(f"Bounding box: {bbox[0]:.1f} x {bbox[1]:.1f} x {bbox[2]:.1f} mm")

    output_dir = Path("data/mounts")
    output_dir.mkdir(exist_ok=True)
    write_obj(output_dir / "outer_sleeve.obj", verts, faces)
    write_stl(output_dir / "outer_sleeve.stl", verts, faces)


if __name__ == "__main__":
    main()
