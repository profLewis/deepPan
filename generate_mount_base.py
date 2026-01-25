#!/usr/bin/env python3
"""
Generate a screw-on mount base that attaches to the notepad cylinder.

Features:
- Internal threads matching the notepad's external threads
- Smooth exterior with grip ridges for hand turning
- Base plate with central platform
- Watertight manifold mesh (two separate manifold objects that intersect)
"""

import numpy as np
from pathlib import Path
import struct
from collections import defaultdict

# Import matching parameters from notepad generator
NOTEPAD_INNER_DIAMETER = 23.5
NOTEPAD_WALL_THICKNESS = 2.5
NOTEPAD_THREAD_DEPTH = 1.0
NOTEPAD_THREAD_PITCH = 2.0

# Derived notepad dimensions
NOTEPAD_WALL_OUTER = NOTEPAD_INNER_DIAMETER + 2 * NOTEPAD_WALL_THICKNESS
NOTEPAD_THREAD_OUTER = NOTEPAD_WALL_OUTER + 2 * NOTEPAD_THREAD_DEPTH

# Mount base parameters
THREAD_CLEARANCE = 0.3
BASE_INNER_DIAMETER = NOTEPAD_THREAD_OUTER + THREAD_CLEARANCE
BASE_THREAD_DEPTH = 1.0
BASE_WALL_THICKNESS = 3.0
BASE_THREAD_HEIGHT = 9.0

# Base plate parameters
BASE_PLATE_THICKNESS = 3.0
PLATFORM_DIAMETER = 12.0
PLATFORM_RECESS = 7.0

# Grip ridge parameters
GRIP_RIDGE_COUNT = 6
GRIP_RIDGE_DEPTH = 1.5

# PCB mount parameters
PCB_SIZE = 20.0             # Square PCB size
PCB_HOLE_GRID = 16.0        # M2 holes on 16mm grid
PCB_HOLE_DIAMETER = 2.2     # M2 clearance hole
PCB_BOSS_DIAMETER = 5.0     # Boss around screw hole
PCB_BOSS_HEIGHT = 3.0       # Height of screw bosses
PCB_MOUNT_DEPTH = 2.0       # Recess depth for PCB

# Resolution
SEGMENTS = 48
BOSS_SEGMENTS = 16          # Resolution for screw bosses


def check_manifold(vertices, faces):
    """Check mesh for non-manifold edges and holes."""
    edge_count = defaultdict(int)

    for face in faces:
        n = len(face)
        for i in range(n):
            v1, v2 = face[i], face[(i + 1) % n]
            edge = tuple(sorted([v1, v2]))
            edge_count[edge] += 1

    non_manifold = []
    boundary = []

    for edge, count in edge_count.items():
        if count > 2:
            non_manifold.append((edge, count))
        elif count == 1:
            boundary.append(edge)

    is_manifold = len(non_manifold) == 0 and len(boundary) == 0
    return is_manifold, non_manifold, boundary


def generate_outer_shell():
    """Generate the outer shell (nut body with threads inside, grip outside)."""
    vertices = []
    faces = []

    inner_r = BASE_INNER_DIAMETER / 2
    thread_inner_r = inner_r - BASE_THREAD_DEPTH
    outer_r = inner_r + BASE_WALL_THICKNESS
    grip_outer_r = outer_r + GRIP_RIDGE_DEPTH

    z_top = 0
    z_base_interior = -BASE_THREAD_HEIGHT
    z_base_bottom = z_base_interior - BASE_PLATE_THICKNESS

    def add_ring(radius, z):
        start = len(vertices)
        for i in range(SEGMENTS):
            angle = 2 * np.pi * i / SEGMENTS
            vertices.append([radius * np.cos(angle), radius * np.sin(angle), z])
        return start

    def connect_rings(ring1_start, ring2_start, reverse=False):
        for i in range(SEGMENTS):
            i0 = ring1_start + i
            i1 = ring1_start + (i + 1) % SEGMENTS
            j0 = ring2_start + i
            j1 = ring2_start + (i + 1) % SEGMENTS

            if reverse:
                faces.append([i0, i1, j1])
                faces.append([i0, j1, j0])
            else:
                faces.append([i0, j0, j1])
                faces.append([i0, j1, i1])

    def connect_annulus(outer_start, inner_start, flip=False):
        for i in range(SEGMENTS):
            o0 = outer_start + i
            o1 = outer_start + (i + 1) % SEGMENTS
            i0 = inner_start + i
            i1 = inner_start + (i + 1) % SEGMENTS

            if flip:
                faces.append([o0, o1, i1])
                faces.append([o0, i1, i0])
            else:
                faces.append([o0, i0, i1])
                faces.append([o0, i1, o1])

    def cap_ring(ring_start, center_z, flip=False):
        center_idx = len(vertices)
        vertices.append([0, 0, center_z])
        for i in range(SEGMENTS):
            p0 = ring_start + i
            p1 = ring_start + (i + 1) % SEGMENTS
            if flip:
                faces.append([p0, p1, center_idx])
            else:
                faces.append([p0, center_idx, p1])

    # Interior threads
    thread_rings_count = int(BASE_THREAD_HEIGHT / NOTEPAD_THREAD_PITCH) * 2 + 1
    interior_rings = []

    for i in range(thread_rings_count):
        z = z_top - (i / (thread_rings_count - 1)) * BASE_THREAD_HEIGHT
        r = inner_r if (i % 2 == 0) else thread_inner_r
        interior_rings.append(add_ring(r, z))

    for i in range(len(interior_rings) - 1):
        connect_rings(interior_rings[i], interior_rings[i + 1], reverse=True)

    # Exterior with grips
    grip_height = BASE_THREAD_HEIGHT / (GRIP_RIDGE_COUNT * 2)
    exterior_rings = []

    for i in range(GRIP_RIDGE_COUNT * 2 + 1):
        z = z_top - i * grip_height
        if z < z_base_interior:
            z = z_base_interior
        r = grip_outer_r if (i % 2 == 1) else outer_r
        exterior_rings.append(add_ring(r, z))

    for i in range(len(exterior_rings) - 1):
        connect_rings(exterior_rings[i], exterior_rings[i + 1])

    # Base plate exterior
    base_ext_bottom = add_ring(outer_r, z_base_bottom)
    connect_rings(exterior_rings[-1], base_ext_bottom)

    # Top annulus
    connect_annulus(exterior_rings[0], interior_rings[0])

    # Interior floor (solid cap at z_base_interior)
    cap_ring(interior_rings[-1], z_base_interior, flip=True)

    # Bottom (solid cap)
    cap_ring(base_ext_bottom, z_base_bottom, flip=True)

    return np.array(vertices), faces


def generate_platform():
    """Generate the platform cylinder."""
    vertices = []
    faces = []

    platform_r = PLATFORM_DIAMETER / 2
    z_top = 0
    z_base_interior = -BASE_THREAD_HEIGHT
    z_base_bottom = z_base_interior - BASE_PLATE_THICKNESS
    z_platform_top = -PLATFORM_RECESS

    def add_ring(radius, z):
        start = len(vertices)
        for i in range(SEGMENTS):
            angle = 2 * np.pi * i / SEGMENTS
            vertices.append([radius * np.cos(angle), radius * np.sin(angle), z])
        return start

    def connect_rings(ring1_start, ring2_start, reverse=False):
        for i in range(SEGMENTS):
            i0 = ring1_start + i
            i1 = ring1_start + (i + 1) % SEGMENTS
            j0 = ring2_start + i
            j1 = ring2_start + (i + 1) % SEGMENTS

            if reverse:
                faces.append([i0, i1, j1])
                faces.append([i0, j1, j0])
            else:
                faces.append([i0, j0, j1])
                faces.append([i0, j1, i1])

    def cap_ring(ring_start, center_z, flip=False):
        center_idx = len(vertices)
        vertices.append([0, 0, center_z])
        for i in range(SEGMENTS):
            p0 = ring_start + i
            p1 = ring_start + (i + 1) % SEGMENTS
            if flip:
                faces.append([p0, p1, center_idx])
            else:
                faces.append([p0, center_idx, p1])

    # Platform cylinder
    bottom_ring = add_ring(platform_r, z_base_bottom)
    top_ring = add_ring(platform_r, z_platform_top)

    connect_rings(bottom_ring, top_ring, reverse=True)

    # Bottom cap
    cap_ring(bottom_ring, z_base_bottom, flip=True)

    # Top cap
    cap_ring(top_ring, z_platform_top)

    return np.array(vertices), faces


def generate_pcb_mount():
    """Generate PCB mount with square base and M2 screw bosses."""
    vertices = []
    faces = []

    z_base_interior = -BASE_THREAD_HEIGHT
    z_base_bottom = z_base_interior - BASE_PLATE_THICKNESS
    z_pcb_surface = z_base_bottom - PCB_MOUNT_DEPTH
    z_boss_bottom = z_pcb_surface - PCB_BOSS_HEIGHT

    half_size = PCB_SIZE / 2
    half_grid = PCB_HOLE_GRID / 2
    boss_r = PCB_BOSS_DIAMETER / 2
    hole_r = PCB_HOLE_DIAMETER / 2

    def add_vertex(x, y, z):
        idx = len(vertices)
        vertices.append([x, y, z])
        return idx

    def add_ring(cx, cy, radius, z, segments=BOSS_SEGMENTS):
        start = len(vertices)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            vertices.append([cx + radius * np.cos(angle), cy + radius * np.sin(angle), z])
        return start

    def connect_rings(ring1_start, ring2_start, segments=BOSS_SEGMENTS, reverse=False):
        for i in range(segments):
            i0 = ring1_start + i
            i1 = ring1_start + (i + 1) % segments
            j0 = ring2_start + i
            j1 = ring2_start + (i + 1) % segments

            if reverse:
                faces.append([i0, i1, j1])
                faces.append([i0, j1, j0])
            else:
                faces.append([i0, j0, j1])
                faces.append([i0, j1, i1])

    def connect_annulus(outer_start, inner_start, segments=BOSS_SEGMENTS, flip=False):
        for i in range(segments):
            o0 = outer_start + i
            o1 = outer_start + (i + 1) % segments
            i0 = inner_start + i
            i1 = inner_start + (i + 1) % segments

            if flip:
                faces.append([o0, o1, i1])
                faces.append([o0, i1, i0])
            else:
                faces.append([o0, i0, i1])
                faces.append([o0, i1, o1])

    # Square base plate
    # Top surface at z_base_bottom (connects to shell bottom)
    # Bottom surface at z_pcb_surface
    corners_top = [
        add_vertex(-half_size, -half_size, z_base_bottom),
        add_vertex(half_size, -half_size, z_base_bottom),
        add_vertex(half_size, half_size, z_base_bottom),
        add_vertex(-half_size, half_size, z_base_bottom),
    ]

    corners_bottom = [
        add_vertex(-half_size, -half_size, z_pcb_surface),
        add_vertex(half_size, -half_size, z_pcb_surface),
        add_vertex(half_size, half_size, z_pcb_surface),
        add_vertex(-half_size, half_size, z_pcb_surface),
    ]

    # Top face (facing up toward shell)
    faces.append([corners_top[0], corners_top[1], corners_top[2]])
    faces.append([corners_top[0], corners_top[2], corners_top[3]])

    # Bottom face (facing down)
    faces.append([corners_bottom[0], corners_bottom[3], corners_bottom[2]])
    faces.append([corners_bottom[0], corners_bottom[2], corners_bottom[1]])

    # Side walls
    for i in range(4):
        t0, t1 = corners_top[i], corners_top[(i + 1) % 4]
        b0, b1 = corners_bottom[i], corners_bottom[(i + 1) % 4]
        faces.append([t0, b0, b1])
        faces.append([t0, b1, t1])

    # Screw bosses at corners of 16mm grid
    boss_positions = [
        (-half_grid, -half_grid),
        (half_grid, -half_grid),
        (half_grid, half_grid),
        (-half_grid, half_grid),
    ]

    for cx, cy in boss_positions:
        # Outer rings
        boss_outer_top = add_ring(cx, cy, boss_r, z_pcb_surface)
        boss_outer_bottom = add_ring(cx, cy, boss_r, z_boss_bottom)

        # Inner rings (hole)
        boss_inner_top = add_ring(cx, cy, hole_r, z_pcb_surface)
        boss_inner_bottom = add_ring(cx, cy, hole_r, z_boss_bottom)

        # Outer wall
        connect_rings(boss_outer_top, boss_outer_bottom)

        # Inner wall (hole, reversed)
        connect_rings(boss_inner_top, boss_inner_bottom, reverse=True)

        # Top annulus (connects outer to inner at top)
        connect_annulus(boss_outer_top, boss_inner_top, flip=True)

        # Bottom annulus
        connect_annulus(boss_outer_bottom, boss_inner_bottom)

    return np.array(vertices), faces


def write_obj(filepath, vertices, faces, name="MountBase"):
    """Write mesh to OBJ file."""
    with open(filepath, 'w') as f:
        f.write(f"# Mount Base\n")
        f.write(f"o {name}\n")

        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for face in faces:
            f.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")

    print(f"Saved: {filepath}")


def write_combined_obj(filepath, objects):
    """Write multiple objects to a single OBJ file."""
    with open(filepath, 'w') as f:
        f.write(f"# Mount Base - {len(objects)} manifold objects\n")

        vertex_offset = 0
        for name, verts, fcs in objects:
            f.write(f"o {name}\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in fcs:
                f.write("f " + " ".join(str(idx + 1 + vertex_offset) for idx in face) + "\n")
            vertex_offset += len(verts)

    print(f"Saved: {filepath}")


def write_stl(filepath, vertices, faces):
    """Write mesh to binary STL file."""
    vertices = np.array(vertices)

    with open(filepath, 'wb') as f:
        header = b'Binary STL - Mount Base' + b'\0' * 57
        f.write(header[:80])

        f.write(struct.pack('<I', len(faces)))

        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

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


def write_combined_stl(filepath, objects):
    """Write multiple objects to a single STL file."""
    all_faces = []

    for name, verts, fcs in objects:
        verts = np.array(verts)
        for face in fcs:
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            all_faces.append((v0, v1, v2))

    with open(filepath, 'wb') as f:
        header = b'Binary STL - Mount Base' + b'\0' * 57
        f.write(header[:80])

        f.write(struct.pack('<I', len(all_faces)))

        for v0, v1, v2 in all_faces:
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
    print("Generating Mount Base")
    print("=" * 60)

    inner_r = BASE_INNER_DIAMETER / 2
    thread_inner_r = inner_r - BASE_THREAD_DEPTH
    outer_r = inner_r + BASE_WALL_THICKNESS
    grip_outer_r = outer_r + GRIP_RIDGE_DEPTH

    z_top = 0
    z_base_interior = -BASE_THREAD_HEIGHT
    z_base_bottom = z_base_interior - BASE_PLATE_THICKNESS

    z_pcb_surface = z_base_bottom - PCB_MOUNT_DEPTH
    z_boss_bottom = z_pcb_surface - PCB_BOSS_HEIGHT

    print(f"Mount base dimensions:")
    print(f"  Inner diameter (thread root): {BASE_INNER_DIAMETER:.1f}mm")
    print(f"  Thread crest inner diameter: {thread_inner_r * 2:.1f}mm")
    print(f"  Outer diameter (wall): {outer_r * 2:.1f}mm")
    print(f"  Outer diameter (grip peaks): {grip_outer_r * 2:.1f}mm")
    print(f"  Total height: {-z_boss_bottom:.1f}mm")
    print(f"  Platform diameter: {PLATFORM_DIAMETER:.1f}mm")
    print(f"  Platform top: {PLATFORM_RECESS:.1f}mm below opening")
    print(f"  PCB mount: {PCB_SIZE:.1f}x{PCB_SIZE:.1f}mm, M2 holes on {PCB_HOLE_GRID:.1f}mm grid")

    # Generate all objects
    shell_verts, shell_faces = generate_outer_shell()
    platform_verts, platform_faces = generate_platform()
    pcb_verts, pcb_faces = generate_pcb_mount()

    print(f"\nShell: {len(shell_verts)} vertices, {len(shell_faces)} faces")
    print(f"Platform: {len(platform_verts)} vertices, {len(platform_faces)} faces")
    print(f"PCB mount: {len(pcb_verts)} vertices, {len(pcb_faces)} faces")

    # Check each for manifold
    print("\nChecking mesh integrity...")

    is_shell_manifold, shell_nm, shell_boundary = check_manifold(shell_verts, shell_faces)
    if is_shell_manifold:
        print("  Shell: watertight and manifold")
    else:
        print(f"  Shell: {len(shell_nm)} non-manifold, {len(shell_boundary)} boundary")

    is_platform_manifold, plat_nm, plat_boundary = check_manifold(platform_verts, platform_faces)
    if is_platform_manifold:
        print("  Platform: watertight and manifold")
    else:
        print(f"  Platform: {len(plat_nm)} non-manifold, {len(plat_boundary)} boundary")

    is_pcb_manifold, pcb_nm, pcb_boundary = check_manifold(pcb_verts, pcb_faces)
    if is_pcb_manifold:
        print("  PCB mount: watertight and manifold")
    else:
        print(f"  PCB mount: {len(pcb_nm)} non-manifold, {len(pcb_boundary)} boundary")

    # Combined bounding box
    all_verts = np.vstack([shell_verts, platform_verts, pcb_verts])
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
        ("Shell", shell_verts, shell_faces),
        ("Platform", platform_verts, platform_faces),
        ("PCB_Mount", pcb_verts, pcb_faces),
    ]

    write_combined_obj(obj_path, objects)
    write_combined_stl(stl_path, objects)

    print(f"\n{'=' * 60}")
    print(f"Generated mount base (3 manifold objects)")
    print(f"  OBJ: {obj_path}")
    print(f"  STL: {stl_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
