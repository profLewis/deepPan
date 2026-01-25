#!/usr/bin/env python3
"""
Generate a screw-on mount base that attaches to the notepad cylinder.

Features:
- Internal threads matching the notepad's external threads
- Base plate with central platform (recessed)
- Knurled exterior for hand grip
"""

import numpy as np
from pathlib import Path
import struct

# Import matching parameters from notepad generator
# These must match the notepad cylinder for proper fit
NOTEPAD_INNER_DIAMETER = 23.5   # Notepad cylinder inner diameter
NOTEPAD_WALL_THICKNESS = 2.5    # Notepad cylinder wall thickness
NOTEPAD_THREAD_DEPTH = 1.0      # Notepad thread depth (outward)
NOTEPAD_THREAD_PITCH = 2.0      # Thread pitch

# Derived notepad dimensions
NOTEPAD_WALL_OUTER = NOTEPAD_INNER_DIAMETER + 2 * NOTEPAD_WALL_THICKNESS  # 28.5mm
NOTEPAD_THREAD_OUTER = NOTEPAD_WALL_OUTER + 2 * NOTEPAD_THREAD_DEPTH      # 30.5mm

# Mount base parameters
THREAD_CLEARANCE = 0.3          # Clearance for thread fit
BASE_INNER_DIAMETER = NOTEPAD_THREAD_OUTER + THREAD_CLEARANCE  # ~30.8mm (thread root)
BASE_THREAD_DEPTH = 1.0         # Internal thread depth
BASE_WALL_THICKNESS = 3.0       # Wall thickness outside threads
BASE_THREAD_HEIGHT = 9.0        # Height of threaded section (matches notepad depth)

# Base plate parameters
BASE_PLATE_THICKNESS = 3.0      # Thickness of the base plate
PLATFORM_DIAMETER = 20.0        # Central platform diameter
PLATFORM_HEIGHT = 5.0           # Platform height above base interior
PLATFORM_RECESS = 7.0           # How far below flush the platform top sits

# Knurl parameters
KNURL_COUNT = 24                # Number of knurls around circumference
KNURL_DEPTH = 1.5               # Depth of knurl grooves
KNURL_WIDTH_RATIO = 0.4         # Ratio of groove width to knurl spacing

# Resolution
SEGMENTS = 48                   # Circumferential segments


def generate_mount_base():
    """
    Generate the screw-on mount base geometry.

    The base is oriented with:
    - Top at Z=0 (where it meets the notepad)
    - Base plate at bottom (negative Z)
    - Platform extending upward from base interior
    """
    vertices = []
    faces = []

    # Key dimensions
    inner_r = BASE_INNER_DIAMETER / 2                    # Inner radius at thread root
    thread_inner_r = inner_r - BASE_THREAD_DEPTH         # Inner radius at thread crest
    outer_r = inner_r + BASE_WALL_THICKNESS              # Outer radius (before knurls)
    knurl_outer_r = outer_r + KNURL_DEPTH                # Outer radius at knurl peaks

    platform_r = PLATFORM_DIAMETER / 2

    # Heights (Z coordinates, with top at 0)
    z_top = 0
    z_base_interior = -BASE_THREAD_HEIGHT                # Inside bottom of threaded section
    z_base_bottom = z_base_interior - BASE_PLATE_THICKNESS  # Very bottom
    z_platform_top = z_base_interior + PLATFORM_HEIGHT - PLATFORM_RECESS  # Platform top

    print(f"Mount base dimensions:")
    print(f"  Inner diameter (thread root): {BASE_INNER_DIAMETER:.1f}mm")
    print(f"  Thread crest inner diameter: {thread_inner_r * 2:.1f}mm")
    print(f"  Outer diameter (wall): {outer_r * 2:.1f}mm")
    print(f"  Outer diameter (knurl peaks): {knurl_outer_r * 2:.1f}mm")
    print(f"  Total height: {-z_base_bottom:.1f}mm")
    print(f"  Threaded section height: {BASE_THREAD_HEIGHT:.1f}mm")
    print(f"  Base plate thickness: {BASE_PLATE_THICKNESS:.1f}mm")
    print(f"  Platform top recess: {PLATFORM_RECESS:.1f}mm below top")

    # Generate threaded interior (internal threads)
    # Thread goes from z_top down to z_base_interior
    thread_turns = BASE_THREAD_HEIGHT / NOTEPAD_THREAD_PITCH
    thread_segments_per_turn = SEGMENTS
    total_thread_segments = int(thread_turns * thread_segments_per_turn) + 1

    # Generate thread profile vertices (helical)
    thread_verts_inner = []  # At thread crest (innermost)
    thread_verts_outer = []  # At thread root (where it meets wall)

    for i in range(total_thread_segments):
        angle = 2 * np.pi * i / thread_segments_per_turn
        z = z_top - (i / thread_segments_per_turn) * NOTEPAD_THREAD_PITCH

        if z < z_base_interior:
            z = z_base_interior

        # Thread profile: triangular, pointing inward
        # At each point, the thread depth varies sinusoidally
        thread_phase = (i % thread_segments_per_turn) / thread_segments_per_turn
        # 0 = root, 0.5 = crest, 1 = root
        thread_offset = BASE_THREAD_DEPTH * (1 - abs(2 * thread_phase - 1))

        r_inner = inner_r - thread_offset

        x = np.cos(angle)
        y = np.sin(angle)

        thread_verts_inner.append([r_inner * x, r_inner * y, z])
        thread_verts_outer.append([inner_r * x, inner_r * y, z])

    # Add thread vertices
    base_idx = len(vertices)
    vertices.extend(thread_verts_inner)
    inner_start = base_idx

    base_idx = len(vertices)
    vertices.extend(thread_verts_outer)
    outer_start = base_idx

    # Create faces for threaded interior
    for i in range(total_thread_segments - 1):
        # Quad from inner to outer
        i0 = inner_start + i
        i1 = inner_start + i + 1
        o0 = outer_start + i
        o1 = outer_start + i + 1

        # Two triangles for the quad (inner surface faces inward, so reverse winding)
        faces.append([i0, o0, o1])
        faces.append([i0, o1, i1])

    # Generate knurled exterior
    # Knurls are vertical ridges around the outside
    knurl_angle_step = 2 * np.pi / KNURL_COUNT
    knurl_groove_angle = knurl_angle_step * KNURL_WIDTH_RATIO

    # Heights for exterior wall
    z_levels = [z_top, z_base_interior, z_base_bottom]

    exterior_rings = []
    for z in [z_top, z_base_interior]:  # Only threaded section gets knurls
        ring = []
        for k in range(KNURL_COUNT):
            base_angle = k * knurl_angle_step

            # Each knurl has: groove start, peak, groove end
            # Groove
            a1 = base_angle
            a2 = base_angle + knurl_groove_angle / 2
            a3 = base_angle + knurl_angle_step / 2  # Peak
            a4 = base_angle + knurl_angle_step - knurl_groove_angle / 2
            a5 = base_angle + knurl_angle_step  # Next groove

            # Groove points (at outer_r)
            ring.append([outer_r * np.cos(a1), outer_r * np.sin(a1), z])
            ring.append([outer_r * np.cos(a2), outer_r * np.sin(a2), z])
            # Peak point (at knurl_outer_r)
            ring.append([knurl_outer_r * np.cos(a3), knurl_outer_r * np.sin(a3), z])
            # Back to groove
            ring.append([outer_r * np.cos(a4), outer_r * np.sin(a4), z])

        exterior_rings.append(ring)

    # Add base plate exterior (smooth, no knurls)
    base_ring = []
    for i in range(SEGMENTS):
        angle = 2 * np.pi * i / SEGMENTS
        base_ring.append([outer_r * np.cos(angle), outer_r * np.sin(angle), z_base_bottom])
    exterior_rings.append(base_ring)

    # Add exterior vertices
    ring_starts = []
    for ring in exterior_rings:
        ring_starts.append(len(vertices))
        vertices.extend(ring)

    # Create faces for knurled exterior (top ring to middle ring)
    n_knurl_verts = KNURL_COUNT * 4
    for i in range(n_knurl_verts):
        i0 = ring_starts[0] + i
        i1 = ring_starts[0] + (i + 1) % n_knurl_verts
        j0 = ring_starts[1] + i
        j1 = ring_starts[1] + (i + 1) % n_knurl_verts

        faces.append([i0, j0, j1])
        faces.append([i0, j1, i1])

    # Create faces for base plate exterior (middle ring to bottom ring)
    # Need to connect knurled ring (n_knurl_verts) to smooth ring (SEGMENTS)
    # Simplified: create a transition ring at z_base_interior with SEGMENTS verts
    transition_ring_start = len(vertices)
    for i in range(SEGMENTS):
        angle = 2 * np.pi * i / SEGMENTS
        vertices.append([outer_r * np.cos(angle), outer_r * np.sin(angle), z_base_interior])

    # Connect transition ring to bottom ring
    for i in range(SEGMENTS):
        t0 = transition_ring_start + i
        t1 = transition_ring_start + (i + 1) % SEGMENTS
        b0 = ring_starts[2] + i
        b1 = ring_starts[2] + (i + 1) % SEGMENTS

        faces.append([t0, b0, b1])
        faces.append([t0, b1, t1])

    # Top ring (annular face at z=0)
    # Inner edge at thread_inner_r, outer edge at knurled exterior
    top_inner_ring_start = len(vertices)
    for i in range(SEGMENTS):
        angle = 2 * np.pi * i / SEGMENTS
        vertices.append([thread_inner_r * np.cos(angle), thread_inner_r * np.sin(angle), z_top])

    # Connect to knurled exterior top (use simplified outer ring)
    top_outer_ring_start = len(vertices)
    for i in range(SEGMENTS):
        angle = 2 * np.pi * i / SEGMENTS
        # Average radius for simplified connection
        r = outer_r
        vertices.append([r * np.cos(angle), r * np.sin(angle), z_top])

    # Top annular faces
    for i in range(SEGMENTS):
        ti0 = top_inner_ring_start + i
        ti1 = top_inner_ring_start + (i + 1) % SEGMENTS
        to0 = top_outer_ring_start + i
        to1 = top_outer_ring_start + (i + 1) % SEGMENTS

        faces.append([ti0, to0, to1])
        faces.append([ti0, to1, ti1])

    # Bottom face (base plate with central platform hole)
    # Outer edge at outer_r, inner edge at platform_r
    bottom_outer_start = ring_starts[2]  # Already have this

    bottom_inner_start = len(vertices)
    for i in range(SEGMENTS):
        angle = 2 * np.pi * i / SEGMENTS
        vertices.append([platform_r * np.cos(angle), platform_r * np.sin(angle), z_base_bottom])

    # Bottom annular faces
    for i in range(SEGMENTS):
        bo0 = bottom_outer_start + i
        bo1 = bottom_outer_start + (i + 1) % SEGMENTS
        bi0 = bottom_inner_start + i
        bi1 = bottom_inner_start + (i + 1) % SEGMENTS

        faces.append([bo0, bi0, bi1])
        faces.append([bo0, bi1, bo1])

    # Central platform
    # Cylinder from z_base_bottom up to z_platform_top
    platform_bottom_start = bottom_inner_start  # Reuse bottom inner ring

    platform_top_start = len(vertices)
    for i in range(SEGMENTS):
        angle = 2 * np.pi * i / SEGMENTS
        vertices.append([platform_r * np.cos(angle), platform_r * np.sin(angle), z_platform_top])

    # Platform outer wall
    for i in range(SEGMENTS):
        pb0 = platform_bottom_start + i
        pb1 = platform_bottom_start + (i + 1) % SEGMENTS
        pt0 = platform_top_start + i
        pt1 = platform_top_start + (i + 1) % SEGMENTS

        # Wall faces outward
        faces.append([pb0, pt0, pt1])
        faces.append([pb0, pt1, pb1])

    # Platform top cap
    platform_center_idx = len(vertices)
    vertices.append([0, 0, z_platform_top])

    for i in range(SEGMENTS):
        pt0 = platform_top_start + i
        pt1 = platform_top_start + (i + 1) % SEGMENTS
        faces.append([pt0, pt1, platform_center_idx])

    # Interior base (floor between platform and thread)
    # Annular face at z_base_interior from inner_r to platform_r
    interior_floor_outer_start = len(vertices)
    for i in range(SEGMENTS):
        angle = 2 * np.pi * i / SEGMENTS
        vertices.append([inner_r * np.cos(angle), inner_r * np.sin(angle), z_base_interior])

    interior_floor_inner_start = len(vertices)
    for i in range(SEGMENTS):
        angle = 2 * np.pi * i / SEGMENTS
        vertices.append([platform_r * np.cos(angle), platform_r * np.sin(angle), z_base_interior])

    # Interior floor faces
    for i in range(SEGMENTS):
        fo0 = interior_floor_outer_start + i
        fo1 = interior_floor_outer_start + (i + 1) % SEGMENTS
        fi0 = interior_floor_inner_start + i
        fi1 = interior_floor_inner_start + (i + 1) % SEGMENTS

        faces.append([fo0, fi0, fi1])
        faces.append([fo0, fi1, fo1])

    # Connect interior floor inner ring to platform wall
    # (The platform wall goes from z_base_bottom to z_platform_top)
    # Need to add a ring at z_base_interior at platform_r and connect to platform wall
    platform_mid_start = interior_floor_inner_start  # At z_base_interior, platform_r

    # Platform interior wall (from z_base_interior down to z_base_bottom)
    for i in range(SEGMENTS):
        pm0 = platform_mid_start + i
        pm1 = platform_mid_start + (i + 1) % SEGMENTS
        pb0 = platform_bottom_start + i
        pb1 = platform_bottom_start + (i + 1) % SEGMENTS

        # Interior wall faces inward
        faces.append([pm0, pb0, pb1])
        faces.append([pm0, pb1, pm1])

    vertices = np.array(vertices)

    return vertices, faces


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


def write_stl(filepath, vertices, faces):
    """Write mesh to binary STL file."""
    vertices = np.array(vertices)

    with open(filepath, 'wb') as f:
        # Header (80 bytes)
        header = b'Binary STL - Mount Base' + b'\0' * 57
        f.write(header[:80])

        # Number of triangles
        f.write(struct.pack('<I', len(faces)))

        # Write triangles
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

            # Compute normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm

            # Write normal
            f.write(struct.pack('<3f', *normal))
            # Write vertices
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            # Attribute byte count
            f.write(struct.pack('<H', 0))

    print(f"Saved: {filepath}")


def main():
    print("=" * 60)
    print("Generating Mount Base")
    print("=" * 60)

    vertices, faces = generate_mount_base()

    print(f"\nMesh: {len(vertices)} vertices, {len(faces)} faces")

    # Bounding box
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min
    print(f"Bounding box: {bbox_size[0]:.1f} x {bbox_size[1]:.1f} x {bbox_size[2]:.1f} mm")

    # Output
    output_dir = Path("data/mounts")
    output_dir.mkdir(exist_ok=True)

    obj_path = output_dir / "mount_base.obj"
    stl_path = output_dir / "mount_base.stl"

    write_obj(obj_path, vertices, faces)
    write_stl(stl_path, vertices, faces)

    print(f"\n{'=' * 60}")
    print(f"Generated mount base")
    print(f"  OBJ: {obj_path}")
    print(f"  STL: {stl_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
