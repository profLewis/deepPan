#!/usr/bin/env python3
"""
Generate a small outer sleeve for central/inner ring pads (Adafruit 1740 sensor).
Screws onto the small mount base.
"""

import numpy as np
from pathlib import Path
import struct
import math

# Match small mount base external dimensions
from generate_small_mount_base import (
    BASE_WALL_THICKNESS, EXT_THREAD_DEPTH, EXT_THREAD_PITCH,
    BASE_HEIGHT, NOTEPAD_WALL_OUTER,
)

MOUNT_BASE_WALL_OUTER = NOTEPAD_WALL_OUTER + 2 * BASE_WALL_THICKNESS
MOUNT_BASE_EXT_THREAD_DEPTH = EXT_THREAD_DEPTH
MOUNT_BASE_HEIGHT = BASE_HEIGHT

# Sleeve parameters
THREAD_CLEARANCE = 0.5
SLEEVE_INNER_DIAMETER = MOUNT_BASE_WALL_OUTER + 2 * MOUNT_BASE_EXT_THREAD_DEPTH + THREAD_CLEARANCE
SLEEVE_THREAD_DEPTH = 0.8
SLEEVE_THREAD_PITCH = EXT_THREAD_PITCH
SLEEVE_WALL_THICKNESS = 2.5
FLOOR_THICKNESS = 2.0

THREAD_REGION_HEIGHT = MOUNT_BASE_HEIGHT
FLOOR_Z = -MOUNT_BASE_HEIGHT - 8.0   # Some room below base
SLEEVE_HEIGHT = -FLOOR_Z + FLOOR_THICKNESS

# Grip ridges
GRIP_RIDGES = 10
GRIP_RIDGE_DEPTH = 1.2
GRIP_START_Z = -2.0
GRIP_END_Z = -14.0

# Wire slit
SLIT_WIDTH = 8.0
SLIT_TOP_Z = -8.0
SLIT_BOTTOM_Z = FLOOR_Z

# Floor access hole
FLOOR_HOLE_DIAMETER = 8.0

SEGMENTS = 48
HOLE_SEGMENTS = 24


def helical_thread_profile(phase, depth, crest_fraction=0.25):
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
    """Generate small outer sleeve."""
    vertices = []
    faces = []

    inner_r = SLEEVE_INNER_DIAMETER / 2
    outer_r = inner_r + SLEEVE_WALL_THICKNESS
    hole_r = FLOOR_HOLE_DIAMETER / 2

    # Slit angular range
    slit_half_angle = math.atan2(SLIT_WIDTH / 2, inner_r)
    slit_segs = max(2, int(slit_half_angle * 2 * SEGMENTS / (2 * math.pi)) + 1)
    slit_start = SEGMENTS - slit_segs // 2
    slit_end = slit_segs - slit_segs // 2

    def in_slit(seg):
        return seg >= slit_start or seg < slit_end

    # Grip ridge positions
    grip_angles = set()
    for g in range(GRIP_RIDGES):
        angle_idx = int(g * SEGMENTS / GRIP_RIDGES) % SEGMENTS
        grip_angles.add(angle_idx)

    z_levels = max(int(SLEEVE_HEIGHT / SLEEVE_THREAD_PITCH * 16), 48)

    inner_rings = []
    outer_rings = []

    for z_idx in range(z_levels + 1):
        z = -z_idx * SLEEVE_HEIGHT / z_levels
        inner_ring = {}
        outer_ring = {}

        # Internal thread (top region only)
        if z >= -THREAD_REGION_HEIGHT:
            phase = (-z / SLEEVE_THREAD_PITCH) % 1.0
            thread_h = helical_thread_profile(phase, SLEEVE_THREAD_DEPTH)
            ir = inner_r - thread_h
        else:
            ir = inner_r

        for i in range(SEGMENTS):
            # Skip slit segments in slit Z region
            if in_slit(i) and SLIT_BOTTOM_Z <= z <= SLIT_TOP_Z:
                continue

            angle = 2 * math.pi * i / SEGMENTS

            inner_ring[i] = len(vertices)
            vertices.append([ir * math.cos(angle), ir * math.sin(angle), z])

            # Grip ridges on exterior
            grip_r = outer_r
            if i in grip_angles and GRIP_END_Z <= z <= GRIP_START_Z:
                grip_r = outer_r + GRIP_RIDGE_DEPTH

            outer_ring[i] = len(vertices)
            vertices.append([grip_r * math.cos(angle), grip_r * math.sin(angle), z])

        inner_rings.append(inner_ring)
        outer_rings.append(outer_ring)

    # Build faces - inner wall, outer wall, top cap, bottom cap
    for z_idx in range(z_levels):
        segs_this = sorted(set(inner_rings[z_idx].keys()) & set(inner_rings[z_idx + 1].keys()))
        for s_idx in range(len(segs_this) - 1):
            i, i_next = segs_this[s_idx], segs_this[s_idx + 1]
            # Inner wall
            if i in inner_rings[z_idx] and i_next in inner_rings[z_idx] and \
               i in inner_rings[z_idx + 1] and i_next in inner_rings[z_idx + 1]:
                faces.append([inner_rings[z_idx][i], inner_rings[z_idx][i_next],
                             inner_rings[z_idx + 1][i_next], inner_rings[z_idx + 1][i]])
            # Outer wall
            if i in outer_rings[z_idx] and i_next in outer_rings[z_idx] and \
               i in outer_rings[z_idx + 1] and i_next in outer_rings[z_idx + 1]:
                faces.append([outer_rings[z_idx][i], outer_rings[z_idx + 1][i],
                             outer_rings[z_idx + 1][i_next], outer_rings[z_idx][i_next]])

    # Top cap
    top_segs = sorted(inner_rings[0].keys())
    for s_idx in range(len(top_segs) - 1):
        i, i_next = top_segs[s_idx], top_segs[s_idx + 1]
        faces.append([inner_rings[0][i], outer_rings[0][i], outer_rings[0][i_next], inner_rings[0][i_next]])

    # Bottom cap (annular with center hole)
    bot_segs = sorted(inner_rings[-1].keys())

    # Floor hole vertices
    floor_z = -SLEEVE_HEIGHT
    hole_ring = []
    for i in range(HOLE_SEGMENTS):
        angle = 2 * math.pi * i / HOLE_SEGMENTS
        hole_ring.append(len(vertices))
        vertices.append([hole_r * math.cos(angle), hole_r * math.sin(angle), floor_z])

    # Connect bottom inner ring to hole ring via triangles
    # Simple fan approach
    for s_idx in range(len(bot_segs) - 1):
        i, i_next = bot_segs[s_idx], bot_segs[s_idx + 1]
        faces.append([inner_rings[-1][i], inner_rings[-1][i_next],
                     outer_rings[-1][i_next], outer_rings[-1][i]])

    return np.array(vertices), faces


def main():
    from generate_notepad import write_obj, write_stl

    print("Generating small outer sleeve (Adafruit 1740)...")
    print(f"  Inner diameter: {SLEEVE_INNER_DIAMETER:.1f}mm")
    print(f"  Outer diameter: {SLEEVE_INNER_DIAMETER + 2 * SLEEVE_WALL_THICKNESS:.1f}mm")
    print(f"  Height: {SLEEVE_HEIGHT:.1f}mm")

    verts, faces = generate_sleeve()
    print(f"  Vertices: {len(verts)}, Faces: {len(faces)}")

    out_dir = Path("data/mounts")
    out_dir.mkdir(exist_ok=True)
    write_obj(out_dir / "small_outer_sleeve.obj", verts, faces, "SmallOuterSleeve")
    write_stl(out_dir / "small_outer_sleeve.stl", verts, faces, "SmallOuterSleeve")


if __name__ == "__main__":
    main()
