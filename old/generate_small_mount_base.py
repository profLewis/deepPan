#!/usr/bin/env python3
"""
Generate a small mount base for Adafruit 1740 piezo (14mm dia, 2.5mm thick).
Screws onto the central/inner ring notepad mount cylinders.
Same design as generate_mount_base.py but scaled for smaller sensor.
"""

import numpy as np
from pathlib import Path
import struct

# Match small notepad mount cylinder dimensions
NOTEPAD_INNER_DIAMETER = 15.0
NOTEPAD_WALL_THICKNESS = 2.0
NOTEPAD_THREAD_DEPTH = 0.8
NOTEPAD_THREAD_PITCH = 1.5

NOTEPAD_WALL_OUTER = NOTEPAD_INNER_DIAMETER + 2 * NOTEPAD_WALL_THICKNESS
NOTEPAD_THREAD_OUTER = NOTEPAD_WALL_OUTER + 2 * NOTEPAD_THREAD_DEPTH

# Mount base parameters
THREAD_CLEARANCE = 0.5
BASE_INNER_DIAMETER = NOTEPAD_THREAD_OUTER + THREAD_CLEARANCE
BASE_THREAD_DEPTH = 0.8
BASE_WALL_THICKNESS = 2.5
THREAD_HEIGHT = 6.0       # Match small mount depth
FLOOR_THICKNESS = 4.0
BASE_HEIGHT = THREAD_HEIGHT + FLOOR_THICKNESS

# External thread (for sleeve)
EXT_THREAD_DEPTH = 0.8
EXT_THREAD_PITCH = 1.5

# Notch (wire routing)
NOTCH_WIDTH = 3.0

# Anti-rotation groove
GROOVE_WIDTH = 2.5
GROOVE_DEPTH = 0.8
GROOVE_ANGLE = np.pi

# Sensor pocket (Adafruit 1740: 14mm dia, 2.5mm thick)
SENSOR_POCKET_DIAMETER = 14.5   # 14mm + 0.5mm clearance
SENSOR_POCKET_DEPTH = 3.0       # 2.5mm sensor + 0.5mm room

SEGMENTS = 48
BOSS_SEGMENTS = 16


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


def generate_cylinder():
    """Generate small mount base cylinder with threads, notch, and groove."""
    import math

    inner_r = BASE_INNER_DIAMETER / 2
    wall_outer_r = inner_r + BASE_WALL_THICKNESS
    ext_thread_outer_r = wall_outer_r + EXT_THREAD_DEPTH

    notch_half_angle = math.atan2(NOTCH_WIDTH / 2, inner_r)
    notch_segments = max(2, int(notch_half_angle * 2 * SEGMENTS / (2 * math.pi)) + 1)
    notch_start = SEGMENTS - notch_segments // 2
    notch_end = notch_segments - notch_segments // 2

    groove_half_angle = math.atan2(GROOVE_WIDTH / 2, inner_r)

    def in_notch(seg):
        return seg >= notch_start or seg < notch_end

    def in_groove(seg):
        angle = 2 * math.pi * seg / SEGMENTS
        diff = abs(angle - GROOVE_ANGLE)
        if diff > math.pi:
            diff = 2 * math.pi - diff
        return diff <= groove_half_angle

    z_levels = max(int(BASE_HEIGHT / NOTEPAD_THREAD_PITCH * 16), 32)
    vertices = []
    faces = []
    inner_rings = []
    outer_rings = []

    for z_idx in range(z_levels + 1):
        z = -z_idx * BASE_HEIGHT / z_levels
        inner_ring = {}
        outer_ring = {}

        # Internal thread (at top portion only)
        if z >= -THREAD_HEIGHT:
            int_phase = (-z / NOTEPAD_THREAD_PITCH) % 1.0
            int_thread_h = helical_thread_profile(int_phase, BASE_THREAD_DEPTH)
            ir = inner_r - int_thread_h
        else:
            ir = inner_r

        # External thread (full height)
        ext_phase = (-z / EXT_THREAD_PITCH) % 1.0
        ext_thread_h = helical_thread_profile(ext_phase, EXT_THREAD_DEPTH)
        ext_r = wall_outer_r + ext_thread_h

        for i in range(SEGMENTS):
            if in_notch(i):
                continue

            angle = 2 * math.pi * i / SEGMENTS

            # Anti-rotation groove
            if in_groove(i) and z >= -THREAD_HEIGHT:
                groove_ir = inner_r + GROOVE_DEPTH
            else:
                groove_ir = ir

            inner_ring[i] = len(vertices)
            vertices.append([groove_ir * math.cos(angle), groove_ir * math.sin(angle), z])

            outer_ring[i] = len(vertices)
            vertices.append([ext_r * math.cos(angle), ext_r * math.sin(angle), z])

        inner_rings.append(inner_ring)
        outer_rings.append(outer_ring)

    valid_segs = sorted([i for i in range(SEGMENTS) if not in_notch(i)])
    first_valid = valid_segs[0]
    last_valid = valid_segs[-1]

    first_angle = 2 * math.pi * first_valid / SEGMENTS
    last_angle = 2 * math.pi * last_valid / SEGMENTS

    # Notch wall vertices
    notch_inner_left = []
    notch_outer_left = []
    notch_inner_right = []
    notch_outer_right = []

    for z_idx in range(z_levels + 1):
        z = -z_idx * BASE_HEIGHT / z_levels
        if z >= -THREAD_HEIGHT:
            int_phase = (-z / NOTEPAD_THREAD_PITCH) % 1.0
            ir = inner_r - helical_thread_profile(int_phase, BASE_THREAD_DEPTH)
        else:
            ir = inner_r

        ext_phase = (-z / EXT_THREAD_PITCH) % 1.0
        ext_r = wall_outer_r + helical_thread_profile(ext_phase, EXT_THREAD_DEPTH)

        notch_inner_left.append(len(vertices))
        vertices.append([ir * math.cos(last_angle), ir * math.sin(last_angle), z])
        notch_outer_left.append(len(vertices))
        vertices.append([ext_r * math.cos(last_angle), ext_r * math.sin(last_angle), z])

        notch_inner_right.append(len(vertices))
        vertices.append([ir * math.cos(first_angle), ir * math.sin(first_angle), z])
        notch_outer_right.append(len(vertices))
        vertices.append([ext_r * math.cos(first_angle), ext_r * math.sin(first_angle), z])

    # === FACES ===

    # Top cap
    for idx in range(len(valid_segs) - 1):
        i, i_next = valid_segs[idx], valid_segs[idx + 1]
        faces.append([inner_rings[0][i], outer_rings[0][i], outer_rings[0][i_next], inner_rings[0][i_next]])

    faces.append([notch_inner_left[0], notch_outer_left[0], outer_rings[0][last_valid], inner_rings[0][last_valid]])
    faces.append([inner_rings[0][first_valid], outer_rings[0][first_valid], notch_outer_right[0], notch_inner_right[0]])

    # Bottom cap (solid floor with sensor pocket)
    # For now, just a solid annular floor
    for idx in range(len(valid_segs) - 1):
        i, i_next = valid_segs[idx], valid_segs[idx + 1]
        faces.append([inner_rings[-1][i], inner_rings[-1][i_next], outer_rings[-1][i_next], outer_rings[-1][i]])

    faces.append([notch_inner_left[-1], inner_rings[-1][last_valid], outer_rings[-1][last_valid], notch_outer_left[-1]])
    faces.append([inner_rings[-1][first_valid], notch_inner_right[-1], notch_outer_right[-1], outer_rings[-1][first_valid]])

    # Inner wall
    for z_idx in range(z_levels):
        for idx in range(len(valid_segs) - 1):
            i, i_next = valid_segs[idx], valid_segs[idx + 1]
            faces.append([inner_rings[z_idx][i], inner_rings[z_idx][i_next],
                         inner_rings[z_idx + 1][i_next], inner_rings[z_idx + 1][i]])

    # Outer wall
    for z_idx in range(z_levels):
        for idx in range(len(valid_segs) - 1):
            i, i_next = valid_segs[idx], valid_segs[idx + 1]
            faces.append([outer_rings[z_idx][i], outer_rings[z_idx + 1][i],
                         outer_rings[z_idx + 1][i_next], outer_rings[z_idx][i_next]])

    # Notch walls
    for z_idx in range(z_levels):
        faces.append([notch_inner_left[z_idx], inner_rings[z_idx][last_valid],
                     inner_rings[z_idx + 1][last_valid], notch_inner_left[z_idx + 1]])
        faces.append([outer_rings[z_idx][last_valid], notch_outer_left[z_idx],
                     notch_outer_left[z_idx + 1], outer_rings[z_idx + 1][last_valid]])
        faces.append([notch_inner_left[z_idx], notch_outer_left[z_idx],
                     notch_outer_left[z_idx + 1], notch_inner_left[z_idx + 1]])

        faces.append([inner_rings[z_idx][first_valid], notch_inner_right[z_idx],
                     notch_inner_right[z_idx + 1], inner_rings[z_idx + 1][first_valid]])
        faces.append([notch_outer_right[z_idx], outer_rings[z_idx][first_valid],
                     outer_rings[z_idx + 1][first_valid], notch_outer_right[z_idx + 1]])
        faces.append([notch_outer_right[z_idx], notch_inner_right[z_idx],
                     notch_inner_right[z_idx + 1], notch_outer_right[z_idx + 1]])

    return np.array(vertices), faces


def main():
    from generate_notepad import write_obj, write_stl

    print("Generating small mount base (Adafruit 1740)...")
    print(f"  Internal bore: {BASE_INNER_DIAMETER:.1f}mm")
    print(f"  Wall outer: {NOTEPAD_WALL_OUTER + 2 * BASE_WALL_THICKNESS:.1f}mm")
    print(f"  Height: {BASE_HEIGHT:.1f}mm")

    verts, faces = generate_cylinder()
    print(f"  Vertices: {len(verts)}, Faces: {len(faces)}")

    out_dir = Path("data/mounts")
    out_dir.mkdir(exist_ok=True)
    write_obj(out_dir / "small_mount_base.obj", verts, faces, "SmallMountBase")
    write_stl(out_dir / "small_mount_base.stl", verts, faces, "SmallMountBase")


if __name__ == "__main__":
    main()
