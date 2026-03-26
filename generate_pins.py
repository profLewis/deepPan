#!/usr/bin/env python3
"""
Generate alignment/test pins that fit through screw holes.

Each pin mimics an M2 flat-head screw (DIN 965):
  - Tapered cone head: 3.8mm top → 2.0mm bottom, 1.0mm tall, 90° included angle
  - Cylindrical shaft: 2.0mm diameter, extends down through the hole

Pins are positioned at each screw hole from notepad_properties.json,
oriented along the pad's surface normal.  Output in body coords
(matches assembly_view.obj and pan_body.obj).

Output:
  data/notepads/pins.obj — all pins as named objects
"""

import numpy as np
import json
import math
from pathlib import Path

from generate_notepad import (
    compute_leveling_rotation, NOTE_BY_INDEX,
)

OBJ_PATH = "data/Tenor Pan only.obj"
OUTPUT = "data/notepads/pins.obj"

# M2 flat-head pin dimensions (DIN 965)
HEAD_TOP_R = 1.9       # mm — head top radius (3.8mm dia)
HEAD_BOT_R = 1.0       # mm — transitions to shaft (2.0mm dia)
HEAD_HEIGHT = 0.9      # mm — cone height (90° included angle)
SHAFT_R = 1.0          # mm — shaft radius (2.0mm dia, M2)
SHAFT_LENGTH = 15.0    # mm — shaft below head (total pin = 15.9mm ≈ 16mm)
SEGMENTS = 24          # angular resolution


def generate_pin_mesh():
    """Generate a single pin mesh centered at origin, pointing along +Z.

    Head cone at top (Z=0 to Z=-HEAD_HEIGHT), shaft below (down to -HEAD_HEIGHT-SHAFT_LENGTH).
    The head top face is at Z=0 (playing surface level).

    Returns (vertices, faces) as numpy arrays.
    """
    verts = []
    faces = []

    # --- Head cone: Z=0 (top, wide) to Z=-HEAD_HEIGHT (bottom, narrow) ---
    # Top ring (head surface)
    top_center = len(verts)
    verts.append([0, 0, 0])  # center of head top
    top_ring_start = len(verts)
    for i in range(SEGMENTS):
        angle = 2 * math.pi * i / SEGMENTS
        verts.append([HEAD_TOP_R * math.cos(angle),
                      HEAD_TOP_R * math.sin(angle), 0])
    # Top face (fan)
    for i in range(SEGMENTS):
        faces.append([top_center, top_ring_start + i,
                      top_ring_start + (i + 1) % SEGMENTS])

    # Bottom of head ring
    head_bot_start = len(verts)
    z_bot = -HEAD_HEIGHT
    for i in range(SEGMENTS):
        angle = 2 * math.pi * i / SEGMENTS
        verts.append([HEAD_BOT_R * math.cos(angle),
                      HEAD_BOT_R * math.sin(angle), z_bot])

    # Head cone side faces
    for i in range(SEGMENTS):
        i1 = (i + 1) % SEGMENTS
        faces.append([top_ring_start + i, head_bot_start + i, head_bot_start + i1])
        faces.append([top_ring_start + i, head_bot_start + i1, top_ring_start + i1])

    # --- Shaft cylinder: Z=-HEAD_HEIGHT to Z=-(HEAD_HEIGHT+SHAFT_LENGTH) ---
    shaft_bot_start = len(verts)
    z_shaft = -(HEAD_HEIGHT + SHAFT_LENGTH)
    for i in range(SEGMENTS):
        angle = 2 * math.pi * i / SEGMENTS
        verts.append([SHAFT_R * math.cos(angle),
                      SHAFT_R * math.sin(angle), z_shaft])

    # Shaft side faces (head_bot_start → shaft_bot_start)
    for i in range(SEGMENTS):
        i1 = (i + 1) % SEGMENTS
        faces.append([head_bot_start + i, shaft_bot_start + i, shaft_bot_start + i1])
        faces.append([head_bot_start + i, shaft_bot_start + i1, head_bot_start + i1])

    # Shaft bottom cap
    bot_center = len(verts)
    verts.append([0, 0, z_shaft])
    for i in range(SEGMENTS):
        faces.append([bot_center, shaft_bot_start + (i + 1) % SEGMENTS,
                      shaft_bot_start + i])

    return np.array(verts, dtype=float), faces


def transform_pin(pin_verts, position, normal):
    """Transform pin from local coords (Z=pin axis) to world coords.

    Pin +Z maps to +normal direction (head points outward from surface).
    Pin -Z maps to -normal (shaft goes into the body).
    """
    # Build rotation from +Z to normal
    z_axis = np.array([0, 0, 1.0])
    n = normal / np.linalg.norm(normal)

    v = np.cross(z_axis, n)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, n)

    if s < 1e-8:
        R = np.eye(3) if c > 0 else np.diag([1, -1, -1])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

    return (R @ pin_verts.T).T + position


def main():
    print("Generating alignment pins for all screw holes...")

    R_level = compute_leveling_rotation(OBJ_PATH)
    with open("data/pan_centroid_offset.json") as f:
        pan_offset = np.array(json.load(f)["centroid_offset_mm"])

    props = json.load(open("data/notepads/notepad_properties.json"))

    # Generate template pin
    pin_v, pin_f = generate_pin_mesh()
    print(f"  Pin template: {len(pin_v)}v, {len(pin_f)}f")
    print(f"  Head: {HEAD_TOP_R*2:.1f}mm → {HEAD_BOT_R*2:.1f}mm, {HEAD_HEIGHT:.1f}mm tall")
    print(f"  Shaft: {SHAFT_R*2:.1f}mm dia, {SHAFT_LENGTH:.1f}mm long")
    print(f"  Total length: {HEAD_HEIGHT + SHAFT_LENGTH:.1f}mm")

    total_pins = 0
    with open(OUTPUT, 'w') as out:
        out.write("# Alignment pins for screw holes\n")
        out.write("# M2 flat-head profile, body coords\n\n")
        vert_offset = 0

        for p in sorted(props, key=lambda x: x['index']):
            idx = p['index']
            normal = np.array(p['normal'])
            holes = p.get('hole_positions', [])
            if not holes:
                continue

            for hi, hp in enumerate(holes):
                hp_3d = np.array(hp)
                # Transform to body coords
                pos_body = R_level @ (hp_3d - pan_offset)
                n_body = R_level @ normal

                # Position pin: head at surface, shaft goes into body
                transformed = transform_pin(pin_v, pos_body, n_body)

                out.write(f"o Pin_{idx}_h{hi}\n")
                for v in transformed:
                    out.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
                for face in pin_f:
                    out.write(f"f {' '.join(str(vi + 1 + vert_offset) for vi in face)}\n")
                vert_offset += len(transformed)
                total_pins += 1

    print(f"\nSaved {OUTPUT}: {total_pins} pins")
    print(f"  Load alongside assembly_view.obj in Blender to verify fit")


if __name__ == "__main__":
    main()
