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
TAPER_HEIGHT = 0.9     # mm — cone height (90° included angle)
PLUG_HEIGHT = 2.0      # mm — cylindrical plug above taper (fits bore recess)
SHAFT_R = 1.0          # mm — shaft radius (2.0mm dia, M2)
SHAFT_LENGTH = 13.1    # mm — shaft below taper (total = plug + taper + shaft = 16mm)
SEGMENTS = 24          # angular resolution


def generate_pin_mesh():
    """Generate a single pin mesh centered at origin, pointing along +Z.

    Profile (top to bottom):
      Z=0                          → top face (plug top, flush with surface)
      Z=0 to Z=-PLUG_HEIGHT       → plug cylinder (HEAD_TOP_R)
      Z=-PLUG to Z=-PLUG-TAPER    → tapered cone (HEAD_TOP_R → HEAD_BOT_R)
      Z=-PLUG-TAPER to Z=-total   → shaft cylinder (SHAFT_R)

    Returns (vertices, faces) as numpy arrays.
    """
    S = SEGMENTS
    verts = []
    faces = []

    z_top = 0
    z_plug_bot = -PLUG_HEIGHT
    z_taper_bot = -(PLUG_HEIGHT + TAPER_HEIGHT)
    z_shaft_bot = -(PLUG_HEIGHT + TAPER_HEIGHT + SHAFT_LENGTH)

    # Top cap center
    top_center = len(verts)
    verts.append([0, 0, z_top])

    # Ring 0: plug top (HEAD_TOP_R at z_top)
    r0_start = len(verts)
    for i in range(S):
        a = 2 * math.pi * i / S
        verts.append([HEAD_TOP_R * math.cos(a), HEAD_TOP_R * math.sin(a), z_top])

    # Ring 1: plug bottom (HEAD_TOP_R at z_plug_bot)
    r1_start = len(verts)
    for i in range(S):
        a = 2 * math.pi * i / S
        verts.append([HEAD_TOP_R * math.cos(a), HEAD_TOP_R * math.sin(a), z_plug_bot])

    # Ring 2: taper bottom (HEAD_BOT_R at z_taper_bot)
    r2_start = len(verts)
    for i in range(S):
        a = 2 * math.pi * i / S
        verts.append([HEAD_BOT_R * math.cos(a), HEAD_BOT_R * math.sin(a), z_taper_bot])

    # Ring 3: shaft bottom (SHAFT_R at z_shaft_bot)
    r3_start = len(verts)
    for i in range(S):
        a = 2 * math.pi * i / S
        verts.append([SHAFT_R * math.cos(a), SHAFT_R * math.sin(a), z_shaft_bot])

    # Bottom cap center
    bot_center = len(verts)
    verts.append([0, 0, z_shaft_bot])

    # Faces
    for i in range(S):
        j = (i + 1) % S
        # Top cap fan
        faces.append([top_center, r0_start + i, r0_start + j])
        # Plug cylinder side (r0 → r1)
        faces.append([r0_start + i, r1_start + i, r1_start + j])
        faces.append([r0_start + i, r1_start + j, r0_start + j])
        # Taper cone side (r1 → r2)
        faces.append([r1_start + i, r2_start + i, r2_start + j])
        faces.append([r1_start + i, r2_start + j, r1_start + j])
        # Shaft cylinder side (r2 → r3)
        faces.append([r2_start + i, r3_start + i, r3_start + j])
        faces.append([r2_start + i, r3_start + j, r2_start + j])
        # Bottom cap fan (reversed)
        faces.append([bot_center, r3_start + j, r3_start + i])

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
    total_len = PLUG_HEIGHT + TAPER_HEIGHT + SHAFT_LENGTH
    print(f"  Pin template: {len(pin_v)}v, {len(pin_f)}f")
    print(f"  Plug: {HEAD_TOP_R*2:.1f}mm dia, {PLUG_HEIGHT:.1f}mm tall")
    print(f"  Taper: {HEAD_TOP_R*2:.1f}mm → {HEAD_BOT_R*2:.1f}mm, {TAPER_HEIGHT:.1f}mm")
    print(f"  Shaft: {SHAFT_R*2:.1f}mm dia, {SHAFT_LENGTH:.1f}mm long")
    print(f"  Total length: {total_len:.1f}mm")

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
