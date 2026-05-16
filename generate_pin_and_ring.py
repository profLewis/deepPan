#!/usr/bin/env python3
"""
Generate two separate hardware parts:

  1. dowel_pin.stl — 4mm Ø × 18mm long cylinder.
     Print 6 copies; one slips into each of the 6 R=110 radial holes
     drilled by add_assembly_holes.py to align central piece to outer
     pieces during assembly.

  2. locking_ring.stl — flat annular plate, OD 130mm, ID 95mm, thickness 4mm.
     12 M4 countersunk through-holes (6 at R=100, 6 at R=120, all at
     sector midpoint angles 74.5/108.5/167/227.5/259.5/340°).
     Screws from BELOW into matching thread bosses on the central piece's
     and outer pieces' undersides. Provides clamping force at the R=110
     joints (the dowel pins only handle alignment).

Outputs:
    pipeline_output/dowel_pin.{obj,stl}
    pipeline_output/locking_ring.{obj,stl}
    pipeline_output/locking_ring_screw_pattern.json
"""
from pathlib import Path
import json
import math
import numpy as np
import trimesh


OUT_DIR = Path("pipeline_output")
OUT_DIR.mkdir(exist_ok=True)

PIN_DIAMETER = 4.0          # mm
PIN_LENGTH = 18.0           # mm (slightly shorter than 20mm hole for clearance)

RING_OD = 130.0             # mm outer diameter
RING_ID = 95.0              # mm inner diameter
RING_THICKNESS = 4.0        # mm

# 12 M4 mounting holes — 6 inner (R=100) and 6 outer (R=120) at sector midpoints
SECTOR_MIDPOINT_ANGLES_DEG = [74.5, 108.5, 167.0, 227.5, 259.5, 340.0]
INNER_HOLE_R = 100.0        # mounts to central-piece bosses
OUTER_HOLE_R = 120.0        # mounts to outer-piece bosses
M4_CLEAR_R = 2.3            # 4.6mm Ø clearance
M4_COUNTER_R = 4.2          # 8.4mm Ø countersink (for M4 head)
M4_COUNTER_DEPTH = 2.5      # mm

# ─────────────────────────────────────────────────────────────────────────
# 1. Dowel pin
# ─────────────────────────────────────────────────────────────────────────
print(f"Generating dowel pin: Ø {PIN_DIAMETER}mm × {PIN_LENGTH}mm long")
pin = trimesh.creation.cylinder(
    radius=PIN_DIAMETER / 2,
    height=PIN_LENGTH,
    sections=32,
)
pin.export(OUT_DIR / "dowel_pin.obj")
pin.export(OUT_DIR / "dowel_pin.stl")
print(f"  Wrote pipeline_output/dowel_pin.{{obj,stl}}  "
      f"({len(pin.vertices)} verts, {len(pin.faces)} faces)")

# ─────────────────────────────────────────────────────────────────────────
# 2. Locking ring
# ─────────────────────────────────────────────────────────────────────────
print(f"\nGenerating locking ring: OD {RING_OD} ID {RING_ID} × {RING_THICKNESS}mm")

# Build the ring as outer cylinder minus inner cylinder
outer = trimesh.creation.cylinder(radius=RING_OD/2, height=RING_THICKNESS, sections=128)
inner = trimesh.creation.cylinder(radius=RING_ID/2, height=RING_THICKNESS + 2, sections=128)
ring = outer.difference(inner)
print(f"  Base ring: {len(ring.vertices)} verts, {len(ring.faces)} faces")

# Drill 12 M4 holes (with countersinks on top face)
holes_pattern = []
for ang_deg in SECTOR_MIDPOINT_ANGLES_DEG:
    for R in [INNER_HOLE_R, OUTER_HOLE_R]:
        a = math.radians(ang_deg)
        x = R * math.cos(a)
        y = R * math.sin(a)
        holes_pattern.append({
            "x": x, "y": y,
            "R": R,
            "angle_deg": ang_deg,
            "engages": "central" if R == INNER_HOLE_R else "outer",
        })

print(f"  Drilling {len(holes_pattern)} M4 holes (with countersinks)...")
for h in holes_pattern:
    # Through-hole
    bore = trimesh.creation.cylinder(
        radius=M4_CLEAR_R, height=RING_THICKNESS + 1, sections=24,
    )
    bore.apply_translation([h["x"], h["y"], 0])
    ring = ring.difference(bore)
    # Countersink on top face (z = +THICKNESS/2)
    cs = trimesh.creation.cylinder(
        radius=M4_COUNTER_R, height=M4_COUNTER_DEPTH + 0.5, sections=24,
    )
    cs.apply_translation([h["x"], h["y"],
                          RING_THICKNESS / 2 - M4_COUNTER_DEPTH / 2 + 0.25])
    ring = ring.difference(cs)

print(f"  Ring after drilling: {len(ring.vertices)} verts, {len(ring.faces)} faces")

ring.export(OUT_DIR / "locking_ring.obj")
ring.export(OUT_DIR / "locking_ring.stl")
print(f"  Wrote pipeline_output/locking_ring.{{obj,stl}}")

# Write screw pattern for downstream tooling (drilling matching boss holes
# on the assembled pan model).
pattern = {
    "ring_od": RING_OD,
    "ring_id": RING_ID,
    "ring_thickness": RING_THICKNESS,
    "sector_angles_deg": SECTOR_MIDPOINT_ANGLES_DEG,
    "inner_R": INNER_HOLE_R,
    "outer_R": OUTER_HOLE_R,
    "m4_clear_r": M4_CLEAR_R,
    "m4_counter_r": M4_COUNTER_R,
    "m4_counter_depth": M4_COUNTER_DEPTH,
    "holes": holes_pattern,
}
with open(OUT_DIR / "locking_ring_screw_pattern.json", "w") as f:
    json.dump(pattern, f, indent=2)
print(f"  Wrote pipeline_output/locking_ring_screw_pattern.json")
