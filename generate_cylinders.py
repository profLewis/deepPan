#!/usr/bin/env python3
"""
Structural cylinder parameters for the inner and outer support cylinders.

These cylinders go vertically through the drum, clipped at the drum surface.
They provide structural support and divide the interior into zones.

The inner cylinder has:
  - 250mm outer diameter, 10mm wall thickness
  - Screw holes to attach to base plate
  - Keyed slot for alignment with outer cylinder

The outer cylinder has:
  - Fits snugly around the inner cylinder with tolerance
  - 10mm wall thickness, ~271mm outer diameter
  - Key protrusion at bottom for alignment
  - 30mm wiring hole through both cylinders

Both are clipped at the drum surface so they don't protrude.
"""

import math
import numpy as np

# ── Inner cylinder ──────────────────────────────────────────────
INNER_CYL_OUTER_R = 125.0       # mm (250mm outer diameter)
INNER_CYL_WALL = 10.0           # mm wall thickness
INNER_CYL_INNER_R = INNER_CYL_OUTER_R - INNER_CYL_WALL  # 115mm

# ── Outer cylinder ──────────────────────────────────────────────
CYL_TOLERANCE = 0.5              # mm gap between inner and outer
OUTER_CYL_INNER_R = INNER_CYL_OUTER_R + CYL_TOLERANCE    # 125.5mm
OUTER_CYL_WALL = 10.0           # mm
OUTER_CYL_OUTER_R = OUTER_CYL_INNER_R + OUTER_CYL_WALL   # 135.5mm

# ── Key/slot alignment ──────────────────────────────────────────
KEY_WIDTH = 10.0                 # mm circumferential width
KEY_DEPTH = CYL_TOLERANCE + 2.0 # mm radial (bridges gap + into inner cyl)
KEY_HEIGHT = 10.0                # mm vertical (at bottom of cylinders)
KEY_ANGLE = 0.0                  # radians — angular position of key

# ── Wiring hole ─────────────────────────────────────────────────
WIRE_HOLE_R = 15.0               # mm (30mm diameter)
WIRE_HOLE_ANGLE = math.pi        # opposite side from key
WIRE_HOLE_CENTER_HEIGHT = 30.0   # mm above base plate

# ── Base attachment screws ──────────────────────────────────────
INNER_CYL_SCREW_N = 8           # M3 screws for inner cylinder
OUTER_CYL_SCREW_N = 8           # M3 screws for outer cylinder
CYL_SCREW_PILOT_R = 1.25        # M3 tapped pilot radius
CYL_SCREW_CLEAR_R = 1.7         # M3 clearance in base plate
CYL_SCREW_DEPTH = 8.0           # pilot hole depth into cylinder wall
# Screw positions: at mid-wall radius of each cylinder
INNER_CYL_SCREW_R = (INNER_CYL_INNER_R + INNER_CYL_OUTER_R) / 2  # 120mm
OUTER_CYL_SCREW_R = (OUTER_CYL_INNER_R + OUTER_CYL_OUTER_R) / 2  # 130.5mm


def is_in_key_arc(angle, key_angle=KEY_ANGLE, key_width=KEY_WIDTH, radius=OUTER_CYL_INNER_R):
    """Check if an angle falls within the key arc."""
    half_arc = key_width / (2 * radius)  # angular half-width in radians
    diff = (angle - key_angle + math.pi) % (2 * math.pi) - math.pi
    return abs(diff) <= half_arc


def is_in_wire_hole(x, z, y, base_y, wire_angle=WIRE_HOLE_ANGLE,
                    wire_r=WIRE_HOLE_R, wire_height=WIRE_HOLE_CENTER_HEIGHT):
    """Check if a point (x,z,y) falls inside the wiring hole cylinder.

    The wiring hole is a horizontal cylinder at a given angle,
    passing through both cylinder walls at a given height above the base.
    """
    wire_center_y = base_y + wire_height
    # The hole axis is along the radial direction at wire_angle
    # Project the point onto the plane perpendicular to the radial direction
    cos_a = math.cos(wire_angle)
    sin_a = math.sin(wire_angle)
    # Radial component (along the hole axis)
    # radial = x * cos_a + z * sin_a  # not needed for distance calc
    # Tangential component (perpendicular to hole axis in XZ plane)
    tangential = -x * sin_a + z * cos_a
    # Vertical component
    vertical = y - wire_center_y
    # Distance from hole axis
    dist_sq = tangential * tangential + vertical * vertical
    return dist_sq <= wire_r * wire_r
